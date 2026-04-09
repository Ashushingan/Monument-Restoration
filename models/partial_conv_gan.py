"""
WEEKS 7-9: Partial Convolution GAN — Monument Reconstruction
=============================================================
Module 3 — Reconstructs damaged monument regions using:
  • Partial Convolutions (handle irregular masks better than regular conv)
  • Style conditioning from Module 1 (ResNet embeddings)
  • Multi-scale PatchGAN Discriminator
  • Combined loss: reconstruction + perceptual (VGG) + style + adversarial

Reference: Liu et al., "Image Inpainting for Irregular Holes Using
           Partial Convolutions", ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.dataset import get_inpainting_loaders


# ══════════════════════════════════════════════════════════════
#  1. PARTIAL CONVOLUTION LAYER
# ══════════════════════════════════════════════════════════════
class PartialConv2d(nn.Module):
    """
    Partial Convolution: only uses unmasked (valid) pixels for convolution.
    After each forward pass, updates the mask automatically.

    mask convention: 1 = valid (intact), 0 = hole (damaged)
    Note: This is the INVERSE of our damage mask (where 1=damaged).
          Convert before passing: valid_mask = 1 - damage_mask
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        # Mask updating conv — fixed weights = 1, no bias
        self.mask_conv = nn.Conv2d(
            1, 1, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        self.slide_winsize = in_channels * kernel_size * kernel_size

    def forward(self, x, mask):
        """
        x:    [B, C, H, W]  — feature map (masked regions can be anything)
        mask: [B, 1+, H, W] — 1=valid, 0=hole (may have >1 ch after skip-concat)
        returns: (output, updated_mask)
        """
        # Always collapse to single channel — all channels carry same spatial validity
        mask_1ch = mask[:, :1, :, :]          # [B, 1, H, W]

        # Normalisation factor: how many valid pixels in each kernel window
        with torch.no_grad():
            mask_sum = self.mask_conv(mask_1ch)          # [B, 1, H', W']
            # Replace 0 with 1 to avoid division by zero
            mask_ratio = self.slide_winsize / (mask_sum + 1e-8)
            mask_ratio = mask_ratio * (mask_sum > 0).float()
            updated_mask = torch.clamp(mask_sum, 0, 1)  # binary {0, 1}

        # Feature conv: zero out holes first (broadcast mask across all feature channels)
        x_masked = x * mask_1ch.expand_as(x)
        out = self.feature_conv(x_masked)
        out = out * mask_ratio  # re-scale by valid pixel ratio

        return out, updated_mask


# ══════════════════════════════════════════════════════════════
#  2. GENERATOR BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════
class PConvBNActiv(nn.Module):
    """PartialConv → BatchNorm → Activation block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 activ="relu", bn=True):
        super().__init__()
        self.pconv = PartialConv2d(in_ch, out_ch, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        if activ == "relu":
            self.activ = nn.ReLU(inplace=True)
        elif activ == "leaky":
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        elif activ == "none":
            self.activ = nn.Identity()
        else:
            self.activ = nn.Tanh()

    def forward(self, x, mask):
        out, updated_mask = self.pconv(x, mask)
        out = self.bn(out)
        out = self.activ(out)
        return out, updated_mask


class StyleConditioningBlock(nn.Module):
    """
    Injects style embedding from the classifier into the generator
    via Adaptive Instance Normalisation (AdaIN).
    """
    def __init__(self, feature_channels, style_dim=2048):
        super().__init__()
        self.norm = nn.InstanceNorm2d(feature_channels, affine=False)
        self.style_scale = nn.Linear(style_dim, feature_channels)
        self.style_bias  = nn.Linear(style_dim, feature_channels)
        nn.init.ones_(self.style_scale.weight)
        nn.init.zeros_(self.style_bias.weight)

    def forward(self, x, style_emb):
        """
        x:         [B, C, H, W]
        style_emb: [B, 2048]
        """
        x_norm = self.norm(x)
        scale = self.style_scale(style_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        bias  = self.style_bias(style_emb).unsqueeze(-1).unsqueeze(-1)
        return x_norm * scale + bias


# ══════════════════════════════════════════════════════════════
#  3. GENERATOR (Partial Conv U-Net with Style Conditioning)
# ══════════════════════════════════════════════════════════════
class PartialConvGenerator(nn.Module):
    """
    Encoder-Decoder U-Net using Partial Convolutions.
    Style embedding injected at bottleneck via AdaIN.

    Input:
        x          — [B, 3, 512, 512] damaged image (holes zeroed out)
        mask       — [B, 1, 512, 512] valid mask  (1=intact, 0=hole)
        style_emb  — [B, 2048] from style classifier (optional)
    Output:
        [B, 3, 512, 512] completed image in range [-1, 1]
    """
    def __init__(self, style_dim=2048):
        super().__init__()

        # ── ENCODER ──────────────────────────────────────────────
        # Each level halves spatial resolution
        self.enc1 = PConvBNActiv(3,   64,  7, stride=2, padding=3, bn=False, activ="relu")
        self.enc2 = PConvBNActiv(64,  128, 5, stride=2, padding=2, activ="relu")
        self.enc3 = PConvBNActiv(128, 256, 5, stride=2, padding=2, activ="relu")
        self.enc4 = PConvBNActiv(256, 512, 3, stride=2, padding=1, activ="relu")
        self.enc5 = PConvBNActiv(512, 512, 3, stride=2, padding=1, activ="relu")
        self.enc6 = PConvBNActiv(512, 512, 3, stride=2, padding=1, activ="relu")

        # ── BOTTLENECK STYLE CONDITIONING ───────────────────────
        self.style_cond = StyleConditioningBlock(512, style_dim)

        # ── DECODER ──────────────────────────────────────────────
        # Each level doubles spatial resolution; skip connections from encoder
        # In channels = current + skip (concatenated)
        self.dec6 = PConvBNActiv(512 + 512, 512, 3, padding=1, activ="leaky")
        self.dec5 = PConvBNActiv(512 + 512, 512, 3, padding=1, activ="leaky")
        self.dec4 = PConvBNActiv(512 + 256, 256, 3, padding=1, activ="leaky")
        self.dec3 = PConvBNActiv(256 + 128, 128, 3, padding=1, activ="leaky")
        self.dec2 = PConvBNActiv(128 + 64,   64, 3, padding=1, activ="leaky")
        self.dec1 = PConvBNActiv(64  +  3,    3, 3, padding=1, activ="tanh", bn=False)

    def _upsample(self, x, mask, target_h, target_w):
        """Bilinear upsample for features; nearest for mask (keep binary)."""
        x    = F.interpolate(x,    size=(target_h, target_w), mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
        return x, mask

    def forward(self, x, mask, style_emb=None):
        # Prepare valid mask (1=intact, 0=hole) from damage mask (1=hole)
        valid_mask = 1.0 - mask  # [B, 1, H, W]

        # ── Encode ───────────────────────────────────────────────
        e1, m1 = self.enc1(x,  valid_mask)    # 256x256
        e2, m2 = self.enc2(e1, m1)            # 128x128
        e3, m3 = self.enc3(e2, m2)            # 64x64
        e4, m4 = self.enc4(e3, m3)            # 32x32
        e5, m5 = self.enc5(e4, m4)            # 16x16
        e6, m6 = self.enc6(e5, m5)            # 8x8

        # ── Style Conditioning at bottleneck ─────────────────────
        bottleneck = e6
        if style_emb is not None:
            bottleneck = self.style_cond(e6, style_emb)

        # ── Decode ───────────────────────────────────────────────
        # Level 6 → 5
        d, dm = self._upsample(bottleneck, m6, e5.shape[2], e5.shape[3])
        d, dm = self.dec6(torch.cat([d, e5], dim=1), torch.cat([dm, m5], dim=1))

        # Level 5 → 4
        d, dm = self._upsample(d, dm, e4.shape[2], e4.shape[3])
        d, dm = self.dec5(torch.cat([d, e4], dim=1), torch.cat([dm, m4], dim=1))

        # Level 4 → 3
        d, dm = self._upsample(d, dm, e3.shape[2], e3.shape[3])
        d, dm = self.dec4(torch.cat([d, e3], dim=1), torch.cat([dm, m3], dim=1))

        # Level 3 → 2
        d, dm = self._upsample(d, dm, e2.shape[2], e2.shape[3])
        d, dm = self.dec3(torch.cat([d, e2], dim=1), torch.cat([dm, m2], dim=1))

        # Level 2 → 1
        d, dm = self._upsample(d, dm, e1.shape[2], e1.shape[3])
        d, dm = self.dec2(torch.cat([d, e1], dim=1), torch.cat([dm, m1], dim=1))

        # Final output at original resolution
        d, dm = self._upsample(d, dm, x.shape[2], x.shape[3])
        out, _ = self.dec1(torch.cat([d, x], dim=1), torch.cat([dm, valid_mask], dim=1))

        # Composite: keep original pixels in intact regions, use generated in holes
        restored = x * (1 - mask) + out * mask
        return restored


# ══════════════════════════════════════════════════════════════
#  4. DISCRIMINATOR (Multi-scale PatchGAN)
# ══════════════════════════════════════════════════════════════
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator — classifies overlapping patches as real/fake.
    More stable than global discriminator for inpainting tasks.
    """
    def __init__(self, in_channels=3, base_filters=64, n_layers=4):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        filters = base_filters
        for _ in range(1, n_layers):
            next_filters = min(filters * 2, 512)
            layers += [
                nn.Conv2d(filters, next_filters, 4, stride=2, padding=1),
                nn.InstanceNorm2d(next_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            filters = next_filters
        layers += [nn.Conv2d(filters, 1, 4, padding=1)]  # patch output
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Two PatchGAN discriminators operating at different scales.
    Catches both fine texture and coarse structure artefacts.
    """
    def __init__(self):
        super().__init__()
        self.d1 = PatchDiscriminator(in_channels=3, n_layers=4)  # full resolution
        self.d2 = PatchDiscriminator(in_channels=3, n_layers=3)  # 2x downscaled
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.d1(x), self.d2(self.downsample(x))


# ══════════════════════════════════════════════════════════════
#  5. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual + style loss using VGG-16 feature maps.
    Ensures realistic texture rather than blurry reconstruction.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Use relu1_2, relu2_2, relu3_3, relu4_3 feature layers
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.features)[:4]),    # relu1_2
            nn.Sequential(*list(vgg.features)[4:9]),   # relu2_2
            nn.Sequential(*list(vgg.features)[9:16]),  # relu3_3
            nn.Sequential(*list(vgg.features)[16:23]), # relu4_3
        ])
        for slice_ in self.slices:
            for p in slice_.parameters():
                p.requires_grad = False

    def _gram(self, x):
        """Gram matrix for style loss."""
        b, c, h, w = x.shape
        f = x.view(b, c, -1)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def forward(self, pred, target, mask):
        """
        pred, target: [B, 3, H, W] in [-1, 1]
        mask:         [B, 1, H, W] — 1=hole
        """
        # Remap to [0, 1] for VGG
        pred_01   = (pred   + 1) / 2
        target_01 = (target + 1) / 2

        perc_loss  = 0.0
        style_loss = 0.0
        hole_loss  = 0.0

        x, y = pred_01, target_01
        for slice_ in self.slices:
            x = slice_(x)
            y = slice_(y)

            # Perceptual: feature map MSE
            perc_loss += F.l1_loss(x, y)

            # Style: gram matrix difference
            style_loss += F.l1_loss(self._gram(x), self._gram(y))

        # Hole reconstruction loss (only on damaged region)
        hole_loss = F.l1_loss(pred_01 * mask, target_01 * mask)

        # Valid region loss (should be perfectly preserved)
        valid_loss = F.l1_loss(pred_01 * (1 - mask), target_01 * (1 - mask))

        return perc_loss, style_loss, hole_loss, valid_loss


class GANLoss(nn.Module):
    """LSGAN loss (mean squared error) — more stable than vanilla BCE."""
    def discriminator_loss(self, real_preds, fake_preds):
        """Train discriminator to output 1 for real, 0 for fake."""
        loss = 0
        for rp, fp in zip(real_preds, fake_preds):
            loss += F.mse_loss(rp, torch.ones_like(rp))
            loss += F.mse_loss(fp, torch.zeros_like(fp))
        return loss / len(real_preds)

    def generator_loss(self, fake_preds):
        """Train generator to fool discriminator (fake → 1)."""
        loss = 0
        for fp in fake_preds:
            loss += F.mse_loss(fp, torch.ones_like(fp))
        return loss / len(fake_preds)


# ══════════════════════════════════════════════════════════════
#  6. GAN TRAINER
# ══════════════════════════════════════════════════════════════
class PartialConvGANTrainer:
    """
    Full training loop for the reconstruction GAN.

    Loss weights (tunable):
        λ_valid = 1.0   — L1 on intact regions (should be preserved perfectly)
        λ_hole  = 6.0   — L1 on damaged regions (main task)
        λ_perc  = 0.05  — perceptual (VGG feature maps)
        λ_style = 120.0 — style (gram matrix)
        λ_adv   = 0.1   — adversarial (GAN)
    """

    LOSS_WEIGHTS = {
        "valid": 1.0,
        "hole":  6.0,
        "perc":  0.05,
        "style": 120.0,
        "adv":   0.1,
    }

    def __init__(self, data_dir, batch_size=4, lr_g=1e-4, lr_d=4e-4,
                 style_dim=2048, save_dir="checkpoints/gan"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[PartialConvGAN] Using device: {self.device}")

        # Models
        self.G = PartialConvGenerator(style_dim=style_dim).to(self.device)
        self.D = MultiScaleDiscriminator().to(self.device)

        # Data
        self.train_loader, self.val_loader = get_inpainting_loaders(data_dir, batch_size)

        # Optimisers — separate for G and D
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

        # LR decay: halve every 50 epochs after initial 100
        self.sched_G = optim.lr_scheduler.StepLR(self.opt_G, step_size=50, gamma=0.5)
        self.sched_D = optim.lr_scheduler.StepLR(self.opt_D, step_size=50, gamma=0.5)

        # Losses
        self.perc_loss_fn = VGGPerceptualLoss().to(self.device)
        self.gan_loss_fn  = GANLoss()

        self.history = {k: [] for k in ["g_total", "g_hole", "g_perc", "g_style", "g_adv", "d_loss"]}
        self.best_g_loss = float("inf")

    def _prepare_inputs(self, batch):
        """Extract and move tensors to device."""
        clean   = batch["clean"].to(self.device)    # ground truth  [B,3,H,W]
        damaged = batch["damaged"].to(self.device)  # model input   [B,3,H,W]
        mask    = batch["mask"].to(self.device)      # damage mask   [B,1,H,W]
        return clean, damaged, mask

    def _train_discriminator(self, clean, restored, mask):
        self.opt_D.zero_grad()
        real_preds = self.D(clean)
        fake_preds = self.D(restored.detach())
        d_loss = self.gan_loss_fn.discriminator_loss(real_preds, fake_preds)
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
        self.opt_D.step()
        return d_loss.item()

    def _train_generator(self, clean, damaged, mask, restored, style_emb=None):
        self.opt_G.zero_grad()

        # Perceptual losses
        perc, style, hole, valid = self.perc_loss_fn(restored, clean, mask)

        # Adversarial loss
        fake_preds = self.D(restored)
        adv = self.gan_loss_fn.generator_loss(fake_preds)

        # Combined weighted loss
        w = self.LOSS_WEIGHTS
        g_loss = (
            w["valid"] * valid +
            w["hole"]  * hole  +
            w["perc"]  * perc  +
            w["style"] * style +
            w["adv"]   * adv
        )

        g_loss.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
        self.opt_G.step()

        return g_loss.item(), hole.item(), perc.item(), style.item(), adv.item()

    def train(self, epochs=100):
        print(f"\n{'='*55}")
        print("Training Partial Convolution GAN (Monument Reconstruction)")
        print(f"Loss weights: {self.LOSS_WEIGHTS}")
        print(f"{'='*55}\n")

        for epoch in range(1, epochs + 1):
            self.G.train()
            self.D.train()

            epoch_stats = {k: 0.0 for k in self.history}
            n_batches = 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch:03d}", leave=False):
                clean, damaged, mask = self._prepare_inputs(batch)

                # Forward: generate restored image
                # (no style embedding here unless classifier is also loaded)
                restored = self.G(damaged, mask, style_emb=None)

                # ─ Train D ─
                d_loss = self._train_discriminator(clean, restored, mask)

                # ─ Train G (2× more than D for stability) ─
                g_total, g_hole, g_perc, g_style, g_adv = self._train_generator(
                    clean, damaged, mask, restored
                )
                g_total2, _, _, _, _ = self._train_generator(
                    clean, damaged, mask, self.G(damaged, mask)
                )
                g_total = (g_total + g_total2) / 2

                epoch_stats["g_total"] += g_total
                epoch_stats["g_hole"]  += g_hole
                epoch_stats["g_perc"]  += g_perc
                epoch_stats["g_style"] += g_style
                epoch_stats["g_adv"]   += g_adv
                epoch_stats["d_loss"]  += d_loss
                n_batches += 1

            # Average stats
            for k in epoch_stats:
                epoch_stats[k] /= n_batches
                self.history[k].append(epoch_stats[k])

            self.sched_G.step()
            self.sched_D.step()

            print(
                f"Epoch [{epoch:03d}/{epochs}] | "
                f"G: {epoch_stats['g_total']:.4f} "
                f"(hole={epoch_stats['g_hole']:.3f}, "
                f"perc={epoch_stats['g_perc']:.4f}, "
                f"style={epoch_stats['g_style']:.2f}, "
                f"adv={epoch_stats['g_adv']:.4f}) | "
                f"D: {epoch_stats['d_loss']:.4f}"
            )

            # Save best generator
            if epoch_stats["g_total"] < self.best_g_loss:
                self.best_g_loss = epoch_stats["g_total"]
                torch.save({
                    "epoch": epoch,
                    "G_state": self.G.state_dict(),
                    "D_state": self.D.state_dict(),
                    "g_loss": epoch_stats["g_total"],
                }, self.save_dir / "best_gan.pth")
                print(f"  ✓ Saved best GAN checkpoint")

            # Save samples every 10 epochs
            if epoch % 10 == 0:
                self._save_samples(epoch)

        self._plot_history()
        print(f"\nBest Generator Loss: {self.best_g_loss:.4f}")

    def _save_samples(self, epoch, n=4):
        """Save reconstruction samples for visual inspection."""
        self.G.eval()
        batch = next(iter(self.val_loader))
        clean, damaged, mask = self._prepare_inputs(batch)

        with torch.no_grad():
            restored = self.G(damaged, mask)

        def denorm(t):
            return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()

        rows = []
        for i in range(min(n, clean.shape[0])):
            d_img = denorm(damaged[i])
            m_img = (mask[i, 0].cpu().numpy() * 255).astype(np.uint8)
            m_rgb = cv2.cvtColor(m_img, cv2.COLOR_GRAY2RGB)
            r_img = denorm(restored[i])
            c_img = denorm(clean[i])
            row   = np.hstack([d_img, m_rgb, r_img, c_img])
            rows.append(row)

        grid = np.vstack(rows)
        out_path = self.save_dir / f"samples_epoch_{epoch:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved samples → {out_path}")

    def _plot_history(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        keys_labels = [
            ("g_total", "G Total Loss"),
            ("g_hole",  "G Hole Loss"),
            ("g_perc",  "G Perceptual Loss"),
            ("g_style", "G Style Loss"),
            ("g_adv",   "G Adversarial Loss"),
            ("d_loss",  "D Loss"),
        ]
        for ax, (key, label) in zip(axes.flat, keys_labels):
            ax.plot(self.history[key])
            ax.set_title(label)
            ax.set_xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(self.save_dir / "gan_training_curves.png", dpi=120)
        plt.close()
        print(f"Saved GAN training curves → {self.save_dir}/gan_training_curves.png")

    def load_best(self):
        ckpt = torch.load(self.save_dir / "best_gan.pth", map_location=self.device)
        self.G.load_state_dict(ckpt["G_state"])
        self.D.load_state_dict(ckpt["D_state"])
        print(f"Loaded best GAN (epoch {ckpt['epoch']}, G loss={ckpt['g_loss']:.4f})")
        return self.G

    def restore(self, image_bgr: np.ndarray, mask_gray: np.ndarray,
                style_emb=None):
        """
        Run inference to restore a single image.
        image_bgr:  OpenCV BGR image
        mask_gray:  uint8 mask [H, W], 255=damaged, 0=intact
        style_emb:  optional [1, 2048] tensor from style classifier
        Returns:    restored BGR image (same size as input)
        """
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)

        mask_resized = cv2.resize(mask_gray, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_tensor  = torch.from_numpy(
            (mask_resized > 127).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 512, 512]

        self.G.eval()
        with torch.no_grad():
            restored = self.G(tensor, mask_tensor, style_emb)

        # Convert back to BGR image
        restored_np = ((restored[0].clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
        restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
        restored_bgr = cv2.resize(restored_bgr, (orig_w, orig_h))
        return restored_bgr


# ══════════════════════════════════════════════════════════════
#  7. ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    trainer = PartialConvGANTrainer(
        data_dir="data/raw",
        batch_size=4,
        lr_g=1e-4,
        lr_d=4e-4,
        epochs=100,
    )
    trainer.train(epochs=100)

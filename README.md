# Digital Restoration of Damaged Monuments
## Walchand College of Engineering, Sangli — Mini Project

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up data folders + auto-download images
python setup_data.py
pip install icrawler
python download_images.py   # auto-created by setup_data.py

# 3. Train all modules
python train_all.py --stage all

# 4. Launch the app
pip install streamlit
streamlit run app.py
```

---

## How the App Works

1. **Upload** any damaged monument photograph
2. **U-Net** automatically detects damaged/missing regions (no manual masking)
3. **Partial Conv GAN** reconstructs what the original structure would have looked like
4. **Style Classifier** identifies the architectural style (Dravidian, Nagara, etc.)
5. **Download** the restored image

---

## Project Structure

```
monument_restoration/
├── data/
│   └── dataset.py           ← Weeks 1-2: Data loading + synthetic damage
├── models/
│   ├── style_classifier.py  ← Weeks 3-4: ResNet-50 style classifier
│   ├── damage_segmentation.py ← Weeks 5-6: U-Net damage detector
│   └── partial_conv_gan.py  ← Weeks 7-9: Partial Conv GAN reconstruction
├── utils/
│   └── metrics.py           ← SSIM, PSNR, visualisation helpers
├── checkpoints/             ← Saved model weights (auto-created)
├── app.py                   ← Streamlit UI (fully automatic pipeline)
├── train_all.py             ← Master training script
├── setup_data.py            ← Data folder scaffolding + download guide
└── requirements.txt
```

---

## Training

```bash
# All stages in sequence (recommended)
python train_all.py --stage all

# Individual stages
python train_all.py --stage classifier --cls_epochs 30
python train_all.py --stage segmenter  --seg_epochs 40
python train_all.py --stage gan        --gan_epochs 100

# Low VRAM (< 6 GB GPU) — already the default
python train_all.py --stage classifier --batch_cls 8 --image_size 224 --grad_accum 4
python train_all.py --stage segmenter  --batch_seg 4
python train_all.py --stage gan        --batch_gan 2
```

---

## Data Preparation

```
data/raw/
├── Dravidian/      ← ~50 images per style class
├── Nagara/
├── Vesara/
├── Islamic/
├── Buddhist/
├── Colonial/
├── Rock_Cut/
├── Stepwell/
└── all_monuments/  ← All images combined (for segmenter + GAN)
```

Run `python setup_data.py` for full instructions and auto-download script.

---

## Target Metrics

| Module              | Metric          | Target   |
|---------------------|-----------------|----------|
| Style Classifier    | Top-1 Accuracy  | > 80%    |
| Damage Segmenter    | IoU             | > 0.70   |
| GAN Reconstruction  | SSIM (hole)     | > 0.75   |
| GAN Reconstruction  | PSNR (hole)     | > 25 dB  |

---

## Bugs Fixed During Development

| Error | Fix |
|-------|-----|
| `num_samples=0` — empty dataset | Added clear error + `setup_data.py` |
| `GaussNoise var_limit` deprecated | Changed to `std_range=(0.02, 0.1)` |
| CUDA OOM at batch=32, 512px | Reduced to batch=8, 224px + gradient accumulation |
| `BatchNorm1d` batch=1 crash | Added `drop_last=True` to train DataLoader |
| `ReduceLROnPlateau verbose=True` | Removed deprecated argument |
| `mask_conv` expects 1ch, got 2ch | Slice `mask[:, :1, ...]` in `PartialConv2d.forward` |

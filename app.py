"""
app.py — Monument Digital Restoration
======================================
Fully automatic pipeline:
  1. Upload a damaged monument photo
  2. U-Net auto-detects damaged regions
  3. Partial Conv GAN reconstructs missing structure
  4. Style classifier labels the architecture

Run:  streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="Monument Restoration",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0e0e0e; }
.title-block { padding: 2.5rem 0 1rem 0; text-align: center; }
.title-block h1 { font-family: 'Cormorant Garamond', serif; font-size: 3rem; font-weight: 700; color: #e8dcc8; letter-spacing: 0.05em; margin: 0; line-height: 1.1; }
.title-block p { color: #7a7060; font-size: 0.85rem; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.5rem; }
.step-pill { display: inline-block; background: #1e1a16; border: 1px solid #3a3530; border-radius: 20px; padding: 0.25rem 0.9rem; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #c9a96e; margin-bottom: 0.6rem; }
.result-label { font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; color: #6a6258; margin-bottom: 0.4rem; }
.style-badge { background: linear-gradient(135deg, #2a2015, #1a1510); border: 1px solid #c9a96e44; border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center; }
.style-badge .style-name { font-family: 'Cormorant Garamond', serif; font-size: 1.8rem; color: #e8dcc8; font-weight: 600; }
.style-badge .confidence { color: #c9a96e; font-size: 0.85rem; margin-top: 0.2rem; }
.metric-row { display: flex; gap: 0.8rem; margin-top: 0.8rem; }
.metric-box { flex: 1; background: #161410; border: 1px solid #2a2520; border-radius: 8px; padding: 0.8rem; text-align: center; }
.metric-box .val { font-family: 'Cormorant Garamond', serif; font-size: 1.4rem; color: #c9a96e; font-weight: 600; }
.metric-box .lbl { font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; color: #5a5248; margin-top: 2px; }
.prob-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
.prob-name { color: #9a9088; font-size: 0.78rem; width: 90px; flex-shrink: 0; }
.prob-track { flex: 1; height: 6px; background: #2a2520; border-radius: 3px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 3px; background: #c9a96e; }
.prob-fill.top { background: linear-gradient(90deg, #c9a96e, #e8c87e); }
.prob-pct { color: #6a6258; font-size: 0.72rem; width: 36px; text-align: right; flex-shrink: 0; }
.damage-stat { background: #161410; border: 1px solid #2a2520; border-radius: 8px; padding: 0.8rem 1rem; margin-top: 0.6rem; display: flex; justify-content: space-between; align-items: center; }
.damage-stat span { color: #6a6258; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
.damage-stat strong { color: #c9a96e; font-size: 0.95rem; }
.divider { border: none; border-top: 1px solid #2a2520; margin: 1.5rem 0; }
.ckpt-status { background: #161410; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; border: 1px solid #2a2520; font-size: 0.8rem; color: #7a7060; }
.ckpt-status .row { display:flex; justify-content:space-between; padding: 3px 0; }
.ok { color: #6abf6a; } .missing { color: #bf6a6a; }
.stButton > button { background: linear-gradient(135deg, #2a2015, #1e1a12) !important; color: #c9a96e !important; border: 1px solid #c9a96e55 !important; border-radius: 8px !important; font-size: 0.82rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; padding: 0.6rem 1.5rem !important; width: 100% !important; }
.stButton > button:hover { background: linear-gradient(135deg, #3a3020, #2a2418) !important; border-color: #c9a96e99 !important; }
div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = {"device": device, "classifier": None, "segmenter": None, "gan": None,
         "style_classes": None}

    cls_path = Path("checkpoints/style/best_style_classifier.pth")
    if cls_path.exists():
        try:
            from models.style_classifier import ArchitecturalStyleClassifier
            from data.dataset import STYLE_CLASSES
            m = ArchitecturalStyleClassifier(num_classes=len(STYLE_CLASSES)).to(device)
            ckpt = torch.load(cls_path, map_location=device, weights_only=False)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            M["classifier"] = m
            M["style_classes"] = STYLE_CLASSES
            M["cls_epoch"] = ckpt.get("epoch", "?")
            M["cls_acc"]   = ckpt.get("val_acc", 0)
        except Exception as e:
            M["cls_error"] = str(e)

    seg_path = Path("checkpoints/segmentation/best_damage_segmenter.pth")
    if seg_path.exists():
        try:
            from models.damage_segmentation import DamageSegmentationModel
            m = DamageSegmentationModel().to(device)
            ckpt = torch.load(seg_path, map_location=device, weights_only=False)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            M["segmenter"] = m
            M["seg_iou"] = ckpt.get("val_iou", 0)
        except Exception as e:
            M["seg_error"] = str(e)

    gan_path = Path("checkpoints/gan/best_gan.pth")
    if gan_path.exists():
        try:
            from models.partial_conv_gan import PartialConvGenerator, MultiScaleDiscriminator
            G = PartialConvGenerator().to(device)
            D = MultiScaleDiscriminator().to(device)
            ckpt = torch.load(gan_path, map_location=device, weights_only=False)
            G.load_state_dict(ckpt["G_state"])
            D.load_state_dict(ckpt["D_state"])
            G.eval()
            M["gan"] = G
            M["gan_epoch"] = ckpt.get("epoch", "?")
            M["gan_gloss"] = ckpt.get("g_loss", 0)
        except Exception as e:
            M["gan_error"] = str(e)

    return M


def auto_segment(image_bgr, segmenter, device, threshold=0.45):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])
    h, w = image_bgr.shape[:2]
    tensor = transform(image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        mask = (torch.sigmoid(segmenter(tensor)) > threshold).float()
    mask_np = cv2.resize(mask[0,0].cpu().numpy(), (w,h), interpolation=cv2.INTER_NEAREST)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = (mask_np * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k2)
    return m


def classify_style(image_bgr, classifier, device, style_classes):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    tensor = transform(image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)[0].cpu().numpy()
        emb   = classifier.get_embedding(tensor)
    return probs, emb


def gan_restore(image_bgr, mask_gray, G, device, style_emb=None):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])
    orig_h, orig_w = image_bgr.shape[:2]
    tensor = transform(image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(device)
    mask_t = torch.from_numpy(
        (cv2.resize(mask_gray,(512,512),interpolation=cv2.INTER_NEAREST)>127).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = G(tensor, mask_t, style_emb)
    out_np = ((out[0].clamp(-1,1)+1)/2*255).byte().permute(1,2,0).cpu().numpy()
    return cv2.resize(cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR), (orig_w, orig_h))


def make_overlay(image_bgr, mask_gray):
    ov = image_bgr.copy(); ov[mask_gray>127] = [40,40,220]
    return cv2.cvtColor(cv2.addWeighted(image_bgr,0.55,ov,0.45,0), cv2.COLOR_BGR2RGB)

def pil_to_bgr(p): return cv2.cvtColor(np.array(p.convert("RGB")), cv2.COLOR_RGB2BGR)
def bgr_to_pil(b): return Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))


# ── Load ───────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    M = load_all_models()

device = M["device"]
has_cls = M["classifier"] is not None
has_seg = M["segmenter"] is not None
has_gan = M["gan"] is not None

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>Monument Digital Restoration</h1>
  <p>Walchand College of Engineering, Sangli &nbsp;·&nbsp; Deep Learning Mini Project</p>
</div>
""", unsafe_allow_html=True)

def icon(ok): return '<span class="ok">●</span>' if ok else '<span class="missing">○</span>'
st.markdown(f"""
<div class="ckpt-status">
  <div class="row"><span>{icon(has_cls)} Style Classifier &nbsp; {'Epoch ' + str(M.get('cls_epoch','?')) + ' · Acc ' + f"{M.get('cls_acc',0)*100:.1f}%" if has_cls else 'not trained — run python train_all.py --stage classifier'}</span></div>
  <div class="row"><span>{icon(has_seg)} Damage Segmenter &nbsp; {'IoU ' + f"{M.get('seg_iou',0):.3f}" if has_seg else 'not trained — run python train_all.py --stage segmenter'}</span></div>
  <div class="row"><span>{icon(has_gan)} Restoration GAN &nbsp; {'Epoch ' + str(M.get('gan_epoch','?')) if has_gan else 'not trained — run python train_all.py --stage gan'}</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a damaged monument photograph",
                             type=["jpg","jpeg","png"], label_visibility="collapsed")

if not uploaded:
    st.markdown("""
    <div style="border:1px dashed #3a3530;border-radius:12px;background:#161410;
                padding:3rem 2rem;text-align:center;color:#5a5248;">
        <div style="font-size:2.5rem;margin-bottom:0.5rem;">🏛️</div>
        <div style="font-size:0.9rem;color:#4a4540;margin-bottom:0.3rem;">Drop a monument photo here</div>
        <div style="font-size:0.75rem;color:#3a3530;">JPG · PNG · The pipeline auto-detects damage and restores it</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

image_pil = Image.open(uploaded)
image_bgr = pil_to_bgr(image_pil)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="step-pill">Input</div>', unsafe_allow_html=True)
    st.image(image_pil, use_container_width=True)
    seg_threshold = st.slider("Detection sensitivity", 0.20, 0.80, 0.45, 0.05,
        help="Lower = detect more damage. Raise if healthy stone is being flagged.")
    run = st.button("⟳  Detect & Restore", key="run_btn")

with col_right:
    if not run:
        st.markdown("""
        <div style="background:#161410;border:1px solid #2a2520;border-radius:12px;
                    padding:3rem 2rem;text-align:center;color:#3a3530;">
            <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.4;">✦</div>
            <div style="font-size:0.82rem;color:#4a4540;">Press <em>Detect &amp; Restore</em> to run the pipeline</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Step 1: Style ────────────────────────────────────────────
    st.markdown('<div class="step-pill">Step 1 — Architectural Style</div>', unsafe_allow_html=True)
    if has_cls:
        with st.spinner("Classifying…"):
            probs, style_emb = classify_style(image_bgr, M["classifier"], device, M["style_classes"])
        top_idx = int(np.argmax(probs))
        top_style = M["style_classes"][top_idx]
        conf = probs[top_idx]
        st.markdown(f"""
        <div class="style-badge">
            <div class="style-name">{top_style}</div>
            <div class="confidence">{conf*100:.1f}% confidence</div>
        </div>""", unsafe_allow_html=True)
        bars = '<div style="margin-top:0.8rem;">'
        for i in np.argsort(probs)[::-1]:
            pct = probs[i]*100
            cls = "top" if i == top_idx else ""
            bars += f'<div class="prob-row"><div class="prob-name">{M["style_classes"][i]}</div><div class="prob-track"><div class="prob-fill {cls}" style="width:{pct:.1f}%"></div></div><div class="prob-pct">{pct:.0f}%</div></div>'
        bars += "</div>"
        st.markdown(bars, unsafe_allow_html=True)
    else:
        style_emb = None
        st.info("Style classifier not trained yet.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 2: Segmentation ──────────────────────────────────────
    st.markdown('<div class="step-pill">Step 2 — Damage Detection</div>', unsafe_allow_html=True)
    mask_gray = None
    damage_pct = 0
    if has_seg:
        with st.spinner("Detecting damaged regions…"):
            mask_gray = auto_segment(image_bgr, M["segmenter"], device, seg_threshold)
        damage_pct = (mask_gray > 127).sum() / mask_gray.size * 100
        st.image(make_overlay(image_bgr, mask_gray),
                 caption="Detected damage (red overlay)", use_container_width=True)
        st.markdown(f'<div class="damage-stat"><span>Damaged area</span><strong>{damage_pct:.1f}%</strong></div>',
                    unsafe_allow_html=True)
    else:
        st.warning("Segmenter not trained. Run: `python train_all.py --stage segmenter`")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 3: GAN Restoration ───────────────────────────────────
    st.markdown('<div class="step-pill">Step 3 — Restored Image</div>', unsafe_allow_html=True)
    restored_pil = None
    if has_gan and mask_gray is not None:
        with st.spinner("Reconstructing damaged structure…"):
            restored_bgr = gan_restore(image_bgr, mask_gray, M["gan"], device, style_emb)
        restored_pil = bgr_to_pil(restored_bgr)
        st.image(restored_pil, use_container_width=True)

        diff = cv2.absdiff(image_bgr, restored_bgr)
        diff_region = diff[mask_gray > 127] if (mask_gray > 127).any() else diff
        mean_change = float(diff_region.mean()) if len(diff_region) > 0 else 0.0

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="val">{damage_pct:.1f}%</div><div class="lbl">Area Restored</div></div>
            <div class="metric-box"><div class="val">{mean_change:.1f}</div><div class="lbl">Avg Pixel Change</div></div>
        </div>""", unsafe_allow_html=True)

        buf = io.BytesIO()
        restored_pil.save(buf, format="PNG")
        st.download_button("⬇  Download Restored Image", data=buf.getvalue(),
                           file_name=f"restored_{Path(uploaded.name).stem}.png", mime="image/png")
    elif not has_gan:
        st.warning("GAN not trained. Run: `python train_all.py --stage gan`")

# ── Full-width comparison ──────────────────────────────────────
if run and has_gan and has_seg and mask_gray is not None and restored_pil is not None:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;"><div class="step-pill">Before / After</div></div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown('<div class="result-label">Original (Damaged)</div>', unsafe_allow_html=True)
        st.image(image_pil, use_container_width=True)
    with c2:
        st.markdown('<div class="result-label">Detected Damage</div>', unsafe_allow_html=True)
        st.image(make_overlay(image_bgr, mask_gray), use_container_width=True)
    with c3:
        st.markdown('<div class="result-label">Restored</div>', unsafe_allow_html=True)
        st.image(restored_pil, use_container_width=True)

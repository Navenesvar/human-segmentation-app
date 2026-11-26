import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageOps

# ── Config (must match training) ─────────────────────────────────────────────
IMG_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CKPT_PATH = Path(__file__).parent / "best_human_seg.pth"
DEFAULT_THRESHOLD = 0.50  # best threshold found on validation set

st.set_page_config(page_title="Human Segmentation", page_icon="🧍", layout="wide")


# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # weights come from our checkpoint
        in_channels=3,
        classes=1,
        activation=None,
    )
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


@torch.no_grad()
def predict_prob(model, device, rgb: np.ndarray) -> np.ndarray:
    """Return a probability map (H, W) in [0, 1] at the original image size."""
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    # upsample the probability (not the binary mask) for smoother edges
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


# ── Rendering helpers ────────────────────────────────────────────────────────
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def make_overlay(rgb, mask, color=(255, 0, 80), alpha=0.5):
    out = rgb.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + c * alpha
    return out.astype(np.uint8)


def make_cutout(rgb, soft_mask):
    """RGBA image with transparent background."""
    alpha = (soft_mask * 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def replace_background(rgb, soft_mask, mode, color=(0, 255, 0), blur=35):
    m = soft_mask[..., None]
    if mode == "Solid color":
        bg = np.full_like(rgb, color)
    else:  # Blur
        k = blur if blur % 2 == 1 else blur + 1
        bg = cv2.GaussianBlur(rgb, (k, k), 0)
    return (rgb * m + bg * (1 - m)).astype(np.uint8)


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def load_image(file) -> np.ndarray:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")  # respect phone rotation
    # cap very large images to keep things fast
    img.thumbnail((2000, 2000))
    return np.array(img)


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🧍 Human Image Segmentation")
st.caption("U-Net with a ResNet34 encoder · trained on the Human Segmentation Dataset · validation Dice ≈ 0.96")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Mask threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05,
                          help="Pixels with probability above this are treated as 'person'.")
    output_mode = st.radio(
        "Output",
        ["Overlay", "Binary mask", "Cutout (transparent)", "Replace background"],
    )
    if output_mode == "Overlay":
        overlay_color = st.color_picker("Overlay color", "#FF0050")
        overlay_alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
    if output_mode == "Replace background":
        bg_mode = st.radio("Background", ["Solid color", "Blur"], horizontal=True)
        if bg_mode == "Solid color":
            bg_color = st.color_picker("Background color", "#00C853")
        else:
            blur_strength = st.slider("Blur strength", 5, 99, 35, 2)

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if file is None:
    st.info("Upload an image to get started.")
    st.stop()

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

rgb = load_image(file)

with st.spinner("Segmenting…"):
    prob = predict_prob(model, device, rgb)

mask = (prob > threshold).astype(np.uint8)
soft = np.where(prob > threshold, prob, 0.0).astype(np.float32)
# smooth alpha for nicer edges in cutout / background replacement
soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)

if output_mode == "Overlay":
    result = make_overlay(rgb, mask, hex_to_rgb(overlay_color), overlay_alpha)
    fname = "overlay.png"
elif output_mode == "Binary mask":
    result = (mask * 255).astype(np.uint8)
    fname = "mask.png"
elif output_mode == "Cutout (transparent)":
    result = make_cutout(rgb, soft)
    fname = "cutout.png"
else:
    if bg_mode == "Solid color":
        result = replace_background(rgb, soft, bg_mode, hex_to_rgb(bg_color))
    else:
        result = replace_background(rgb, soft, bg_mode, blur=blur_strength)
    fname = "background_replaced.png"

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(rgb, use_container_width=True)
with col2:
    st.subheader("Result")
    st.image(result, use_container_width=True, clamp=True)

coverage = mask.mean() * 100
st.caption(f"Person covers ~{coverage:.1f}% of the image · running on {device.upper()}")

st.download_button("⬇️ Download result (PNG)", to_png_bytes(result), file_name=fname, mime="image/png")
if output_mode != "Binary mask":
    st.download_button("⬇️ Download mask (PNG)", to_png_bytes((mask * 255).astype(np.uint8)),
                       file_name="mask.png", mime="image/png")

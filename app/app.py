"""Streamlit demo: upload a chest X-ray, get a pneumonia prediction + Grad-CAM.

Run locally:
    streamlit run app/app.py

Live demo deployed on Streamlit Community Cloud.
"""
import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

try:
    from pytorch_gradcam import GradCAM
    from pytorch_gradcam.utils.image import show_cam_on_image
    from pytorch_gradcam.utils.model_targets import ClassifierOutputTarget
except ImportError:  # grad-cam >= 1.5 renamed the package to pytorch_grad_cam
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import get_transforms  # noqa: E402
from model import build_model  # noqa: E402

st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="\U0001fa7b",
    layout="wide",
)

# ---------------------------------------------------------------- styling ---
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 2.2rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0 0 0.4rem 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: 0.85; font-size: 1.05rem; }
    .card {
        background: #ffffff;
        border: 1px solid #e6e8eb;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .verdict {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: white;
        margin: 1rem 0;
    }
    .verdict .prob { font-size: 2.6rem; font-weight: 700; margin: 0; }
    .verdict .label { font-size: 1.25rem; font-weight: 600; margin: 0.2rem 0 0 0; }
    .verdict .sub { opacity: 0.9; margin: 0.3rem 0 0 0; }
    .verdict-red { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
    .verdict-amber { background: linear-gradient(135deg, #b7791f, #e0a437); }
    .verdict-green { background: linear-gradient(135deg, #1d7a4f, #35b579); }
    .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; }
    .metric-box {
        flex: 1 1 140px;
        background: #f6f8fa;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-box .v { font-size: 1.5rem; font-weight: 700; color: #203a43; }
    .metric-box .k { font-size: 0.85rem; color: #666; }
    .disclaimer {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        font-size: 0.9rem;
        color: #7a5c00;
        margin-top: 1.5rem;
    }
    .footer { text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ header ---
st.markdown(
    """
    <div class="hero">
        <h1>\U0001fa7b Chest X-ray Pneumonia Detector</h1>
        <p>Upload a chest X-ray — the AI predicts pneumonia risk and shows
        <b>exactly which lung regions</b> drove its decision.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    ckpt = os.path.join(os.path.dirname(__file__), "..",
                        "models", "best_model.pth")
    if not os.path.exists(ckpt):
        # Fallback to the as-uploaded filename (GitHub web UI can't rename binaries).
        ckpt = os.path.join(os.path.dirname(__file__), "..",
                            "models", "best_model_fp16_0_bdxz.pth")
    state = torch.load(ckpt, map_location=device)
    # Weights are stored in fp16 to keep the file small; cast back for inference.
    state = {k: v.float() if torch.is_tensor(v) and v.is_floating_point()
             else v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, device


model, device = load_model()
transform = get_transforms(train=False)

# ------------------------------------------------------------------ upload ---
uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="A frontal chest radiograph as JPG or PNG.",
)

if not uploaded:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("How it works")
    st.markdown(
        "1. **Upload** a chest X-ray using the box above.\n"
        "2. **Get a prediction** — the DenseNet-121 model outputs a pneumonia probability.\n"
        "3. **See the evidence** — a Grad-CAM heatmap highlights the lung regions "
        "behind the decision."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model performance")
    st.markdown(
        '<div class="metric-row">'
        '<div class="metric-box"><div class="v">0.977</div><div class="k">Test AUC-ROC</div></div>'
        '<div class="metric-box"><div class="v">92%</div><div class="k">Accuracy</div></div>'
        '<div class="metric-box"><div class="v">97%</div><div class="k">Pneumonia recall</div></div>'
        '<div class="metric-box"><div class="v">5,863</div><div class="k">Training X-rays</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    pil = Image.open(uploaded).convert("RGB").resize((224, 224))
    rgb = np.array(pil).astype(np.float32) / 255.0
    tensor = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    with st.spinner("Analyzing X-ray..."):
        cam = GradCAM(model=model,
                      target_layers=[model.features.denseblock4])
        heatmap = cam(input_tensor=tensor,
                      targets=[ClassifierOutputTarget(0)])[0]
        overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)

    # ------------------------------------------------------------- verdict ---
    if prob >= 0.70:
        cls, label, sub = ("verdict-red", "Pneumonia likely",
                           "High predicted probability — this is a screening aid, not a diagnosis.")
    elif prob >= 0.30:
        cls, label, sub = ("verdict-amber", "Uncertain",
                           "Borderline result — a radiologist should review this scan.")
    else:
        cls, label, sub = ("verdict-green", "Likely normal",
                           "Low predicted probability of pneumonia.")
    st.markdown(
        f'<div class="verdict {cls}">'
        f'<p class="prob">{prob:.1%}</p>'
        f'<p class="label">{label}</p>'
        f'<p class="sub">{sub}</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.progress(int(prob * 100))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input X-ray")
        st.image(pil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What the AI looked at")
        st.image(overlay, use_container_width=True)
        st.caption("Grad-CAM heatmap — brighter regions contributed more to the prediction.")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ footer ---
st.markdown(
    """
    <div class="disclaimer">
        \u26a0\ufe0f <b>Educational demo — not a medical device.</b>
        This model is a student research project. Do not use it for diagnosis
        or clinical decisions.
    </div>
    <div class="footer">
        DenseNet-121 &middot; PyTorch &middot; Grad-CAM &middot;
        Built by Sree Divya
    </div>
    """,
    unsafe_allow_html=True,
)

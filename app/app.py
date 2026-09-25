"""Streamlit demo: upload a chest X-ray, get a pneumonia prediction + Grad-CAM.

Run locally:
    streamlit run app/app.py

Deploy free on Hugging Face Spaces (Streamlit SDK, CPU hardware).
"""
import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image
from pytorch_gradcam import GradCAM
from pytorch_gradcam.utils.image import show_cam_on_image
from pytorch_gradcam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import get_transforms  # noqa: E402
from model import build_model  # noqa: E402

st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("Chest X-ray Pneumonia Detector")
st.caption("Educational demo — not a medical device.")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    ckpt = os.path.join(os.path.dirname(__file__), "..",
                        "results", "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, device


model, device = load_model()
transform = get_transforms(train=False)

uploaded = st.file_uploader("Upload a chest X-ray",
                            type=["jpg", "jpeg", "png"])
if uploaded:
    pil = Image.open(uploaded).convert("RGB").resize((224, 224))
    rgb = np.array(pil).astype(np.float32) / 255.0
    tensor = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    st.image(pil, caption="Input X-ray", width=300)
    st.metric("Pneumonia probability", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    with st.spinner("Generating explanation..."):
        cam = GradCAM(model=model,
                      target_layers=[model.features.denseblock4])
        heatmap = cam(input_tensor=tensor,
                      targets=[ClassifierOutputTarget(0)])[0]
        overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
    st.image(overlay,
             caption="Grad-CAM: regions driving the prediction",
             width=300)

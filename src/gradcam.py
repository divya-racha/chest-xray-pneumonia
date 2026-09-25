"""Generate Grad-CAM heatmaps showing which lung regions drive predictions.

    python src/gradcam.py --num-images 8

Heatmaps are saved to results/gradcam/ — these are the images for your
README and LinkedIn post.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from pytorch_gradcam import GradCAM
from pytorch_gradcam.utils.image import show_cam_on_image
from pytorch_gradcam.utils.model_targets import ClassifierOutputTarget

from dataset import CLASS_NAMES, ChestXrayDataset, get_transforms
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/chest_xray")
    p.add_argument("--checkpoint", default="results/best_model.pth")
    p.add_argument("--output-dir", default="results/gradcam")
    p.add_argument("--num-images", type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Last convolutional block of DenseNet-121.
    cam = GradCAM(model=model, target_layers=[model.features.denseblock4])
    transform = get_transforms(train=False)

    test_ds = ChestXrayDataset(args.data_dir, "test")
    for i in range(min(args.num_images, len(test_ds))):
        path, label = test_ds.samples[i]
        pil = Image.open(path).convert("RGB").resize((224, 224))
        rgb = np.array(pil).astype(np.float32) / 255.0
        tensor = transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        pred = "PNEUMONIA" if prob >= 0.5 else "NORMAL"

        # Category 0 = the single pneumonia logit.
        grayscale_cam = cam(input_tensor=tensor,
                            targets=[ClassifierOutputTarget(0)])[0]
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        fname = (f"{i:02d}_true-{CLASS_NAMES[label]}_"
                 f"pred-{pred}_{prob:.2f}.png")
        Image.fromarray(overlay).save(os.path.join(args.output_dir, fname))
    print(f"Saved heatmaps to {args.output_dir}/")


if __name__ == "__main__":
    main()

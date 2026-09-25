# Chest X-ray Pneumonia Detector

A deep learning classifier that detects pneumonia from chest X-rays —
with **Grad-CAM explainability** showing which lung regions drive each
prediction, deployed as an interactive web app.

> **Live demo:** _coming soon — deploy on Hugging Face Spaces (see below)_
> *(Replace this line with your demo GIF once you have one.)*

## Results

| Metric | Score |
|---|---|
| Test AUC-ROC | _fill in after `python src/evaluate.py`_ |
| Precision (pneumonia) | _fill in_ |
| Recall (pneumonia) | _fill in_ |

*(Add your `results/roc_curve.png` and a Grad-CAM example here.)*

## How it works

1. **Data** — ~5,800 chest X-rays (Kaggle pneumonia dataset), split into
   train/val/test with augmentation (rotation, horizontal flip).
2. **Model** — ImageNet-pretrained DenseNet-121, fine-tuned with a binary
   classification head and class-weighted loss for imbalance.
3. **Explainability** — Grad-CAM heatmaps over the final convolutional
   block, so every prediction comes with a visual explanation.
4. **Deployment** — Streamlit app served on Hugging Face Spaces (free CPU tier).

## Quick start

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Data (see data/README.md for details)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/chest_xray

# 3. Train (use a GPU — Google Colab's free tier works)
python src/train.py --epochs 15

# 4. Evaluate
python src/evaluate.py

# 5. Generate Grad-CAM explanations
python src/gradcam.py --num-images 8

# 6. Launch the demo
streamlit run app/app.py
```

## Deploy to Hugging Face Spaces

1. Create a free account at huggingface.co and go to **New Space**.
2. Choose SDK **Streamlit**, visibility **Public**, hardware **CPU (free)**.
3. Upload `app/app.py` (as `app.py`), `requirements.txt`, the `src/` folder,
   and your trained `results/best_model.pth`.
4. Paste the live URL at the top of this README.

## What I'd do next

- Scale to NIH ChestX-ray14: multi-label classification over 14 pathologies
- Compare architectures (EfficientNet, Vision Transformer)
- Add uncertainty estimation so the model can say "I'm not sure"

## Disclaimer

Educational portfolio project — not a medical device, not for clinical use.

## Project structure

```
├── app/            # Streamlit demo
├── data/           # Dataset download instructions (images git-ignored)
├── src/
│   ├── dataset.py  # PyTorch dataset + augmentation
│   ├── model.py    # DenseNet-121 binary classifier
│   ├── train.py    # Training loop with class-weighted loss
│   ├── evaluate.py # Test metrics: AUC, precision/recall, ROC curve
│   └── gradcam.py  # Grad-CAM heatmap generation
├── results/        # Checkpoints, plots, heatmaps (git-ignored)
└── requirements.txt
```

"""Model definition: ImageNet-pretrained DenseNet-121 with a binary head."""
import torch.nn as nn
from torchvision import models


def build_model(freeze_features: bool = False) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier.in_features
    # Single logit: P(pneumonia). Trained with BCEWithLogitsLoss.
    model.classifier = nn.Linear(in_features, 1)
    return model

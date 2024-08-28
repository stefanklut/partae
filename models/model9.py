import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics import Accuracy
from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    VGG16_BN_Weights,
    ViT_B_16_Weights,
)
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from models.model_base import ClassificationModel
from models.text_features_array import TextFeaturesArray
from utils.text_utils import combine_texts


class LazyLinearBlock(nn.Module):
    def __init__(self, out_features, dropout=0.0):
        super(LazyLinearBlock, self).__init__()
        self.fc = nn.LazyLinear(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, merge_to_batch=True, resize_size=(512, 512)):
        super(ImageEncoder, self).__init__()

        self.merge_to_batch = merge_to_batch
        self.resize_size = resize_size

        imagenet = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.imagenet = nn.Sequential(*list(imagenet.children())[:-2])

    @property
    @lru_cache
    def device(self):
        return next(self.parameters()).device

    def encode_image(self, image: torch.Tensor):
        image = image.to(self.device)  # (B, C, H, W)
        encoded_image = F.interpolate(image, self.resize_size)  # (B, C, resize_size[0], resize_size[1])
        encoded_image = self.imagenet(encoded_image)  # (B, 2048, ...)
        return encoded_image

    def forward(self, x: torch.Tensor):
        # x: (B, N, C, H, W) B: Batch size, N: Number of images, C: Channels, H: Height, W: Width
        B, N, C, H, W = x.size()

        if self.merge_to_batch:
            # Chunk the input into individual images
            chunked_x: tuple[torch.Tensor, ...] = x.chunk(N, dim=1)  # N x (B, 1, C, H, W)
            batched_x: torch.Tensor = torch.concat(chunked_x, dim=0).squeeze(dim=1)  # (B * N, C, H, W)

            # Encode each image
            batched_encoded_image = self.encode_image(batched_x)  # (B * N, 512)

            # Chunk the encoded images back into the original batch size
            chunked_encoded_images: tuple[torch.Tensor, ...] = batched_encoded_image.chunk(N, dim=0)  # N x (B, )
            encoded_images = torch.stack(chunked_encoded_images, dim=1)  # (B, N, 512)

        else:
            # Loop through the images and encode them
            encoded_images_list = []
            for i in range(N):
                image = x[:, i]

                # Encode the individual images per batch
                encoded_image = self.encode_image(image)  # (B, 512)
                encoded_images_list.append(encoded_image)

            # Recombine the encoded images
            encoded_images = torch.stack(encoded_images_list, dim=1)  # (B, N, 512)

        # if torch.isnan(encoded_images).any():
        #     raise ValueError("NaN values in the encoded images")

        return encoded_images


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.text_feature_array = TextFeaturesArray(512, 16, 16, 512, 512, mode="baseline")

    def forward(self, x: list[list[dict]]):
        return self.text_feature_array(x)


class DocumentSeparator(ClassificationModel):
    def __init__(
        self,
        image_encoder=ImageEncoder(merge_to_batch=True),
        text_encoder=TextEncoder(),
        output_size=2,
        dropout=0.5,
        label_smoothing=0.0,
        **kwargs,
    ):
        super(DocumentSeparator, self).__init__(**kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.output_size = output_size
        self.dropout = dropout
        self.label_smoothing = label_smoothing

    def forward(self, x: dict):
        images = x["images"]
        texts = x["texts"]
        shapes = x["shapes"]

        encoded_images = self.image_encoder(images)
        encoded_texts = self.text_encoder(texts)

        print(encoded_images.shape, encoded_texts.shape)


if __name__ == "__main__":
    # Test ImageEncoder
    image_encoder = ImageEncoder()
    x = torch.randn(2, 3, 3, 512, 512)
    encoded_images = image_encoder(x)
    print(encoded_images.shape)

    x2 = [
        [
            {
                "id": {
                    "text": "Hello, world!",
                    "baseline": np.array([[1, 1], [2, 2]]),
                    "bbox": np.array([1, 1, 1, 1]),
                    "coords": np.array([[1, 2], [2, 3]]),
                }
            }
        ]
    ]

    text_encoder = TextFeaturesArray(512, 16, 16, 512, 512, mode="baseline")

    encoded_text = text_encoder(x2)

    document_separator = DocumentSeparator()
    data = {"images": x, "texts": x2, "shapes": torch.randn(2, 3, 2)}
    output = document_separator(data)
    print(output)

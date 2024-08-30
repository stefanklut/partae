# Cosine similarity between scans from the same document should be high, add extra conv to image encoder and larger image size
from functools import lru_cache

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

from models.model_base import ClassificationModel
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
        self.conv2d = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(1, -1)
        self.fc = nn.Sequential(
            LazyLinearBlock(2048),
            LinearBlock(2048, 1024),
            nn.Linear(1024, 512),
        )

    @property
    @lru_cache
    def device(self):
        return next(self.parameters()).device

    def encode_image(self, image: torch.Tensor):
        image = image.to(self.device)  # (B, C, H, W)
        encoded_image = F.interpolate(image, self.resize_size)  # (B, C, resize_size[0], resize_size[1])
        encoded_image = self.imagenet(encoded_image)  # (B, 2048, ...)
        encoded_image = self.flatten(encoded_image)  # (B, 2048 * ...)
        encoded_image = self.fc(encoded_image)  # (B, 512)
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
            chunked_encoded_images: tuple[torch.Tensor, ...] = batched_encoded_image.chunk(N, dim=0)  # N x (B, 512)
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
    def __init__(self, merge_to_batch=True):
        super(TextEncoder, self).__init__()

        self.merge_to_batch = merge_to_batch

        self.tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        self.roberta = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base", add_pooling_layer=False)

        self.fc = nn.Sequential(
            LazyLinearBlock(512),
            nn.Linear(512, 512),
        )

    @property
    @lru_cache
    def device(self):
        return next(self.parameters()).device

    def encode_text(self, text: list[str]):
        # Tokenize the text
        encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")  # (B, S)
        encoded_text = {key: tensor.to(self.device) for key, tensor in encoded_text.items()}

        # Encode the text

        encoded_text = self.roberta(**encoded_text).last_hidden_state  # (B, S, 768)
        encoded_text = encoded_text[:, 0]  # (B, 768)
        encoded_text = self.fc(encoded_text)  # (B, 512)

        return encoded_text

    def forward(self, x: list[list[str]]):
        N = len(x[0])  # Number of documents
        if self.merge_to_batch:
            # Flatten the input
            batched_x: list[str] = [text for texts in x for text in texts]

            # Encode the text
            batched_encoded_texts: torch.Tensor = self.encode_text(batched_x)  # (B * N, 512)

            # Chunk the encoded texts back into the original batch size
            chunked_encoded_texts: tuple[torch.Tensor, ...] = batched_encoded_texts.chunk(N, dim=0)  # N x (B, 512)
            encoded_texts: torch.Tensor = torch.stack(chunked_encoded_texts, dim=1)  # (B, N, 512)

        else:
            # Loop through the documents and encode them
            encoded_texts_list = []
            for i in range(N):
                texts = [x[j][i] for j in range(len(x))]

                # Encode the individual documents per batch
                encoded_text = self.encode_text(texts)  # (B, 512)
                encoded_texts_list.append(encoded_text)

            # Recombine the encoded documents
            encoded_texts: torch.Tensor = torch.stack(encoded_texts_list, dim=1)  # (B, N, 512)

        # if torch.isnan(encoded_texts).any():
        #     raise ValueError("NaN values in the encoded texts")

        return encoded_texts


class DocumentSeparator(ClassificationModel):
    def __init__(
        self,
        image_encoder=ImageEncoder(merge_to_batch=True),
        text_encoder=TextEncoder(merge_to_batch=True),
        output_size=2,
        dropout=0.5,
        label_smoothing=0.0,
        **kwargs,
    ):
        super(DocumentSeparator, self).__init__(**kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            LazyLinearBlock(1024, dropout=dropout),
            LinearBlock(1024, 512, dropout=dropout),
            nn.Linear(512, output_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.label_smoothing = label_smoothing

        self.accuracy = Accuracy(task="multiclass", num_classes=output_size)
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

    @property
    @lru_cache
    def device(self):
        return next(self.parameters()).device

    def cosine_with_targets(self, x_tensor, y_tensor, x_target, y_target):
        target = torch.where(y_target == 1, torch.tensor(-1.0), torch.tensor(1.0)).to(self.device)
        return self.cosine_embedding_loss(x_tensor, y_tensor, target)

    def forward(self, x):
        images = x["images"]
        shapes = x["shapes"]
        texts = x["texts"]
        for i in range(len(texts)):
            for j in range(len(texts[i])):
                texts[i][j] = combine_texts(x["text"] for x in texts[i][j].values())

        images = images.to(self.device)
        shapes = shapes.to(self.device)

        images = self.image_encoder(images)
        texts = self.text_encoder(texts)

        if torch.logical_xor(shapes[..., 0:1] == 0, shapes[..., 1:2] == 0).any():
            raise ValueError("One of the shapes is 0, both should be 0 (no image) or both should be non-zero (normal image)")
        both_zero = torch.logical_and(shapes[..., 0:1] == 0, shapes[..., 1:2] == 0)
        inverted_shapes = 1 / shapes
        inverted_shapes = torch.where(both_zero, torch.tensor(0, device=both_zero.device), inverted_shapes)
        ratio_shapes = shapes[..., 0:1] / shapes[..., 1:2]
        ratio_shapes = torch.where(both_zero, torch.tensor(0, device=both_zero.device), ratio_shapes)

        encoded_features = torch.cat([images, texts], dim=2)  # (B, N, 1027)
        output = self.dropout(encoded_features)
        output, _ = self.lstm(output)  # (B, N, 1024)
        output = self.fc(output)
        output_center = output[:, output.shape[1] // 2]

        if "targets" in x:
            targets = x["targets"]
            targets_center = targets[:, targets.shape[1] // 2]
            cross_entropy_loss = F.cross_entropy(output_center, targets_center, label_smoothing=self.label_smoothing)
            cosine_loss_total = 0
            assert encoded_features.shape[1] > 1, "Number of documents must be greater than 1 for cosine loss"
            for i in range(encoded_features.shape[1] - 1):
                cosine_loss = self.cosine_with_targets(
                    encoded_features[:, i], encoded_features[:, i + 1], targets[:, i], targets[:, i + 1]
                )
                cosine_loss_total += cosine_loss
            losses = {
                "cross_entropy_loss": cross_entropy_loss,
                "cosine_loss": cosine_loss_total,
            }

            acc = self.accuracy(output_center, targets_center)
            metrics = {
                "acc": acc,
            }
            return output_center, losses, metrics

        return output_center, None, None


if __name__ == "__main__":
    # Test ImageEncoder
    image_encoder = ImageEncoder()
    x = torch.randn(1, 3, 3, 512, 512)
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
            },
            {
                "id": {
                    "text": "Hello, world!",
                    "baseline": np.array([[1, 1], [2, 2]]),
                    "bbox": np.array([1, 1, 1, 1]),
                    "coords": np.array([[1, 2], [2, 3]]),
                }
            },
            {
                "id": {
                    "text": "Hello, world!",
                    "baseline": np.array([[1, 1], [2, 2]]),
                    "bbox": np.array([1, 1, 1, 1]),
                    "coords": np.array([[1, 2], [2, 3]]),
                }
            },
        ],
    ]

    text_encoder = TextEncoder()

    encoded_text = text_encoder(x2)
    print(encoded_text.shape)

    document_separator = DocumentSeparator()
    data = {"images": x, "texts": x2, "shapes": torch.randn(1, 3, 2)}
    output, _, _ = document_separator(data)
    print(output.shape)

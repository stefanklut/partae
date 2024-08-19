from functools import lru_cache

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
    def __init__(self, merge_to_batch=True, resize_size=(224, 224)):
        super(ImageEncoder, self).__init__()

        self.merge_to_batch = merge_to_batch
        self.resize_size = resize_size

        imagenet = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.imagenet = nn.Sequential(*list(imagenet.children())[:-2])
        self.flatten = nn.Flatten(1, -1)
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            LazyLinearBlock(512),
            nn.Linear(512, 16),
        )

    @property
    @lru_cache
    def device(self):
        return next(self.parameters()).device

    def encode_image(self, image: torch.Tensor):
        image = image.to(self.device)  # (B, C, H, W)
        encoded_image = F.interpolate(image, self.resize_size)  # (B, C, resize_size[0], resize_size[1])
        encoded_image = self.imagenet(encoded_image)  # (B, 2048, ...)
        encoded_image = self.conv(encoded_image)  # (B, 1024, ...)
        encoded_image = self.flatten(encoded_image)  # (B, 1024 * ...)
        encoded_image = self.fc(encoded_image)  # (B, 16)
        return encoded_image

    def forward(self, x: torch.Tensor):
        # x: (B, N, C, H, W) B: Batch size, N: Number of images, C: Channels, H: Height, W: Width
        B, N, C, H, W = x.size()

        if self.merge_to_batch:
            # Chunk the input into individual images
            chunked_x: tuple[torch.Tensor, ...] = x.chunk(N, dim=1)  # N x (B, 1, C, H, W)
            batched_x: torch.Tensor = torch.concat(chunked_x, dim=0).squeeze(dim=1)  # (B * N, C, H, W)

            # Encode each image
            batched_encoded_image = self.encode_image(batched_x)  # (B * N, 16)

            # Chunk the encoded images back into the original batch size
            chunked_encoded_images: tuple[torch.Tensor, ...] = batched_encoded_image.chunk(N, dim=0)  # N x (B, 16)
            encoded_images = torch.stack(chunked_encoded_images, dim=1)  # (B, N, 16)

        else:
            # Loop through the images and encode them
            encoded_images_list = []
            for i in range(N):
                image = x[:, i]

                # Encode the individual images per batch
                encoded_image = self.encode_image(image)  # (B, 16)
                encoded_images_list.append(encoded_image)

            # Recombine the encoded images
            encoded_images = torch.stack(encoded_images_list, dim=1)  # (B, N, 16)

        # if torch.isnan(encoded_images).any():
        #     raise ValueError("NaN values in the encoded images")

        return encoded_images


class TextEncoder(nn.Module):
    def __init__(self, merge_to_batch=True):
        super(TextEncoder, self).__init__()

        self.merge_to_batch = merge_to_batch

        self.tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        self.roberta = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base")

        self.fc = nn.Sequential(
            LazyLinearBlock(512),
            nn.Linear(512, 16),
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
        encoded_text = self.fc(encoded_text)  # (B, 16)

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


class DocumentSeparator(nn.Module):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        output_size=2,
        dropout=0.5,
        label_smoothing=0.0,
        turn_off_image=False,
        turn_off_text=False,
        turn_off_shapes=False,
    ):
        super(DocumentSeparator, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.lstm_combined = nn.LSTM(
            input_size=35,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.lstm_text = nn.LSTM(
            input_size=16,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.lstm_image = nn.LSTM(
            input_size=16,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.lstm_shapes = nn.LSTM(
            input_size=3,
            hidden_size=2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc_combined = nn.Sequential(
            LazyLinearBlock(16, dropout=dropout),
            nn.Linear(16, output_size),
        )

        self.fc_text = nn.Sequential(
            LazyLinearBlock(16, dropout=dropout),
            nn.Linear(16, output_size),
        )

        self.fc_image = nn.Sequential(
            LazyLinearBlock(16, dropout=dropout),
            nn.Linear(16, output_size),
        )

        self.fc_shapes = nn.Sequential(
            LazyLinearBlock(4, dropout=dropout),
            nn.Linear(4, output_size),
        )

        self.dropout = nn.Dropout(dropout)
        self.label_smoothing = label_smoothing

        self.accuracy = Accuracy(task="multiclass", num_classes=output_size)

        self.turn_off_image = turn_off_image
        self.turn_off_text = turn_off_text
        self.turn_off_shapes = turn_off_shapes

    @property
    @lru_cache
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        images = x["images"]
        shapes = x["shapes"]
        texts = x["texts"]
        for i in range(len(texts)):
            for j in range(len(texts[i])):
                texts[i][j] = combine_texts(x["text"] for x in texts[i][j].values())

        images = images.to(self.device)
        shapes = shapes.to(self.device)

        if self.turn_off_image:
            B, N, C, H, W = images.size()
            images = torch.zeros((B, N, 16), device=self.device)
        else:
            images = self.image_encoder(images)

        if self.turn_off_text:
            B, N = len(texts), len(texts[0])
            texts = torch.zeros((B, N, 16), device=self.device)
        else:
            texts = self.text_encoder(texts)

        if torch.logical_xor(shapes[..., 0:1] == 0, shapes[..., 1:2] == 0).any():
            raise ValueError("One of the shapes is 0, both should be 0 (no image) or both should be non-zero (normal image)")

        if self.turn_off_shapes:
            B, N = len(shapes), len(shapes[0])
            inverted_shapes = torch.zeros((B, N, 2), device=self.device)
            ratio_shapes = torch.zeros((B, N, 1), device=self.device)
        else:
            both_zero = torch.logical_and(shapes[..., 0:1] == 0, shapes[..., 1:2] == 0)
            inverted_shapes = 1 / shapes
            inverted_shapes = torch.where(both_zero, torch.tensor(0, device=both_zero.device), inverted_shapes)
            ratio_shapes = shapes[..., 0:1] / shapes[..., 1:2]
            ratio_shapes = torch.where(both_zero, torch.tensor(0, device=both_zero.device), ratio_shapes)

        # IDEA Add the image height and width to the embedding, but maybe invert them to keep them close to 0
        # x = torch.cat([images, texts], dim=2)  # (B, N, 1024)

        output_combined = torch.cat([images, texts, inverted_shapes, ratio_shapes], dim=2)  # (B, N, 35)

        output_combined = self.dropout(output_combined)
        output_combined, _ = self.lstm_combined(output_combined)
        output_combined = output_combined[:, output_combined.shape[1] // 2]
        output_combined = self.fc_combined(output_combined)

        output_text = self.dropout(texts)
        output_text, _ = self.lstm_text(output_text)
        output_text = output_text[:, output_text.shape[1] // 2]
        output_text = self.fc_text(output_text)

        output_image = self.dropout(images)
        output_image, _ = self.lstm_image(output_image)
        output_image = output_image[:, output_image.shape[1] // 2]
        output_image = self.fc_image(output_image)

        output_shapes = torch.cat([inverted_shapes, ratio_shapes], dim=2)
        output_shapes = self.dropout(output_shapes)
        output_shapes, _ = self.lstm_shapes(output_shapes)
        output_shapes = output_shapes[:, output_shapes.shape[1] // 2]
        output_shapes = self.fc_shapes(output_shapes)

        if "targets" in x:
            targets = x["targets"]
            targets_center = targets[:, targets.shape[1] // 2]
            loss_combined = F.cross_entropy(output_combined, targets_center, label_smoothing=self.label_smoothing)
            loss_text = F.cross_entropy(output_text, targets_center)
            loss_image = F.cross_entropy(output_image, targets_center)
            loss_shapes = F.cross_entropy(output_shapes, targets_center)

            losses = {
                "loss_combined": loss_combined * 0.2,
                "loss_text": loss_text * 0.3,
                "loss_image": loss_image * 0.3,
                "loss_shapes": loss_shapes * 0.2,
            }

            acc_combined = self.accuracy(output_combined, targets_center)
            acc_text = self.accuracy(output_text, targets_center)
            acc_image = self.accuracy(output_image, targets_center)
            acc_shapes = self.accuracy(output_shapes, targets_center)

            metrics = {
                "acc": acc_combined,
                "acc_text": acc_text,
                "acc_image": acc_image,
                "acc_shapes": acc_shapes,
            }
            return output_combined, losses, metrics
        return output_combined, None, None


if __name__ == "__main__":
    image_encoder = ImageEncoder(merge_to_batch=True)
    image_encoder2 = ImageEncoder(merge_to_batch=False)
    images = torch.randn(2, 3, 3, 224, 224)
    encoded_images = image_encoder(images)
    encoded_images2 = image_encoder2(images)
    print(encoded_images.size())
    print(encoded_images2.size())

    text_encoder = TextEncoder()
    texts = [["Hello", "world", "Holy"], ["This", "is", "test"]]
    encoded_texts = text_encoder(texts)
    print(encoded_texts.size())

    data = {"image": images, "text": texts}
    document_separator = DocumentSeparator(image_encoder, text_encoder)
    output = document_separator(data)
    print(output.size())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    VGG16_BN_Weights,
    ViT_B_16_Weights,
)
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


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
        self.fc = nn.Sequential(
            LazyLinearBlock(2048),
            LinearBlock(2048, 1024),
            nn.Linear(1024, 512),
        )

    def encode_image(self, image: torch.Tensor):
        encoded_image = F.interpolate(image, self.resize_size)  # (B, C, resize_size[0], resize_size[1])
        with torch.no_grad():
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

        return encoded_images


class TextEncoder(nn.Module):
    def __init__(self, merge_to_batch=True):
        super(TextEncoder, self).__init__()

        self.merge_to_batch = merge_to_batch

        self.tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        self.roberta = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base")

        self.fc = nn.Sequential(
            LazyLinearBlock(512),
            nn.Linear(512, 512),
        )

    def encode_text(self, text: list[str]):
        # Tokenize the text
        encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")  # (B, S)
        encoded_text = {key: tensor.to(self.roberta.device) for key, tensor in encoded_text.items()}

        # Encode the text
        with torch.no_grad():
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
        return encoded_texts


class DocumentSeparator(nn.Module):
    def __init__(self, image_encoder, text_encoder, output_size=2):
        super(DocumentSeparator, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            LazyLinearBlock(1024),
            LinearBlock(1024, 512),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        images = x["images"]
        texts = x["texts"]
        shapes = x["shapes"]
        images = self.image_encoder(images)
        texts = self.text_encoder(texts)

        # IDEA Add the image height and width to the embedding, but maybe invert them to keep them close to 0
        x = torch.cat([images, texts], dim=2)  # (B, N, 1024)
        x, _ = self.lstm(x)  # (B, N, 1024)
        inverted_shapes = 1 / shapes
        ratio_shapes = shapes[..., 0:0] / shapes[..., 1:1]
        x = torch.concat([x, inverted_shapes, ratio_shapes], dim=2)
        x = self.fc(x)
        return x


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

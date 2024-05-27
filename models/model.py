import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights


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
    def __init__(self, merge_to_batch=False, resize_size=(224, 224)):
        super(ImageEncoder, self).__init__()

        self.merge_to_batch = merge_to_batch
        self.resize_size = resize_size

        imagenet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.imagenet = nn.Sequential(*list(imagenet.children())[:-2])
        self.flatten = nn.Flatten(1, -1)
        self.fc = nn.Sequential(
            LazyLinearBlock(2048),
            LinearBlock(2048, 1024),
            LinearBlock(1024, 512),
        )

    def encode_image(self, image: torch.Tensor):
        encoded_image = F.interpolate(image, self.resize_size)  # (B, C, resize_size[0], resize_size[1])
        encoded_image = self.imagenet(image)  # (B, 2048, ...)
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
                self.encode_image(image)  # (B, 512)

            # Recombine the encoded images
            encoded_images = torch.stack(encoded_images_list, dim=1)  # (B, N, 512)

        return encoded_images


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.fc = nn.Linear(300, 512)


class DocumentSeparator(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super(DocumentSeparator, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        image, text = x
        image = self.image_encoder(image)
        text = self.text_encoder(text)
        x = torch.cat([image, text], dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    image_encoder = ImageEncoder(merge_to_batch=True)
    image_encoder2 = ImageEncoder(merge_to_batch=False)
    images = torch.randn(2, 3, 3, 224, 224)
    print(images)
    encoded_images = image_encoder(images)
    encoded_images2 = image_encoder2(images)
    print(encoded_images.size())
    print(encoded_images2.size())

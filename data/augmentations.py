import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL.Image import Image


class _ApplyToList(torch.nn.Module):
    def __init__(self, fn):
        super(_ApplyToList, self).__init__()
        self.fn = fn

    def forward(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if not isinstance(tensor_list, list):
            return self.fn(tensor_list)
        return [self.fn(item) for item in tensor_list]


class PadToMaxSize(_ApplyToList):
    def __init__(self, fn):
        fn = None
        super(PadToMaxSize, self).__init__(fn)

    def forward(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if not isinstance(tensor_list, list):
            return tensor_list

        height, width = np.max([image.size()[-2:] for image in tensor_list])

        for i in range(len(tensor_list)):
            height_difference = height - tensor_list[i].size()[-2]
            width_difference = width - tensor_list[i].size()[-1]
            padding = [0, 0, int(width_difference), int(height_difference)]
            tensor_list[i] = F.pad(
                tensor_list[i],
                padding,
                fill=0,
            )
        return tensor_list

    def repr(self):
        return f"{self.__class__.__name__}()"


class SmartCompose(torch.nn.Module):
    def __init__(self, transforms):
        super(SmartCompose, self).__init__()
        self.transforms = transforms

    def forward(self, image: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(image, torch.Tensor):
            for transform in self.transforms:
                image = transform(image)
            return image
        if isinstance(image, list):
            for transform in self.transforms:
                if not isinstance(transform, _ApplyToList):
                    transform = _ApplyToList(transform)
                image = transform(image)
        return image

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class SquarePad(torch.nn.Module):
    def __init__(self):
        super(SquarePad, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            width, height = image.size()[-1], image.size()[-2]
        if not isinstance(image, torch.Tensor):
            width, height = image.size  # type: ignore
        max_size = max(width, height)
        height_difference = max_size - height
        width_difference = max_size - width
        padding = [0, 0, int(width_difference), int(height_difference)]
        return F.pad(image, padding, 0, "constant")

    def __repr__(self):
        return f"{self.__class__.__name__}()"

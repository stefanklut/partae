from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL.Image import Image


class _ApplyToList(torch.nn.Module):
    """
    Apply a transformation to a list of tensors
    """

    def __init__(self, fn):
        super(_ApplyToList, self).__init__()
        self.fn = fn

    def forward(self, tensor_list: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
        if not isinstance(tensor_list, list):
            return self.fn(tensor_list)
        # if all(tensor is None for tensor in tensor_list):
        #     raise ValueError("All tensors are None")
        for i in range(len(tensor_list)):
            if tensor_list[i] is None:
                continue
            tensor_list[i] = self.fn(tensor_list[i])
        return tensor_list


class PadToMaxSize(_ApplyToList):
    """
    Apply padding to a list of tensors to make them the same size
    """

    def __init__(self):
        fn = None
        super(PadToMaxSize, self).__init__(fn)

    def pil_pad(self, image_list: list[Optional[Image]]) -> list[Optional[Image]]:
        """
        Pad a list of PIL images to make them the same size

        Args:
            image_list (list[Optional[Image]]): the list of images to pad

        Returns:
            list[Optional[Image]]: the list of padded images
        """
        height, width = np.max([image.size for image in image_list if image is not None], axis=0)
        for i in range(len(image_list)):
            if image_list[i] is None:
                continue
            height_difference = height - image_list[i].size[1]  # type: ignore
            width_difference = width - image_list[i].size[0]  # type: ignore
            padding = [0, 0, int(width_difference), int(height_difference)]
            image_list[i] = F.pad(  # type: ignore
                image_list[i],  # type: ignore
                padding,
                fill=0,
            )
        return image_list

    def tensor_pad(self, tensor_list: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
        """
        Pad a list of tensors to make them the same size

        Args:
            tensor_list (list[Optional[torch.Tensor]]): the list of tensors to pad

        Returns:
            list[Optional[torch.Tensor]]: the list of padded tensors
        """
        if all(tensor is None for tensor in tensor_list):
            return tensor_list
        height, width = np.max([image.size()[-2:] for image in tensor_list if image is not None], axis=0)
        for i in range(len(tensor_list)):
            if tensor_list[i] is None:
                continue
            height_difference = height - tensor_list[i].size()[-2]  # type: ignore
            width_difference = width - tensor_list[i].size()[-1]  # type: ignore
            padding = [0, 0, int(width_difference), int(height_difference)]
            tensor_list[i] = F.pad(
                tensor_list[i],  # type: ignore
                padding,
                fill=0,
            )
        return tensor_list

    def use_pil(self, tensor_list: list[Optional[torch.Tensor]]) -> bool:
        """
        Determine if the input is a list of PIL images, thus if the padding should be done with PIL

        Args:
            tensor_list (list[Optional[torch.Tensor]]): the list of tensors to check

        Returns:
            bool: True if the input is a list of PIL images, False otherwise
        """

        for tensor in tensor_list:
            if tensor is not None:
                return not isinstance(tensor, torch.Tensor)

        return False

    def forward(self, tensor_list: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
        if not isinstance(tensor_list, list):
            return tensor_list

        if self.use_pil(tensor_list):
            return self.pil_pad(tensor_list)  # type: ignore
        return self.tensor_pad(tensor_list)

    def repr(self):
        return f"{self.__class__.__name__}()"


class SmartCompose(torch.nn.Module):
    """
    Apply a list of transformations to an image or list of images
    """

    def __init__(self, transforms):
        """
        Apply a list of transformations to an image or list of images

        Args:
            transforms (Transform/_ApplyToList): the transformations to apply
        """
        super(SmartCompose, self).__init__()
        self.transforms = transforms

    def forward(self, image: torch.Tensor | list[Optional[torch.Tensor]]) -> torch.Tensor | list[Optional[torch.Tensor]]:
        if isinstance(image, torch.Tensor):
            for transform in self.transforms:
                image = transform(image)
            return image
        elif isinstance(image, list):
            for transform in self.transforms:
                if not isinstance(transform, _ApplyToList):
                    transform = _ApplyToList(transform)
                image = transform(image)
            return image
        else:
            raise ValueError(f"Expected input to be torch.Tensor or list, got {type(image)}")

    def __repr__(self) -> str:
        # The format string is initialized with the class name and adds all the transformations as new lines
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class SquarePad(torch.nn.Module):
    """
    Pad an image to make it square
    """

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

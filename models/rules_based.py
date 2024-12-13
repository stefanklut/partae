import re

import numpy as np
import torch


class RulesBased:
    def __init__(self):
        pass

    def device(self):
        """
        Set the device to CPU.
        """
        return torch.device("cpu")

    def get_size_match(self, image_size1: np.ndarray, image_size2: np.ndarray, margin: float) -> bool:
        """
        Compare the size of two images. If the size of the images is within the margin, return True.

        Args:
            image_size1 (np.ndarray): The size of the first image
            image_size2 (np.ndarray): The size of the second image
            margin (float): The margin of the difference between the sizes

        Returns:
            bool: True if the sizes are within the margin, False otherwise
        """
        min_multiplier = 1.0 - margin
        max_multiplier = 1.0 + margin
        border_multiplier = 0.01

        assert image_size1.shape == (2,), f"Expected shape (2,) but got {image_size1.shape}"
        assert image_size2.shape == (2,), f"Expected shape (2,) but got {image_size2.shape}"

        image1_width = image_size1[0]
        image2_width = image_size2[0]
        image1_height = image_size1[1]
        image2_height = image_size2[1]
        similar_height_width = (
            image1_width * min_multiplier < image2_width < image1_width * max_multiplier
            and image1_height * min_multiplier < image2_height < image1_height * max_multiplier
        )
        if similar_height_width:
            return True
        similar_height_half_width = (
            image1_width * (1 - border_multiplier) * min_multiplier
            < image2_width / 2
            < image1_width * (1 - border_multiplier) * max_multiplier
            and image1_height * min_multiplier < image2_height < image1_height * max_multiplier
        )
        if similar_height_half_width:
            return True
        similar_height_double_width = (
            image1_width * min_multiplier < image2_width * (2 - border_multiplier) < image1_width * max_multiplier
            and image1_height * min_multiplier < image2_height < image1_height * max_multiplier
        )
        if similar_height_double_width:
            return True
        return False

    def get_median_color_match(self, image1: np.ndarray, image2: np.ndarray, margin: float) -> bool:
        """
        Compare the median color of two images. If the median color of the images is within the margin, return True.

        Args:
            image1 (np.ndarray): The first image
            image2 (np.ndarray): The second image
            margin (float): The margin of the difference between the median colors

        Returns:
            bool: True if the median colors are within the margin, False otherwise
        """
        median1 = np.median(image1, axis=(1, 2)) / 255
        median2 = np.median(image2, axis=(1, 2)) / 255
        color_median_diff = (
            np.abs(median1[0] - median2[0]) + np.abs(median1[1] - median2[1]) + np.abs(median1[2] - median2[2])
        ) / 3
        if color_median_diff < margin:
            return True
        return False

    def __call__(self, x):
        """
        Create a prediction based on the image shapes and paths. The prediction is based on the size of the images. And on the path of the images.


        Args:
            x (dict): Dictionary containing the shapes, image paths and images

        Raises:
            ValueError: If the inventory number in the directory does not match the inventory number in the file
            ValueError: If the path does not match the expected format

        Returns:
            torch.Tensor: Predictions
        """
        shapes = x["shapes"]
        image_paths = x["image_paths"]
        # images = x["images"]

        B = shapes.shape[0]
        N = shapes.shape[1]
        assert N > 1, "There should be at least 2 images"

        center_index = N // 2

        predictions = torch.zeros((B, 2))
        for i in range(B):
            center_path = image_paths[i][center_index]
            inventory_number_dir = center_path.parent.name
            # Check if the path matches the expected format
            if check := re.match(r"(.+)_(.+)_(\d+)(_deelopname\d+)?", center_path.stem):
                # Get the inventory number and page number from the path
                inventory_number_file = check.group(2)
                if inventory_number_dir != inventory_number_file:
                    raise ValueError(
                        f"Inventory number in dir {inventory_number_dir} does not match with inventory number in file {inventory_number_file}. Path: {center_path}"
                    )
                page_number = int(check.group(3))

                # If the page number is 1, the image is the first scan, thus the scan in a document
                if page_number == 1:
                    predictions[i, 1] = 1
                    continue

                # If the page has _deelopname in the name, the image is a scan of a part of a document, thus not the first scan
                if check.group(4):
                    predictions[i, 0] = 1
                    continue
            else:
                raise ValueError(f"Path {center_path} does not match the expected format")

            # Compare the size of the images
            center_shape = np.asarray(shapes[i, center_index])[::-1]
            prev_shape = np.asarray(shapes[i, center_index - 1])[::-1]
            # center_image = (images[i, center_index] * 255).numpy().astype(np.uint8)
            # prev_image = (images[i, center_index - 1] * 255).numpy().astype(np.uint8)

            if self.get_size_match(prev_shape, center_shape, 0.05):
                predictions[i, 0] = 1
            else:
                predictions[i, 1] = 1
        return predictions

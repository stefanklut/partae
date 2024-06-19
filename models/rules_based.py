import re

import numpy as np
import torch


class RulesBased:
    def __init__(self):
        pass

    def device(self):
        return torch.device("cpu")

    def get_size_match(self, image_size1, image_size2, margin):
        min_multiplier = 1.0 - margin
        max_multiplier = 1.0 + margin
        border_multiplier = 0.01
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

    def __call__(self, x):
        shapes = x["shapes"]
        image_paths = x["image_paths"]

        B = shapes.shape[0]
        N = shapes.shape[1]
        assert N > 1, "There should be at least 2 images"

        center_index = N // 2

        predictions = torch.zeros((B, 2))
        for i in range(B):
            center_path = image_paths[i][center_index]
            inventory_number_dir = center_path.parent.name
            if check := re.match(r"(.+)_(.+)_(\d+)(_deelopname\d+)?", center_path.stem):
                inventory_number_file = check.group(2)
                if inventory_number_dir != inventory_number_file:
                    raise ValueError(
                        f"Inventory number in dir {inventory_number_dir} does not match with inventory number in file {inventory_number_file}. Path: {path}"
                    )
                page_number = int(check.group(3))
                if page_number == 1:
                    predictions[i, 1] = 1
                    continue

                if check.group(4):
                    predictions[i, 0] = 1
                    continue
            else:
                raise ValueError(f"Path {center_path} does not match the expected format")

            center_shape = np.asarray(shapes[i, center_index])[::-1]
            prev_shape = np.asarray(shapes[i, center_index - 1])[::-1]
            if self.get_size_match(prev_shape, center_shape, 0.1):
                predictions[i, 0] = 1
            else:
                predictions[i, 1] = 1
        return predictions

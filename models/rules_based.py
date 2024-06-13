import torch


class RulesBased:
    def __init__(self):
        pass

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
        B = shapes.shape[0]
        N = shapes.shape[1]
        assert N > 1, "There should be at least 2 images"

        center_index = N // 2 + 1
        center_shape = shapes[:, center_index]
        prev_shape = shapes[:, center_index - 1]

        predictions = torch.zeros((B, 2))
        for i in range(B):
            if self.get_size_match(prev_shape[i], center_shape[i], 0.1):
                predictions[i, 1] = 1
        return predictions

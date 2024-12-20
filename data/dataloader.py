from collections import defaultdict

import numpy as np
import torch


def collate_fn(batch):
    _images = []
    shapes = []
    texts = []
    targets = defaultdict(list)
    image_paths = []
    idcs = []
    for item in batch:
        _images.append(item["images"])
        shapes.append(item["shapes"])
        texts.append(item["texts"])
        for key, value in item["targets"].items():
            targets[key].append(value)

        image_paths.append(item["image_paths"])
        idcs.append(item["idcs"])

    # If all images are None, images are zeros
    if all(image is None for sub_images in _images for image in sub_images):
        # Keep the batch size and number of images
        batch_size = len(batch)
        images_size = len(_images[0])

        # Create a tensor of zeros
        images = torch.zeros((batch_size, images_size, 3, 1, 1))
    else:
        # Pad to the same size
        max_shape = np.max([image.size()[-2:] for sub_images in _images for image in sub_images if image is not None], axis=0)
        images = []
        for i in range(len(_images)):
            for j in range(len(_images[i])):
                if _images[i][j] is None:
                    _images[i][j] = torch.zeros((3, max_shape[0], max_shape[1]))
                _images[i][j] = torch.nn.functional.pad(
                    _images[i][j],
                    (0, int(max_shape[1] - _images[i][j].size()[-1]), 0, int(max_shape[0] - _images[i][j].size()[-2])),
                    value=0,
                )
            images.append(torch.stack(_images[i]))

        images = torch.stack(images)

    shapes = torch.tensor(shapes)
    targets = {key: torch.tensor(value) for key, value in targets.items()}

    return {"images": images, "shapes": shapes, "texts": texts, "targets": targets, "image_paths": image_paths, "idcs": idcs}

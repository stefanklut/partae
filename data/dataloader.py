import numpy as np
import torch


def collate_fn(batch):
    images = []
    shapes = []
    texts = []
    targets = []
    for item in batch:
        # Pad to the same size
        images.append(item["images"])
        shapes.append(item["shapes"])
        texts.append(item["texts"])
        targets.append(item["targets"])

    max_shape = np.max([image.size()[-2:] for image in images], axis=0)
    for i in range(len(images)):
        images[i] = torch.nn.functional.pad(
            images[i],
            (0, int(max_shape[1] - images[i].size()[-1]), 0, int(max_shape[0] - images[i].size()[-2])),
            value=0,
        )
    images = torch.stack(images)
    shapes = torch.tensor(shapes)
    targets = torch.tensor(targets)

    return {"images": images, "shapes": shapes, "texts": texts, "targets": targets}

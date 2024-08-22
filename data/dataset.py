import functools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from data.augmentations import SmartCompose
from page_xml.xmlPAGE import PageData
from utils.image_utils import load_image_array_from_path
from utils.path_utils import check_path_accessible, image_path_to_xml_path


class DocumentSeparationDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Sequence[Sequence[Path]]],
        mode: str = "train",
        target: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        number_of_images=3,
        randomize_document_order=False,
        sample_same_inventory=True,
        wrap_round=False,
        transform=None,
    ):
        self.image_paths = image_paths
        assert mode in ["train", "val", "test"], "Mode must be one of 'train', 'val', 'test'"
        self.mode = mode
        self.idx_to_idcs = {}
        idx = 0
        self.doc_lengths = defaultdict(list)
        self.inventory_lengths = []

        for i, inventory_i in enumerate(image_paths):
            for j, doc_j in enumerate(inventory_i):
                if len(doc_j) < 1:
                    raise ValueError(f"Document {i} in inventory {inventory_i} has no images")
                for k, path_i in enumerate(doc_j):
                    check_path_accessible(path_i)
                    xml_path_i = image_path_to_xml_path(path_i)
                    self.idx_to_idcs[idx] = (i, j, k)
                    idx += 1

        self.len = idx

        assert number_of_images > 0, "Number of images must be greater than 0"
        self.number_of_images = number_of_images
        self.randomize_document_order = randomize_document_order
        self.sample_same_inventory = sample_same_inventory
        self.wrap_round = wrap_round
        self.transform = transform

        if self.mode in ["train", "val"]:
            if target is not None:
                self.target = target
            else:
                self.target = []
                # If no target is provided, assume that the first image in the document is the target
                for i in range(len(image_paths)):
                    _target_inventory = []
                    for j in range(len(image_paths[i])):
                        _target_document = [1] + [0] * (len(image_paths[i][j]) - 1)
                        _target_inventory.append(_target_document)
                    self.target.append(_target_inventory)

    def __len__(self):
        return self.len

    @functools.lru_cache(maxsize=16)
    def get_image(self, i, j, k):
        image_path = self.image_paths[i][j][k].resolve()
        try:
            image = Image.open(image_path)
            image.load()
        except OSError as e:
            print(f"Could not open image {image_path}")
            return None
        return image

    @functools.lru_cache(maxsize=16)
    def get_text(self, i, j, k):
        xml_path = image_path_to_xml_path(self.image_paths[i][j][k])
        page_data = PageData.from_file(xml_path)
        text = page_data.get_transcription_dict()
        return text

    def out_of_bounds(self, i, j, k):
        return (
            i < 0
            or j < 0
            or k < 0
            or i >= len(self.image_paths)
            or j >= len(self.image_paths[i])
            or k >= len(self.image_paths[i][j])
        )

    def start_of_inventory(self, i, j, k):
        return j == 0 and k == 0

    def end_of_inventory(self, i, j, k):
        return j == len(self.image_paths[i]) - 1 and k == len(self.image_paths[i][j]) - 1

    def start_of_document(self, i, j, k):
        return k == 0

    def end_of_document(self, i, j, k):
        return k == len(self.image_paths[i][j]) - 1

    def is_first_document(self, i, j, k):
        return i == 0 and j == 0 and k == 0

    def is_last_document(self, i, j, k):
        return i == len(self.image_paths) - 1 and j == len(self.image_paths[i]) - 1 and k == len(self.image_paths[i][j]) - 1

    def get_next_scan(self, i, j, k):
        if self.end_of_document(i, j, k):
            if self.end_of_inventory(i, j, k):
                if self.is_last_document(i, j, k):
                    if self.wrap_round:
                        return 0, 0, 0
                    else:
                        return i, j, k + 1
                else:
                    return i + 1, 0, 0
            else:
                return i, j + 1, 0
        else:
            return i, j, k + 1

    def get_previous_scan(self, i, j, k):
        if self.start_of_document(i, j, k):
            if self.start_of_inventory(i, j, k):
                if self.is_first_document(i, j, k):
                    if self.wrap_round:
                        return len(self.image_paths) - 1, len(self.image_paths[-1]) - 1, len(self.image_paths[-1][-1]) - 1
                    else:
                        return i, j, k - 1
                else:
                    return i - 1, len(self.image_paths[i - 1]) - 1, len(self.image_paths[i - 1][-1]) - 1
            else:
                return i, j - 1, len(self.image_paths[i][j - 1]) - 1
        else:
            return i, j, k - 1

    # https://stackoverflow.com/a/64015315
    @staticmethod
    def random_choice_except(high: int, excluding: int, size=None, replace=True):
        assert isinstance(high, int), "high must be an integer"
        assert isinstance(excluding, int), "excluding must be an integer"
        assert excluding < high, "excluding value must be less than high"
        # generate random values in the range [0, high-1)
        choices = np.random.choice(high - 1, size, replace=replace)
        # shift values to avoid the excluded number
        return choices + (choices >= excluding)

    def get_random_next_scan(self, i, j, k):
        if self.end_of_document(i, j, k):
            if not self.sample_same_inventory:
                random_i = self.random_choice_except(len(self.image_paths), i)
            else:
                random_i = i
            random_j = np.random.choice(len(self.image_paths[random_i]))
            return random_i, random_j, 0
        else:
            return i, j, k + 1

    def get_random_previous_scan(self, i, j, k):
        if self.start_of_document(i, j, k):
            if not self.sample_same_inventory:
                random_i = self.random_choice_except(len(self.image_paths), i)
            else:
                random_i = i
            random_j = np.random.choice(len(self.image_paths[random_i]))
            return random_i, random_j, len(self.image_paths[random_i][random_j]) - 1
        else:
            return i, j, k - 1

    def __getitem__(self, idx):
        steps_back = self.number_of_images // 2
        steps_forward = self.number_of_images - 1 - steps_back

        if self.randomize_document_order:
            idcs = []
            i, j, k = self.idx_to_idcs[idx]

            # Steps back
            prev_i, prev_j, prev_k = i, j, k
            for _ in range(steps_back):
                prev_i, prev_j, prev_k = self.get_random_previous_scan(prev_i, prev_j, prev_k)
                idcs.append((prev_i, prev_j, prev_k))

            idcs.reverse()

            # Current
            idcs.append((i, j, k))

            # Steps forward
            next_i, next_j, next_k = i, j, k
            for _ in range(steps_forward):
                next_i, next_j, next_k = self.get_random_next_scan(next_i, next_j, next_k)
                idcs.append((next_i, next_j, next_k))
        else:
            idcs = []
            i, j, k = self.idx_to_idcs[idx]

            # Steps back
            prev_i, prev_j, prev_k = i, j, k
            for _ in range(steps_back):
                prev_i, prev_j, prev_k = self.get_previous_scan(prev_i, prev_j, prev_k)
                idcs.append((prev_i, prev_j, prev_k))

            idcs.reverse()

            # Current
            idcs.append((i, j, k))

            # Steps forward
            next_i, next_j, next_k = i, j, k
            for _ in range(steps_forward):
                next_i, next_j, next_k = self.get_next_scan(next_i, next_j, next_k)
                idcs.append((next_i, next_j, next_k))

        targets = []
        _images = []
        texts = []
        shapes = []
        targets = []
        image_paths = []
        _idcs = []
        for i, j, k in idcs:
            _idcs.append((i, j, k))
            if self.out_of_bounds(i, j, k):
                image = None
                shape = (0, 0)
                text = {
                    None: {
                        "text": "",
                        "coords": None,
                        "bbox": None,
                        "baseline": None,
                    }
                }
                if self.mode in ["train", "val"]:
                    target = 0
                image_path = ""

            else:
                image = self.get_image(i, j, k)
                if image is None:
                    shape = (0, 0)
                    text = self.get_text(i, j, k)
                    if self.mode in ["train", "val"]:
                        target = self.target[i][j][k]
                    image_path = self.image_paths[i][j][k]
                else:
                    shape = image.size[1], image.size[0]  # H, W
                    text = self.get_text(i, j, k)
                    if self.mode in ["train", "val"]:
                        target = self.target[i][j][k]
                    image_path = self.image_paths[i][j][k]

            image_paths.append(image_path)
            _images.append(image)
            shapes.append(shape)
            texts.append(text)
            if self.mode in ["train", "val"]:
                targets.append(target)

        images = []
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if isinstance(self.transform, SmartCompose):
            images = self.transform(_images)
        else:
            for image in _images:
                if image is None:
                    images.append(None)
                    continue
                image = self.transform(image)
                images.append(image)

        output = {"images": images, "shapes": shapes, "texts": texts, "image_paths": image_paths}

        if self.mode in ["train", "val"]:
            output["targets"] = targets
            output["idcs"] = _idcs

        return output


if __name__ == "__main__":
    test_image_paths = [
        [
            [
                Path("/home/stefan/Downloads/ushmm_test/113I/NL-HaNA_2.09.09_113I_0001.jpg"),
                Path("/home/stefan/Downloads/ushmm_test/113I/NL-HaNA_2.09.09_113I_0002.jpg"),
            ],
            [Path("/home/stefan/Downloads/ushmm_test/113I/NL-HaNA_2.09.09_113I_0003.jpg")],
            [Path("/home/stefan/Downloads/ushmm_test/113I/NL-HaNA_2.09.09_113I_0004.jpg")],
            [
                Path("/home/stefan/Downloads/ushmm_test/113I/NL-HaNA_2.09.09_113I_0005.jpg"),
                Path("/home/stefan/Downloads/ushmm_test/113I/NL-HaNA_2.09.09_113I_0006.jpg"),
                Path("/home/stefan/Downloads/ushmm_test/113I/NL-HaNA_2.09.09_113I_0007.jpg"),
            ],
            [Path("/home/stefan/Downloads/ushmm_test/113I/NL-HaNA_2.09.09_113I_0008.jpg")],
        ]
    ]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    dataset = DocumentSeparationDataset(test_image_paths, transform=transform)
    import torch.utils.data

    from data.dataloader import collate_fn

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
    item = next(iter(dataloader))
    print(item["images"].size())
    print(item["shapes"].size())
    print(item["targets"].size())
    # Text size is not fixed, so it is a list of lists
    print(len(item["texts"]), len(item["texts"][0]))

    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(len(item["images"])):
        for j in range(len(item["images"][i])):
            print(item["texts"][i][j])
            print(item["targets"][i][j])
            print(item["shapes"][i][j])
            print(item["image_paths"][i][j])
            print(item["idcs"][i][j])
            image = item["images"][i][j]
            image = image.permute(1, 2, 0).numpy()
            plt.imshow(image)
            plt.show()

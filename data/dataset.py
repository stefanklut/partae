import functools
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

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
        target: Optional[dict[str, Sequence[Sequence[Sequence[int]]]]] = None,
        number_of_images=3,
        randomize_document_order=False,
        sample_same_inventory=True,
        wrap_round=False,
        transform=None,
        check_files=False,
        # percentage value 0-100
        shuffle_doc_chance: int = 0,
        # percentage value 0-100
        random_scan_insert_chance: int = 0
    ):
        self.image_paths = image_paths
        assert mode in ["train", "val", "test"], "Mode must be one of 'train', 'val', 'test'"
        self.mode = mode
        self.idx_to_idcs = {}
        self.icds_to_idx = {}
        idx = 0
        self.doc_lengths = defaultdict(list)
        self.inventory_lengths = []
        self.shuffle_doc_chance = shuffle_doc_chance
        self.random_scan_insert_chance = random_scan_insert_chance

        total_scans = sum(sum(len(doc) for doc in inventory) for inventory in image_paths)
        with tqdm(total=total_scans, desc="Checking files") as pbar:
            idx = 0
            for i, inventory_i in enumerate(image_paths):
                for j, doc_j in enumerate(inventory_i):
                    if len(doc_j) < 1:
                        raise ValueError(f"Document {i} in inventory {inventory_i} has no images")
                    for k, path_i in enumerate(doc_j):
                        if check_files:
                            check_path_accessible(path_i)
                            xml_path_i = image_path_to_xml_path(path_i)
                        self.idx_to_idcs[idx] = (i, j, k)
                        idx += 1
                        pbar.update()

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
                self.target = {"start": [], "end": [], "middle": []}
                # If no target is provided, assume that the first image in the document is the start, and the last is the end, and all others are in between

                for i in range(len(image_paths)):
                    _target_inventory_start = []
                    _target_inventory_end = []
                    _target_inventory_middle = []

                    for j in range(len(image_paths[i])):
                        _target_document_start = [1] + [0] * (len(image_paths[i][j]) - 1)
                        _target_document_end = [0] * (len(image_paths[i][j]) - 1) + [1]
                        _target_document_middle = [
                            int(not (x or y)) for x, y in zip(_target_document_start, _target_document_end)
                        ]

                        _target_inventory_start.append(_target_document_start)
                        _target_inventory_end.append(_target_document_end)
                        _target_inventory_middle.append(_target_document_middle)

                    self.target["start"].append(_target_inventory_start)
                    self.target["end"].append(_target_inventory_end)
                    self.target["middle"].append(_target_inventory_middle)

    def __len__(self):
        return self.len

    @functools.lru_cache(maxsize=16)
    def get_image(self, inventory, document, scan):
        image_path: Path = self.image_paths[inventory][document][scan]

        # Check if thumbnail exists
        thumbnail_path = Path("/data/thumbnails/").joinpath(str(image_path.relative_to(Path("/"))) + ".thumbnail.jpg")
        try:
            image = Image.open(thumbnail_path)
            image.load()
            image = image.convert("RGB")
        except OSError as e:
            print(f"Could not open thumbnail {thumbnail_path}. Trying to open original image")
            try:
                image = Image.open(image_path.resolve())
                image.load()
                image = image.convert("RGB")
            except OSError as e:
                print(f"Could not open image {image_path}")
                return None
        return image

    @functools.lru_cache(maxsize=16)
    def get_text(self, inventory, document, scan):
        xml_path = image_path_to_xml_path(self.image_paths[inventory][document][scan])
        page_data = PageData.from_file(xml_path)
        text = page_data.get_transcription_dict()
        shape = page_data.get_size()
        return text, shape

    def out_of_bounds(self, inventory, document, scan):
        return (
            inventory < 0
            or document < 0
            or scan < 0
            or inventory >= len(self.image_paths)
            or document >= len(self.image_paths[inventory])
            or scan >= len(self.image_paths[inventory][document])
        )

    def start_of_inventory(self, inventory, document, scan):
        return document == 0 and scan == 0

    def end_of_inventory(self, inventory, document, scan):
        return document == len(self.image_paths[inventory]) - 1 and scan == len(self.image_paths[inventory][document]) - 1

    def start_of_document(self, inventory, document, scan):
        return scan == 0

    def end_of_document(self, inventory, document, scan):
        return scan == len(self.image_paths[inventory][document]) - 1

    def is_first_document(self, inventory, document, scan):
        return inventory == 0 and document == 0 and scan == 0

    def is_last_document(self, inventory, document, scan):
        return (
            inventory == len(self.image_paths) - 1
            and document == len(self.image_paths[inventory]) - 1
            and scan == len(self.image_paths[inventory][document]) - 1
        )

    def get_indices_of_document(self, inventory, document):
        indices = []
        for scan in range(len(self.image_paths[inventory][document])):
            indices.append(self.icds_to_idx[(inventory, document, scan)])

        return indices

    def get_next_scan(self, inventory, document, scan):
        if self.end_of_document(inventory, document, scan):
            if self.end_of_inventory(inventory, document, scan):
                if self.is_last_document(inventory, document, scan):
                    if self.wrap_round:
                        return 0, 0, 0
                    else:
                        return inventory, document, scan + 1
                else:
                    return inventory + 1, 0, 0
            else:
                return inventory, document + 1, 0
        else:
            return inventory, document, scan + 1

    def get_previous_scan(self, inventory, document, scan):
        if self.start_of_document(inventory, document, scan):
            if self.start_of_inventory(inventory, document, scan):
                if self.is_first_document(inventory, document, scan):
                    if self.wrap_round:
                        return len(self.image_paths) - 1, len(self.image_paths[-1]) - 1, len(self.image_paths[-1][-1]) - 1
                    else:
                        return inventory, document, scan - 1
                else:
                    return inventory - 1, len(self.image_paths[inventory - 1]) - 1, len(self.image_paths[inventory - 1][-1]) - 1
            else:
                return inventory, document - 1, len(self.image_paths[inventory][document - 1]) - 1
        else:
            return inventory, document, scan - 1

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

    def get_random_next_scan(self, inventory, document, scan):
        if self.end_of_document(inventory, document, scan):
            if not self.sample_same_inventory:
                random_inventory = self.random_choice_except(len(self.image_paths), inventory)
            else:
                random_inventory = inventory
            random_document = np.random.choice(len(self.image_paths[random_inventory]))
            return random_inventory, random_document, 0
        else:
            return inventory, document, scan + 1

    def get_random_previous_scan(self, inventory, document, scan):
        if self.start_of_document(inventory, document, scan):
            if not self.sample_same_inventory:
                random_inventory = self.random_choice_except(len(self.image_paths), inventory)
            else:
                random_inventory = inventory
            random_document = np.random.choice(len(self.image_paths[random_inventory]))
            return random_inventory, random_document, len(self.image_paths[random_inventory][random_document]) - 1
        else:
            return inventory, document, scan - 1

    def __getitem__(self, idx):
        # Add the previous and next images to the current image
        steps_back = self.number_of_images // 2
        steps_forward = self.number_of_images - 1 - steps_back

        lowest_index_of_doc = -1
        highest_index = -1
        idx_prev = []
        idx_next = []
        if np.random.randint(1, 100) <= self.shuffle_doc_chance:
            (inv, doc, scan) = self.idx_to_idcs[idx]
            indices_of_doc = self.get_indices_of_document(inv, doc)
            highest_index = indices_of_doc[-1]
            lowest_index_of_doc = indices_of_doc[0]
            np.random.shuffle(indices_of_doc)
            idx_in_doc = indices_of_doc.index(idx)
            idx_prev = indices_of_doc[:idx_in_doc]
            idx_next = indices_of_doc[idx_in_doc + 1:]


        if self.randomize_document_order:
            next_function = self.get_random_next_scan
            prev_function = self.get_random_previous_scan
        else:
            next_function = self.get_next_scan
            prev_function = self.get_previous_scan

        idcs = []
        # Get the current inventory, document and scan
        inventory, document, scan = self.idx_to_idcs[idx]

        # Get the previous inventory, document and scans based on the number of steps back
        if lowest_index_of_doc > -1:
            prev_inventory, prev_document, prev_scan = self.idx_to_idcs[lowest_index_of_doc]
        else:
            prev_inventory, prev_document, prev_scan = inventory, document, scan
        for step_back in range(steps_back):
            if len(idx_prev) > step_back:
                idcs.append(self.idx_to_idcs[idx_prev[step_back]])
            else:
                prev_inventory, prev_document, prev_scan = prev_function(prev_inventory, prev_document, prev_scan)
                idcs.append((prev_inventory, prev_document, prev_scan))


        idcs.reverse()  # Reverse the list to get the previous images in the correct order

        # Add the current inventory, document and scan
        idcs.append((inventory, document, scan))

        # Get the next inventory, document and scans based on the number of steps forward
        if highest_index > -1:
            next_inventory, next_document, next_scan = self.idx_to_idcs[highest_index]
        else:
            next_inventory, next_document, next_scan = inventory, document, scan
        for steps_forward in range(steps_forward):
            if len(idx_next) > steps_forward:
                idcs.append(self.idx_to_idcs[idx_next[steps_forward]])
            else:
                next_inventory, next_document, next_scan = next_function(next_inventory, next_document, next_scan)
                idcs.append((next_inventory, next_document, next_scan))


        # insert random scan
        if np.random.randint(1, 100) <= self.random_scan_insert_chance:
            rand_idc = (inventory, document, scan)
            while rand_idc in idcs:
                rand_idc = list(self.idx_to_idcs.values())[np.random.randint(0, len(self.idx_to_idcs))]

            insert_pos = np.random.randint(0, len(rand_idc) - 1)
            # if is middle item
            if insert_pos == (len(idcs) // 2):
                insert_pos += 1
            idcs[insert_pos] = rand_idc

        print(idcs)
        # From the obtained indices, get the images, texts and targets
        targets = defaultdict(list)
        _images = []
        texts = []
        shapes = []
        image_paths = []
        _idcs = []
        for inventory, document, scan in idcs:
            _idcs.append((inventory, document, scan))
            if self.out_of_bounds(inventory, document, scan):
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
                    target = {
                        "start": 0,
                        "end": 0,
                        "middle": 0,
                    }
                image_path = ""

            else:
                image = self.get_image(inventory, document, scan)
                if image is None:
                    image_path = self.image_paths[inventory][document][scan]
                    text, shape = self.get_text(inventory, document, scan)
                    if self.mode in ["train", "val"]:
                        target = {
                            "start": self.target["start"][inventory][document][scan],
                            "end": self.target["end"][inventory][document][scan],
                            "middle": self.target["middle"][inventory][document][scan],
                        }
                else:
                    image_path = self.image_paths[inventory][document][scan]

                    text, shape = self.get_text(inventory, document, scan)

                    if self.mode in ["train", "val"]:
                        target = {
                            "start": self.target["start"][inventory][document][scan],
                            "end": self.target["end"][inventory][document][scan],
                            "middle": self.target["middle"][inventory][document][scan],
                        }

            image_paths.append(image_path)
            _images.append(image)
            shapes.append(shape)
            texts.append(text)
            if self.mode in ["train", "val"]:
                for key, value in target.items():
                    targets[key].append(value)

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

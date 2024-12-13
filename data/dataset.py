import functools
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from queue import Queue
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
    """
    A PyTorch Dataset for document separation. The dataset returns a sequence of images, texts and targets.
    """

    def __init__(
        self,
        image_paths: Sequence[Sequence[Sequence[Path]]],
        mode: str = "train",
        number_of_images=3,
        sample_same_inventory=True,
        wrap_round=False,
        transform=None,
        check_files=False,
        # percentage value 0-1
        prob_shuffle_document: float = 0.0,
        prob_randomize_document_order: float = 0.0,
        prob_random_scan_insert: float = 0.0,
    ):
        self.image_paths = image_paths
        assert mode in ["train", "val", "test"], "Mode must be one of 'train', 'val', 'test'"
        self.mode = mode
        self.idx_to_idcs = {}
        idx = 0
        self.doc_lengths = defaultdict(list)

        # Check if all images are accessible, and create a mapping from idx to inventory, document and scan
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

        # Set the length of the dataset
        self.len = idx

        # Calculate the number of steps back and forward
        assert number_of_images > 0, "Number of images must be greater than 0"
        self.number_of_images = number_of_images
        self.steps_back = (self.number_of_images // 2) + 1
        self.steps_forward = self.number_of_images + 1 - self.steps_back

        # Initialize the next and previous scans, these are used in the shuffling of the current document
        self.next_scans = []
        self.prev_scans = []

        # Augmentation parameters
        self.prob_shuffle_document = prob_shuffle_document
        self.prob_randomize_document_order = prob_randomize_document_order
        self.prob_random_scan_insert = prob_random_scan_insert
        self.sample_same_inventory = sample_same_inventory
        self.wrap_round = wrap_round
        self.transform = transform

    def __len__(self):
        return self.len

    @functools.lru_cache(maxsize=16)
    def get_image(self, inventory: int, document: int, scan: int) -> Optional[Image.Image]:
        """
        Get the image at the given inventory, document and scan. If the image is not found, None is returned. Uses lru_cache to cache the images.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Returns:
            Optional[Image.Image]: The image at the given inventory, document and scan
        """
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
        """
        Get the text at the given inventory, document and scan. Uses lru_cache to cache the text.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Returns:
            Tuple[Dict[str, str], Tuple[int, int]]: The text at the given inventory, document and scan, and the shape of the image as found in the page xml
        """
        xml_path = image_path_to_xml_path(self.image_paths[inventory][document][scan])
        page_data = PageData.from_file(xml_path)
        text = page_data.get_transcription_dict()
        shape = page_data.get_size()
        return text, shape

    def is_out_of_bounds(self, inventory: int, document: int, scan: int) -> bool:
        """
        Check if the given inventory, document and scan is out of bounds.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Returns:
            bool: True if the inventory, document or scan is out of bounds, False otherwise
        """
        return (
            inventory < 0
            or document < 0
            or scan < 0
            or inventory >= len(self.image_paths)
            or document >= len(self.image_paths[inventory])
            or scan >= len(self.image_paths[inventory][document])
        )

    def scan_in_document(self, inventory: int, document: int, scan: int) -> bool:
        """
        Checks if the scan index is within the valid range for the specified document in the inventory.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Returns:
            bool: True if the scan index is valid, False otherwise.
        """

        return 0 <= scan < len(self.image_paths[inventory][document])

    def get_first_document(self, inventory: int) -> tuple[int, int]:
        """
        Retrieve the first document in the given inventory.

        Args:
            inventory (int): The inventory index

        Returns:
            tuple[int, int]: The inventory and document index of the first document
        """
        return inventory, 0

    def get_last_document(self, inventory: int) -> tuple[int, int]:
        """
        Retrieve the last document for a given inventory.

        Args:
            inventory (int): The inventory index

        Returns:
            tuple[int, int]: A tuple containing the inventory index and the index of the last document.
        """

        return inventory, len(self.image_paths[inventory]) - 1

    def get_next_document(self, inventory: int, document: int) -> Optional[tuple[int, int]]:
        """
        Retrieve the next document in the inventory.

        Args:
            inventory (int): The inventory index
            document (int): The document index

        Returns:
            Optional[tuple[int, int]]: A tuple containing the inventory index and the index of the next document. None if the next document is out of bounds.
        """
        if self.is_out_of_bounds(inventory, document + 1, 0):
            return None
        return inventory, document + 1

    def get_prev_document(self, inventory: int, document: int) -> Optional[tuple[int, int]]:
        """
        Retrieve the previous document in the inventory.

        Args:
            inventory (int): The inventory index
            document (int): The document index

        Returns:
            Optional[tuple[int, int]]: A tuple containing the inventory index and the index of the previous document. None if the previous document is out of bounds.
        """
        if self.is_out_of_bounds(inventory, document - 1, 0):
            return None
        return inventory, document - 1

    def get_first_inventory(self) -> int:
        """
        Get the index of the first inventory.

        Returns:
            int: The index of the first inventory
        """
        return 0

    def get_last_inventory(self) -> int:
        """
        Get the index of the last inventory.

        Returns:
            int: The index of the last inventory
        """
        return len(self.image_paths) - 1

    def get_next_inventory(self, inventory: int) -> Optional[int]:
        """
        Get the index of the next inventory.

        Args:
            inventory (int): The inventory index

        Returns:
            Optional[int]: The index of the next inventory. None if the next inventory is out of bounds.
        """
        if self.is_out_of_bounds(inventory + 1, 0, 0):
            return None
        return inventory + 1

    def get_prev_inventory(self, inventory: int) -> Optional[int]:
        """
        Get the index of the previous inventory.

        Args:
            inventory (int): The inventory index

        Returns:
            Optional[int]: The index of the previous inventory. None if the previous inventory is out of bounds.
        """
        if self.is_out_of_bounds(inventory - 1, 0, 0):
            return None
        return inventory - 1

    def get_all_scans_in_document(self, inventory: int, document: int) -> list[tuple[int, int, int]]:
        """
        Get all scans in the given document in a given inventory as a list of tuples.

        Args:
            inventory (int): The inventory index
            document (int): The document index

        Returns:
            list[tuple[int, int, int]]: A list of tuples containing the inventory, document and scan indices. Empty list if the document is out of bounds.
        """
        if self.is_out_of_bounds(inventory, document, 0):
            return []
        return [(inventory, document, i) for i in range(len(self.image_paths[inventory][document]))]

    def fill_next_prev_scans(self, inventory: int, document: int, scan: int) -> None:
        """
        Fill the next and previous scans for the given inventory, document and scan. The next and previous scans are used in the shuffling of the current document.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Raises:
            ValueError: If the document is out of bounds
        """
        self.next_scans = []
        self.prev_scans = []

        if self.prob_shuffle_document > np.random.rand():
            all_scans = self.get_all_scans_in_document(inventory, document)
            if all_scans is None:
                raise ValueError("Document is out of bounds")
            if len(all_scans) == 1:
                return
            np.random.shuffle(all_scans)

            before_scan = True
            after_scan = False

            for i in range(len(all_scans)):
                if all_scans[i] == (inventory, document, scan):
                    before_scan = False
                    after_scan = True
                    continue
                if before_scan:
                    self.prev_scans.append(all_scans[i])
                elif after_scan:
                    self.next_scans.append(all_scans[i])

    def get_next_scans_in_document(self, inventory: int, document: int, scan: int) -> list[tuple[int, int, int]]:
        """
        Get the next scans in the given document in the given inventory. If the next scans are already filled, the filled scans are returned.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Returns:
            list[tuple[int, int, int]]: A list of tuples containing the inventory, document and scan indices of the next scans
        """
        if self.next_scans or self.prev_scans:
            return self.next_scans
        assert self.scan_in_document(inventory, document, scan), "Scan is out of bounds"
        output = [(inventory, document, i) for i in range(scan + 1, len(self.image_paths[inventory][document]))]
        return output

    def get_prev_scans_in_document(self, inventory: int, document: int, scan: int) -> list[tuple[int, int, int]]:
        """
        Get the previous scans in the given document in the given inventory. If the previous scans are already filled, the filled scans are returned.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Returns:
            list[tuple[int, int, int]]: A list of tuples containing the inventory, document and scan indices of the previous scans
        """
        if self.next_scans or self.prev_scans:
            return self.prev_scans
        assert self.scan_in_document(inventory, document, scan), f"Scan is out of bounds {inventory} {document} {scan}"
        output = [(inventory, document, i) for i in range(0, scan)]
        return output

    # https://stackoverflow.com/a/64015315
    @staticmethod
    def random_choice_except(high: int, excluding: int, size=None, replace=True) -> np.ndarray | int:
        """
        Generate random values in the range [0, high-1) without the value excluding

        Args:
            high (int): The upper bound of the range
            excluding (int): The value to exclude
            size (int, optional): The number of random values to generate. Defaults to None.
            replace (bool, optional): Whether to sample with replacement. Defaults to True.

        Returns:
            np.ndarray | int: The random values in the range [0, high-1) without the value excluding
        """
        assert isinstance(high, int), "high must be an integer"
        assert isinstance(excluding, int), "excluding must be an integer"
        assert excluding < high, "excluding value must be less than high"
        # generate random values in the range [0, high-1)
        choices = np.random.choice(high - 1, size, replace=replace)
        # shift values to avoid the excluded number
        return choices + (choices >= excluding)

    def get_scans_in_next_document(
        self, inventory: int, document: int
    ) -> tuple[Optional[list[tuple[int, int, int]]], tuple[int, int]]:
        """
        Get the scans in the next document in the given inventory and document.

        Args:
            inventory (int): The inventory index
            document (int): The document index

        Returns:
            list[tuple[int, int, int]]: A list of tuples containing the inventory, document and scan indices of the scans in the next document
        """
        if self.prob_randomize_document_order > np.random.rand():
            # Randomize document order
            if self.sample_same_inventory:
                if len(self.image_paths[inventory]) == 1:
                    next_inventory_document = None
                else:
                    next_inventory_document = inventory, self.random_choice_except(len(self.image_paths[inventory]), document)
            else:
                next_inventory = np.random.choice(len(self.image_paths))
                if next_inventory == inventory:
                    if len(self.image_paths[next_inventory]) == 1:
                        return None, (inventory, document)
                    next_document = self.random_choice_except(len(self.image_paths[next_inventory]), document)
                else:
                    next_document = np.random.choice(len(self.image_paths[next_inventory]))
                next_inventory_document = next_inventory, next_document
        else:
            # Get next document
            next_inventory_document = self.get_next_document(inventory, document)

        # If the next document is out of bounds, check if we should wrap around
        if next_inventory_document is None:
            if self.wrap_round:
                if self.sample_same_inventory:
                    next_inventory_document = self.get_first_document(inventory)
                else:
                    next_inventory = self.get_next_inventory(inventory)
                    if next_inventory is None:
                        next_inventory = self.get_first_inventory()
                    next_inventory_document = self.get_first_document(next_inventory)
            else:
                return None, (inventory, document)

        scans = self.get_all_scans_in_document(*next_inventory_document)

        if self.prob_shuffle_document > np.random.rand():
            np.random.shuffle(scans)

        return scans, next_inventory_document

    def get_scans_in_prev_document(
        self, inventory: int, document: int
    ) -> tuple[Optional[list[tuple[int, int, int]]], tuple[int, int]]:
        # TODO Remove slight bias towards smaller inventories. Sampling inventories first creates this problem. Sample should be uniform
        if self.prob_randomize_document_order > np.random.rand():
            # Randomize document order
            if self.sample_same_inventory:
                if len(self.image_paths[inventory]) == 1:
                    prev_inventory_document = None
                else:
                    prev_inventory_document = inventory, self.random_choice_except(len(self.image_paths[inventory]), document)

            else:
                prev_inventory = np.random.choice(len(self.image_paths))

                if prev_inventory == inventory:
                    if len(self.image_paths[prev_inventory]) == 1:
                        prev_inventory_document = None
                        return None, (inventory, document)
                    prev_document = self.random_choice_except(len(self.image_paths[prev_inventory]), document)
                else:
                    prev_document = np.random.choice(len(self.image_paths[prev_inventory]))
                prev_inventory_document = prev_inventory, prev_document
        else:
            # Get previous document
            prev_inventory_document = self.get_prev_document(inventory, document)

        # If the previous document is out of bounds, check if we should wrap around
        if prev_inventory_document is None:
            if self.wrap_round:
                if self.sample_same_inventory:
                    prev_inventory_document = self.get_last_document(inventory)
                else:
                    prev_inventory = self.get_prev_inventory(inventory)
                    if prev_inventory is None:
                        prev_inventory = self.get_last_inventory()
                    prev_inventory_document = self.get_last_document(prev_inventory)
            else:
                return None, (inventory, document)

        scans = self.get_all_scans_in_document(*prev_inventory_document)

        if self.prob_shuffle_document > np.random.rand():
            np.random.shuffle(scans)

        return scans, prev_inventory_document

    def get_next_idcs(self, inventory: int, document: int, scan: int) -> list[tuple[int, int, int] | None]:
        """
        Get the next indices in the dataset.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Returns:
            list[tuple[int, int, int]]: A list of tuples containing the inventory, document and scan indices of the next scans
        """
        next_idcs = []
        next_idcs.extend(self.get_next_scans_in_document(inventory, document, scan))
        next_document = inventory, document
        while len(next_idcs) < self.steps_forward:
            next_scans, next_document = self.get_scans_in_next_document(*next_document)
            if next_scans is None:
                next_idcs.append(None)
                continue
            next_idcs.extend(next_scans)
        return next_idcs[: self.steps_forward]

    def get_previous_idcs(self, inventory: int, document: int, scan: int) -> list[tuple[int, int, int] | None]:
        """
        Get the previous indices in the dataset.

        Args:
            inventory (int): The inventory index
            document (int): The document index
            scan (int): The scan index

        Returns:
            list[tuple[int, int, int]]: A list of tuples containing the inventory, document and scan indices of the previous scans
        """
        prev_idcs = []
        prev_idcs.extend(self.get_prev_scans_in_document(inventory, document, scan))
        prev_document = inventory, document
        while len(prev_idcs) < self.steps_back:
            prev_scans, prev_document = self.get_scans_in_prev_document(*prev_document)
            if prev_scans is None:
                prev_idcs = [None] + prev_idcs
                continue
            prev_idcs = prev_scans + prev_idcs

        return prev_idcs[-self.steps_back :]

    def insert_random_scan(self, idcs: list[tuple[int, int, int]]) -> None:
        """
        Insert a random scan somewhere in the list of indices. The random scan cannot be inserted in the middle of the list.

        Args:
            idcs (list[tuple[int, int, int]]): A list of tuples containing the inventory, document and scan indices
        """
        if self.prob_random_scan_insert > np.random.rand():
            middle_position = len(idcs) // 2
            remaining_positions = list(set(range(len(idcs))) - set([middle_position]))
            insert_position = np.random.choice(remaining_positions)

            if self.sample_same_inventory:
                inventory, document, _ = idcs[middle_position]
                possible_documents = set(range(len(self.image_paths[inventory])))
                possible_documents.remove(document)
                random_inventory = inventory
                if len(possible_documents) == 0:
                    return
                random_document = np.random.choice(list(possible_documents))
                possible_scans = list(range(len(self.image_paths[random_inventory][random_document])))
                random_scan = np.random.choice(possible_scans)
            else:
                inventory, document, _ = idcs[middle_position]
                random_inventory = np.random.choice(list(range(len(self.image_paths))))
                possible_documents = set(range(len(self.image_paths[random_inventory])))
                if random_inventory == inventory:
                    possible_documents.remove(document)
                if len(possible_documents) == 0:
                    return
                random_document = np.random.choice(list(possible_documents))
                possible_scans = list(range(len(self.image_paths[random_inventory][random_document])))
                random_scan = np.random.choice(possible_scans)

            idcs[insert_position] = (random_inventory, random_document, random_scan)

    def __getitem__(self, idx: int) -> dict:
        """
        Get the images, texts and targets for the given index.

        Args:
            idx (int): The index of the dataset

        Returns:
            dict: A dictionary containing the images, texts and targets (if in train or val mode) for the given index
        """
        idcs = []
        # Get the current inventory, document and scan
        inventory, document, scan = self.idx_to_idcs[idx]

        self.fill_next_prev_scans(inventory, document, scan)

        # Add the precceding scans
        idcs.extend(self.get_previous_idcs(inventory, document, scan))

        # Add the current scan
        idcs.append((inventory, document, scan))

        # Add the following scans
        idcs.extend(self.get_next_idcs(inventory, document, scan))

        self.insert_random_scan(idcs)

        # print(idcs)

        # From the obtained indices, get the images, texts and targets
        targets = defaultdict(list)
        _images = []
        texts = []
        shapes = []
        image_paths = []
        for i in range(self.number_of_images):
            prev_idx = idcs[i]
            idx = idcs[i + 1]
            next_idx = idcs[i + 2]
            # If the index is out of bounds, set the image to None and the text to an empty string
            if idx is None:
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
            else:
                inventory, document, scan = idx
                image = self.get_image(inventory, document, scan)

            # Set the target based on the previous and next indices
            prev_inventory = prev_idx[0] if prev_idx is not None else None
            next_inventory = next_idx[0] if next_idx is not None else None
            prev_document = prev_idx[1] if prev_idx is not None else None
            next_document = next_idx[1] if next_idx is not None else None

            # If the previous document is None, the current scan is the first scan of a document.
            # Otherwise, if the previous document is different from the current document, the current scan is the first scan in a document.
            # The same logic applies to determining if the current scan is the last scan in a document, but done by comparing with the next document.
            start = prev_document is None or (prev_document != document or prev_inventory != inventory)
            end = next_document is None or (next_document != document or next_inventory != inventory)
            middle = not start and not end

            target = {
                "start": int(start),
                "middle": int(middle),
                "end": int(end),
            }

            # If the image did not load or is out of bounds, set the image path to None
            if image is None:
                image_path = None
            else:
                image_path = self.image_paths[inventory][document][scan]
            text, shape = self.get_text(inventory, document, scan)

            # Append the image, text, target and image path to the lists
            image_paths.append(image_path)
            _images.append(image)
            shapes.append(shape)
            texts.append(text)
            if self.mode in ["train", "val"]:
                for key, value in target.items():
                    targets[key].append(value)

        # Transform the images
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
            output["idcs"] = idcs[1:-1]

        return output


if __name__ == "__main__":

    test_image_paths = [
        [
            [
                Path("/home/stefank/Downloads/mini-republic/train/NL-HaNA_1.01.02_62_0109.jpg"),
                Path("/home/stefank/Downloads/mini-republic/train/NL-HaNA_1.01.02_62_0118.jpg"),
            ],
            [Path("/home/stefank/Downloads/mini-republic/train/NL-HaNA_1.01.02_62_0258.jpg")],
            [Path("/home/stefank/Downloads/mini-republic/train/NL-HaNA_1.01.02_62_0504.jpg")],
            [
                Path("/home/stefank/Downloads/mini-republic/train/NL-HaNA_1.01.02_3097_0039.jpg"),
                Path("/home/stefank/Downloads/mini-republic/train/NL-HaNA_1.01.02_3097_0079.jpg"),
                Path("/home/stefank/Downloads/mini-republic/train/NL-HaNA_1.01.02_3097_0114.jpg"),
            ],
        ],
        [
            [Path("/home/stefank/Downloads/mini-republic/train/NL-HaNA_1.01.02_3097_0137.jpg")],
        ],
    ]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    dataset = DocumentSeparationDataset(
        test_image_paths,
        transform=transform,
        prob_randomize_document_order=0,
        prob_random_scan_insert=1,
        sample_same_inventory=True,
        wrap_round=False,
    )
    import torch.utils.data

    from data.dataloader import collate_fn

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
    item = next(iter(dataloader))
    print("Images tensor size:", item["images"].size())
    print("Shapes tensor size:", item["shapes"].size())
    print("Targets start tensor size:", item["targets"]["start"].size())
    # Text size is not fixed, so it is a list of lists
    print("Text size:", len(item["texts"]), len(item["texts"][0]))

    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(len(item["images"])):
        for j in range(len(item["images"][i])):
            # print(item["texts"][i][j])
            print("Target start:", item["targets"]["start"][i][j])
            print("Target middle:", item["targets"]["middle"][i][j])
            print("Target end:", item["targets"]["end"][i][j])
            print("Shape:", item["shapes"][i][j])
            print("Image path:", item["image_paths"][i][j])
            print("Indices:", item["idcs"][i][j])
            image = item["images"][i][j]
            image = image.permute(1, 2, 0).numpy()
            plt.imshow(image)
            plt.show()

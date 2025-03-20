# Modified from P2PaLA

import datetime
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import NoneType
from typing import Iterable, TypedDict

import numpy as np

from utils.text_utils import combine_texts

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger_name
from utils.tempdir import AtomicFileName

_VALID_TYPES = {tuple, list, str, int, float, bool, NoneType}


class PageData:
    """Class to process PAGE xml files"""

    def __init__(self, creator=None):
        """
        Args:
            filepath (string): Path to PAGE-xml file.
        """
        self.logger = logging.getLogger(get_logger_name())

        self.creator = "partae" if creator is None else creator

        # REVIEW should this be replaced with the newer pageXML standard?
        self.XMLNS = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": " ".join(
                [
                    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                    " http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
                ]
            ),
        }
        self.size = None
        self.filepath = None
        self.name = None

        self.root = None
        self.base = None

    @classmethod
    def from_file(cls, filepath: Path, creator=None):
        instance = cls(creator=creator)
        instance.filepath = filepath
        instance.name = filepath.stem

        tree = ET.parse(filepath)
        instance.root = tree.getroot()
        instance.base = "".join([instance.root.tag.rsplit("}", 1)[0], "}"])
        return instance

    @classmethod
    def from_string(cls, xml_string: str, filepath: Path, creator=None):
        instance = cls(creator=creator)
        instance.filepath = filepath
        instance.name = filepath.stem

        tree = ET.ElementTree(ET.fromstring(xml_string))
        instance.root = tree.getroot()
        instance.base = "".join([instance.root.tag.rsplit("}", 1)[0], "}"])
        return instance

    def set_size(self, size: tuple[int, int]):
        self.size = size

    def get_region(self, region_name):
        """
        get all regions in PAGE which match region_name
        """
        return self.root.findall("".join([".//", self.base, region_name])) or None

    def get_zones(self, region_names):
        to_return = {}
        idx = 0
        for element in region_names:
            for node in self.root.findall("".join([".//", self.base, element])):
                to_return[idx] = {
                    "coords": self.get_coords(node),
                    "type": self.get_region_type(node),
                    "id": self.get_id(node),
                }
                idx += 1
        if to_return:
            return to_return
        else:
            return None

    def get_id(self, element) -> str:
        """
        get Id of current element
        """
        return str(element.attrib.get("id"))

    def get_region_type(self, element):
        """
        Returns the type of element
        """
        try:
            re_match = re.match(r".*structure {.*type:(.*);.*}", element.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No region type defined for {self.get_id(element)} at {self.filepath}")
            return None
        if re_match is None:
            self.logger.warning(f"No region type defined for {self.get_id(element)} at {self.filepath}")
            return None
        e_type = re_match.group(1)

        return e_type

    def get_size(self):
        """
        Get Image size defined on XML file
        """
        if self.size is not None:
            return self.size

        img_width = int(self.root.findall("".join(["./", self.base, "Page"]))[0].get("imageWidth"))
        img_height = int(self.root.findall("".join(["./", self.base, "Page"]))[0].get("imageHeight"))
        self.size = (img_height, img_width)

        return self.size

    def get_coords(self, element) -> np.ndarray:
        """
        Get the coordinates of the element
        """
        str_coords = element.findall("".join(["./", self.base, "Coords"]))[0].attrib.get("points").split()
        return np.array([i.split(",") for i in str_coords]).astype(np.int32)

    def get_baseline_coords(self, element) -> np.ndarray:
        """
        Get the coordinates of the baseline of the element
        """
        str_coords = element.findall("".join(["./", self.base, "Baseline"]))[0].attrib.get("points").split()
        return np.array([i.split(",") for i in str_coords]).astype(np.int32)

    def get_polygons(self, element_name):
        """
        returns a list of polygons for the element desired
        """
        polygons = []
        for element in self._iter_element(element_name):
            # --- get element type
            e_type = self.get_region_type(element)
            if e_type is None:
                self.logger.warning(f'Element type "{e_type}" undefined, set to "other"')
                e_type = "other"

            polygons.append([self.get_coords(element), e_type])

        return polygons

    def _iter_element(self, element):
        return self.root.iterfind("".join([".//", self.base, element]))

    def iter_class_coords(self, element, class_dict):
        for node in self._iter_element(element):
            element_type = self.get_region_type(node)
            if element_type is None or element_type not in class_dict:
                self.logger.warning(f'Element type "{element_type}" undefined in class dict {self.filepath}')
                continue
            element_class = class_dict[element_type]
            element_coords = self.get_coords(node)

            # Ignore lines
            if element_coords.shape[0] < 3:
                continue

            yield element_class, element_coords

    def iter_baseline_coords(self):
        for node in self._iter_element("Baseline"):
            str_coords = node.attrib.get("points")
            if str_coords is None:
                continue
            split_str_coords = str_coords.split()
            # REVIEW currently ignoring empty baselines, doubling single value baselines (otherwise they are not drawn)
            if len(split_str_coords) == 0:
                continue
            if len(split_str_coords) == 1:
                split_str_coords = split_str_coords * 2  # double for cv2.polyline
            coords = np.array([i.split(",") for i in split_str_coords]).astype(np.int32)
            yield coords

    def iter_class_baseline_coords(self, element, class_dict):
        for class_node in self._iter_element(element):
            element_type = self.get_region_type(class_node)
            if element_type is None or element_type not in class_dict:
                self.logger.warning(f'Element type "{element_type}" undefined in class dict {self.filepath}')
                continue
            element_class = class_dict[element_type]
            for baseline_node in class_node.iterfind("".join([".//", self.base, "Baseline"])):
                str_coords = baseline_node.attrib.get("points")
                if str_coords is None:
                    continue
                split_str_coords = str_coords.split()
                if len(split_str_coords) == 0:
                    continue
                if len(split_str_coords) == 1:
                    split_str_coords = split_str_coords * 2  # double for cv2.polyline
                coords = np.array([i.split(",") for i in split_str_coords]).astype(np.int32)
                yield element_class, coords

    def iter_text_line_coords(self):
        for node in self._iter_element("TextLine"):
            coords = self.get_coords(node)
            yield coords

    def get_text(self, element):
        """
        get Text defined for element
        """
        text_node = element.find("".join(["./", self.base, "TextEquiv"]))
        if text_node is None:
            self.logger.info(f"No Text node found for line {self.get_id(element)} at {self.filepath}")
            return ""
        else:
            child_node = text_node.find("*")
            if child_node is None or child_node.text is None:
                self.logger.info(f"No text found in line {self.get_id(element)} at {self.filepath}")
                return ""
            else:
                return child_node.text

    def get_transcription(self):
        """Extracts text from each line on the XML file"""
        data = {}
        for element in self.root.findall("".join([".//", self.base, "TextRegion"])):
            r_id = self.get_id(element)
            for line in element.findall("".join([".//", self.base, "TextLine"])):
                l_id = self.get_id(line)
                data["_".join([r_id, l_id])] = self.get_text(line)

        return data

    def get_transcription_dict(self):
        data = {}
        for element in self.root.findall("".join([".//", self.base, "TextRegion"])):
            r_id = self.get_id(element)
            for line in element.findall("".join([".//", self.base, "TextLine"])):
                l_id = self.get_id(line)
                coords = self.get_coords(line)
                data["_".join([r_id, l_id])] = {
                    "text": self.get_text(line),
                    "coords": coords,
                    "bbox": np.asarray([np.min(coords, axis=0), np.max(coords, axis=0)]),
                    "baseline": self.get_baseline_coords(line),
                }
        return data

    def get_combined_transcription(self):
        """Extracts text from each line on the XML file and combines them"""
        text = self.get_transcription()
        return combine_texts(text.values())

    def write_transcriptions(self, out_dir):
        """write out one txt file per text line"""
        # for line, text in self.get_transcription().iteritems():
        for line, text in list(self.get_transcription().items()):
            fh = open(os.path.join(out_dir, "".join([self.name, "_", line, ".txt"])), "w")
            fh.write(text + "\n")
            fh.close()

    ## NEW PAGEXML

    def new_page(self, name, rows, cols):
        """create a new PAGE xml"""
        self.xml = ET.Element("PcGts")
        self.xml.attrib = self.XMLNS
        self.metadata = ET.SubElement(self.xml, "Metadata")
        ET.SubElement(self.metadata, "Creator").text = self.creator
        ET.SubElement(self.metadata, "Created").text = datetime.datetime.today().strftime("%Y-%m-%dT%X")
        ET.SubElement(self.metadata, "LastChange").text = datetime.datetime.today().strftime("%Y-%m-%dT%X")
        self.page = ET.SubElement(self.xml, "Page")
        self.page.attrib = {
            "imageFilename": name,
            "imageWidth": cols,
            "imageHeight": rows,
        }

    def add_processing_step(self, git_hash: str, uuid: str):
        if git_hash is None:
            raise TypeError(f"git_hash is None")
        if uuid is None:
            raise TypeError(f"uuid is None")
        if self.metadata is None:
            raise TypeError(f"self.metadata is None")

        processing_step = ET.SubElement(self.metadata, "MetadataItem")
        processing_step.attrib = {
            "type": "processingStep",
            "name": "layout-analysis",
            "value": "partae",
        }
        labels = ET.SubElement(processing_step, "Labels")
        git_hash_element = ET.SubElement(labels, "Label")
        git_hash_element.attrib = {
            "type": "githash",
            "value": git_hash,
        }

        uuid_element = ET.SubElement(labels, "Label")
        uuid_element.attrib = {
            "type": "uuid",
            "value": uuid,
        }

    def add_element(self, region_class, region_id, region_type, region_coords, parent=None):
        """add element to parent node"""
        parent = self.page if parent is None else parent
        t_reg = ET.SubElement(parent, region_class)
        t_reg.attrib = {
            "id": str(region_id),
            "custom": f"structure {{type:{region_type};}}",
        }
        ET.SubElement(t_reg, "Coords").attrib = {"points": region_coords}
        return t_reg

    def remove_element(self, element, parent=None):
        """remove element from parent node"""
        parent = self.page if parent is None else parent
        parent.remove(element)

    def add_baseline(self, b_coords, parent):
        """add baseline element ot parent line node"""
        ET.SubElement(parent, "Baseline").attrib = {"points": b_coords}

    def save_xml(self, filepath):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        with AtomicFileName(filepath) as path:
            tree.write(path, encoding="UTF-8", xml_declaration=True)

    def _indent(self, elem, level=0):
        """
        Function borrowed from:
            http://effbot.org/zone/element-lib.htm#prettyprint
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

import json
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, request
from PIL import Image
from prometheus_client import Counter, Gauge, generate_latest
from torchvision.transforms import Resize, ToTensor

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))  # noqa: E402
from data.augmentations import PadToMaxSize, SmartCompose
from inference import Predictor
from page_xml.xmlPAGE import PageData
from utils.image_utils import load_image_array_from_bytes
from utils.logging_utils import get_logger_name

logger = logging.getLogger(get_logger_name())

# Reading environment files
try:
    max_queue_size_string: str = os.environ["PARTAE_MAX_QUEUE_SIZE"]
    model_base_path_string: str = os.environ["PARTAE_MODEL_BASE_PATH"]
    output_base_path_string: str = os.environ["PARTAE_OUTPUT_BASE_PATH"]
except KeyError as error:
    raise KeyError(f"Missing PARTAE Environment variable: {error.args[0]}")

# Convert
max_queue_size = int(max_queue_size_string)
model_base_path = Path(model_base_path_string)
output_base_path = Path(output_base_path_string)

# Checks if ENV variable exist
if not model_base_path.is_dir():
    raise FileNotFoundError(f"PARTAE_MODEL_BASE_PATH: {model_base_path} is not found in the current filesystem")
if not output_base_path.is_dir():
    raise FileNotFoundError(f"PARTAE_OUTPUT_BASE_PATH: {output_base_path} is not found in the current filesystem")

app = Flask(__name__)

predictor = None


class PredictorWrapper:
    """
    Wrapper around the page generation code
    """

    def __init__(self) -> None:
        self.model_name: Optional[str] = None
        self.predictor: Optional[Predictor] = None

    def setup_model(self, model_name: str):
        """
        Create the model and post-processing code

        Args:
            model_name (str): Model name, used to determine what model to load from models present in base path
            args (DummyArgs): Dummy version of command line arguments, to set up config
        """
        # If model name matches current model name return without init
        if model_name is not None and self.predictor is not None and model_name == self.model_name:
            return

        self.model_name = model_name
        model_path = model_base_path.joinpath(self.model_name)

        self.predictor = Predictor(model_path)


predict_wrapper = PredictorWrapper()

max_workers = 1
max_queue_size = max_workers + max_queue_size

# Run a separate thread on which the GPU runs and processes requests put in the queue
executor = ThreadPoolExecutor(max_workers=max_workers)

# Prometheus metrics to be returned
queue_size_gauge = Gauge("queue_size", "Size of worker queue").set_function(lambda: executor._work_queue.qsize())
images_processed_counter = Counter("images_processed", "Total number of images processed")
exception_predict_counter = Counter("exception_predict", "Exception thrown in predict() function")


def get_middle_path(paths: list[list[str]]) -> Path:
    """
    Get the middle path of a list of paths

    Args:
        paths (list[list[Path]]): List of paths to get the middle path of

    Returns:
        Path: Middle path of the list
    """
    return Path(paths[0][len(paths[0]) // 2])


def predict_class(
    data: dict[str, Any],
    identifier: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Run the prediction for the given image

    Args:
        data (dict[str, Any]): Data to predict on
        identifier (str): Unique identifier for the image
        model_name (str): Name of the model to use for prediction

    Raises:
        TypeError: If the current Predictor is not initialized

    Returns:
        dict[str, Any]: Information about the processed image
    """
    input_args = locals()
    try:
        predict_wrapper.setup_model(model_name=model_name)

        image_path = get_middle_path(data["image_paths"])

        output_path = output_base_path.joinpath(identifier, image_path.name).with_suffix(".json")
        if predict_wrapper.predictor is None:
            raise TypeError("The current Predictor is not initialized")

        if not output_path.parent.is_dir():
            output_path.parent.mkdir()

        output = predict_wrapper.predictor(data)
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                output[key] = value.cpu().item()

        with open(output_path, "w") as file:
            json.dump(output, file)

        images_processed_counter.inc()
        return input_args
    except Exception as exception:
        # Catch CUDA out of memory errors
        if isinstance(exception, torch.cuda.OutOfMemoryError) or (
            isinstance(exception, RuntimeError) and "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in str(exception)
        ):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # HACK remove traceback to prevent complete halt of program, not sure why this happens
            exception = exception.with_traceback(None)

        return input_args | {"exception": exception}


class ResponseInfo(TypedDict, total=False):
    """
    Template for what fields are allowed in the response
    """

    status_code: int
    identifier: str
    images: list[str]
    texts: list[str]
    whitelist: list[str]
    added_queue_position: int
    remaining_queue_size: int
    added_time: str
    model_name: str
    error_message: str


def abort_with_info(
    status_code: int,
    error_message: str,
    info: Optional[ResponseInfo] = None,
):
    """
    Abort while still providing info about what went wrong

    Args:
        status_code (int): Error type code
        error_message (str): Message
        info (Optional[ResponseInfo], optional): Response info. Defaults to None.
    """
    if info is None:
        info = ResponseInfo(status_code=status_code)  # type: ignore
    info["error_message"] = error_message
    info["status_code"] = status_code
    response = jsonify(info)
    response.status_code = status_code
    abort(response)


def check_exception_callback(future: Future):
    """
    Log on exception

    Args:
        future (Future): Results from other thread
    """
    results = future.result()
    if "exception" in results:
        logger.exception(results, exc_info=results["exception"])


@app.route("/predict", methods=["POST"])
@exception_predict_counter.count_exceptions()
def predict() -> tuple[Response, int]:
    """
    Run the prediction on a submitted image

    Returns:
        Response: Submission response
    """
    if request.method != "POST":
        abort(405)

    response_info = ResponseInfo(status_code=500)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    response_info["added_time"] = current_time

    try:
        identifier = request.form["identifier"]
        response_info["identifier"] = identifier
    except KeyError as error:
        abort_with_info(400, "Missing identifier in form", response_info)

    try:
        model_name = request.form["model"]
        response_info["model_name"] = model_name
    except KeyError as error:
        abort_with_info(400, "Missing model in form", response_info)

    try:
        post_file = request.files.getlist("images[]")
        if len(post_file) == 0:
            raise KeyError("No image in form")
    except KeyError as error:
        abort_with_info(400, "Missing image in form", response_info)

    try:
        post_text = request.files.getlist("texts[]")
        if len(post_text) == 0:
            raise KeyError("No text in form")
    except KeyError as error:
        abort_with_info(400, "Missing text in form", response_info)

    if len(post_file) != len(post_text):
        abort_with_info(400, "Number of images and texts do not match", response_info)

    # TODO Maybe make slightly more stable/predicable, https://docs.python.org/3/library/threading.html#threading.Semaphore https://gist.github.com/frankcleary/f97fe244ef54cd75278e521ea52a697a
    queue_size = executor._work_queue.qsize()
    response_info["added_queue_position"] = queue_size
    response_info["remaining_queue_size"] = max_queue_size - queue_size
    if queue_size > max_queue_size:
        abort_with_info(429, "Exceeding queue size", response_info)

    transform = SmartCompose(
        [
            ToTensor(),
            PadToMaxSize(),
            Resize((224, 224)),
        ]
    )

    # Load the multiple images
    _images = []
    image_names = []
    for post_file_i in post_file:
        if post_file_i.filename:
            image_name = Path(post_file_i.filename)
            image_names.append(str(image_name))
        else:
            abort_with_info(400, "Missing filename", response_info)

        image_bytes = post_file_i.read()

        if post_file_i.filename == "null" and image_bytes == b"":
            _images.append(None)
            continue

        image_data = Image.open(BytesIO(image_bytes))

        if image_data is None:
            abort_with_info(500, "Image could not be loaded correctly", response_info)

        _images.append(image_data)

    response_info["images"] = image_names

    if len(_images) != 3:
        abort_with_info(400, "Number of images should be 3", response_info)

    # Load the multiple XML files
    texts = []
    text_names = []
    for post_text_i in post_text:
        if post_text_i.filename:
            text_name = Path(post_text_i.filename)
            text_names.append(str(text_name))
        else:
            abort_with_info(400, "Missing filename", response_info)

        text_bytes = post_text_i.read()
        if post_text_i.filename == "null" and text_bytes == b"":
            texts.append("")
            continue
        xml = text_bytes.decode("utf-8")
        page_data = PageData.from_string(xml, text_name)
        text = page_data.get_transcription_dict()

        texts.append(text)

    response_info["texts"] = text_names

    if len(texts) != 3:
        abort_with_info(400, "Number of texts should be 3", response_info)

    # Check if the image was send correctly. If the image is empty, the text should also be empty
    for image, text in zip(_images, texts):
        if image is None and not text:
            continue
        if image is None and text:
            abort_with_info(400, "Empty image was send with text", response_info)

    # Find the shapes of the images
    shapes = []
    for image in _images:
        if image is None:
            shapes.append((0, 0))
            continue
        shapes.append((image.size[1], image.size[0]))

    _images = transform(_images)

    # Pad the images to the same size
    max_shape = np.max([image.size()[-2:] for image in _images if image is not None], axis=0)

    for i in range(len(_images)):
        if _images[i] is None:
            _images[i] = torch.zeros((3, max_shape[0], max_shape[1]))
        _images[i] = torch.nn.functional.pad(
            _images[i],
            (0, int(max_shape[1] - _images[i].size()[-1]), 0, int(max_shape[0] - _images[i].size()[-2])),
            value=0,
        )

    # Add batch dimension
    images = torch.stack(_images).unsqueeze(0)
    shapes = torch.tensor(shapes).unsqueeze(0)

    data = {
        "images": images,
        "shapes": shapes,
        "texts": [texts],
        "image_paths": [image_names],
    }

    future = executor.submit(predict_class, data, identifier, model_name)
    future.add_done_callback(check_exception_callback)

    response_info["status_code"] = 202
    # Return response and status code
    return jsonify(response_info), 202


@app.route("/prometheus", methods=["GET"])
def metrics() -> bytes:
    """
    Return the Prometheus metrics for the running flask application

    Returns:
        bytes: Encoded string with the information
    """
    if request.method != "GET":
        abort(405)
    return generate_latest()


@app.route("/health", methods=["GET"])
def health_check() -> tuple[str, int]:
    """
    Health check endpoint for Kubernetes checks

    Returns:
        tuple[str, int]: Response and status code
    """
    return "OK", 200


if __name__ == "__main__":
    app.run()

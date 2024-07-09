import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, request
from prometheus_client import Counter, Gauge, generate_latest

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))  # noqa: E402
from page_xml.xmlPAGE import PageData
from run import Predictor
from utils.image_utils import load_image_array_from_bytes
from utils.logging_utils import get_logger_name

logger = logging.getLogger(get_logger_name())

# Reading environment files
try:
    max_queue_size_string: str = os.environ["SEPARATION_MAX_QUEUE_SIZE"]
    model_base_path_string: str = os.environ["SEPARATION_MODEL_BASE_PATH"]
    output_base_path_string: str = os.environ["SEPARATION_OUTPUT_BASE_PATH"]
except KeyError as error:
    raise KeyError(f"Missing Separation Environment variable: {error.args[0]}")

# Convert
max_queue_size = int(max_queue_size_string)
model_base_path = Path(model_base_path_string)
output_base_path = Path(output_base_path_string)

# Checks if ENV variable exist
if not model_base_path.is_dir():
    raise FileNotFoundError(f"SEPARATION_MODEL_BASE_PATH: {model_base_path} is not found in the current filesystem")
if not output_base_path.is_dir():
    raise FileNotFoundError(f"SEPARATION_OUTPUT_BASE_PATH: {output_base_path} is not found in the current filesystem")

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


def get_middle_scan(y: np.ndarray):
    return y[:, y.shape[1] // 2]


def get_middle_path(paths: list[list[Path]]):
    return paths[0][len(paths[0]) // 2]


def predict_class(
    data: dict[str, Any],
    identifier: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Run the prediction for the given image

    Args:
        image (np.ndarray): Image array send to model prediction
        dpi (Optional[int]): DPI (dots per inch) of the image
        image_path (Path): Path to the image file
        identifier (str): Unique identifier for the image
        model_name (str): Name of the model to use for prediction
        whitelist (list[str]): List of characters to whitelist during prediction

    Raises:
        TypeError: If the current GenPageXML is not initialized
        TypeError: If the current Predictor is not initialized

    Returns:
        dict[str, Any]: Information about the processed image
    """
    input_args = locals()
    try:
        predict_wrapper.setup_model(model_name=model_name)

        image_path = get_middle_path(data["image_paths"])

        output_path = output_base_path.joinpath(identifier, image_path)
        if predict_wrapper.predictor is None:
            raise TypeError("The current Predictor is not initialized")

        if not output_path.parent.is_dir():
            output_path.parent.mkdir()

        outputs = predict_wrapper.predictor(data)

        output_class = torch.argmax(outputs, dim=2).cpu().numpy()
        output_class = get_middle_scan(output_class)

        with open(output_path, "w") as file:
            file.write(str(output_class.item()))

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
    filename: str
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
    except KeyError as error:
        abort_with_info(400, "Missing image in form", response_info)

    try:
        post_text = request.files.getlist("texts[]")
    except KeyError as error:
        abort_with_info(400, "Missing text in form", response_info)

    assert len(post_file) == len(post_text), "Number of images and texts must be the same"

    # TODO Maybe make slightly more stable/predicable, https://docs.python.org/3/library/threading.html#threading.Semaphore https://gist.github.com/frankcleary/f97fe244ef54cd75278e521ea52a697a
    queue_size = executor._work_queue.qsize()
    response_info["added_queue_position"] = queue_size
    response_info["remaining_queue_size"] = max_queue_size - queue_size
    if queue_size > max_queue_size:
        abort_with_info(429, "Exceeding queue size", response_info)

    images = []
    image_names = []
    for post_file_i in post_file:
        if (image_name := post_file_i.filename) is not None:
            image_name = Path(image_name)
            image_names.append(image_name)
            response_info["filename"] = str(image_name)
        else:
            abort_with_info(400, "Missing filename", response_info)

        img_bytes = post_file_i.read()
        image_data = load_image_array_from_bytes(img_bytes, image_path=image_name)

        if image_data is None:
            abort_with_info(500, "Image could not be loaded correctly", response_info)

    texts = []
    for post_text_i in post_text:
        if (text_name := post_text_i.filename) is not None:
            text_name = Path(text_name)
            response_info["filename"] = str(text_name)
        else:
            abort_with_info(400, "Missing filename", response_info)

        text_bytes = post_text_i.read()
        page_data = PageData(text_bytes)
        page_data.parse()
        text = page_data.get_transcription()
        total_text = ""
        for _, text_line in text.items():
            # If line ends with - then add it to the next line, otherwise add a space
            text_line = text_line.strip()
            if len(text_line) > 0:
                if text_line[-1] == "-":
                    text_line = text_line[:-1]
                else:
                    text_line += " "

            total_text += text_line

        texts.append(total_text)

    data = {
        "images": torch.stack(images, dim=0).unsqueeze(0),
        "shapes": torch.stack([image.shape[:2] for image in images], dim=0).unsqueeze(0),
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

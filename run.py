# DEPRECATED This file has been replaced by the inference.py file

import logging

from inference import get_arguments, main
from utils.logging_utils import get_logger_name

if __name__ == "__main__":
    args = get_arguments()
    logger = logging.getLogger(get_logger_name())
    logger.warning(
        (
            "DeprecationWarning: The use of the run.py file is deprecated. "
            "Replace it with the inference.py file in any scrips or commands you are using."
        )
    )

    main(args)

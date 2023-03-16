import logging
import os

from data_acquisition.modules.utils.FileHandler import create_folder


def Logger(name):
    # instantiate logger
    dir_path = os.path.dirname(os.path.realpath("data_acquisition"))
    logs_path = f"{dir_path}/logs"

    # Create a logs folder if it is not exist
    create_folder(logs_path)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # define handler and formatter
    stdout_handler = logging.StreamHandler()  # This for stdout logging
    file_handler = logging.FileHandler(f"{logs_path}/{name}.log", mode='w')  # This for file logging
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)2d - %(message)s | %(threadName)s:%(thread)d")

    # add formatter to handler
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)

    return logger


app_logger = Logger("app")

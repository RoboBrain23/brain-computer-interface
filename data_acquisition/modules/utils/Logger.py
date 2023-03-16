import logging
import os


def Logger(name):
    # instantiate logger
    dir_path = os.path.dirname(os.path.realpath("data_acquisition"))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # define handler and formatter
    stdout_handler = logging.StreamHandler()  # This for stdout logging
    file_handler = logging.FileHandler(f"{dir_path}/logs/{name}.log", mode='a')  # This for file logging
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

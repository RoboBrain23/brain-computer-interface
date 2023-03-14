import logging


def Logger(name):
    # instantiate logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # define handler and formatter
    # handler = logging.StreamHandler()     # This for stdout logging
    handler = logging.FileHandler(f"{dir}/logs/{name}.log", mode='a')  # This for file logging
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(handler)

    return logger

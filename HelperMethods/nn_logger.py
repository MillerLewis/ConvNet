import logging
# import sys

def get_nn_logger(name, level):
    LOGGING = logging.getLogger()
    LOGGING.setLevel(level)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s -- %(name)s -- %(message)s")
    stream_handler.setFormatter(formatter)
    LOGGING.addHandler(stream_handler)

    return LOGGING
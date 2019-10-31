import logging

LOGGER = logging.Logger(__name__)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s -- %(name)s -- %(message)s")
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
LOGGER.addHandler(stream_handler)

LOGGER.debug("TESTMESSAGE")
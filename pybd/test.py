from pybd_in.logger import logger
from pybd_in.custom_exception import invalidexception

logger.info("Test log message from test.py")
# def test_logger():
#     logger.info("Test log message")
#     assert logger is not None

try:
    raise invalidexception("Test exception")
except invalidexception as e:
    logger.error(e)
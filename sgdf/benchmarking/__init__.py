import logging
import time
from contextlib import contextmanager

_log = logging.getLogger(__name__)


@contextmanager
def log_timer(name):
    before = time.time()
    yield
    after = time.time()
    _log.debug("log_timer(%s): %s seconds", repr(name), repr(after - before))

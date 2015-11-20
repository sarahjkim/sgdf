import logging
import time
from contextlib import contextmanager
from os.path import dirname, join
from sgdf.benchmarking.headless import fusion_from_file

LOG = logging.getLogger(__name__)
EXAMPLES_ROOT = join(dirname(__file__), "examples")


@contextmanager
def log_timer(name, level=None):
    before = time.time()
    yield
    after = time.time()
    logger = {"info": LOG.info, "debug": LOG.debug}.get(level, LOG.debug)
    logger("log_timer(%s):%.8fms", repr(name), 1000 * (after - before))


def benchmark_default(algorithm="reference"):
    with log_timer("benchmark: easy", level="info"):
        fusion_from_file(algorithm, join(EXAMPLES_ROOT, "easy-source.jpg"),
                         join(EXAMPLES_ROOT, "easy-target.jpg"),
                         join(EXAMPLES_ROOT, "easy-mask.jpg"))

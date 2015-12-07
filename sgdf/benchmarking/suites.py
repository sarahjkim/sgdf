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


def benchmark_default(algorithm="reference", algorithm_kwargs=None):
    """Runs some basic benchmarking cases."""

    if algorithm_kwargs is None:
        algorithm_kwargs = {}

    with log_timer("benchmark: gradient", level="info"):
        fusion_from_file(algorithm, join(EXAMPLES_ROOT, "gradient-source.png"),
                         join(EXAMPLES_ROOT, "gradient-target.png"),
                         join(EXAMPLES_ROOT, "gradient-mask.png"),
                         algorithm_kwargs=algorithm_kwargs)
    with log_timer("benchmark: easy", level="info"):
        fusion_from_file(algorithm, join(EXAMPLES_ROOT, "easy-source.jpg"),
                         join(EXAMPLES_ROOT, "easy-target.jpg"),
                         join(EXAMPLES_ROOT, "easy-mask.jpg"),
                         algorithm_kwargs=algorithm_kwargs)
    with log_timer("benchmark: brick", level="info"):
        fusion_from_file(algorithm, join(EXAMPLES_ROOT, "brick-source.jpg"),
                         join(EXAMPLES_ROOT, "brick-target.jpg"),
                         join(EXAMPLES_ROOT, "brick-mask.jpg"),
                         offset=[81, 137],
                         algorithm_kwargs=algorithm_kwargs)
    with log_timer("benchmark: window", level="info"):
        fusion_from_file(algorithm, join(EXAMPLES_ROOT, "window.jpg"),
                         join(EXAMPLES_ROOT, "window.jpg"),
                         join(EXAMPLES_ROOT, "window-mask.jpg"),
                         offset=[-1, -226],
                         algorithm_kwargs=algorithm_kwargs)
    with log_timer("benchmark: penguin", level="info"):
        fusion_from_file(algorithm, join(EXAMPLES_ROOT, "penguin-source3.jpg"),
                         join(EXAMPLES_ROOT, "penguin-target.jpg"),
                         join(EXAMPLES_ROOT, "penguin-mask3.jpg"),
                         offset=[356, 287],
                         algorithm_kwargs=algorithm_kwargs)
    with log_timer("benchmark: bear", level="info"):
        fusion_from_file(algorithm, join(EXAMPLES_ROOT, "bear-source3.jpg"),
                         join(EXAMPLES_ROOT, "bear-target.jpg"),
                         join(EXAMPLES_ROOT, "bear-mask3.jpg"),
                         offset=[0, 600],
                         algorithm_kwargs=algorithm_kwargs)
    with log_timer("benchmark: winterfell", level="info"):
        fusion_from_file(algorithm, join(EXAMPLES_ROOT, "winterfell-source.jpg"),
                         join(EXAMPLES_ROOT, "winterfell-target.jpg"),
                         join(EXAMPLES_ROOT, "winterfell-mask.jpg"),
                         offset=[287,421],
                         algorithm_kwargs=algorithm_kwargs)

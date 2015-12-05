import logging
import numpy as np
from sgdf.benchmarking import log_timer
from sgdf.fusion.alternatives.reference import ReferenceFusion

try:
    import sgdf.fusion.alternatives._quickdescent as _quickdescent
except ImportError:
    raise RuntimeError("Failed to import _quickdescent module.\n\n"
                       "    You need to build the _quickdescent native extension first.\n"
                       "    Please run './setup.py build_ext --inplace' in the project root.\n")

LOG = logging.getLogger(__name__)


class QuickdescentFusion(ReferenceFusion):
    def __init__(self):
        """
        An algorithm for the Gradient Domain Fusion problem, using gradient descent implemented
        natively.

        """
        ReferenceFusion.__init__(self)

    def update_blend(self, mask_ndarray):
        assert mask_ndarray.dtype == np.bool
        assert mask_ndarray.shape == self.canvas.shape[:2]
        self.active = True
        assert self.active_mask is None or id(self.active_mask) == id(mask_ndarray), \
            "The quickdescent algorithm requires that you don't change the mask address."
        self.active_mask = mask_ndarray

    def poisson_blend(self, source, mask, tinyt, max_iterations=100):
        assert source.shape == mask.shape == tinyt.shape
        assert len(source.shape) == 2
        solution = np.zeros(tinyt.shape, dtype=np.float32)
        scratch = np.zeros(tinyt.shape, dtype=np.float32)
        errorlog = np.zeros(max_iterations, dtype=np.float32)
        with log_timer("%s.native" % self.__class__.__name__):
            q = _quickdescent.QuickdescentContext(source, mask, tinyt, solution, scratch, errorlog,
                                                  0.00001, max_iterations)
            q.initializeGuess()
            q.blend()
        LOG.debug("Quickdescent iterations: %d" % (errorlog.nonzero()[0][-1] + 1))
        LOG.debug("Final error value: %f" % (errorlog[errorlog.nonzero()[0][-1]]))
        return solution.clip(0, 1)

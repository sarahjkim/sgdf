import numpy as np
import sgdf.fusion.alternatives._quickdescent as _quickdescent
from sgdf.fusion.alternatives.reference import ReferenceFusion


class QuickdescentFusion(ReferenceFusion):
    def __init__(self):
        """
        An algorithm for the Gradient Domain Fusion problem, using gradient descent implemented
        natively.

        """
        ReferenceFusion.__init__(self)

    def poisson_blend(self, source, mask, tinyt):
        assert source.shape == mask.shape == tinyt.shape
        assert len(source.shape) == 2
        solution = np.ndarray(tinyt.shape, dtype=np.float32)
        _quickdescent.poisson_blend(source, mask, tinyt, solution)
        return solution

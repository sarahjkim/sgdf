import numpy as np
from sgdf.fusion.alternatives.reference import ReferenceFusion

try:
    import sgdf.fusion.alternatives._quickdescent as _quickdescent
except ImportError:
    raise RuntimeError("Failed to import _quickdescent module.\n\n"
                       "    You need to build the _quickdescent native extension first.\n"
                       "    Please run './setup.py build_ext --inplace' in the project root.\n")


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

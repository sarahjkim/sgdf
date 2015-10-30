import numpy as np
from unittest2 import TestCase
from sgdf.fusion.alternatives.reference import ReferenceFusion


class ReferenceFusionTest(TestCase):
    def test_base_cases(self):
        fusion = ReferenceFusion()
        im = np.array([[[1.0, 0.0, 0.0], [0.5, 0.2, 0.4], [0.3, 0.5, 0.25]],
                       [[0.5, 0.2, 0.4], [0.3, 0.5, 0.25], [1.0, 0.0, 0.0]],
                       [[0.3, 0.5, 0.25], [1.0, 0.0, 0.0], [0.5, 0.2, 0.4]]])
        fusion.set_image(im)
        np.testing.assert_array_equal(fusion.get_fusion(), im)

    def test_simple(self):
        fusion = ReferenceFusion()
        target = np.array([[[1.0, 0.0, 0.0], [0.5, 0.2, 0.4], [0.3, 0.5, 0.25]],
                           [[0.5, 0.2, 0.4], [0.3, 0.5, 0.25], [1.0, 0.0, 0.0]],
                           [[0.3, 0.5, 0.25], [1.0, 0.0, 0.0], [0.5, 0.2, 0.4]]])
        source = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        mask = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
        displacement = np.array([0, 0])
        fusion.set_image(target)
        fusion.update_blend(source, mask, displacement)
        expected = np.array([[[1.0, 0.0, 0.0], [0.5, 0.2, 0.4], [0.3, 0.5, 0.25]],
                             [[0.5, 0.2, 0.4], [0.75, 0.1, 0.2], [1.0, 0.0, 0.0]],
                             [[0.3, 0.5, 0.25], [1.0, 0.0, 0.0], [0.5, 0.2, 0.4]]])
        np.testing.assert_array_equal(fusion.get_fusion(), expected)

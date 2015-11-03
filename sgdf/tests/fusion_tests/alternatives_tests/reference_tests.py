import numpy as np
from unittest2 import TestCase
from sgdf.fusion.alternatives.reference import ReferenceFusion


class ReferenceFusionTest(TestCase):
    def test_base_cases(self):
        fusion = ReferenceFusion()
        im = np.array([[[1.0, 0.0, 0.0], [0.5, 0.2, 0.4], [0.3, 0.5, 0.25]],
                       [[0.5, 0.2, 0.4], [0.3, 0.5, 0.25], [1.0, 0.0, 0.0]],
                       [[0.3, 0.5, 0.25], [1.0, 0.0, 0.0], [0.5, 0.2, 0.4]]], dtype=np.float32)
        fusion.set_source_image(im)
        np.testing.assert_array_equal(fusion.get_fusion(), im)

    def test_simple(self):
        fusion = ReferenceFusion()
        target = np.array([[[1.0, 0.0, 0.0], [0.5, 0.2, 0.4], [0.3, 0.5, 0.25]],
                           [[0.5, 0.2, 0.4], [0.3, 0.5, 0.25], [1.0, 0.0, 0.0]],
                           [[0.3, 0.5, 0.25], [1.0, 0.0, 0.0], [0.5, 0.2, 0.4]]], dtype=np.float32)
        source = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        mask = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
        displacement = np.array([0, 0])
        fusion.set_source_image(target)
        fusion.set_target_image(target)
        fusion.set_anchor_points(np.array([0, 0]), np.array([0, 0]))
        fusion.update_blend(mask)
        expected = np.array([[[1.0, 0.0, 0.0], [0.5, 0.2, 0.4], [0.3, 0.5, 0.25]],
                             [[0.5, 0.2, 0.4], [0.75, 0.1, 0.2], [1.0, 0.0, 0.0]],
                             [[0.3, 0.5, 0.25], [1.0, 0.0, 0.0], [0.5, 0.2, 0.4]]], dtype=np.float32)
        np.testing.assert_array_equal(fusion.get_fusion(), expected)

import numpy as np
from unittest2 import TestCase
from sgdf.fusion.alternatives.reference import ReferenceFusion


class ReferenceFusionTest(TestCase):
    def test_simple(self):
        fusion = ReferenceFusion()
        target = np.tile(np.float32(0.5), (9, 9, 3))
        target[2:7, 2:7] = 0.75
        target[4, 4] = 1.0
        source = np.tile(np.float32(0.1), (9, 9, 3))
        mask = np.tile(False, (9, 9))
        mask[3:6, 3:6] = True
        displacement = np.array([0, 0])
        expected = np.tile(np.float32(0.5), (9, 9, 3))
        expected[2:7, 2:7] = 0.75
        fusion.set_source_image(source)
        fusion.set_target_image(target)
        fusion.set_anchor_points(np.array([0, 0]), displacement)
        fusion.update_blend(mask)
        np.testing.assert_array_almost_equal(fusion.get_fusion(), expected)

    def test_with_nonzero_displacement(self):
        fusion = ReferenceFusion()
        target = np.tile(np.float32(0.5), (9, 9, 3))
        target[2:7, 2:7] = 0.75
        target[4, 4] = 1.0
        source = np.tile(np.float32(0.1), (5, 5, 3))
        mask = np.tile(False, (5, 5))
        mask[1:4, 1:4] = True
        displacement = np.array([2, 2])
        expected = np.tile(np.float32(0.5), (9, 9, 3))
        expected[2:7, 2:7] = 0.75
        fusion.set_source_image(source)
        fusion.set_target_image(target)
        fusion.set_anchor_points(np.array([0, 0]), displacement)
        fusion.update_blend(mask)
        # print fusion.get_fusion()[:, :, 0]
        np.testing.assert_array_almost_equal(fusion.get_fusion(), expected)


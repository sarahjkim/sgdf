import numpy as np
from sgdf.fusion.alternatives.base import BaseFusion


class CopypasteFusion(BaseFusion):
    def __init__(self):
        """
        An example fusion algorithm that just copies and pastes images with no blending.

        """
        self.canvas = None
        self.active = False
        self.active_source = None
        self.active_mask = None

    def set_source_image(self, source_ndarray):
        assert source_ndarray.dtype == np.float32
        self.active_source = source_ndarray

    def set_target_image(self, target_ndarray):
        assert target_ndarray.dtype == np.float32
        self.canvas = target_ndarray

    def set_anchor_points(self, source_anchor, target_anchor):
        self.source_anchor = source_anchor
        self.target_anchor = target_anchor

        s_anchor_row, s_anchor_col = self.source_anchor
        t_anchor_row, t_anchor_col = self.target_anchor

        s_height, s_width, _ = self.active_source.shape
        t_height, t_width, _ = self.canvas.shape

        space_above_anchor = min(s_anchor_row, t_anchor_row)
        space_below_anchor = min(s_height - s_anchor_row, t_height - t_anchor_row)
        space_left_anchor = min(s_anchor_col, t_anchor_col)
        space_right_anchor = min(s_width - s_anchor_col, t_width - t_anchor_col)

        self.source_bounds = (s_anchor_row - space_above_anchor, s_anchor_row + space_below_anchor,
                              s_anchor_col - space_left_anchor, s_anchor_col + space_right_anchor)
        self.target_bounds = (t_anchor_row - space_above_anchor, t_anchor_row + space_below_anchor,
                              t_anchor_col - space_left_anchor, t_anchor_col + space_right_anchor)

    def commit_blend(self):
        if self.active:
            self.canvas = self.get_fusion()
            self.active = False

    def update_blend(self, mask_ndarray):
        assert mask_ndarray.dtype == np.bool
        assert mask_ndarray.shape == self.canvas.shape[:2]
        self.active = True
        self.active_mask = mask_ndarray

    def get_fusion(self):
        target = np.copy(self.canvas)
        if not self.active:
            return target

        s_top, s_bottom, s_left, s_right = self.source_bounds
        t_top, t_bottom, t_left, t_right = self.target_bounds
        source = self.active_source[s_top:s_bottom, s_left:s_right]
        tinyt = target[t_top:t_bottom, t_left:t_right, :]
        s_mask = self.active_mask[t_top:t_bottom, t_left:t_right]

        for channel in range(3):
            solution = np.choose(s_mask, [tinyt[:, :, channel], source[:, :, channel]])
            target[t_top:t_bottom, t_left:t_right, channel] = solution

        return target

import numpy as np
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import lsqr
from sgdf.benchmarking import log_timer
from sgdf.fusion import BaseFusion


class ReferenceFusion(BaseFusion):
    def __init__(self):
        """
        A reference solution to the Gradient Domain Fusion problem, borrowed from our project
        mentor, Rachel Albert.

        Source: https://github.com/rachelalbert/CS294-26_code/tree/master/project4_code

        """
        self.canvas = None
        self.active = False
        self.active_source = None
        self.active_mask = None
        self.source_anchor = None
        self.target_anchor = None
        self.source_bounds = None
        self.target_bounds = None

    def set_source_image(self, source_ndarray):
        assert source_ndarray.dtype == np.float32
        self.active_source = source_ndarray

    def set_target_image(self, target_ndarray):
        assert target_ndarray.dtype == np.float32
        self.canvas = np.copy(target_ndarray)

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

    def update_blend(self, mask_ndarray):
        assert mask_ndarray.dtype == np.bool
        assert mask_ndarray.shape == self.canvas.shape[:2]
        self.active = True
        self.active_mask = mask_ndarray

    def commit_blend(self):
        if self.active:
            self.canvas = self.get_fusion()
            self.active = False

    def get_fusion(self):
        with log_timer("ReferenceFusion.get_fusion"):
            # TODO is this inefficient? OH WELL
            target = np.copy(self.canvas)
            if not self.active:
                return target

            s_top, s_bottom, s_left, s_right = self.source_bounds
            t_top, t_bottom, t_left, t_right = self.target_bounds
            source = self.active_source[s_top:s_bottom, s_left:s_right]
            s_mask = self.active_mask[t_top:t_bottom, t_left:t_right]
            tinyt = target[t_top:t_bottom, t_left:t_right, :]
            tinyt_topleft = (t_top, t_left)

            for channel in range(3):
                solution = self.poisson_blend(source[:, :, channel],
                                              s_mask,
                                              tinyt[:, :, channel],
                                              target[:, :, channel],
                                              tinyt_topleft)
                target[:, :, channel] = solution
            return target

    def shift(self, im, amounts):
        """
        Shifts an image. The resulting image will be the same size as the original.

        """
        original_dimensions = im.shape
        dy, dx = amounts
        im = np.pad(im, ((max(0, dy), max(0, -dy)),
                         (max(0, dx), max(0, -dx))), mode="constant")
        if dy < 0:
            if dx < 0:
                im = im[-dy:, -dx:]
            elif dx == 0:
                im = im[-dy:, :]
            else:
                im = im[-dy:, :-dx]
        elif dy == 0:
            if dx < 0:
                im = im[:, -dx:]
            elif dx == 0:
                pass
            else:
                im = im[:, :-dx]
        else:
            if dx < 0:
                im = im[:-dy, -dx:]
            elif dx == 0:
                im = im[:-dy, :]
            else:
                im = im[:-dy, :-dx]
        assert im.shape == original_dimensions, "This code has a bug, FIX IT"
        return im

    def inside(self, mask):
        return (self.shift(mask, (-1, 0)) &
                self.shift(mask, (0, -1)) &
                self.shift(mask, (1, 0)) &
                self.shift(mask, (0, 1)))

    def construct_A4(self, source, mask_border=[[]]):
        source_h, source_w = source.shape
        sy, sx = np.nonzero(mask_border)
        source_size = source_h*source_w
        # [x,x+1], [x,x-1], [y,y+1], [y,y-1]
        all_offsets = [[0, -1], [0, 1], [0, -source_w], [0, source_w]]
        As = []
        for offset in all_offsets:
            A = diags(diagonals=[1, -1],
                      offsets=offset,
                      shape=[source_size, source_size],
                      format='csr',
                      dtype=float)
            r, c = (A[source_w*sy + sx, :] < 0).nonzero()
            A[(source_w*sy + sx)[r], c] = 0
            r, c = A[source_w*sy + sx, :].nonzero()
            As.append(A)
        return vstack(As)

    def set_b(self, b, mask, values):
        bigmask = np.concatenate([mask, mask, mask, mask])
        b[bigmask] = values[bigmask]
        return b

    def poisson_blend(self, source, mask, tinyt, t, tinyt_topleft, maximum=False):
        """
        Modifies the target image to blend the source image at the designated mask pixels using a
        Poisson blend.

        Args:
            source (2D float np.array): Values of source image at relevant blending pixels;
                    same size as mask array (may be smaller than source image)
            mask (2D bool np.array): Mask with 1 (True) values for source pixels
            tinyt (2D float np.array): Values of target image at relevant blending pixels;
                    same size as mask array (may be smaller than target image)
            t (2D float np.array): Full size target image
            tinyt_topleft ((int, int)): (row, col) designating top-left coordinate of tinyt
                    with respect to image t

        Returns:
            (2D float np.array): Channel of modified target image with section of source image

        """
        mask_inside = self.inside(mask)
        mask_border = mask & ~mask_inside

        A4 = self.construct_A4(source)
        t_prime = A4.dot(tinyt.ravel())
        s_prime = A4.dot(source.ravel())

        b = t_prime.copy()
        if maximum:
            max_prime = np.maximum(s_prime, t_prime)
            b = self.set_b(b, mask_inside.ravel(), max_prime)
        else:
            b = self.set_b(b, mask_inside.ravel(), s_prime)
        tinyt_values = np.concatenate([tinyt.ravel(), tinyt.ravel(), tinyt.ravel(), tinyt.ravel()])
        b = self.set_b(b, mask_border.ravel(), tinyt_values)

        A4 = self.construct_A4(source, mask_border=mask_border)
        imh, imw = source.shape
        v = lsqr(A4, b)[0]
        out = v.reshape((imh, imw))

        tttly, tttlx = tinyt_topleft
        tty, ttx = tinyt.shape

        t[tttly:tttly + tty, tttlx:tttlx + ttx] = out

        return t

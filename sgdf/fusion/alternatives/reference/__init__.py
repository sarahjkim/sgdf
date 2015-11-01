import numpy as np
from scipy.sparse import diags
from scipy.sparse import vstack
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
        self.active_source = source_ndarray

    def set_target_image(self, target_ndarray):
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
        self.active = True
        self.active_mask = mask_ndarray

    def commit_blend(self):
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
            s = self.active_source[s_top:s_bottom, s_left:s_right]
            s_mask = self.active_mask[t_top:t_bottom, t_left:t_right]
            tinyt = target[t_top:t_bottom, t_left:t_right, :]
            tinyt_topleft = (t_top, t_left)

            for channel in range(3):
                solution = self.poisson_blend(s[:, :, channel],
                                              s_mask,
                                              tinyt[:, :, channel],
                                              target[:, :, channel],
                                              tinyt_topleft)
                target[:, :, channel] = solution
            return target

    def shift(self, m, direction):
        padded = np.pad(m, [(d, 0) if d > 0 else (0, -d) for d in direction], mode='constant')
        return padded[[np.s_[:sh] if d > 0 else np.s_[-sh:] for sh, d in zip(m.shape, direction)]]

    def inside(self, mask):
        return (self.shift(mask, (-1, 0)) &
                self.shift(mask, (0, -1)) &
                self.shift(mask, (1, 0)) &
                self.shift(mask, (0, 1)))

    def construct_A4(self, s, s_border=[[]]):
        imh, imw = s.shape
        sy, sx = np.where(s_border)
        npx = imh*imw
        # [x,x+1], [x,x-1], [y,y+1], [y,y-1]
        all_offsets = [[0, -1], [0, 1], [0, -imw], [0, imw]]
        As = []
        for offset in all_offsets:
            A = diags(
                diagonals=[1, -1],
                offsets=offset,
                shape=[npx, npx],
                format='csr',
                dtype=float)
            r, c = (A[imw*sy + sx, :] < 0).nonzero()
            A[(imw*sy + sx)[r], c] = 0
            r, c = A[imw*sy + sx, :].nonzero()
            As.append(A)
        return vstack(As)

    def set_b(self, b, mask, values):
        bigmask = np.concatenate([mask, mask, mask, mask])
        b[bigmask] = values[bigmask]
        return b

    def poisson_blend(self, s, s_mask, tinyt, t, tinyt_topleft, maximum=False):
        """
        Modifies the target image to blend the source image at the designated mask pixels using a
        Poisson blend.

        Args:
            s (2D float np.array): Values of source image at relevant blending pixels;
                    same size as s_mask array (may be smaller than source image)
            s_mask (2D bool np.array): Mask with 1 (True) values for source pixels
            tinyt (2D float np.array): Values of target image at relevant blending pixels;
                    same size as s_mask array (may be smaller than target image)
            t (2D float np.array): Full size target image
            tinyt_topleft ((int, int)): (row, col) designating top-left coordinate of tinyt
                    with respect to image t

        Returns:
            (2D float np.array): Channel of modified target image with section of source image

        """
        s_inside = self.inside(s_mask)
        s_border = s_mask & ~s_inside

        A4 = self.construct_A4(s)
        t_prime = A4.dot(tinyt.ravel())
        s_prime = A4.dot(s.ravel())

        b = t_prime.copy()
        if maximum:
            max_prime = np.maximum(s_prime, t_prime)
            b = self.set_b(b, s_inside.ravel(), max_prime)
        else:
            b = self.set_b(b, s_inside.ravel(), s_prime)
        tinyt_values = np.concatenate([tinyt.ravel(), tinyt.ravel(), tinyt.ravel(), tinyt.ravel()])
        b = self.set_b(b, s_border.ravel(), tinyt_values)

        A4 = self.construct_A4(s, s_border=s_border)
        imh, imw = s.shape
        v = lsqr(A4, b)[0]
        out = v.reshape((imh, imw))

        tttly, tttlx = tinyt_topleft
        tty, ttx = tinyt.shape
        t[tttly:tttly + tty, tttlx:tttlx + ttx] = out
        return t

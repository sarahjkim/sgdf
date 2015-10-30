import numpy as np
from scipy.sparse import diags
from scipy.sparse import vstack
from scipy.sparse.linalg import lsqr
from sgdf.fusion import BaseFusion


class ReferenceFusion(BaseFusion):
    def __init__(self):
        """
        A reference solution to the Gradient Domain Fusion problem, borrowed from our project
        mentor, Rachel Albert.

        Source: https://github.com/rachelalbert/CS294-26_code/tree/master/project4_code

        """
        self.canvas = None
        self.active_source = None
        self.active_mask = None
        self.active_displacement = None

    def set_target_image(self, ndarray):
        self.canvas = np.copy(ndarray)

    def update_blend(self, source_ndarray, mask_ndarray, displacement):
        self.active_source = source_ndarray
        self.active_mask = mask_ndarray
        self.active_displacement = displacement

    def commit_blend(self):
        self.canvas = self.get_fusion()

    def get_fusion(self):
        # TODO is this inefficient? OH WELL
        target = np.copy(self.canvas)
        source_height, source_width, _ = self.active_source.shape
        mask_y, mask_x = self.active_displacement
        tinyt = target[mask_y:mask_y + source_height,
                       mask_x:mask_x + source_width, :]
        return self.poisson_blend(self.active_source, self.active_mask, tinyt, target,
                                  self.active_displacement)

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

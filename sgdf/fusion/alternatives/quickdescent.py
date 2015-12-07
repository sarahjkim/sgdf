import logging
import numpy as np
import threading
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
    def __init__(self, epsilon=0.00005, max_iterations=200, threading=True):
        """
        An algorithm for the Gradient Domain Fusion problem, using gradient descent implemented
        natively.

        """
        ReferenceFusion.__init__(self)

        self.cache_mask = None
        self.cache_native = None
        self.cache_errorlog = None
        self.cache_target = None
        self.cache_source_bounds = None
        self.cache_target_bounds = None

        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.threading = threading

    def get_errorlog(self):
        return self.cache_errorlog

    def update_blend(self, mask_ndarray):
        assert mask_ndarray.dtype == np.bool
        assert mask_ndarray.shape == self.canvas.shape[:2]
        self.active = True
        assert self.active_mask is None or id(self.active_mask) == id(mask_ndarray), \
            "The quickdescent algorithm requires that you don't change the mask address."
        if self.active_mask is None:
            self.active_mask = mask_ndarray
        if self.cache_mask is not None:
            t_top, t_bottom, t_left, t_right = self.target_bounds
            self.cache_mask[:, :] = self.active_mask[t_top:t_bottom, t_left:t_right]

    def commit_blend(self):
        if self.active:
            self.canvas = self.get_fusion()
            self.active = False
            self.cache_mask = None
            self.cache_native = None
            self.cache_errorlog = None
            self.cache_target = None
            self.cache_source_bounds = None
            self.cache_target_bounds = None

    def get_fusion(self):
        with log_timer("%s.get_fusion" % self.__class__.__name__):
            if not self.active:
                return np.copy(self.canvas)

            s_top, s_bottom, s_left, s_right = self.source_bounds
            t_top, t_bottom, t_left, t_right = self.target_bounds

            if (self.source_bounds != self.cache_source_bounds or
                    self.target_bounds != self.cache_target_bounds):
                with log_timer("%s.setup" % self.__class__.__name__):
                    # We copy cache_mask here, because the native code would need to copy it anyway
                    # (since it is a view, not an actual ndarray).  However, we don't copy source or
                    # tinyt, because it is cloned per-channel in the "for channel in range(3)" loop
                    # below.
                    source = self.active_source[s_top:s_bottom, s_left:s_right]
                    tinyt = self.canvas[t_top:t_bottom, t_left:t_right, :]
                    self.cache_mask = np.copy(self.active_mask[t_top:t_bottom, t_left:t_right],
                                              order="C")

                    # Used for storing our QuickdescentContext instances
                    self.cache_native = []

                    # Used for storing all our errorlogs
                    self.cache_errorlog = []

                    # Used for storing the fusion result (we can reuse this buffer)
                    self.cache_target = np.copy(self.canvas)

                    for channel in range(3):
                        solution = np.zeros(tinyt.shape[:2], dtype=np.float32, order="C")
                        errorlog = np.zeros(self.max_iterations, dtype=np.float32, order="C")
                        q = _quickdescent.QuickdescentContext(source[:, :, channel].copy(order="C"),
                                                              self.cache_mask,
                                                              tinyt[:, :, channel].copy(order="C"),
                                                              solution, errorlog)
                        q.initializeGuess()
                        self.cache_native.append(q)
                        self.cache_errorlog.append(errorlog)

                    self.cache_source_bounds = self.source_bounds
                    self.cache_target_bounds = self.target_bounds

            def blend_channel(channel):
                with log_timer("%s.native" % self.__class__.__name__):
                    solution = self.cache_native[channel].blend(self.epsilon, self.max_iterations)
                self.cache_target[t_top:t_bottom, t_left:t_right, channel] = solution
                errorlog = self.cache_errorlog[channel]
                LOG.debug("Quickdescent iterations: %d" % (errorlog.nonzero()[0][-1] + 1))
                LOG.debug("Final error value: %f" % (errorlog[errorlog.nonzero()[0][-1]]))

            if self.threading:
                threads = []
                for channel in range(3):
                    thread = threading.Thread(target=blend_channel, args=(channel,))
                    threads.append(thread)

                for thread in threads:
                    thread.start()

                for thread in threads:
                    thread.join()
            else:
                for channel in range(3):
                    blend_channel(channel)

            return self.cache_target.clip(0, 1)

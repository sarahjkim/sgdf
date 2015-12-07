import logging
import matplotlib
import numpy as np
from sgdf.fusion import get_fusion_algorithm
from sgdf.util.io import imread, imsave
from sgdf.util.preprocessing import to_mask

matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)
RED = (1., 0., 0., 1.)
GREEN = (0., 1., 0., 1.)
BLUE = (0., 0., 1., 1.)


def save_image(fig, name, dont_clear=False):
    LOG.info("save_image: Saving graph to %s" % name)
    fig.savefig(name, bbox_inches='tight', dpi=144)
    if not dont_clear:
        fig.clear()


def learning_curve(fusion, output_path, title=None, basename="output"):
    if not hasattr(fusion, "get_errorlog"):
        LOG.info("Fusion algorithm does not support errorlog. Skipping learning_curve().")
    errorlog = fusion.get_errorlog()
    iterations = errorlog[0].nonzero()[0][-1] + 1
    fig, ax = plt.subplots()
    if title is None:
        ax.set_title("Learning curve")
    else:
        ax.set_title("Learning curve\n%s" % title)
    ax.set_yscale("log")
    ax.set_ylabel("Error")
    ax.set_xlim(left=0, right=iterations + 1)
    ax.set_xlabel("Iteration")
    ax.scatter(np.arange(iterations), errorlog[0], linewidth=0, s=3, c=RED)
    ax.scatter(np.arange(iterations), errorlog[1], linewidth=0, s=3, c=GREEN)
    ax.scatter(np.arange(iterations), errorlog[2], linewidth=0, s=3, c=BLUE)
    output_path = "%s-%s" % (basename, output_path)
    save_image(fig, output_path)


def fusion_with_charts(algorithm, source_path, target_path, mask_path, offset=None, output_path=None,
                       algorithm_kwargs=None, chart_kwargs=None):
    if offset is None:
        offset = np.array([0, 0])
    if algorithm_kwargs is None:
        algorithm_kwargs = {}
    if chart_kwargs is None:
        chart_kwargs = {}
    fusion = get_fusion_algorithm(algorithm)(**algorithm_kwargs)
    mask = to_mask(imread(mask_path))
    im_source = imread(source_path)
    im_target = imread(target_path)
    fusion.set_target_image(im_target)
    fusion.set_source_image(im_source)
    fusion.set_anchor_points(np.array([0, 0]), np.array(offset))
    fusion.update_blend(mask)
    im_result = (fusion.get_fusion() * 255.).astype(np.uint8)
    if output_path:
        imsave(output_path, im_result)
    learning_curve(fusion, "learning-curve.png", **chart_kwargs)
    return im_result

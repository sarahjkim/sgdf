import numpy as np
from matplotlib.image import imread, imsave
from sgdf.fusion import get_fusion_algorithm


def fusion_from_file(algorithm, source_path, target_path, mask_path, output_path=None):
    fusion = get_fusion_algorithm(algorithm)()
    mask = imread(mask_path)
    mask = np.mean(mask, 2)
    mask = (mask > np.average(mask))
    im_source = imread(source_path)
    if np.max(im_source) > 1.0:
        im_source = im_source.astype(np.float32) / 255.
    im_target = imread(target_path)
    if np.max(im_target) > 1.0:
        im_target = im_target.astype(np.float32) / 255.
    fusion.set_target_image(im_target)
    fusion.set_source_image(im_source)
    fusion.set_anchor_points(np.array([0, 0]), np.array([0, 0]))
    fusion.update_blend(mask)
    im_result = (fusion.get_fusion() * 255.).astype(np.uint8)
    if output_path:
        imsave(output_path, im_result)
    return im_result

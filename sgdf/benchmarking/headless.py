import numpy as np
from sgdf.fusion import get_fusion_algorithm
from sgdf.util.io import imread, imsave
from sgdf.util.preprocessing import to_mask


def fusion_from_file(algorithm, source_path, target_path, mask_path, offset=None, output_path=None,
                     algorithm_kwargs=None):
    if offset is None:
        offset = np.array([0, 0])
    if algorithm_kwargs is None:
        algorithm_kwargs = {}
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
    return im_result

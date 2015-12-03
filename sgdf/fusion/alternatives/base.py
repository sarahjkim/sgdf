import importlib


def get_fusion_algorithm(alternative, class_name=None):
    """
    Returns a reference to the class that represents a fusion algorithm. Looks in
    sgdf.fusion.alternative.* for a module containing the algorithm. This function expects the
    class to be named $(alternative.capitalize())Fusion, but you can override it with the class_name
    keyword argument.

    ARGUMENTS
        alternative -- The name of the algorithm
        class_name  -- (Optional) The class name to load

    """
    module = importlib.import_module("sgdf.fusion.alternatives.%s" % alternative)
    if class_name is None:
        class_name = "%sFusion" % alternative.capitalize()
    return getattr(module, class_name)


class BaseFusion(object):
    def set_source_image(self, source_ndarray):
        """
        Sets the source image (H * W * C). Moving the cursor over the target image will blend source
        pixels into the target image. The source_ndarray must not be modified.

        """
        raise RuntimeError("Unimplemented method")

    def set_target_image(self, target_ndarray):
        """
        Sets the canvas to a target image (H * W * C). This image will be the base image, upon which
        other images will be blended. The target_ndarray must not be modified.

        """
        raise RuntimeError("Unimplemented method")

    def set_anchor_points(self, source_anchor, target_anchor):
        """
        Updates the anchor points in the source and destination images. These anchor points are
        used to translate masked source coordinates to canvas (target) coordinates.

        This method should be called when the user first selects a target anchor point.

        """
        raise RuntimeError("Unimplemented method")

    def update_blend(self, mask_ndarray):
        """
        Updates the currently active blending operation (the "current blend"). The current blend
        consists of a source image, a mask (np.float32, with same dimensions as the source image),
        and anchor points specified with a previous call to set_anchor_points(). This update should
        replace the currently active blend, if one exists.

        This method should be called as the user updates the current mask.

        """
        raise RuntimeError("Unimplemented method")

    def commit_blend(self):
        """
        Commits the current blend to the canvas (see the update_blend method). Resets the current
        blend, to prepare for the next operation.

        This method should be called when the user is finished with a blend.

        """
        raise RuntimeError("Unimplemented method")

    def get_fusion(self):
        """
        Returns the current canvas, including the result of blending the current blend. This may be
        a partially-computed solution.

        """
        raise RuntimeError("Unimplemented method")

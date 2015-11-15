import argparse
import logging
import numpy as np
from matplotlib.image import imread, imsave
from sgdf.gui.editor import EditorView
from sgdf.fusion import get_fusion_algorithm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="gui", choices=["gui", "benchmark", "headless"],
                        help="Program mode")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Turns on debug logging (extra verbose)")
    parser.add_argument("-s", "--source", help="Source image path")
    parser.add_argument("-t", "--target", help="Target image path")
    parser.add_argument("-m", "--mask", help="Mask image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("-a", "--algorithm", default="reference",
                        help="Fusion algorithm to use")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.command == "gui":
        editor_view = EditorView()
        if args.source:
            editor_view.handle_loadsource(image_path=args.source)
        if args.target:
            editor_view.handle_loadtarget(image_path=args.target)
        editor_view.mainloop()
    elif args.command == "benchmark":
        print "Jiggaflops: 9999+"
    elif args.command == "headless":
        # TODO Find a better home for this code
        fusion = get_fusion_algorithm(args.algorithm)()
        mask = imread(args.mask)
        mask = np.mean(mask, 2)
        mask = (mask > np.average(mask))
        source_im = imread(args.source)
        if np.max(source_im) > 1.0:
            source_im = source_im.astype(np.float32) / 255.
        target_im = imread(args.target)
        if np.max(target_im) > 1.0:
            target_im = target_im.astype(np.float32) / 255.
        fusion.set_target_image(target_im)
        fusion.set_source_image(source_im)
        fusion.set_anchor_points(np.array([0, 0]), np.array([0, 0]))
        fusion.update_blend(mask)
        imsave(args.output, (fusion.get_fusion() * 255.).astype(np.uint8))

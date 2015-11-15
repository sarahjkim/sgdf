import argparse
import logging
from sgdf.benchmarking.suites import benchmark_default
from sgdf.benchmarking.headless import fusion_from_file
from sgdf.gui.editor import EditorView


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
        benchmark_default()
    elif args.command == "headless":
        assert args.algorithm
        assert args.source
        assert args.target
        assert args.mask
        fusion_from_file(args.algorithm, args.source, args.target, args.mask, args.output)

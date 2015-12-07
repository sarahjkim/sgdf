import argparse
import logging
import os
import re
from sgdf.benchmarking.suites import benchmark_default
from sgdf.benchmarking.headless import fusion_from_file
from sgdf.cli.extra import parse_extra
from sgdf.gui.editor import EditorView


def main():
    command_modes = ["gui", "benchmark", "headless"]

    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="gui", choices=command_modes,
                        help="Program mode")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Turns on debug logging (extra verbose)")
    parser.add_argument("--wait", action="store_true", default=False,
                        help="Wait for keyboard input (useful for debugging)")
    parser.add_argument("-a", "--algorithm", default="reference",
                        help="Fusion algorithm to use")
    parser.add_argument("-X", "--extra", nargs="*", default=[],
                        help="Extra options for algorithm")
    parser.add_argument("-s", "--source", help="Source image path")
    parser.add_argument("-t", "--target", help="Target image path")
    parser.add_argument("-m", "--mask", help="Mask image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("--offset", help="Offset for source image (--offset y,x)")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    algorithm_kwargs = {}
    if args.extra:
        for extra_option in args.extra:
            algorithm_kwargs.update(parse_extra(extra_option))

    if args.wait:
        raw_input(("Wait mode (--wait) enabled.\n"
                   "This feature lets you easily attach a debugger or profiler before starting.\n"
                   "PID: %d\n\n"
                   "Press any key to continue...") % os.getpid())

    if args.command == "gui":
        editor_view = EditorView(args.algorithm, algorithm_kwargs=algorithm_kwargs)
        if args.source:
            editor_view.handle_loadsource(image_path=args.source)
        if args.target:
            editor_view.handle_loadtarget(image_path=args.target)
        editor_view.mainloop()
    elif args.command == "benchmark":
        benchmark_default(args.algorithm, algorithm_kwargs=algorithm_kwargs)
    elif args.command == "headless":
        assert args.algorithm
        assert args.source
        assert args.target
        assert args.mask
        if args.offset:
            offset_match = re.match(r"^(?P<y>-?\d+),(?P<x>-?\d+)$", args.offset)
            assert offset_match, "Syntax for --offset should be \"--offset 41,62\""
            offset = [int(offset_match.group("y")), int(offset_match.group("x"))]
        else:
            offset = None
        fusion_from_file(args.algorithm, args.source, args.target, args.mask,
                         offset=offset,
                         output_path=args.output,
                         algorithm_kwargs=algorithm_kwargs)

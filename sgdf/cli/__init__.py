import argparse
import logging
from sgdf.gui.editor import EditorView


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="gui", choices=["gui", "benchmark"],
                        help="Program mode")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Turns on debug logging (extra verbose)")
    parser.add_argument("-s", nargs=1, help="Source image path")
    parser.add_argument("-t", nargs=1, help="Target image path")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.command == "gui":
        editor_view = EditorView()
        if args.s:
            editor_view.handle_loadsource(image_path=args.s[0])
        if args.t:
            editor_view.handle_loadtarget(image_path=args.t[0])
        editor_view.mainloop()
    elif args.command == "benchmark":
        print "Jiggaflops: 9999+"

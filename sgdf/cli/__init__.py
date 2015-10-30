import argparse
import logging
from sgdf.gui.editor import EditorView


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="gui", choices=["gui", "benchmark"],
                        help="Program mode")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Turns on debug logging (extra verbose)")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.command == "gui":
        editor_view = EditorView()
        editor_view.mainloop()
    elif args.command == "benchmark":
        print "Jiggaflops: 9999+"

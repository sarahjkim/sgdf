import argparse
from sgdf.gui.editor import EditorView


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="gui", choices=["gui", "benchmark"],
                        help="Program mode")
    args = parser.parse_args()

    if args.command == "gui":
        editor_view = EditorView()
        editor_view.mainloop()
    elif args.command == "benchmark":
        print "Jiggaflops: 9999+"

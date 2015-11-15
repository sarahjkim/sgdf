#!/usr/bin/env python2.7

import glob
import os
from setuptools import setup


if __name__ == "__main__":
    # Borrowed from https://bitbucket.org/django/django/src/tip/setup.py
    def fullsplit(path, result=None):
        """
        Split a pathname into components (the opposite of os.path.join) in a
        platform-neutral way.
        """
        if result is None:
            result = []
        head, tail = os.path.split(path)
        if head == "":
            return [tail] + result
        if head == path:
            return result
        return fullsplit(head, [tail] + result)

    packages, data_files = [], []
    root_dir = os.path.dirname(__file__)
    if root_dir != "":
        os.chdir(root_dir)
    for dirpath, dirnames, filenames in os.walk("sgdf"):
        # Ignore dirnames that start with "."
        for i, dirname in enumerate(dirnames):
            if dirname.startswith("."):
                del dirnames[i]
        if "__init__.py" in filenames:
            packages.append(".".join(fullsplit(dirpath)))
        elif filenames:
            data_files.append([dirpath, [os.path.join(dirpath, f) for f in filenames]])

    setup(name="sgdf",
          version="0.1.dev0",
          description="Streaming gradient domain fusion",
          url="https://github.com/sarajkim/sgdf",
          scripts=glob.glob("bin/sgdf*"),
          packages=packages,
          data_files=data_files,
          install_requires=["matplotlib", "numpy", "scipy", "Pillow"],
          test_suite="nose.collector",
          tests_require=["nose", "unittest2"])

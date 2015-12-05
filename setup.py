#!/usr/bin/env python2.7

import glob
import numpy as np
import os
from setuptools import setup, Extension


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


def find_tbb():
    include_path = glob.glob("vendor/tbb*/include")
    assert include_path, "Can't find TBB include directory."
    assert len(include_path) == 1, "Multiple versions of TBB installed? Ambiguous include path"
    library_path = glob.glob("vendor/tbb*/build/*_release/")
    assert library_path, ("Can't find TBB library directory.\n\n"
                          "    Please build the included version of TBB first.\n"
                          "    Go into vendor/tbb44_20151115oss/ and run 'make'.")
    assert len(library_path) == 1, "Multiple versions of TBB built? Ambiguous library path"
    return include_path[0], library_path[0]


if __name__ == "__main__":
    packages, data_files, extensions = [], [], []
    root_dir = os.path.dirname(__file__)
    if root_dir != "":
        os.chdir(root_dir)
    for dirpath, dirnames, filenames in os.walk("sgdf"):
        if os.path.basename(dirpath).startswith("_"):
            c_sources = filter(lambda name: name.endswith(".c") or name.endswith(".cpp"), filenames)
            if c_sources:
                extension_name = ".".join(fullsplit(dirpath))
                print "Extension:", extension_name
                tbb_include_dir, tbb_library_path = find_tbb()
                extension = Extension(extension_name,
                                      map(lambda name: os.path.join(dirpath, name), c_sources),
                                      include_dirs=[np.get_include(), tbb_include_dir],
                                      library_dirs=[tbb_library_path],
                                      libraries=["tbb"],
                                      extra_compile_args=["-std=c++11", "-g", "-O2", "-pthread"],
                                      extra_link_args=["-pthread"])
                extensions.append(extension)
        elif "__init__.py" in filenames:
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
          ext_modules=extensions,
          install_requires=["matplotlib", "numpy", "scipy", "Pillow"],
          test_suite="nose.collector",
          tests_require=["nose", "unittest2"])

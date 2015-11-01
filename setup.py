#!/usr/bin/env python2.7

import glob
import numpy as np
import os
import sys
from distutils.command.build_ext import build_ext
from setuptools import setup
from setuptools.extension import Extension


try:
    import _tkinter
except (ImportError, OSError):
    # pypy emits an oserror
    _tkinter = None


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


def _add_directory(path, dir, where=None):
    if dir is None:
        return
    dir = os.path.realpath(dir)
    if os.path.isdir(dir) and dir not in path:
        if where is None:
            path.append(dir)
        else:
            path.insert(where, dir)


class sgdf_build_ext(build_ext):
    def _find_library_file(self, library):
        # Fix for 3.2.x <3.2.4, 3.3.0, shared lib extension is the python shared
        # lib extension, not the system shared lib extension: e.g. .cpython-33.so
        # vs .so. See Python bug http://bugs.python.org/16754
        if 'cpython' in self.compiler.shared_lib_extension:
            existing = self.compiler.shared_lib_extension
            self.compiler.shared_lib_extension = "." + existing.split('.')[-1]
            ret = self.compiler.find_library_file(
                self.compiler.library_dirs, library)
            self.compiler.shared_lib_extension = existing
            return ret
        else:
            return self.compiler.find_library_file(
                self.compiler.library_dirs, library)

    def build_extensions(self):
        # Extra support for _numpytk native library
        # Borrowed from https://github.com/python-pillow/Pillow/blob/master/setup.py

        exts = []

        if _tkinter:
            TCL_VERSION = _tkinter.TCL_VERSION[:3]

        version = TCL_VERSION[0] + TCL_VERSION[2]
        if self._find_library_file("tcl" + version):
            tcl_feature = "tcl" + version
        elif self._find_library_file("tcl" + TCL_VERSION):
            tcl_feature = "tcl" + TCL_VERSION
        if self._find_library_file("tk" + version):
            tk_feature = "tk" + version
        elif self._find_library_file("tk" + TCL_VERSION):
            tk_feature = "tk" + TCL_VERSION

        if sys.platform == "darwin":
            # locate Tcl/Tk frameworks
            frameworks = []
            framework_roots = ["/Library/Frameworks",
                               "/System/Library/Frameworks"]
            for root in framework_roots:
                root_tcl = os.path.join(root, "Tcl.framework")
                root_tk = os.path.join(root, "Tk.framework")
                if (os.path.exists(root_tcl) and os.path.exists(root_tk)):
                    print("--- using frameworks at %s" % root)
                    frameworks = ["-framework", "Tcl", "-framework", "Tk"]
                    dir = os.path.join(root_tcl, "Headers")
                    _add_directory(self.compiler.include_dirs, dir, 0)
                    dir = os.path.join(root_tk, "Headers")
                    _add_directory(self.compiler.include_dirs, dir, 1)
                    break
            assert frameworks, "Cannot find Tcl/Tk headers"
            if frameworks:
                exts.append(Extension("sgdf.gui._numpytk",
                                      glob.glob(os.path.join("sgdf", "gui", "_numpytk", "*")),
                                      extra_compile_args=frameworks,
                                      extra_link_args=frameworks,
                                      include_dirs=[np.get_include(), "/usr/X11/include"]))
        else:
            assert tcl_feature and tk_feature
            exts.append(Extension("sgdf.gui._numpytk",
                                  glob.glob(os.path.join("sgdf", "gui", "_numpytk", "*")),
                                  libraries=[tcl_feature, tk_feature],
                                  include_dirs=[np.get_include()]))

        self.extensions[:] = exts

        build_ext.build_extensions(self)


if __name__ == "__main__":
    packages, extensions, data_files = [], [], []
    root_dir = os.path.dirname(__file__)
    if root_dir != "":
        os.chdir(root_dir)

    for dirpath, dirnames, filenames in os.walk("sgdf"):
        if os.path.basename(dirpath).startswith("_"):
            # if os.path.basename(dirpath) == "_numpytk":
            #     # The _numpytk library is configured manually
            #     continue
            c_sources = filter(lambda name: name.endswith(".c"), filenames)
            print "Extension:", ".".join(fullsplit(dirpath))
            extension = Extension(".".join(fullsplit(dirpath)),
                                  map(lambda name: os.path.join(dirpath, name), c_sources),
                                  include_dirs=[np.get_include()])
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
          install_requires=["matplotlib", "numpy", "scipy"],
          test_suite="nose.collector",
          tests_require=["nose", "unittest2"],
          cmdclass={"build_ext": sgdf_build_ext},
          ext_modules=extensions)

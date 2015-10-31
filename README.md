Streaming Gradient Domain Fusion
================================

A graphical user interface for combining images in the gradient domain.

Installation
------------

This program requires:

* Python 2.7
* Python setuptools
* Python.h development libraries
* C++ compiler

To install, run the following:

```shell
# Installs the sgdf package and executable
python2.7 setup.py install
```

You may want to run the installation inside of a virtualenv. Note that if you choose to do this,
give the VM access to the system packages. The option looks like this for `virtualenv`:

```shell
# (Optional) Create VM with access to system packages
virtualenv --system-site-packages NAME
```

The application can be started by
running `sgdf` on the command line.

While writing code for sgdf, you can also run the program without installing. In your shell, change
directories to the root of this repository. Then, run the bundled script:

```shell
# (Optional) Update the native extensions when necessary
python2.7 setup.py build_ext --inplace

# (Optional) Run the unit tests
python2.7 setup.py test

# Run the sgdf executable
./bin/sgdf
```

Contributing
------------

PEP8
100 line width

You can run the linter with this command:

```shell
flake8 --max-line-length=100 bin/ sgdf/
```

Authors
-------

This project is part of the CS194-15 course at UC Berkeley.

* Gilbert Ghang (@gilbertghang)
* Jason Zhang (@jasonzhang4628)
* Roger Chen (@rogerhub)
* Sarah Kim (@sarahjkim)
* Quinn Romanek (@quinnromanek)

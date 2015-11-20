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

You can just set up sgdf's dependencies and run the script directly:

```shell
# Installs sgdf's dependencies
pip install -r requirements.txt

# (Optional) Update the native extensions when necessary
python2.7 setup.py build_ext --inplace

# (Optional) Run the unit tests
python2.7 setup.py test

# (Optional) Run the unit tests with pytest, with debugger
py.test --pdb sgdf/

# Run sgdf
./bin/sgdf
```

Or, if you would like to install sgdf for use anywhere, run the following:

```shell
# Installs the sgdf package and executable
python2.7 setup.py install
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

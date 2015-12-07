import re


def parse_extra(text):
    """
    A crappy options parser for options in the form of:

        -Xmultithreading
        -Xno-multithreading
        -Xthreads=4
        -Xepsilon=0.0005

    Returns a dictionary in the form of {key: value}, suitable for kwargs.

    """
    if "=" not in text:
        if text.startswith("no-"):
            key = text[len("no-"):]
            value = False
        else:
            key = text
            value = True
    else:
        key_length = text.find("=")
        assert key_length > 0
        key = text[:key_length]
        value = text[key_length + 1:]
        if re.match(r"^-?\d+$", value):
            value = int(value)
        elif re.match(r"^-?(\d|\.)+$", value):
            value = float(value)

    return {key: value}

import numpy as np


class colors:
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'
    end = '\033[0m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def get_dict():
    return {
        range(0, 21): (colors.fg.green, 'VERY GOOD'),
        range(21, 41): (colors.fg.lightgreen, 'GOOD'),
        range(41, 51): (colors.fg.yellow, 'OK'),
        range(51, 76): (colors.fg.orange, 'BAD'),
        range(76, 101): (colors.fg.red, 'VERY BAD')
    }


def print_signs(val_new, val_test):
    if np.round(val_new) != val_test:
        print("Original: ", val_test, " ---- Predicted: ", val_new, f"{colors.bg.red}",
              " -------------------- Not match ---------", f"{colors.end}")
    else:
        result = get_dict()
        for key, value in result.items():
            if np.ceil(val_new * 100) in key:
                print("Original: ", val_test, " ---- Predicted: ", val_new, " -------------------- ",
                      f"{value[0]} {value[1]} {colors.end}", "---------")

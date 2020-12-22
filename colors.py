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


def get_semaphore():
    return {
        range(0, 21): ('|', colors.fg.green, 'O', '|O|O|O|O|', 'VERY GOOD'),
        range(21, 41): ('|O|', colors.fg.lightgreen, 'O', '|O|O|O|', 'GOOD'),
        range(41, 51): ('|O|O|', colors.fg.yellow, 'O', '|O|O|', 'OK'),
        range(51, 76): ('|O|O|O|', colors.fg.orange, 'O', '|O|', 'BAD'),
        range(76, 101): ('|O|O|O|O|', colors.fg.red, 'O', '|', 'VERY BAD')
    }


def print_signs(val_new, val_test):
    if np.round(val_new) != val_test:
        print("Original: ", val_test, " ---- Predicted: ", val_new, f"{colors.bg.red}",
              " -------------------- Not match ---------", f"{colors.end}")
    else:
        result = get_semaphore()
        for key, value in result.items():
            if np.ceil(val_new * 100) in key:
                print("Original: ", val_test, " ---- Predicted: ", val_new, "---------",
                      f"{colors.fg.lightgrey}{value[0]}{colors.end}{value[1]}{value[2]}{colors.end}"  # Semaphore color
                      f"{colors.fg.lightgrey}{value[3]}{colors.end}", "-------",  # Semaphore color
                      f"{value[1]}{value[4]}{colors.end}", "---------")  # Text color

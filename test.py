from utils import *

if __name__ == '__main__':
    rl, ru = get_CI(0.542, 1800, 0.05)
    print("%.3f %.3f" % (rl, ru))

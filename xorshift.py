import numpy as np
def xorshift_generator(x, max_num, a=13, b=7, c=17):
    x = np.uint64(x); a = np.uint64(a); b = np.uint64(b); c = np.uint64(c)
    while True:
        x ^= (x << a)
        x ^= (x >> b)
        x ^= (x << c)
        yield x % max_num

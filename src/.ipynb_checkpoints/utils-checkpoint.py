import time

def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        shape = results[0].shape
        print('Function', func.__name__, 'Shape', shape, 'time:', round((te -ts),1), 's')
        print()
        return result
    return timed
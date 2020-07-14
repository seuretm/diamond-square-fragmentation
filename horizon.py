import numpy
import random

def horizon(length, roughness=1, amplitude=1):
    res = [0 for x in range(length)]
    res[0]  = random.uniform(0, 1)
    res[-1] = random.uniform(0, 1)
    horizon_step(res, 0, length-1, roughness, roughness)
    a = min(res)
    b = max(res)
    res = numpy.round((numpy.array(res)-a) / (b-a) * amplitude).astype(int)
    return res

def horizon_step(arr, a, c, noise, roughness):
    if c-a<=1:
        return
    b = int((c-a)/2+.5) + a
    arr[b] = (arr[a]+arr[c])/2.0 + random.uniform(-noise/2, noise/2)
    horizon_step(arr, a, b, noise*roughness, roughness)
    horizon_step(arr, b, c, noise*roughness, roughness)

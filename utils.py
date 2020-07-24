import math
import numpy as np 
def squareGridShape(n):
    return [math.ceil(n**0.5)] * 2

def getClosestFactors(n):
    i = int(n ** 0.5)
    while (n % i != 0):
        i -= 1
    return (i, int(n/i))

def getBoundary(x, r, n):
    """returns in the form [lower, upper)"""
    lower = x - r
    upper = x + r + 1
    if lower < 0:
        lower = 0
    if upper > n:
        upper = n
    return (lower, upper)


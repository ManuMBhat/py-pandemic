import numpy as np

def getRandomSample(array, n):
    """returns in the form (x, y, array[x, y])"""
    if n > array.size:
        raise ValueError("Sample size must be smaller than number of elements in array")
    else:
        idx = np.random.choice(array.shape[0], size=n, replace=False)
        idy = np.random.choice(array.shape[1], size=n, replace=False)
        sample = array[idx, idy]
        return list(zip(idx, idy, sample))

def getNeighbours(array, randomSample, maxNeighbours):
    neighbours = np.zeros(len(randomSample), maxNeighbours)
    idx = list(zip(*randomSample))[0]
    idy = list(zip(*randomSample))[1]
    
    neighbours = np.array(array[])


community = np.arange(100).reshape(10, 10)

print(getRandomSample(community, 6))
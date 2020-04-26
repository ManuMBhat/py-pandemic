import numpy as np

def getBoundary(x, r, n):
    """returns in the form [lower, upper)"""
    lower = x - r
    upper = x + r + 1
    if lower < 0:
        lower = 0
    if upper > n:
        upper = n
    return (lower, upper)

def getRandomSample(array, n):
    """returns in the form (x, y, array[x, y])"""
    if n > array.size:
        raise ValueError("Sample size must be smaller than number of elements in array")
    else:
        idx = np.random.choice(array.shape[0], size=n, replace=False)
        idy = np.random.choice(array.shape[1], size=n, replace=False)
        sample = array[idx, idy]
        return list(zip(idx, idy, sample))

def getNeighbours(array, randomSample, radius):
    """Get the neighbours of randomSample[:, 2] within a radius.
    Border cases include -1 for missing neighbours."""

    maxNeighbours = (2*radius + 1)**2 - 1
    sampleSize = len(randomSample)
    neighbours = np.full((sampleSize, maxNeighbours), -1)
    height, width = array.shape

    idx = list(zip(*randomSample))[0]
    idy = list(zip(*randomSample))[1]
    xspans = np.array([getBoundary(x, radius, height) for x in idx], dtype=np.uint32)
    yspans = np.array([getBoundary(y, radius, width) for y in idy], dtype=np.uint32)
    
    for i in range(sampleSize):
        subgrid = np.ix_(range(*xspans[i]), range(*yspans[i]))
        x_rel = idx[i] - xspans[i, 0]
        y_rel = idy[i] - yspans[i, 0]
        
        #get rid of patient zero in subarray
        surrounding = np.delete(array[subgrid], x_rel*subgrid[1].shape[1] + y_rel)
        neighbours[i, :surrounding.shape[0]] = surrounding

    return neighbours

def updateGrid(array, community):
    """shuffle array based on Mersenne Twister algorithm in np.random"""
    
    #shuffle grid along both axes
    np.apply_along_axis(np.random.shuffle, 1, array)
    np.random.shuffle(array)
    
    #update locations of individuals
    getLoc = lambda x : (x // array.shape[0], x % array.shape[1])
    r = array.ravel()
    for i in range(array.size):
        community.people[r[i]].updateLoc(getLoc(i))
    
    return array

#testing
community = np.arange(100).reshape(10, 10)
print(community)
sample = getRandomSample(community, 7)
print(sample)
x = getNeighbours(community, sample, 1)
print(x)
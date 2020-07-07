import numpy as np

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

def equalGridCrossing(grid1, grid2, n):
    """Shuffle n randomly selected individuals between grid1 and grid2.
    Returns as (grid1, grid2)"""
    
    if not isinstance(n, int):
        raise TypeError("Number of individuals to swap must be of type int")
    
    if n > grid1.size or n > grid2.size:
        raise ValueError("number of individuals must be less than size of grid")

    id1x = np.random.choice(grid1.shape[0], size=n, replace=False)
    id1y = np.random.choice(grid1.shape[1], size=n, replace=False)
    id2x = np.random.choice(grid2.shape[0], size=n, replace=False)
    id2y = np.random.choice(grid2.shape[1], size=n, replace=False)
    grid1[id1x, id1y], grid2[id2x, id2y] = grid2[id2x, id2y], grid1[id1x, id1y]

    return (grid1, grid2)

def unequalGridCrossing(grid1, grid2, outGrid1, outGrid2):
    """Shuffle in a way that one grid loses abs(outGrid1 - outGrid2) individuals.
    If outGrid1 is equal to outGrid2 call equalGridCrossing."""
    
    if not (isinstance(outGrid1, int) or isinstance(outGrid2, int)):
        raise TypeError("Number of individuals to swap must be of type int")

    if (outGrid1 > grid1.size or outGrid2 > grid2.size):
        raise ValueError("Cannot relocate more than grid population")
    
    id1x = np.random.choice(grid1.shape[0], size=outGrid1, replace=False)
    id1y = np.random.choice(grid1.shape[1], size=outGrid1, replace=False)
    id2x = np.random.choice(grid2.shape[0], size=outGrid2, replace=False)
    id2y = np.random.choice(grid2.shape[1], size=outGrid2, replace=False)
    excess = abs(outGrid1 - outGrid2)

    if outGrid1 > outGrid2:
        #swap individuals that can be relocated in place
        grid1[id1x[:-excess], id1y[:-excess]], grid2[id2x, id2y] = grid2[id2x, id2y], grid1[id1x[:-excess], id1y[:-excess]]
        #swap excess
        nrow = np.full(grid2.shape[1], -1)
        nrow[:excess] = grid1[id1x[outGrid2:], id1y[outGrid2:]]
        #mark lost individuals in grid1 as -1
        grid1[id1x[outGrid2:], id1y[outGrid2:]] = -1
        #stack the new row created
        grid2 = np.vstack((grid2, nrow))

    elif outGrid2 > outGrid1:
        grid2[id2x[:-excess], id2y[:-excess]], grid1[id1x, id1y] = grid1[id1x, id1y], grid2[id2x[:-excess], id2y[:-excess]]
        nrow = np.full(grid1.shape[1], -1)
        nrow[:excess] = grid2[id2x[excess:], id2y[excess:]]
        grid2[id2x[excess:], id2y[excess:]] = -1
        grid1 = np.vstack((grid1, nrow))
    
    else :
        return equalGridCrossing(grid1, grid2, outGrid1)
    
    return (grid1, grid2)

#testing
community1 = np.arange(100).reshape(10, 10)
community2 = np.arange(100).reshape(10, 10)
a, b = unequalGridCrossing(community1, community2, 7, 2)
print(a, b, sep='\n')
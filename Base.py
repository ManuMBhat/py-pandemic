import math
import numpy as np
import random
from utils import squareGridShape, getBoundary

MAX_GDP = 1000000000000
random.seed(0)
deathChance = 0.05
THRESHOLD_FOR_HOTSPOT = 0.01
R0list = [2, 3, 4]
#recoveryTimes = [i for i in range(14, 7*5)]
recoveryTimes = [i for i in range(4, 12)]

class Person:
    state = 'S'
    timeInfected = 0

    def __init__(self, city, community, location):
        self.city = city
        self.community = community
        self.loc = location
        self.recoveryTime = random.choice(recoveryTimes)
        self.R0 = random.choice(R0list)

    def __str__(self):
        return str(self.state)

    def getCommunity(self):
        return self.community

    def updateState(self, val):
        self.state = val

    def getState(self):
        return self.state

class Community:
    """A larger unit of epidemic modelling (possibly a small sized community in India)."""
    __totalInfected = 0
    __totalRecovered = 0

    def __init__(self, city, name, N):
        self.city = city
        self.name = name
        self.shape = squareGridShape(N)
        self.people = [Person(self.city, self.name, (i // self.shape[0], i % self.shape[1])) for i in range(N)]
        self.__numPeople = N

        self.grid = np.arange(N).reshape(self.shape)
        #mark blanks
        if N < self.grid.size:
            self.grid.ravel()[N:] = -1

        self.outsiders = dict()

    def getNumPeople(self):
        return self.__numPeople

    def getTotalInfected(self):
        return self.__totalInfected

    def getTotalRecovered(self):
        return self.__totalRecovered

    def getTotalSusceptible(self):
        return (self.__numPeople - self.__totalInfected - self.__totalRecovered)

    def getInfected(self, city=None):
        """returns a list of people infected in the form (x, y, grid[x, y])"""
        
        base = np.arange(self.__numPeople).reshape(self.grid.shape)
        getLoc = lambda x : (x // base.shape[0], x % base.shape[1])
        infected = list()
        ifx, ify = list(), list()

        if city is None:            
            ifInfected = np.vectorize(lambda x : True if self.people[x].state == 'I' else False)
            mask = ifInfected(self.grid)
            infected = self.grid[mask]
            ifx, ify = getLoc(base[mask])
        
        else:
            for i in range(self.grid.size):
                p = self.grid.ravel()[i]
                try:
                    pstate = self.people[p].state
                except IndexError:
                    #deal with outsiders
                    communityID, localID = divmod(p, self.__numPeople)
                    pstate = city.Communities[communityID].people[localID].state
                if pstate == 'I':
                    infected.append(p)
                    pLoc = getLoc(i)
                    ifx.append(pLoc[0]); ify.append(pLoc[1])

            infected = np.array(infected)
        
        if infected.size == 0 :
            raise ValueError("No individual is currently infected")

        return list(zip(ifx, ify, infected))

    def showState(self, city=None):
        rgrid = self.grid.ravel()
        stateGrid = np.empty(self.grid.shape, dtype=np.dtype('<U1'))
        sgview = stateGrid.ravel()
        for i in range(rgrid.size):
            p = rgrid[i]
            try:
                if p != -1:
                    sgview[i] = self.people[p].state
                else:
                    sgview[i] = p

            #deal with outsiders
            except IndexError:
                if city is not None:
                    communityID, localID = divmod(p, self.__numPeople)
                    sgview[i] = city.Communities[communityID].people[localID].state
                else:
                    print("this is embarrassing, pls fix") 
        
        #print(stateGrid)

    def getRandomSample(self, n):
        """Returns in a list of randomly selected individuals in the form (x, y, grid[x, y]).
        Does not return blanks ie -1 entries in grid"""

        if n > self.grid.size:
            raise ValueError("Sample size must be smaller than number of elements in array")
        
        else:
            #check for blank entries before grabbing location
            flattened = np.arange(self.grid.size)[np.where(self.grid.ravel() > -1)]
            
            ids = np.random.choice(flattened, size=n, replace=False)
            idx = ids // self.grid.shape[0]
            idy = ids % self.grid.shape[1]
            sample = self.grid[idx, idy]
            
            return list(zip(idx, idy, sample))

    def getNeighbours(self, randomSample, radius):
        """Get the neighbours of randomSample[:, 2] within a radius.
        Border cases include -1 for missing neighbours."""

        maxNeighbours = (2*radius + 1)**2 - 1
        sampleSize = len(randomSample)
        neighbours = np.full((sampleSize, maxNeighbours), -1)
        height, width = self.grid.shape

        idx = list(zip(*randomSample))[0]
        idy = list(zip(*randomSample))[1]
        xspans = np.array([getBoundary(x, radius, height) for x in idx], dtype=np.uint32)
        yspans = np.array([getBoundary(y, radius, width) for y in idy], dtype=np.uint32)
        
        for i in range(sampleSize):
            subgrid = np.ix_(range(*xspans[i]), range(*yspans[i]))
            x_rel = idx[i] - xspans[i, 0]
            y_rel = idy[i] - yspans[i, 0]
            
            #get rid of patient zero in subgrid
            surrounding = np.delete(self.grid[subgrid], x_rel*subgrid[1].shape[1] + y_rel)
            neighbours[i, :surrounding.shape[0]] = surrounding

        return neighbours

    def updateLocations(self, city=None):
        #generate a meshgrid and zip the flattened arrays for coords
        x = np.arange(self.grid.shape[0])
        y = np.arange(self.grid.shape[1])
        xx, yy = np.meshgrid(x, y)
        locs = list(zip(yy.ravel(), xx.ravel()))

        r = self.grid.ravel()
        for i in range(r.size):
            p = r[i]
            if p != -1:
                try:
                    self.people[p].loc = locs[i]
                #deal with outsiders
                except IndexError:
                    if city is not None:
                        communityID, localID = divmod(p, self.__numPeople)
                        city.Communities[communityID].people[localID].loc = locs[i]
                    else:
                        print("this is embarrassing, pls fix")

    def updateGrid(self, city=None):
        """Shuffle array based on Mersenne Twister algorithm in np.random;
        Update the location and state of each individual in community."""

        #infect neighbours of those infected based upon individual's R0
        infected = self.getInfected(city=city)
        neighboursOfInfected = self.getNeighbours(infected, 1)
        infectedPeople = list(zip(*infected))[2]

        infectedR0s = list()
        for x in infectedPeople:
            try:
                infectedR0s.append(self.people[x].R0)
            #deal with pesky outsiders
            except IndexError:
                if city is not None:
                    communityID, localID = divmod(x, self.__numPeople)
                    infectedR0s.append(city.Communities[communityID].people[localID].R0)
                else:
                    print("this is embarrassing, pls fix")
        infectedR0s = np.array(infectedR0s)

        for k in range(len(infected)):
            #spread the disease
            currentNeighbours = neighboursOfInfected[k, neighboursOfInfected[k] > -1]
            spread = np.random.choice(currentNeighbours, size=min(currentNeighbours.size,
                                      infectedR0s[k]), replace=False)
            for l in spread:
                try:
                    unluckyNeighbour = self.people[l]
                #deal with pesky outsiders
                except IndexError:
                    if city is not None:
                        communityID, localID = divmod(l, self.__numPeople)
                        unluckyNeighbour = city.Communities[communityID].people[localID]
                    else:
                        print("this is embarrassing, pls fix")
                if unluckyNeighbour.getState() == 'S':
                    unluckyNeighbour.updateState('I')
                    self.__totalInfected += 1
            
            try:
                p = self.people[infectedPeople[k]]
            except IndexError:
                communityID, localID = divmod(infectedPeople[k], self.__numPeople)
                p = city.Communities[communityID].people[localID]
            
            #increase individual's infected time
            p.timeInfected += 1
            #mark individuals as recovered/deceased if they've crossed their recovery time
            if p.timeInfected >= p.recoveryTime:
                p.updateState('R')
                self.__totalRecovered += 1
                self.__totalInfected -= 1

        #update state and locations of individuals
        self.updateLocations()
        #show the state
        self.showState(city=city)
        #shuffle grid along both axes
        np.apply_along_axis(np.random.shuffle, 1, self.grid)
        np.random.shuffle(self.grid)
    def isHotspot(self,threshold=THRESHOLD_FOR_HOTSPOT):
        if self.__totalInfected >= self.__numPeople:
            return True 
        return False
class City:
    __noOfTravellers = 0
    __numInfected = 0
    __numRecovered = 0

    def __init__(self, name, population, gdp, healthCare):
        self.name = name
        self.population = population
        self.gdp = gdp
        self.healthCare = healthCare
        #Community size fixed to 100 as of now
        self.numCommunities = population // 100
        self.Communities = [Community(self.name, _, 100) for _ in range(self.numCommunities)]
        #self.avgPeopleContact = 15
        #self.transmission = 0.2
        

    def __str__(self):
        return self.name

    def equalGridCrossing(self, c1, c2, n):
        """Shuffle n randomly selected individuals between grid1 and grid2.
        Returns as (grid1, grid2)"""
        
        if not isinstance(n, int):
            raise TypeError("Number of individuals to swap must be of type int")
        
        grid1, grid2 = self.Communities[c1].grid, self.Communities[c2].grid
        if n > grid1.size or n > grid2.size:
            raise ValueError("number of individuals must be less than size of grid")

        #way to differentiate travellers from residents - local v global ID
        globalIDext = lambda c : c * self.Communities[c]._Community__numPeople

        flattened1 = np.arange(grid1.size)[np.where(grid1.ravel() > -1)]
        id1 = np.random.choice(flattened1, size=n, replace=False)
        id1x = id1 // grid1.shape[0]
        id1y = id1 % grid1.shape[1]
        out1 = grid1[id1x, id1y] + globalIDext(c1)

        flattened2 = np.arange(grid2.size)[np.where(grid2.ravel() > -1)]
        id2 = np.random.choice(flattened2, size=n, replace=False)
        id2x = id2 // grid2.shape[0]
        id2y = id2 % grid2.shape[1]
        out2 = grid2[id2x, id2y] + globalIDext(c2)

        grid1[id1x, id1y], grid2[id2x, id2y] = out2, out1

        return (grid1, grid2)

    def unequalGridCrossing(self, C1, C2, outC1, outC2):
        """Shuffle in a way that one grid loses abs(outC1 - outC2) individuals.
        If outC1 is equal to outC2 call equalGridCrossing."""
        
        if not (isinstance(outC1, int) or isinstance(outC2, int)):
            raise TypeError("Number of individuals to swap must be of type int")

        grid1, grid2 = self.Communities[C1].grid, self.Communities[C2].grid
        if (outC1 > grid1.size or outC2 > grid2.size):
            raise ValueError("Cannot relocate more than grid population")

        if (outC1 == outC2):
            self.equalGridCrossing(grid1, grid2, outC1)
        
        excess = abs(outC1 - outC2)
        flattened1 = np.arange(grid1.size)[np.where(grid1.ravel() > -1)]
        id1 = np.random.choice(flattened1, size=outC1, replace=False)
        id1x = id1 // grid1.shape[0]
        id1y = id1 % grid1.shape[1]

        flattened2 = np.arange(grid2.size)[np.where(grid2.ravel() > -1)]
        id2 = np.random.choice(flattened2, size=outC2, replace=False)
        id2x = id2 // grid2.shape[0]
        id2y = id2 % grid2.shape[1]

        #change local to global IDs
        globalIDext = lambda c : c * self.Communities[c]._Community__numPeople
        outGrid1 = grid1[id1x, id1y] + globalIDext(C1)
        outGrid2 = grid2[id2x, id2y] + globalIDext(C2)

        if outC1 > outC2:
            #swap individuals that can be relocated in place
            grid1[id1x[:-excess], id1y[:-excess]], grid2[id2x, id2y] = outGrid2, outGrid1[:-excess]
            #swap excess
            nrow = np.full(grid2.shape[1], -1)
            nrow[:excess] = outGrid1[outC2:]
            #mark lost individuals in grid1 as -1
            grid1[id1x[outC2:], id1y[outC2:]] = -1
            #stack the new row created
            grid2 = np.vstack((grid2, nrow))

        else :
            grid2[id2x[:-excess], id2y[:-excess]], grid1[id1x, id1y] = outGrid1, outGrid2[:-excess]
            nrow = np.full(grid2.shape[1], -1)
            nrow[:excess] = outGrid2[outC1:]
            grid2[id2x[outC1:], id2y[outC1:]] = -1
            grid1 = np.vstack((grid1, nrow))
        
        return (grid1, grid2)
    
    def updateCity(self):
        #unequal traveling not implemented
        
        #self.__numSusceptible is computed using getNumSusceptible()
        self.__numInfected = 0
        self.__numRecovered = 0
        
        for c in self.Communities:
            c.updateGrid(city=self)
            self.__numInfected += c.getTotalInfected()
            self.__numRecovered += c.getTotalRecovered()

    def getNumInfected(self):
        return self.__numInfected

    def getNumSusceptible(self):
        return (self.population - self.__numInfected - self.__numRecovered)

    def getNumRecovered(self):
        return self.__numRecovered

    def get_population(self):
        return self.population
    
    def get_gdp(self):
        return self.gdp
    
    def get_avgPeopleContact(self):
        return self.avgPeopleContact
    
    def get_healthCare(self):
        return self.healthCare

    def set_healthCare(self,val):
        self.healthCare = val

    def social_distancing_protocol(self, socialDistanceVal):
        self.avgPeopleContact = math.floor((1 - socialDistanceVal) * self.avgPeopleContact)

    def setTravellingPopulation(self, restriction=1):
        self.__noOfTravellers = (restriction * self.gdp * self.population)//MAX_GDP

import math
import numpy as np
import random
import functools
import matplotlib.pyplot as plt

from utils import *

MAX_GDP = 1000000000000
random.seed(0)
deathChance = 0.05

R0list = [2, 3, 4]
#recoveryTimes = [i for i in range(14, 7*5)]
recoveryTimes = [i for i in range(4, 12)]

"""
    Helper function
    Chooses a destination city and adds traveller to that city
"""
def travel_to_destination(cities, originCityName, traveller):
    destination = originCityName
    while(destination == originCityName):
        destination = random.choice(cities)
    destination.add_person(traveller)

class Person:
    state = 'S'
    timeInfected = 0

    def __init__(self, community, location):
        self.community = community
        self.loc = location
        self.isTravelling = False
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
    """A larger unit of epidemic modelling (possibly in India)."""
    __totalInfected = 0
    __totalRecovered = 0

    def __init__(self, city, name, N):
        self.city = city
        self.name = name
        self.shape = getClosestFactors(N)
        self.people = [Person(self.name, (i // self.shape[0], i % self.shape[1])) for i in range(N)]
        self.__numPeople = N
        self.grid = np.arange(N).reshape(self.shape)

    def getTotalInfected(self):
        return self.__totalInfected

    def getTotalRecovered(self):
        return self.__totalRecovered

    def getTotalSusceptible(self):
        return (self.__numPeople - self.__totalInfected - self.__totalRecovered)

    def getInfected(self):
        """returns in the form (x, y, array[x, y])"""
        
        base = np.arange(self.__numPeople).reshape(self.grid.shape)
        ifInfected = np.vectorize(lambda x : True if self.people[x].state == 'I' else False)
        mask = ifInfected(self.grid)
        infected = self.grid[mask]
        
        if infected.size == 0 :
            raise ValueError("No individual is currently infected")

        getLoc = np.vectorize(lambda x : (x // base.shape[0], x % base.shape[1]))
        ifx, ify = getLoc(base[mask])

        return list(zip(ifx, ify, infected))

    def showState(self):
        grid = self.grid
        stateGrid = np.chararray(self.grid.shape, unicode=True)
        for i in range(stateGrid.shape[0]):
            for j in range(stateGrid.shape[1]):
                p = grid[i, j]
                if p != -1:
                    stateGrid[i, j] = p
                else:
                    stateGrid = -1

        print(stateGrid)

    def getRandomSample(self, n):
        """returns in the form (x, y, array[x, y])"""

        if n > self.grid.size:
            raise ValueError("Sample size must be smaller than number of elements in array")
        else:
            idx = np.random.choice(self.grid.shape[0], size=n, replace=False)
            idy = np.random.choice(self.grid.shape[1], size=n, replace=False)
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

    def updateLocations(self):
        #generate a meshgrid and zip the flattened arrays for coords
        x = np.arange(self.grid.shape[0])
        y = np.arange(self.grid.shape[1])
        xx, yy = np.meshgrid(x, y)
        locs = list(zip(yy.ravel(), xx.ravel()))

        r = self.grid.ravel()[self.grid.ravel() > -1]
        for i in range(self.grid.size):
            fella = self.people[r[i]]
            if not fella.isTravelling():
                fella.loc = locs[i]
            else:
                #update for outsiders
                eval(fella.community).people[r[i]].loc = locs[i]

    def updateGrid(self):
        """shuffle array based on Mersenne Twister algorithm in np.random;
        Also update the location and state of each individual in community."""
        
        #mark individuals as recovered/deceased if they've crossed their recovery time
        maskfunc = lambda x : True if self.people[x].timeInfected >= self.people[x].recoveryTime else False
        maskfunc = np.vectorize(maskfunc)
        recovered = self.grid[maskfunc(self.grid)]
        for i in recovered:
            self.people[i].updateState('R')
            self.__totalRecovered += 1
            self.__totalInfected -= 1
        
        #infect neighbours of those infected based upon individual's R0
        infected = self.getInfected()
        neighboursOfInfected = self.getNeighbours(infected, 1)
        infectedPeople = list(zip(*infected))[2]
        infectedR0s = np.array([self.people[x].R0 for x in infectedPeople])

        for k in range(len(infected)):
            #increase individual's infected time
            self.people[infectedPeople[k]].timeInfected += 1
            #spread the disease
            currentNeighbours = neighboursOfInfected[k, neighboursOfInfected[k] > -1]
            spread = np.random.choice(currentNeighbours, size=min(currentNeighbours.size, 
                                      infectedR0s[k]), replace=False)
            for l in spread:
                if self.people[l].getState() == 'S':
                    self.people[l].updateState('I')
                    self.__totalInfected += 1

        #update state and locations of individuals
        self.updateLocations()
        #show the state
        self.showState()

        #shuffle grid along both axes
        np.apply_along_axis(np.random.shuffle, 1, self.grid)
        np.random.shuffle(self.grid)

class City(object):
    def __init__(self, name, population, gdp, healthCare):
        self.name = name
        self.population = population
        self.gdp = gdp
        self.healthCare = healthCare
        self.noOfTravellers = 0
        self.people = [Person(self.name,0) for _ in range(self.population)]
        self.avgPeopleContact = 15
        self.transmission = 0.2
        self.infected = list()
        self.dead = list()

    def __str__(self):
        return self.name
    
    def get_population(self):
        return self.population
    
    def get_gdp(self):
        return self.gdp
    
    def get_avgPeopleContact(self):
        return self.avgPeopleContact
    
    def get_healthCare(self):
        return self.healthCare

    def get_person(self,index):
        return self.people[index]

    def get_infected_num(self):
        return len(self.infected)

    def set_healthCare(self,val):
        self.healthCare = val

    def add_person(self,person):
        self.population += 1
        self.people.append(person)

    def get_travellers(self):
        return self.travellers

    def social_distancing_protocol(self, socialDistanceVal):
        self.avgPeopleContact = math.floor((1 - socialDistanceVal) * self.avgPeopleContact)

    def travelling_population(self, restriction=1):
        self.noOfTravellers = (restriction * self.gdp * self.population)//MAX_GDP
        self.travellers = random.sample(self.people, self.noOfTravellers)
        self.population -= self.noOfTravellers
        for i in self.travellers:
            self.people.remove(i)

    def infection_run(self):
        R0 = self.avgPeopleContact * self.transmission//1
        num_infection = int((R0 * len(self.infected))//1)
        sample = random.sample(self.people,num_infection)
        for i in sample:
            i.set_infected(1)
        self.infected = list(filter(lambda x : x.infected == 1, self.people))


    def health_run(self):
        for i in self.infected:
            probabilityOfCure = random.SystemRandom().uniform(0,1)
            if probabilityOfCure <= self.healthCare:
                i.set_infected(0)
            else:
                probabilityOfDeath = random.SystemRandom().uniform(0,1)
                if probabilityOfDeath <= deathChance:
                    i.set_alive(0)
                    i.set_infected(0)
                    self.people.remove(i)
                    self.population -=1
        
        self.infected = list(filter(lambda x: x.infected == 1, self.people))

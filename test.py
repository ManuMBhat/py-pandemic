from Base import City
import math, random
import matplotlib.pyplot as plt

def test():
    "Does not include travellers"

    #City of population 1000 with 10 communities each of size 100
    blore = City("Bangalore", 1000, gdp=1e9, healthCare=1)
    nS, nI, nR = [], [], []
    
    #create patient zero in each community
    for _ in range(len(blore.Communities)):
        currentCommunity = blore.Communities[_]
        pZero = random.randint(0, currentCommunity.grid.size - 1)
        currentCommunity.people[pZero].state = 'I'
    
    #Initial states
    nS.append(blore.getNumSusceptible())
    nI.append(blore.getNumInfected())
    nR.append(blore.getNumRecovered())
    
    #run for 15 iterations of the model
    for _ in range(15):
        blore.updateCity()
        nS.append(blore.getNumSusceptible())
        nI.append(blore.getNumInfected())
        nR.append(blore.getNumRecovered())
    
    plt.plot(nS, 'b', nI, 'r', nR, 'grey')
    plt.show()

if __name__ == "__main__":
    test()

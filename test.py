from Base import City
import math, random
import matplotlib.pyplot as plt

def test():
    "Does not include traveling"

    #City of population 1000 with 10 communities each of size 100
    blore = City("Bangalore", 1000, 10, 1)
    nS, nI, nR = [], [], []
    
    #create patient zero in each community
    for _ in range(100):
        pZero = math.trunc(random.uniform(0, 100))
        blore.Communities[_].people[pZero].state = 'I'
    
    nS.append(blore.getNumSusceptible())
    nI.append(blore.getNumInfected())
    nR.append(blore.getNumRecovered())
    for _ in range(15):
        blore.updateCity()
        nS.append(blore.getNumSusceptible())
        nI.append(blore.getNumInfected())
        nR.append(blore.getNumRecovered())
    
    plt.plot(nS, 'b', nI, 'r', nR, 'grey')
    plt.show()

if __name__ == "__main__":
    test()

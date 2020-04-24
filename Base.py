import numpy as np 
import random
import functools
MAX_GDP = 1000000000000
random.seed(0)
class Person(object):
    def __init__(self,city,infected):
        self.city = city
        self.infected = infected

    def __str__(self):
        return str(self.infected)

    def set_city(self,city):
        self.city = city

    def get_city(self):
        return self.city

    def set_infected(self,val):
        self.infected = val

    def get_infected(self):
        return self.infected



class City(object):
    def __init__(self,name,population,gdp,healthCare):
        self.name = name
        self.population = population
        self.gdp = gdp
        self.healthCare = healthCare
        self.noOfTravellers = 0
        self.people = [Person(self.name,0) for _ in range(self.population)]
        self.avgPeopleContact = 15
        self.transmission = 0.2

    def __str__(self):
        return self.name
    
    def get_population(self):
        return self.population
    
    def get_gdp(self):
        return self.gdp

    def get_healthCare(self):
        return self.healthCare

    def add_person(self,person):
        self.population += 1
        self.people.append(person)

    def travelling_population(self,restriction = 1):
        self.noOfTravellers = (restriction * self.gdp * self.population)//MAX_GDP
        self.travellers = random.sample(self.people,self.noOfTravellers)
        for i in self.travellers:
            self.people.remove(i)

    def infection_run(self):
        infected = list(reduce(lambda x: x.infected == 1 , self.people))
        R0 = self.avgPeopleContact * self.transmission
        sample = random.sample(self.people,R0*len(infected))
        for i in sample:
            i.set_infected(1)




def test():
    a = City("Milan",10000000,100000000,0.5)     
    a.travelling_population()  
if __name__ == "__main__":
    test()
    


    
import numpy as np 
import random
import functools
import matplotlib.pyplot as plt 
import os
MAX_GDP = 1000000000000
random.seed(0)
deathChance = 0.05


"""
    Helper function
    Chooses a destination city and adds traveller to that city
"""
def travel_to_destination(cities,originCityName,traveller):
    destination = originCityName
    while(destination == originCityName):
        destination = random.choice(cities)
    destination.add_person(traveller)


class Person(object):
    def __init__(self,city,infected):
        self.city = city
        self.infected = infected
        self.alive = 1

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
    
    def set_alive(self,val):
        self.alive = val



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

    def social_distancing_protocol(self,socialDistanceVal):
        self.avgPeopleContact = (1 - socialDistanceVal) * self.avgPeopleContact
        self.avgPeopleContact = int(self.avgPeopleContact//1)

    def travelling_population(self,restriction = 1):
        self.noOfTravellers = (restriction * self.gdp * self.population)//MAX_GDP
        self.travellers = random.sample(self.people,self.noOfTravellers)
        self.population -= self.noOfTravellers
        for i in self.travellers:
            self.people.remove(i)

    def infection_run(self):
        
        R0 = self.avgPeopleContact * self.transmission//1
        num_infection = int((R0 * len(self.infected))//1)
        sample = random.sample(self.people,num_infection)
        for i in sample:
            i.set_infected(1)
        self.infected = list(filter(lambda x: x.infected == 1 , self.people))


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







    
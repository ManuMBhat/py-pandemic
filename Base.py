import numpy as np 

MAX_GDP = 1000000000000

class Person(object):
    def __init__(self,city,infected):
        self.city = city
        self.infected = infected
        

class City(object):
    def __init__(self,name,population,gdp,healthCare):
        self.name = name
        self.population = population
        self.gdp = gdp
        self.healthCare = healthCare
        self.travellers = 0

    def __str__(self):
        return self.name
    
    def get_population(self):
        return self.population
    
    def get_gdp(self):
        return self.gdp

    def get_healthCare(self):
        return self.healthCare

    def travelling_population(self):
        self.travellers = (self.gdp * self.population)//MAX_GDP
        return self.travellers

    


    
from Base import City,Person
import matplotlib.pyplot as plt


def test():
    nums = 100
    output = list()
    death = list()
    
    a = City("Mumbai",1000000,240000000000,0.3)
    person = a.get_person(10)
    person.set_infected(1)
    for i in range(nums):
        a.infection_run()
        temp = a.get_population()
        if i > 6 and i%5 == 1 and a.get_avgPeopleContact() >= 0: 
            a.set_healthCare(0.7)
            a.social_distancing_protocol(0.5)
        a.health_run()
        output.append(a.get_infected_num())
        death.append(temp - a.get_population())
        print(output[-1])
        if(output[-1] <= 0):
            break
    plt.plot(output)
    plt.plot(death)
    cumalativeDeath = [sum(death[0:i]) for i in range(len(death))]
    plt.plot(cumalativeDeath)
    plt.show()
        
if __name__ == "__main__":
    test()
    

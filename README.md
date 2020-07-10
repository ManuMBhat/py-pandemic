# Py-Pandemic
A pandemic simulator based on the [**SIR**](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model) (**S**usceptible, **I**nfectious, **R**ecovered/Dead) compartmental model in epidemiology.

## Currently supported features

- [x] Population shuffling
- [x] Obtain people in contact with an individual
- [x] Scalable from communities to cities
- [x] Support for travellers
- [ ] R0s based on social distancing parameters
- [ ] Recovery time based on health care parameters
- [ ] Contact tracing
- [ ] Hotspots

### Dependencies
* [NumPy](https://pypi.org/project/numpy/)

### Example

To simulate a city of population 1000, create a `City` object. Currently the community size is fixed to 100.
```python
blore = City("Bangalore", 1000, gdp=1e9, healthCare=1)
```

Before calling the `updateCity()` method make sure there's at least one infected person in each community of the city.
```python
for _ in range(len(blore.Communities)):
        currentCommunity = blore.Communities[_]
        pZero = random.randint(0, currentCommunity.grid.size - 1)
        currentCommunity.people[pZero].state = 'I'
```

You can call `updateCity()` a few times until everyone reaches state **R**. Other helpful methods such as `getNumInfected()` and `getNumSusceptible()` let you track infectious and susceptible people respectively during each iteration.

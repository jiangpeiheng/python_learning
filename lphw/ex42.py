
## Animal is-a object (yes, sort of confusing) look at the extra credit
class Animal(object):
    phass

## Dog is-a Animal
class Dog(Animal):
    def __init(self, name):
        ## Dog has-a __init__ function with self, name

## cat is a Animal
class Cat(Animal):
    def __init__(self, name):
        ## cat has a __init__ function with self, name
        self.name = name

## Person is a object
class Person(object):
    def __init__(self, name):
        ## person has a __init__ function with para self, name
        self.name = name

        ## person has a pet of some kind
        self.pet = None

## Employee is a person
class Employee(Person):
    def __init__(self, name, salary):
        super(Employee, self).__init__(name)
        self.salary = salary

class Fish(object):
    pass

class Salmon(Fish):
    pass

class Halibut(Fish):
    pass

rover = Dog("Rover")
satan = Cat("Satan")
mary = Person("Mary")
mary.pet = satan
frank = Employee("Frank", 120000)
frank.pet = rover
flipper = Fish()
crouse = Salmon()
harry = Halibut()

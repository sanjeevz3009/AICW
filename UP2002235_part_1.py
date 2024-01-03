from collections import namedtuple
# Partial functions in Python is a function that is created by fixing a certain number
# of arguments of another function
# I will be using partial functions in this code to create specialised callbacks
# for cleaner interface for the rest of the code and to help reduce code duplication
from functools import partial
# Importing typing module for type hinting purposes
from typing import List, Callable, Tuple
# Importing choices so we can randomly select values from a specified list
# and set the probability of something being selected
# Importing randint to select a random value
# Importing randrange to randomly select values from a specified range
# Importing random to randomly select floating numbers between 0 and 1
from random import choices, randint, randrange, random


# Type hinting for variables, tuples and partial functions
Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
Thing = namedtuple("Thing", ["name", "value", "weight"])


things = [
    Thing("Laptop", 500, 2200),
    Thing("Headphones", 150, 160),
    Thing("Coffee Mug", 60, 350),
    Thing("Notepad", 40, 333),
    Thing("Water Bottle", 30, 192)
]


more_things = [
    Thing("Mints", 5, 25),
    Thing("Socks", 10, 38),
    Thing("Tissues", 15, 80),
    Thing("Phone", 500, 200),
    Thing("Baseball Cap", 100, 70)
] + things



def generate_genome(length: int) -> Genome:
    """
    Generate a genome where in this case it's just 1s and 0s 1 being fit and 0 being not
    so. The function below will help us generate a random list of 1s and 0s for the length specified.

    :param length: Length of the genome to generate in 0s and 1s
    :type length: int
    :return: A genome containing 0s and 1s will be returned in a list
    :rtype: Genome
    """
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    """
    Create a population which essentially is a list of genomes. We will call the generate_genome function
    however many times as the size specified until our population has the desired size.

    :param size: Size of the population
    :type size: int
    :param genome_length: Length of genome to generate in 0s and 1s.
    :type genome_length: int
    :return: Lists of genome 
    :rtype: Population
    """
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:
    """
    To take the genome and calculate the fitness value.
    The fitness function below takes the genome and returns a fitness value

    :param genome: Genome in a list which contains 0s and 1s.
    :type genome: Genome
    :param things: _description_
    :type things: [Thing]
    :param weight_limit: _description_
    :type weight_limit: int
    :raises ValueError: _description_
    :return: Fitness value returned
    :rtype: int
    """
    if len(genome) != len(things):
        raise ValueError("genome and things must be of the same length")
    
    weight = 0
    value = 0

    # To calculate the fitness we iterate through all things and check
    # if it's part of the solution by checking if the genome has a 1
    # at the given index
    for i, thing in enumerate(things):
        # If it has we add the wight and the value of this item to our accumulation
        # variables
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

            # As soon the weight exceeds the weight limit, we abort iteration
            # and we return a fitness value of 0
            if weight > weight_limit:
                return 0

    # If we managed to get to the end of the list of things without exceeding
    # the weight limit, we are returning the accumulated value
    return value


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    """
    Selects the pair of solutions, which will be the parents of two new solutions of the next generation.

    :param population: Lists of genome
    :type population: Population
    :param fitness_func: A partial function to calculate the fitness value
    :type fitness_func: FitnessFunc
    :return: A pair of genomes as lists will be returned
    :rtype: Population
    """
    # Solutions with the higher fitness should be more likely to be chosen
    # We use the function choices to specify the population, weights and length
    # By handing over the fitness of a genome as it's weight the fittest solutions
    # are most likely to be chosen for reproduction
    # k=2 specifies that we draw twice from our population to get a pair
    return choices (
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    """
    The single point crossover function takes two genomes are parameters and returns two genomes as output.

    :param a: Genome a
    :type a: Genome
    :param b: Genome b
    :type b: Genome
    :raises ValueError: _description_
    :return: Return two new genomes as output
    :rtype: Tuple[Genome, Genome]
    """
    # Making sure out genomes are the same length, as otherwise crossover would fail
    if len(a) != len(b):
        raise ValueError("Genomes a and be must be of the same length")
    
    length = len(a)

    # The length of the genome has to be at least two as if it's not there wouldn't
    # be a point to cut them in half/ it's not possible
    if length < 2:
        return a, b

    # We randomly chose a index to cut it in half 
    p = randint(1, length - 1)
    # We take the first half of genome a and the second half genome b
    # and put them together and return this as our first new solution
    # For the second solution we take first half of genome b and second half
    # of genome a and put them together
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    """
    The mutation function takes genome and a certain probability to change 1s to 0s and 0s to 1s at random positions.

    :param genome: Genome in a list which contains 0s and 1s.
    :type genome: Genome
    :param num: Number is times to mutate, defaults to 1
    :type num: int, optional
    :param probability: Probability of mutation, defaults to 0.5
    :type probability: float, optional
    :return: Return a mutated genome
    :rtype: Genome
    """
    for _ in range(num):
        # We chose a random index and if random returns a value higher than probability
        # we leave it alone
        index = randrange(len(genome))
        # Otherwise it is in our mutation probability and we need to change it to the absolute 
        # value of the current value minus one
        # This is because e.g. abs(1 - 1) = abs(0) = 0, abs(0 - 1) = abs(-1) = 1
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)

    return genome


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
) -> Tuple[Population, int]:
    """
    The function that pieces everything together and runs the evolution

    :param populate_func: A partial function that generates the population
    :type populate_func: PopulateFunc
    :param fitness_func: Calculates the fitness values for the population
    :type fitness_func: FitnessFunc
    :param fitness_limit: Defines one condition as if the fitness limit of the best solution
    exceeds this limit we are done and reached our goal
    :type fitness_limit: int
    :param selection_func: Selects the pair of solutions, which will be the parents of two new solutions of the next generation.
    , defaults to selection_pair
    :type selection_func: SelectionFunc, optional
    :param crossover_func: The single point crossover function takes two genomes are parameters and returns two genomes as output.
    , defaults to single_point_crossover
    :type crossover_func: CrossoverFunc, optional
    :param mutation_func: The mutation function takes genome and a certain probability to change 1s to 0s and 0s to 1s at random positions.
    , defaults to mutation
    :type mutation_func: MutationFunc, optional
    :param generation_limit: The maximum number of generations our evolution runs for if it's not reaching the fitness limit before that, defaults to 100
    :type generation_limit: int, optional
    :return: _description_
    :rtype: Tuple[Population, int]
    """
    population = populate_func()

    # We first sort out population by fitness
    # This way we know that our top solutions are inhabiting of the first indices of our list of
    # genomes

    # The loops runs till if we reached the fitness limit/ looped for generation limit
    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )

        # Check if we have already reached the fitness limit
        if fitness_func(population[0]) >= fitness_limit:
            break

        # We keep our top two solutions for our next generation
        next_generation = population[0:2]

        # Generate all new solutions for our next generation
        # We pick two parents and get two new solutions every time
        # So we loop for half the length of a generation to get as many solutions in our next generation as before
        # 
        for j in range(int(len(population) / 2) - 1):
            # We call the selection functions to get our parents
            parents = selection_func(population, fitness_func)
            # We put the parents into the crossover function to get two child solutions
            # for our next generation
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            # We also apply the mutation function for each offspring
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        # We then replace our current population with the next_generation
        population = next_generation


    # We finally return the current population in a sorted manner
    population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )

    # Return the population and how many generations we ran for
    return population, i

# We execute the script now by passing all the necessary data
# We call run evolution and store the last population and the amount of generations it took
# get there
population, generations = run_evolution(
    # The partial populate_func helps us preset the parameters which are specific to our current problem
    # That's how we can adjust our population function without handing the population size and genome length
    # to the run_evolution function and without the need to write a completely new populate function
    populate_func=partial(generate_population, size=10, genome_length=len(more_things)),
    # We hand over the list of things to our fitness function and predefined the weight to be 3KG
    fitness_func=partial(fitness, things=more_things, weight_limit=3000),
    fitness_limit=1310,
    generation_limit=100
)

def genome_to_things(genome: Genome, things: [Thing]) -> [Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]

    return result


print(f"Number of generations: {generations}")
print(f"Best solution: {genome_to_things(population[0], more_things)}")
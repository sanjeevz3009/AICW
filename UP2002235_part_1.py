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
# Importing randrange to randomly select values within the valued range of indices
# Importing random to select floating numbers between 0 and 1
from random import choices, randint, randrange, random


# We don't need this as we aren't doing anything with this
# initial_chromosome = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
# Define the target chromosome
target_chromosome = [1] * 32


# Type hinting for variables, tuples, functions partial functions
Chromosome = List[int]
Population = List[Chromosome]
FitnessFunc = Callable[[Chromosome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Chromosome, Chromosome]]
CrossoverFunc = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]
MutationFunc = Callable[[Chromosome], Chromosome]


def generate_chromosome(length: int) -> Chromosome:
    """
    Generate a chromosome where in this case it's just 1s and 0s. 1 being fit and 0 being not
    so. The function below will help us generate a random list of 1s and 0s for the length specified.

    :param length: Length of the chromosome
    :type length: int
    :return: A Chromosome containing 0s and 1s will be returned in a list
    :rtype: Chromosome
    """
    print("generate_chromosome: ", choices([0, 1], k=length))
    print("\n")
    return choices([0, 1], k=length)


def generate_population(size: int, chromosome_length: int) -> Population:
    """
    Create a population which essentially is a lists of chromosome. We will call the generate_chromosome function
    however many times from the size specified until our population has the desired size.

    :param size: Size of the population
    :type size: int
    :param chromosome_length: Length of chromosome to generate
    :type chromosome_length: int
    :return: 2D List containing chromosomes
    :rtype: Population
    """
    return [generate_chromosome(chromosome_length) for _ in range(size)]


def fitness(chromosome: Chromosome, target_chromosome: Chromosome) -> int:
    """
    Calculate fitness based on the number of matching bits with the target chromosome.

    :param chromosome: A Chromosome containing 0s and 1s in a list
    :type chromosome: Chromosome
    :param target_chromosome: The target chromosome to reach
    :type target_chromosome: Chromosome
    :return: Fitness value
    :rtype: int
    """
    # Calculate fitness based on the number of matching bits with the target chromosome
    print("fitness: ", sum(g == t for g, t in zip(chromosome, target_chromosome)))
    return sum(g == t for g, t in zip(chromosome, target_chromosome))


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    """
    Selects the pair of solutions, which will be the parents of two new solutions of the next generation.

    :param population: Lists of chromosome
    :type population: Population
    :param fitness_func: A partial function to calculate the fitness value
    :type fitness_func: FitnessFunc
    :return: A pair of chromosomes as lists will be returned
    :rtype: Population
    """
    # Solutions with the higher fitness should be more likely to be chosen
    # We use the function choices to specify the population, weights and length
    # By handing over the fitness of a Chromosome as it's weight the fittest solutions
    # are most likely to be chosen for reproduction
    # k=2 specifies that we draw twice from our population to get a pair
    print("selection_pair: ", choices (
        population=population,
        weights=[fitness_func(Chromosome) for Chromosome in population],
        k=2
        )
    )
    print("\n")
    return choices (
        population=population,
        weights=[fitness_func(Chromosome) for Chromosome in population],
        k=2
    )


def single_point_crossover(a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
    """
    The single point crossover function takes two chromosomes as parameters and returns two chromosomes as output.

    :param a: Chromosome a
    :type a: Chromosome
    :param b: Chromosome b
    :type b: Chromosome
    :raises ValueError: Raises an errors if chromosome a and b aren't the same length
    :return: Return two new Chromosomes as output
    :rtype: Tuple[Chromosome, Chromosome]
    """
    # Making sure out chromosome are the same length, as otherwise crossover would fail
    if len(a) != len(b):
        raise ValueError("Chromosome a and be must be of the same length")
    
    length = len(a)

    # The length of the chromosome has to be at least two as if it's not there wouldn't
    # be a point to cut them in half/ it's not possible
    if length < 2:
        return a, b

    # We randomly choose an index to cut it in half 
    p = randint(1, length - 1)
    # We take the first half of chromosome a and the second half chromosome b
    # and put them together and return this as our first new solution
    # For the second solution we take first half of chromosome b and second half
    # of chromosome a and put them together
    print("single_point_crossover: ", a[0:p] + b[p:], b[0:p] + a[p:])
    print("\n")
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(chromosome: Chromosome, num: int = 1, mutation_rate: float = 0.1) -> Chromosome:
    """
    The mutation function takes a chromosome and a certain probability to change 1s to 0s and 0s to 1s at random positions.

    :param chromosome: Chromosome in a list
    :type chromosome: Chromosome
    :param num: Number of times to mutate, defaults to 1
    :type num: int, optional
    :param mutation_rate: Probability of mutation, defaults to 0.1
    :type mutation_rate: float, optional
    :return: Return a mutated chromosome
    :rtype: Chromosome
    """
    for _ in range(num):
        # We chose a random index and if random returns a value higher than probability
        # we leave it alone
        index = randrange(len(chromosome))
        # Otherwise it is in our mutation probability and we need to change it to the absolute 
        # value of the current value minus one
        # This is because e.g. abs(1 - 1) = abs(0) = 0, abs(0 - 1) = abs(-1) = 1
        chromosome[index] = chromosome[index] if random() > mutation_rate else abs(chromosome[index] - 1)
    
    print("single_point_crossover: ", chromosome)
    print("\n")
    return chromosome


def genetic_algorithm(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        target_chromosome: Chromosome,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
) -> Tuple[Population, int]:
    """
    The function that pieces everything together and runs the genetic algorithm.

    :param populate_func: A partial function that generates the population
    :type populate_func: PopulateFunc
    :param fitness_func: Calculates the fitness values for the population
    :type fitness_func: FitnessFunc
    :param target_chromosome: Defines the target chromosome
    :type target_chromosome: Chromosome
    :param selection_func: Selects the pair of solutions, which will be the parents of two new solutions of the next generation
    , defaults to selection_pair
    :type selection_func: SelectionFunc, optional
    :param crossover_func: The single point crossover function takes two chromosomes as parameters and returns two chromosomes as output
    , defaults to single_point_crossover
    :type crossover_func: CrossoverFunc, optional
    :param mutation_func: The mutation function takes a chromosome and a certain probability to change 1s to 0s and 0s to 1s at random positions
    , defaults to mutation
    :type mutation_func: MutationFunc, optional
    :param generation_limit: The maximum number of generations our algorithm runs for if it's not reaching the fitness limit before that, defaults to 100
    :type generation_limit: int, optional
    :return: Returns the final generation/ population of chromosomes and the amount of generations it took to get there
    :rtype: Tuple[Population, int]
    """
    population = populate_func()

    # We first sort out population by fitness
    # This way we know that our top solutions are inhabiting of the first indices of our list of
    # chromosomes

    # The loops runs till if we reached the fitness limit/ looped for generation limit
    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda chromosome: fitness_func(chromosome),
            reverse=True
        )

        print(f"Generation {i + 1}: ")
        print("\n")
        # print(f"Generation {population}:\n\n")

        # Check if the best solution matches the target chromosome
        if population[0] == target_chromosome:
            break
        
        # We keep our top two solutions for our next generation
        next_generation = population[0:2]

        # Generate all new solutions for our next generation
        # We pick two parents and get two new solutions every time
        # So we loop for half the length of a generation to get as many solutions in our next generation as before
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
            key=lambda chromosome: fitness_func(chromosome),
            reverse=True
        )

    # Return the population and how many generations we ran for
    return population, i+1

# We execute the script now by passing all the necessary data
# We call run evolution and store the last population and the amount of generations it took
# get there
population, generations = genetic_algorithm(
    # The partial populate_func helps us preset the parameters which are specific to our current problem
    # That's how we can adjust our population function without handing the population size and chromosome length
    # to the genetic_algorithm function and without the need to write a completely new populate function
    populate_func=partial(generate_population, size=10, chromosome_length=32),
    # We hand over the list of things to our fitness function and predefined the weight to be 3KG
    fitness_func=partial(fitness, target_chromosome=target_chromosome),
    target_chromosome=target_chromosome,
    generation_limit=100
)


print(f"Number of generations: {generations}")
print(f"Best solution: {population[0]}")

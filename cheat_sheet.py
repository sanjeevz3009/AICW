# Roulette wheel implementation 1
###################################################################################################### 
import random

def roulette_wheel_selection(population, fitness_values):
    """
    In this example, population represents the individuals in your population, and fitness_values
    represents the corresponding fitness values for each individual.
    The roulette_wheel_selection function selects an individual based on their fitness using the
    roulette wheel selection method.
    The roulette_spin function is a helper function that performs the actual spinning of the wheel based
    on the normalized fitness values.
    The roulette_wheel_selection function takes the population and corresponding fitness values as input.

    total_fitness is calculated by summing up all the fitness values in the population.

    normalized_fitness is calculated by dividing each fitness value by the total fitness, turning them into probabilities.

    The roulette_spin function is a helper function that performs the actual spinning of the roulette wheel. It takes a list of probabilities and returns the index of the selected individual.

    In the roulette_wheel_selection function, the index of the selected individual is obtained using the roulette_spin function, and the corresponding individual is returned.

    The example usage section demonstrates how to use the functions with a sample population and fitness values, printing the selected individual.
    """
    # Calculate the total fitness of the population
    total_fitness = sum(fitness_values)
    
    # Normalize fitness values to probabilities
    normalized_fitness = [fit / total_fitness for fit in fitness_values]

    # Perform the roulette wheel spin and get the index of the selected individual
    selected_index = roulette_spin(normalized_fitness)

    # Return the selected individual from the population
    return population[selected_index]


def roulette_spin(probabilities):
    # Generate a random spin between 0 and 1
    spin = random.uniform(0, 1)
    
    # Initialize cumulative probability
    cumulative_probability = 0

    # Iterate through the probabilities and find the selected index
    for i, probability in enumerate(probabilities):
        cumulative_probability += probability
        
        # If the spin is less than or equal to the cumulative probability, select this index
        if spin <= cumulative_probability:
            return i

# Example usage
population = ['Individual1', 'Individual2', 'Individual3', 'Individual4']
fitness_values = [0.2, 0.5, 0.3, 0.7]

# Use the roulette_wheel_selection function to select an individual based on fitness
selected_individual = roulette_wheel_selection(population, fitness_values)

# Print the selected individual
print("Selected individual:", selected_individual)


# Roulette wheel implementation 2
######################################################################################################
import random as rd

def create_roulette_wheel(self):
    """
    Creates a roulette wheel by calculating cumulative fitness values.
    This method is separated from spin_roulette_wheel to improve performance.
    Returns the list of cumulative fitness values, representing a roulette wheel for selection.

    Calculates the total fitness of the population by summing up the fitness scores of all chromosomes.

    Normalizes the fitness scores to ensure they range between 0 and 1.

    Computes the cumulative fitness values, representing a roulette wheel for selection.
    """
    total_fitness = sum(chromosome.fitness_score for chromosome in self.population)  # Sum up all the chromosome fitness scores in the population
    normalized = [chromosome.fitness_score / total_fitness for chromosome in self.population]  # Normalize fitness scores to be between 0 to 1
    cumulative_fitness = []
    cumulative_total = 0

    # Calculate cumulative fitness values
    for proportion in normalized:
        cumulative_total += proportion
        cumulative_fitness.append(cumulative_total)
    return cumulative_fitness


def spin_roulette_wheel(self, cumulative_fitness):
    """
    This method randomly selects a chromosome from the population in a slightly biased manner.
    The method simulates spinning a roulette wheel to randomly select a parent based on its cumulative fitness.
    Returns a chromosome object.

    Generates a random value between 0 and 1 (selected_value) simulating the spin of a roulette wheel.

    Iterates through the cumulative fitness values and returns the chromosome corresponding to the first cumulative
    fitness value greater than or equal to the random value.

    Raises an exception if the ball drops out of the roulette wheel unexpectedly (this might indicate an issue in the code).
    """

    selected_value = rd.random()  # Generates a random value between 0 and 1

    # Iterates through the cumulative fitness values (population) and If the cumulative fitness value is greater than or equal to the random value,
    # it returns the corresponding individual from the population
    for i in range(len(cumulative_fitness)):
        value = cumulative_fitness[i]
        if value >= selected_value:
            return self.population[i]

    # Raises an exception if the ball drops out of the roulette wheel unexpectedly
    raise Exception("Exception: ball dropped out of roulette wheel :(")


# Roulette wheel implementation 3
######################################################################################################
from random import choices
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    """
    Selects the pair of solutions, which will be the parents of two new solutions of the next generation.
    Roulette Wheel Selection method is used here (Fitness Proportionate Selection).

    :param population: Lists of chromosome
    :type population: Population
    :param fitness_func: A partial function to calculate the fitness value
    :type fitness_func: FitnessFunc
    :return: A pair of chromosomes as lists will be returned
    :rtype: Population
    """
    # Solutions with the higher fitness should be more likely to be chosen
    # We use the function choices to specify the population, weights and length
    # By handing over the fitness of a Chromosome as it's weight the fittest solutions
    # are most likely to be chosen for reproduction
    # k=2 specifies that we draw twice from our population to get a pair

    # Below is easy way to do similar thing to a roulette wheel selection
    # print(
    #     "selection_pair: ",
    #     choices(
    #         population=population,
    #         weights=[fitness_func(Chromosome) for Chromosome in population],
    #         k=2,
    #     ),
    # )
    # print("\n")

    # return choices(
    #     population=population,
    #     weights=[fitness_func(Chromosome) for Chromosome in population],
    #     k=2,
    # )

    # Calculate the total fitness of the population
    total_fitness = sum(fitness_func(chromosome) for chromosome in population)

    # Normalise the fitness scores for each individual
    fitnesses = [fitness_func(chromosome) / total_fitness for chromosome in population]

    print("Normalised fitness:")
    for chromosome, fitness in zip(population, fitnesses):
        print(f"Chromosome: {chromosome}, Normalised fitness: {fitness}")

    # Select two individuals using the normalised fitness scores as weights
    parent1 = choices(population, weights=fitnesses, k=1)[0]
    parent2 = choices(population, weights=fitnesses, k=1)[0]

    parents = [parent1, parent2]

    return parents

# Tournament selection
######################################################################################################
import random

def tournament_selection(population, fitness_values, tournament_size, num_selections):
    """
    This code defines the tournament_selection function, which takes the population, corresponding
    fitness values, tournament size, and the number of individuals to select as input.
    It returns a list of selected individuals.

    The tournament_selection function iterates for the desired number of selections.

    For each selection, a random subset of the population (tournament_indices) of size tournament_size is chosen.

    The fitness values of individuals in the tournament are extracted.

    The index of the individual with the highest fitness in the tournament is determined.

    The selected individual is added to the list of selected_individuals.

    The function returns the list of selected individuals.

    The example usage section demonstrates how to use the function with a sample population, fitness values,
    tournament size, and the number of selections, printing the selected individuals.
    """
    selected_individuals = []

    for _ in range(num_selections):
        # Randomly select individuals for the tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness_values = [fitness_values[i] for i in tournament_indices]

        # Find the index of the individual with the highest fitness in the tournament
        winner_index = tournament_indices[tournament_fitness_values.index(max(tournament_fitness_values))]

        # Add the selected individual to the list of selected individuals
        selected_individuals.append(population[winner_index])

    return selected_individuals

# Example usage
population = ['Individual1', 'Individual2', 'Individual3', 'Individual4', 'Individual5']
fitness_values = [0.2, 0.5, 0.3, 0.7, 0.4]
tournament_size = 2
num_selections = 3

# Use the tournament_selection function to select individuals based on fitness
selected_individuals = tournament_selection(population, fitness_values, tournament_size, num_selections)

# Print the selected individuals
print("Selected individuals:", selected_individuals)


# Rank selection
######################################################################################################
import random

def rank_selection(population, num_selections):
    """
    Rank selection is a method used in genetic algorithms for selecting individuals from a
    population based on their ranks rather than their absolute fitness values.
    The key idea is to assign a rank to each individual in the population based on their fitness,
    and then select individuals with probabilities proportional to their ranks.
    This approach aims to strike a balance between favoring high-fitness individuals and maintaining diversity in the population.
    
    The rank_selection function takes the population and the number of individuals to select as input.

    The population is sorted based on fitness in ascending order, and the indices of the sorted individuals are stored in sorted_indices.

    Selection probabilities are calculated based on the rank of each individual in the sorted population.

    The roulette_spin function, which was used in the previous example, is reused here to perform roulette wheel selection with the calculated selection probabilities.

    For each selection, an individual is selected using the roulette wheel method, and the corresponding individual from the sorted population is added to the list of selected_individuals.

    The function returns the list of selected individuals.

    The example usage section demonstrates how to use the function with a sample population and the number of selections, printing the selected individuals.
    """
    # Sort the population based on fitness (ascending order)
    sorted_indices = sorted(range(len(population)), key=lambda k: population[k]['fitness'])

    # Calculate selection probabilities based on rank
    selection_probabilities = [(2 * (len(population) - i)) / (len(population) * (len(population) + 1)) for i in range(len(population))]

    selected_individuals = []

    for _ in range(num_selections):
        # Use roulette wheel selection with calculated probabilities
        selected_index = roulette_spin_rank_selection(selection_probabilities)
        selected_individuals.append(population[sorted_indices[selected_index]])

    return selected_individuals


def roulette_spin_rank_selection(probabilities):
    spin = random.uniform(0, 1)
    cumulative_probability = 0

    for i, probability in enumerate(probabilities):
        cumulative_probability += probability
        if spin <= cumulative_probability:
            return i

# Example usage
# Assume population is a list of dictionaries with each dictionary having a 'fitness' key
population = [{'fitness': 0.2}, {'fitness': 0.5}, {'fitness': 0.3}, {'fitness': 0.7}]
num_selections = 3

# Use the rank_selection function to select individuals based on fitness rank
selected_individuals = rank_selection(population, num_selections)

# Print the selected individuals
print("Selected individuals:", selected_individuals)


######################################################################################################

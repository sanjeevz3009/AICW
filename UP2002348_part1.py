import random as rd

target = "1" * 10
chromosome_length = len(target)
max_population = 10
max_generation = 10
crossover_rate = 0.1
mutation_rate = 0.1

class Chromosome:
    def __init__(self, chromosome_length):
        self.gene = rd.choices(['0','1'], k=chromosome_length)
        self.fitness_score = 0

    def __repr__(self):
        return ''.join(self.gene) # Returns string of bits representing the chromosome object

    def calc_fitness(self, target):
        score = 0
        for this_gene,target_gene in zip(self.gene, target):
            if this_gene == target_gene: score += 1
        self.fitness_score = score / len(target) * 100

    def crossover(self, partner):
        # Should they fuck in here or in population?
        pass

    def mutate(self, mutation_rate):
        pass

class Population:
    def __init__(self, max_population, max_generation, crossover_rate, mutation_rate, chromosome_length, target):
        # Initialises the population and initial chromosomes
        self.population = [Chromosome(chromosome_length) for _ in range(max_population)] # Creates chromosomes
        self.max_population = max_population
        self.max_generation = max_generation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.target = target

        self.generations = 0
        self.best_chromosome = None

        self.cumulative_proportions = None

    def calc_fitness(self):
        # Calculates fitness of the entire population
        # Records the best 'fit' chromosome
        # Returns the most fit chromosome object  
        self.best_chromosome = self.population[0]
        for chromosome in self.population:
            chromosome.calc_fitness(target) # Asks the chromosome to calculate its fitness
            if self.best_chromosome.fitness_score < chromosome.fitness_score: self.best_chromosome = chromosome
            if self.best_chromosome == 100.0 : break # Break early when the desired target is found / might not need this! or maybe add a variable saying found target
        return self.best_chromosome
    

    def roulette_wheel(self):        
        # print(cumulative_proportions, cumulative_total)

        selectedValue = rd.random()
        
        for i in range(len(self.cumulative_proportions)):
            
            value = self.cumulative_proportions[i]
            if value >= selectedValue:
                return self.population[i]
            
        print("Something went wrong")
        # return       
    
    def cumulative_probability_generation(self):
        total_fitness = sum(chromosome.fitness_score for chromosome in self.population)
        normalised = [chromosome.fitness_score / total_fitness for chromosome in self.population] # Normalises fitness scores to be between 0 and 1

        self.cumulative_proportions = []
        cumulative_total = 0
        for proportion in normalised:
            cumulative_total+=proportion
            self.cumulative_proportions.append(cumulative_total)

    def selectParents(self):
        return [self.roulette_wheel(), self.roulette_wheel()]
    
    def reproduction(self):       
        self.cumulative_probability_generation()
        
        offspring = []

        for _ in range(self.max_population):
            # Parents needed to fuck
            parentA, parentB = self.selectParents()

            while (parentA == parentB):
                # Handles if both parents at random or selected the same
                # assexual is forbidden in this universe you must fuck someone else
                parentB = self.selectParents()    
                    

def main(target, chromosome_length, max_population, max_generation, crossover_rate, mutation_rate):
    # Instantiates population object
    population = Population(max_population, max_generation, crossover_rate, mutation_rate, chromosome_length, target)
    print(f'Best Chromosome: {population.calc_fitness()}, Fitness Score: {population.calc_fitness().fitness_score}')

    # Loop to reproduce here
    population.reproduction()

main(target, chromosome_length, max_population, max_generation, crossover_rate, mutation_rate)
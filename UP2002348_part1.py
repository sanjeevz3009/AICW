import random as rd

pi = list('10010111001010010011000000011010') # Given Chromosome Reference
target = [1] * len(pi) # Goal
chromosome_length = len(target) # size of a chromosome
max_population = 10 # Maximum size of the population
max_generation = 50 # Maximum number of reproductions
mutation_rate = 0.1 # Variation value of offspring

class Chromosome:
    """
        The Chromosome class represents a basic genetic structure of a chromosome - a member in the population.
    """
    def __init__(self, chromosome_length):
        self.gene = rd.choices([0,1], k=chromosome_length) # Variable that saves an encoding of the chromosome (initialy this is random)
        self.fitness_score = 0 # The chromsomes fitness value

    def __repr__(self):
        return ''.join([str(x) for x in self.gene]) # Returns a string of bits representing the chromosome object

    def calc_fitness(self, target):
        """
            This method calculates the chromosomes fitness by matching bits between the chromosome and target chromosome

            Variables:
                - 'score' is a temp variable to sum the fitness score
                - 'this_gene' is the current objects single gene encoding 
                - 'target_gene' is the single gene encoding of the target 
            
            zip(self.gene, target):
                - The zip function is used to combine corresponding elements from self.gene and target
        """
        score = 0
        for this_gene,target_gene in zip(self.gene, target):
            if this_gene == target_gene: score += 1 # Increments the score if the genes match
        self.fitness_score = score / len(target) * 100 # The fitness score is the ratio of matching genes to the total number of genes in the target genome

    def crossover(self, partner):
        """
            This crossover function applies a genetic crossover creating two offspring chromosomes with genes formed from two parents.
            Returns the two offspring chromosome objects.
        """
        cut_point=rd.randint(1, len(self.gene) -1) # A random index position to act as a cut point between the two parents

        # Combines the genes of one chromosome up to the cut point and with the genes of the partner chromosome in order to create a better fit chromosome
        geneA = self.gene[:cut_point] + partner.gene[cut_point:]
        geneB = partner.gene[:cut_point] + self.gene[cut_point:] # Same operation but in reverse

        # creates the offspring by instantiating a Chromosome class. The current genes of the offspring are randomly selected
        offspringA = Chromosome(len(geneA)) 
        offspringB = Chromosome(len(geneB))

        # The genes of the offspring are replaced with the combined genes of the parents
        offspringA.gene = geneA
        offspringB.gene = geneB
        
        return offspringA, offspringB
        

    def mutate(self, mutation_rate):
        """
            This method iterates through each gene of the chromosome and flips the gene value between 1 and 0 if the mutation should occur. Multiple flips could happen
        """
        for i in range(len(self.gene)):
            if rd.random() < mutation_rate: 
                self.gene = self.gene[:i] + [1-self.gene[i]] + self.gene[i + 1:] # 1-self.gene[i] - if gene is 1 this would flip it to 0 (1-1=0). if gene is 0 it will flip it to 1 (1-0=1)

class Population:
    """
        The Population class stores each chromosome in the population and provides an interface to interact with the population
    """
    def __init__(self, max_population, mutation_rate, chromosome_length, target):
        self.population = [Chromosome(chromosome_length) for _ in range(max_population)] # generates and stores a list of random Chromosome objects
        self.max_population = max_population
        self.mutation_rate = mutation_rate
        self.target = target

        self.generation = 0
        self.best_chromosome = None

    def calc_fitness(self):
        """
            Calculates fitness of the entire population
            Records the best 'fit' chromosome
            Returns the most 'fit' chromosome object  
        """
        self.best_chromosome = self.population[0]
        # iterates throught the entire population and requests each chromosome to calculate its fitness
        for chromosome in self.population:
            chromosome.calc_fitness(target) # Asks the chromosome to calculate its fitness
            if self.best_chromosome.fitness_score < chromosome.fitness_score: self.best_chromosome = chromosome
        return self.best_chromosome
    

    def spin_roulette_wheel(self, cumulative_fitness):        
        """
            this method randomly selects a chromosome from the population in a slightly biased manner.
            the method simulates spinning a roulette wheel to randomly select a parent based on its cumulative fitness
            returns a chromosome object
        """

        selectedValue = rd.random() # Generates a random value between 0 and 1
        # Iterates through the cumulative fitness values (population) and If the cumulative fitness value is greater than or equal to the random value,
        # it returns the corresponding individual from the population
        for i in range(len(cumulative_fitness)):
            
            value = cumulative_fitness[i]
            if value >= selectedValue:
                return self.population[i]
        
        raise Exception("Exception: ball dropped out of roulette wheel :(") # Raises an exception if the ball drops out of the roulette wheel unexpectedly  
    
    def selectParents(self, cumulative_fitness):
        """
            Selects two parents from the population using roulette wheel selection.
            Returns list containing two chromosome objects
        """
        parent_a = self.spin_roulette_wheel(cumulative_fitness)
        parent_b = self.spin_roulette_wheel(cumulative_fitness)

        # Handles if both parents are selected at random are the same
        while (parent_a == parent_b):
                parent_b= self.spin_roulette_wheel(cumulative_fitness) # asks for a new chromosome to be selected

        return [parent_a, parent_b]
    
    def create_roulette_wheel(self):
        """
            Creates a roulette wheel by calculating cumulative fitness values.
            This method is seperated from the spin_roulette_wheel to improve performance
            Returns the list of cumulative fitness values, representing a roulette wheel for selection.
        """
        total_fitness = sum(chromosome.fitness_score for chromosome in self.population) # Sums up all the chromosome fitness scores in the population
        normalised = [chromosome.fitness_score / total_fitness for chromosome in self.population] # Normalises fitness scores to be between 0 to 1
        cumulative_fitness = []
        cumulative_total = 0
        
        # Calculate cumulative fitness values
        for proportion in normalised:
            cumulative_total+=proportion
            cumulative_fitness.append(cumulative_total)
        return cumulative_fitness
    
    def reproduction(self):
        """
            Performs the reproduction phase of the genetic algorithm
        """
        cumulitave_fitness = self.create_roulette_wheel() # Creates a roulette wheel based on cumulative fitness

        # iterates through the population to generate 2 new individuals
        for _ in range(self.max_population):
            parentA, parentB = self.selectParents(cumulitave_fitness) # Selects two parents by spinning the roulette wheel selection
    
            offspringA, offspringB = parentA.crossover(parentB) # Performs crossover to create offspring

            # Applys mutation to the offspring
            offspringA.mutate(self.mutation_rate) 
            offspringB.mutate(self.mutation_rate)
            
            # Adds the offspring to the population
            self.population.append(offspringA) 
            self.population.append(offspringB)

        self.calc_fitness() # updates the new fitness of the entire population
        
        self.population = sorted(self.population, key=lambda x: x.fitness_score, reverse=True)[:max_population] # sorts the population from most fit to least fit chromosomes and cuts off the population when the max_population is exceeded
        self.generation += 1

def genetic_algorithm(target, chromosome_length, max_population, max_generation, mutation_rate):
    # Instantiates population object
    population = Population(max_population, mutation_rate, chromosome_length, target) # generates a population of chromosomes
    population.calc_fitness() # calculates the initial fitness of the population
    # Displays information about the initial population
    print('________________________', end='\n'*2)
    print("Initial Population:")   
    print('________________________', end='\n'*2) 
    print(*[f'chromosome: {chromosome} - fitness score: {chromosome.fitness_score}' for chromosome in population.population], sep='\n')
    print('________________________', end='\n'*2)
    print(f'Best Chromosome in Initial Population:')
    print('________________________', end='\n'*2)
    print(f'Best Chromosome: {population.best_chromosome}, Fitness Score: {population.best_chromosome.fitness_score}')

    # Main loop for reproduction and evolution
    while population.generation < max_generation:
        population.reproduction()

        print('________________________', end='\n'*2)
        print("Creating New Generation:", population.generation)
        print('________________________', end='\n'*2)

        print(*[f'chromosome: {chromosome} - fitness score: {chromosome.fitness_score}' for chromosome in population.population], sep='\n')
        print('________________________', end='\n'*2)
        print(f'Best Chromosome in generation {population.generation}:')
        print('________________________', end='\n'*2)
        print(f'Best Chromosome: {population.best_chromosome}, Fitness Score: {population.best_chromosome.fitness_score} for generation: {population.generation}')
        
        if population.best_chromosome.fitness_score == 100.0: break # Checks if the target is achieved

    print('________________________', end='\n'*2)
    print("Final Population:")
    print('________________________', end='\n'*2)
    print(*[f'chromosome: {chromosome} - fitness score: {chromosome.fitness_score}' for chromosome in population.population], sep='\n')
    print('________________________', end='\n'*2)
    print(f'Best Chromosome generated:')
    print('________________________', end='\n'*2)
    print(f'Best Chromosome: {population.best_chromosome}, Fitness Score: {population.best_chromosome.fitness_score} at generation: {population.generation}')

try:
    genetic_algorithm(target, chromosome_length, max_population, max_generation, mutation_rate)
except Exception as e:
    print(e)
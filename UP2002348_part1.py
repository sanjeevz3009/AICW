import random as rd

target = "1" * 32
chromosome_length = len(target)
max_population = 50
max_generation = 100
crossover_rate = 0.1
mutation_rate = 0.1

class Chromosome:
    def __init__(self, chromosome_length):
        self.gene = rd.choices(['0','1'], k=chromosome_length)
        self.fitness_score = 0
        self.id = rd.randint(0, 30)

    def __repr__(self):
        return ''.join(self.gene) # Returns string of bits representing the chromosome object

    def calc_fitness(self, target):
        score = 0
        for this_gene,target_gene in zip(self.gene, target):
            if this_gene == target_gene: score += 1
        self.fitness_score = score / len(target) * 100

    def crossover(self, partner):
        # Should they fuck in here or in population?
        cut_point=rd.randint(1, len(self.gene) -1)

        geneA = self.gene[:cut_point] + partner.gene[cut_point:]
        geneB = partner.gene[:cut_point] + self.gene[cut_point:]

        offspringA = Chromosome(len(geneA))
        offspringB = Chromosome(len(geneB))

        offspringA.gene = geneA
        offspringB.gene = geneB
        
        return offspringA, offspringB
        

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
            # if self.best_chromosome == 100.0 : break # Break early when the desired target is found / might not need this! or maybe add a variable saying found target
        return self.best_chromosome
    

    def spin_roulette_wheel(self):        
        # print(cumulative_proportions, cumulative_total)

        selectedValue = rd.random()
        
        for i in range(len(self.cumulative_proportions)):
            
            value = self.cumulative_proportions[i]
            if value >= selectedValue:
                return self.population[i]
            
        print("ball dropped out")
        # return       
    
    def selectParents(self):
        return [self.spin_roulette_wheel(), self.spin_roulette_wheel()]
    
    def create_roulette_wheel(self):
        total_fitness = sum(chromosome.fitness_score for chromosome in self.population)
        normalised = [chromosome.fitness_score / total_fitness for chromosome in self.population] # Normalises fitness scores to be between 0 and 1

        self.cumulative_proportions = []
        cumulative_total = 0
        for proportion in normalised:
            cumulative_total+=proportion
            self.cumulative_proportions.append(cumulative_total)
    
    def reproduction(self):       
        self.create_roulette_wheel()

        for _ in range(self.max_population):
            # Parents needed to fuck
            parentA, parentB = self.selectParents()

            while (parentA == parentB):
                # Handles if both parents at random or selected the same
                # assexual is forbidden in this universe you must fuck someone else
                parentB = self.spin_roulette_wheel() # need to change this
            
            offspringA, offspringB = parentA.crossover(parentB)
            self.population.append(offspringA)
            self.population.append(offspringB)

        self.calc_fitness() # calc fitness of new population
        
        self.population = sorted(self.population, key=lambda x: x.fitness_score, reverse=True)[:max_population] # new population get the best 10 chromosomes

def main(target, chromosome_length, max_population, max_generation, crossover_rate, mutation_rate):
    # Instantiates population object
    population = Population(max_population, max_generation, crossover_rate, mutation_rate, chromosome_length, target)
   
    # population.calc_fitness()
        
    # population.reproduction()


    # print(population.population[0].crossover(population.population[1])[0])

    # Loop to reproduce here
    generation = 0
    while generation != max_generation:
       
        population.calc_fitness() # Calculates initial populations fitness
        
        population.reproduction()

        print(f'Best Chromosome: {population.calc_fitness()}, Fitness Score: {population.calc_fitness().fitness_score}, generation: {generation}')

        if population.best_chromosome.fitness_score == 100.0: break

        generation+=1
    print(population.population)



main(target, chromosome_length, max_population, max_generation, crossover_rate, mutation_rate)
import random as rd
import numpy as np
import math

def i_pop(size, chromosome):
    pop=[]
    for inx in range(size):
        pop.append(rd.choices(range(2), k=chromosome))
        #      
    return pop

def posi(step,pos):
    up=[0,0]
    right=[0,1]
    down=[1,0]
    left=[1,1]
    if step==up: pos[1]+=1
    elif step==right: pos[0]+=1
    elif step==down: pos[1]-=1
    elif step==left: pos[0]-=1

    return pos

def fitness_f(pos,goal):
    return math.dist(goal, pos)

def find_fitness(bob,goal):
    i=0
    steps=[]
    temp2=[]
    pos=[0,0]
    #
    
    while i<len(bob):
        #
        temp2=[bob[i], bob[i+1]]
        pos=posi(temp2,pos)
        f=10-fitness_f(pos,goal)
        #
        #
        #
        #
        #
        i+=2
    return f

def print_fpop(f_pop):
    for indexp in f_pop:
        print(indexp)
    
def mating_crossover(parent_a,parent_b):
    offspring=[]
    cut_point=rd.randint(1, len(parent_a) -1)
    #
    offspring.append(parent_a[:cut_point] + parent_b[cut_point:])
    offspring.append(parent_b[:cut_point] + parent_a[cut_point:])
    return offspring

def mutate(chromo):  
    for idx in range(len(chromo)):
        if rd.random() < 0.3:  #this is quite high, usually it should be 0.1 
            chromo = chromo[:idx] + [1-chromo[idx]] + chromo[idx + 1:]
    return chromo

def Roulette_wheel(pop,fitness):
    parents=[]
    fitotal=sum(fitness)
    normalized=[x/fitotal for x in fitness]

    print('normalized fitness')
    print('________________________')
    print_fpop(normalized)
    print('________________________')
    f_cumulative=[]
    index=0
    for n_value in normalized:
        index+=n_value
        f_cumulative.append(index)

    pop_size=len(pop)
    print('cumulative fitness')
    print('________________________')
    print_fpop(f_cumulative)
    print('________________________')
    for index2 in range(pop_size):
        rand_n=rd.uniform(0,1)
        individual_n=0
        for fitvalue in f_cumulative:
            if(rand_n<=fitvalue):
                parents.append(pop[individual_n])
                break
            individual_n+=1
    return parents
    

psize=8
ch=10
fgoal=[3,3]

print('###########################') 
pop=i_pop(psize,ch)
print('population')
print('________________________')
print_fpop(pop)
print('________________________')
#
#
#
#

fitall=[find_fitness(indi,fgoal) for indi in pop]
#
#
#
#    
#
print('All fitnesses calculated')
print('________________________')
print_fpop(fitall)
print('________________________')
#
#
#
print('population & corresponding fitness')
print('________________________')
pop_fit=list(zip(pop,fitall))
print_fpop(pop_fit)
print('________________________')
#
parents_p=Roulette_wheel(pop,fitall)
print('these are the parents')
print_fpop(parents_p)
print('________________________')
print('These are two offspring of the first pair as an example')
print('This is after single point cross over with random point in the chromosome')
off=mating_crossover(parents_p[1],parents_p[2])
print('________________________')
print_fpop(off)
print('________________________')
print('apply example mutation on the two offspring')
print('________________________')
print(mutate(off[0]))
print(mutate(off[1]))
print('________________________')

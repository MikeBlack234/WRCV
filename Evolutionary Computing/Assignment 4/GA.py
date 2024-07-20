import numpy as np
import random
import csv



class Individual:
    def __init__(self, weights, fitness):
        self.weights = weights
        self.fitness = fitness
    

#Calculate sigmoid derivative
def Sigmoid(x):
    return 1/(1 + np.exp(-x))


def Beantype(boontjie):
    index = np.argmax(boontjie)
    if(index == 0):
        return "DERMASON"
    if(index == 1):
        return "SIRA"
    if(index == 2):
        return "SEKER"
    if(index == 3):
        return "BARBUNYA"
    if(index == 4):
        return "HOROZ"
    if(index == 5):
        return "CALI"
    if(index == 6):
        return "BOMBAY"

def forward_pass(inputs, weights_biases):
    input_weights = weights_biases[:64].reshape(16, 4)
    input_biases = weights_biases[64:68]
    hidden_weights = weights_biases[68:96].reshape(4, 7)
    hidden_biases = weights_biases[96:]

    hidden_layer = Sigmoid(np.dot(inputs, input_weights) + input_biases)
    outputs = Sigmoid(np.dot(hidden_layer, hidden_weights) + hidden_biases)

    return outputs

def fitness(output, real):
    error = output - real
    return np.sum(error**2)


def two_point_crossover(parent1, parent2):
    # Determine the length of the chromosomes (number of elements)
    chromosome_length = len(parent1.weights)
    
    # Select two crossover points
    crossover_point1 = random.randint(1, chromosome_length - 2)
    crossover_point2 = random.randint(crossover_point1 + 1, chromosome_length - 1)

    # Ensure point2 is greater than point1
    if crossover_point2 < crossover_point1:
        crossover_point1, crossover_point2 = crossover_point2, crossover_point1

    #Parent weights
    p1 = parent1.weights
    p2 = parent2.weights

    # Create two empty weights for offspring
    weights1 = p1[:crossover_point1] + p2[crossover_point1:crossover_point2] + p1[crossover_point2:]
    weights2 = p2[:crossover_point1] + p1[crossover_point1:crossover_point2] + p2[crossover_point2:]

    child1 = Individual(weights=weights1, fitness=0)
    child2 = Individual(weights=weights2, fitness=0)

    return child1, child2
""" def one_point_crossover(parent1, parent2):
    # Determine the length of the chromosomes (number of elements)
    chromosome_length = 103

    # Choose a random crossover point (excluding the endpoints)
    crossover_point = random.randint(1, chromosome_length - 1)
    
    #Parent weights
    p1 = parent1.weights
    p2 = parent2.weights

    # Create two empty weights
    weights1 = []
    weights2 = []

    weights1[:crossover_point] = p1[:crossover_point]
    weights1[crossover_point:] = p2[crossover_point:]

    weights2[:crossover_point] = p2[:crossover_point]
    weights2[crossover_point:] = p1[crossover_point:]
    

    child1 = Individual(weights=weights1, fitness = 0)
    child2 = Individual(weights=weights2, fitness = 0)

    return child1, child2
 """
def one_point_crossover(parent1, parent2):
    # Determine the length of the chromosomes (number of elements)
    chromosome_length = len(parent1.weights)
    
    #Parent weights
    p1 = parent1.weights
    p2 = parent2.weights

    # Create two empty weights for offspring
    weights1 = []
    weights2 = []

    # Perform uniform crossover
    for i in range(chromosome_length):
        if random.random() < 0.5:  # 50% probability
            weights1.append(p1[i])
            weights2.append(p2[i])
        else:
            weights1.append(p2[i])
            weights2.append(p1[i])

    child1 = Individual(weights=weights1, fitness=0)
    child2 = Individual(weights=weights2, fitness=0)

    return child1, child2



def mutate(individual, mutation_rate, mutation_amount):
    if random.random() < mutation_rate:
        index = random.randint(0, len(individual.weights) - 1)
        individual.weights[index] += mutation_amount * (2 * random.random() - 1)

# Read CSV data into numpy array
data = np.loadtxt('SimpleBeanData.csv', delimiter=',', dtype=str, skiprows=1)

demo = np.loadtxt('Demo2.csv', delimiter=',', dtype=str, skiprows=1)
demo_inputs = data[np.arange(0,50)][:, np.arange(0,16)]
demo_inputs = demo_inputs.astype(float)
mean = np.mean(demo_inputs, axis=0)
std = np.std(demo_inputs, axis=0)
# Normalize the data
demo_inputs = (demo_inputs - mean) / std

buffer = []

#One Hot encoding
DERMASON = [1,0,0,0,0,0,0]
SIRA = [0,1,0,0,0,0,0]
SEKER = [0,0,1,0,0,0,0]
BARBUNYA = [0,0,0,1,0,0,0]
HOROZ = [0,0,0,0,1,0,0]
CALI = [0,0,0,0,0,1,0]
BOMBAY = [0,0,0,0,0,0,1]


for i in range(0,11465):
    if(data[i, 16] == 'DERMASON'):
        buffer.append(DERMASON)
    if(data[i, 16] == 'SIRA'):
        buffer.append(SIRA)
    if(data[i,16] == 'SEKER'):
        buffer.append(SEKER)
    if(data[i,16] == 'BARBUNYA'):
        buffer.append(BARBUNYA)
    if(data[i,16] == 'HOROZ'):
        buffer.append(HOROZ)
    if(data[i,16] == 'CALI'):
        buffer.append(CALI)
    if(data[i,16] == 'BOMBAY'):
        buffer.append(BOMBAY)

buffer = np.array(buffer)

all_inputs = data[np.arange(0,11465)][:, np.arange(0,16)]
#Load data into numpy array
inputs = data[np.arange(0,9000)][:, np.arange(0,16)]
outputs = buffer[np.arange(0,9000)][:,np.arange(0,7)]

test_inputs = data[np.arange(9000,11465)][:, np.arange(0,16)]
test_outputs = buffer[np.arange(9000, 11465)][:, np.arange(0,7)]

all_inputs = all_inputs.astype(float)
inputs = inputs.astype(float)
outputs = outputs.astype(float)

test_inputs = test_inputs.astype(float)

mean = np.mean(all_inputs, axis=0)
std = np.std(all_inputs, axis=0)
# Normalize the data
inputs = (inputs - mean) / std
test_inputs = (test_inputs - mean) / std



individuals = 300
dimension = 103
iterations = 1000


population = []
#Create population
for i in range(individuals):
    obj = Individual(weights = np.random.uniform(-10,30, dimension), fitness = 0) #initialize fitness to 0
    population.append(obj)

#Sort based on fitness
#

for _ in range(iterations):
    #Calculate FITNESS
    for i in range(individuals):
        out = forward_pass(inputs, population[i].weights)
        population[i].fitness = fitness(out, outputs)

    #Sort population
    population.sort(key=lambda individual: individual.fitness)
    
    print(int(population[0].fitness))

    new_gen = []
    
        #Elitism at 10%
    s = int((10*individuals)/100)
    new_gen.extend(population[:s]) #Add 10% of fittest individuals to next generation
        
        #Use the 50% of fittest to reproduce
    s = int((90*individuals)/100)

    for _ in range(135):
        index1 = random.randint(0,150)
        index2 = random.randint(0,150)
        parent1 = population[index1]
        parent2 = population[index2]
            
        child1, child2 = one_point_crossover(parent1, parent2)

        new_gen.extend([child1, child2])
    
    m_rate = 0.15
    m_amount = 5

    for individual in new_gen:
        mutate(individual, m_rate, m_amount)
        individual.weights = np.array(individual.weights)

    population = new_gen
    
        
    
population.sort(key=lambda individual: individual.fitness)

#Choose Individual with Highest fitness
best = population[0]


out = forward_pass(inputs, best.weights)
count = 0

for t in range(0, 9000):
    bean = Beantype(out[t])
    if(bean == data[t, 16]):
        count += 1


accuracy = (count/9000)*100
print(count, " beans out of 9000 classified correctly.")
print(accuracy, " percent accurate.") 

out = forward_pass(test_inputs, best.weights)


count = 0

for t in range(9000, 11465):
    bean = Beantype(out[t-9000])
    if(bean == data[t, 16]):
        count += 1


accuracy = (count/2465)*100
print(count, " beans out of 2465 classified correctly.")
print(accuracy, " percent accurate.") 

out = forward_pass(demo_inputs, best.weights)
    

for t in range(50):
    bean = Beantype(out[t])
    print(t, " : ", bean)






        



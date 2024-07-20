import numpy as np
import csv


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

def differential_evolution(func, bounds, popsize, mutate, recombination, max_gen, target_error):
    dimensions = len(bounds)
    
    # Initialize the population
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff

    # Calculate initial fitness
    fitness = np.asarray([func(ind, None) for ind in population])

    for _ in range(max_gen):
        print(_)
        for i in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mutate * (b - c), min_b, max_b)
            cross_points = np.random.rand(dimensions) < recombination
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = func(trial, None)
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial

        if min(fitness) < target_error:
            break

    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]


def objective(weights, _):
    out = forward_pass(inputs, weights)
    return fitness(out, outputs)

# Read CSV data into numpy array
data = np.loadtxt('SimpleBeanData.csv', delimiter=',', dtype=str, skiprows=1)


buffer = []

#One Hot encoding
DERMASON = [1,0,0,0,0,0,0]
SIRA = [0,1,0,0,0,0,0]
SEKER = [0,0,1,0,0,0,0]
BARBUNYA = [0,0,0,1,0,0,0]
HOROZ = [0,0,0,0,1,0,0]
CALI = [0,0,0,0,0,1,0]
BOMBAY = [0,0,0,0,0,0,1]


for i in range(0,11400):
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

all_inputs = data[np.arange(0,11400)][:, np.arange(0,16)]
#Load data into numpy array
inputs = data[np.arange(0,7000)][:, np.arange(0,16)]
outputs = buffer[np.arange(0,7000)][:,np.arange(0,7)]

test_inputs = data[np.arange(7000,11400)][:, np.arange(0,16)]
test_outputs = buffer[np.arange(7000, 11400)][:, np.arange(0,7)]

all_inputs = all_inputs.astype(float)
inputs = inputs.astype(float)
outputs = outputs.astype(float)

test_inputs = test_inputs.astype(float)

mean = np.mean(all_inputs, axis=0)
std = np.std(all_inputs, axis=0)
# Normalize the data
inputs = (inputs - mean) / std
test_inputs = (test_inputs - mean) / std


bounds = [(-40, 40) for _ in range(103)]
max_gen = 2000
popsize = 300
mutate = 2
recombination = 0.7
target_error = 1e-6

best_weights, best_error = differential_evolution(objective, bounds, popsize, mutate, recombination, max_gen, target_error)

out = forward_pass(test_inputs, best_weights)

count = 0

for t in range(7000, 11400):
    bean = Beantype(out[t - 7000])
    if bean == data[t, 16]:
        count += 1

accuracy = (count / 4400) * 100
print(count, " beans out of 4400 classified correctly.")
print(accuracy, " percent accurate.")



import numpy as np
import csv


#Calculate sigmoid derivative
def Sigmoid(x):
    return 1/(1 + np.exp(-x))
    


def Beantype(bean):
    index = np.argmax(bean)
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

print(data[0, 16])
# Parameters
n_particles = 300
dimension = 103
n_iterations = 100
alpha = 0.5
beta = 1.5
gamma = 1.9


particles_position = np.random.uniform(0, 1, (n_particles, dimension))
particles_velocity = np.random.uniform(0, 1, (n_particles, dimension))
particles_best_position = np.copy(particles_position)
particles_best_score = np.full((n_particles,), float('inf'))
g_best_position = np.random.uniform(0, 10, dimension)
g_best_score = float(100000)



for iteration in range(n_iterations):
    print(int(g_best_score))
    for i in range(n_particles):
        o = forward_pass(inputs, particles_position[i])  # Objective: Minimize sum of squares
        err = abs(o - outputs)
        current_score = np.sum(err**2)
        
        # Update personal best
        if current_score < particles_best_score[i]:
            particles_best_score[i] = current_score
            particles_best_position[i] = particles_position[i]
            
        # Update global best
        if current_score < g_best_score:
            g_best_score = current_score
            g_best_position = particles_position[i]
    
    # Update velocities and positions
    for i in range(n_particles):
        inertia = alpha * particles_velocity[i]
        personal_attraction = beta * np.random.random() * (particles_best_position[i] - particles_position[i])
        global_attraction = gamma * np.random.random() * (g_best_position - particles_position[i])
        
        particles_velocity[i] = inertia + personal_attraction + global_attraction
        particles_position[i] += particles_velocity[i]





out = forward_pass(test_inputs, g_best_position)




count = 0

for t in range(7000, 11400):
    bean = Beantype(out[t-7000])
    if(bean == data[t, 16]):
        count += 1


accuracy = (count/4400)*100
print(count, " beans out of 4400 classified correctly.")
print(accuracy, " percent accurate.") 




    
    

    
    








        



import numpy as np
import csv


def Sigmoid(x):
    return 1/(1 + np.exp(-x))

def SigmoidDerivative(x):
    return x*(1-x)


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

unknown_inputs = np.loadtxt('Demo.csv', delimiter=',', dtype=str)
unknown_inputs = unknown_inputs.astype(float)
# Read CSV data into numpy array
data = np.loadtxt('BeanData.csv', delimiter=',', dtype=str )

#Load data into numpy array
inputs = data[np.arange(0,10000)][:, np.arange(0,16)]
test_inputs = data[np.arange(10000, 12000)][:, np.arange(0,16)]

buffer = []

#One Hot encoding
DERMASON = [1,0,0,0,0,0,0]
SIRA = [0,1,0,0,0,0,0] 
SEKER = [0,0,1,0,0,0,0]
BARBUNYA = [0,0,0,1,0,0,0]
HOROZ = [0,0,0,0,1,0,0]
CALI = [0,0,0,0,0,1,0]
BOMBAY = [0,0,0,0,0,0,1]


for i in range(0,12311):
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

outputs = buffer[np.arange(0,10000)][:,np.arange(0,7)]
test_outputs = buffer[np.arange(10000, 12000)][:, np.arange(0,7)]

inputs = inputs.astype(float)
outputs = outputs.astype(float)
test_inputs = test_inputs.astype(float)
test_outputs = test_outputs.astype(float)


print(inputs.shape)
print(outputs.shape)
I = 16 #Inputs
H = 20 #Hidden neurons
O = 7 #Output neurons

#Noise injection
noise = np.random.rand(1000, 16)
print(noise)
#Standardize inputs
mean = np.mean(inputs, axis=0)
std = np.std(inputs, axis=0)
inputs = (inputs - mean) / std

mean = np.mean(test_inputs, axis=0)
std = np.std(test_inputs, axis=0)
test_inputs = (test_inputs - mean) / std 

mean = np.mean(unknown_inputs, axis=0)
std = np.std(unknown_inputs, axis=0)
unknown_inputs = (unknown_inputs - mean) / std

m = 0.0 #mean of weights
std_dev = 0.1
#Initialize weights
h_weight = np.random.normal(m, std_dev, size=(H, I))
h_bias =np.random.rand(H)

o_weight = np.random.normal(m, std_dev, size= (O, H))
o_bias = np.random.rand(O)

N = 1000 #iterations
learning_rate = 0.0001 #learning rate
P = 10000 # no of patterns

""" # Initialize weights and biases random distribution
h_weight = np.random.rand(H, I)
h_bias = np.random.rand(H)
o_weight = np.random.rand(O, H)
o_bias = np.random.rand(O)   """


# Loop through iterations
SSE = []    
for n in range(N): #Using batch learning
    print(n)
    # Calculate feedforward outputs for all patterns
    y = Sigmoid(np.dot(h_weight, inputs.T) + h_bias[:, np.newaxis])
    o = Sigmoid(np.dot(o_weight, y) + o_bias[:, np.newaxis])
    
    # Calculate errors for all patterns
    output_error = o - outputs.T
    hidden_error = np.dot(o_weight.T, output_error)
    
    # Calculate MSE for all patterns and accumulate
    sse = np.sum(output_error**2)
    SSE.append(sse)
    mse = np.mean(output_error**2, axis=0)
    
    # Update H-O weights
    gradient_output = learning_rate * output_error * o * (1 - o)
    adjust = np.dot(gradient_output, y.T)
    o_weight -= adjust
    o_bias -= np.sum(gradient_output, axis=1)
    
    # Update I-H weights
    gradient_input = learning_rate * hidden_error * y * (1 - y)
    adjust2 = np.dot(gradient_input, inputs)
    h_weight -= adjust2
    h_bias -= np.sum(gradient_input, axis=1)

   
se = np.array(SSE)
se = se.astype(int)


with open("Values.csv", mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows([[value] for value in se])  # Each value is a row of its own


count = 0
for j in range(2001):
    y = Sigmoid(np.dot(h_weight, inputs[j]) + h_bias)
    o = Sigmoid(np.dot(o_weight, y) + o_bias)
    test = Beantype(o)
    if(test == data[j, 16]):
        count = count + 1

percentage_accurate = (count/2000)*100   
print(count," out of 2000 beans correctly classified on training set: ")
print(percentage_accurate, " percent accurate")

count = 0
for j in range(2000):
    y = Sigmoid(np.dot(h_weight, test_inputs[j]) + h_bias)
    o = Sigmoid(np.dot(o_weight, y) + o_bias)
    test = Beantype(o)
    if(test == data[j+10000, 16]):
        count = count + 1

percentage_accurate = (count/2000)*100   
print(count," out of 2000 beans correctly classified on test set: ")
print(percentage_accurate, " percent accurate")

""" #Unknown data
for j in range(50):
    y = Sigmoid(np.dot(h_weight, unknown_inputs[j]) + h_bias)
    o = Sigmoid(np.dot(o_weight, y) + o_bias)
    print(j+1, " ", Beantype(o))
     """
print(h_weight)
print(h_bias)
print(o_weight)
print(o_bias)
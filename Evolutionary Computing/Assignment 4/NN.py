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


# Read CSV data into numpy array
data = np.loadtxt('BeanData.csv', delimiter=',', dtype=str, skiprows=1 )

#Load data into numpy array
inputs = data[np.arange(3000,12000)][:, np.arange(0,16)]
test_inputs = data[np.arange(0, 3000)][:, np.arange(0,16)]

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

outputs = buffer[np.arange(3000,12000)][:,np.arange(0,7)]
test_outputs = buffer[np.arange(0, 3000)][:, np.arange(0,7)]

inputs = inputs.astype(float)
outputs = outputs.astype(float)
test_inputs = test_inputs.astype(float)
test_outputs = test_outputs.astype(float)

 

I = 16 #Inputs
H = 14 #Hidden neurons
O = 7 #Output neurons

#Standardize inputs
mean = np.mean(inputs, axis=0)
std = np.std(inputs, axis=0)
inputs = (inputs - mean) / std

mean = np.mean(test_inputs, axis=0)
std = np.std(test_inputs, axis=0)
test_inputs = (test_inputs - mean) / std  

""" 
noise_std = 0.2 # Adjust the noise level according to your preference

# Add noise to inputs
inputs = inputs + np.random.normal(0, noise_std, inputs.shape)

# Add noise to test inputs
test_inputs = test_inputs + np.random.normal(0, noise_std, test_inputs.shape) 
 """

m = 0.0 #mean of weights
std_dev = 0.1

 #Initialize weights
h_weight = np.random.normal(m, std_dev, size=(H, I))
h_bias =np.random.rand(H)

o_weight = np.random.normal(m, std_dev, size= (O, H))
o_bias = np.random.rand(O) 

h_weight = np.random.rand(H, I)
h_bias = np.random.rand(H)
o_weight = np.random.rand(O, H)
o_bias = np.random.rand(O) 

N = 300 #iterations
learning_rate = 0.001 #learning rate
P = 9000 # no of patterns


SSE = []
for n in range(N):
    print(n)
    for p in range(P):
        #Calculate feedforward outputs
        y = Sigmoid(np.dot(h_weight, inputs[p]) + h_bias)
        o = Sigmoid(np.dot(o_weight, y) + o_bias)

       
        #Calculate errors
        #Output layer error
        output_error = o - outputs[p]
        hidden_error = np.dot(o_weight.T, output_error)
        sse = np.sum(output_error**2)
        
        #Adjust all the weights
        #Update H-O weights
        gradient_output = learning_rate*output_error*(o+(1-o))
        gradient1 = gradient_output.reshape((O,1)) #This adjustment is made so that numpy can perform the dot product
        yT = y.reshape((1,H)) #This adjustment is made so that numpy can perform the dot product
        adjust = np.dot(gradient1, yT) #Add these values to the weights to update them
        o_weight += -adjust
        o_bias += -gradient_output
        #Update I-H weights
        gradient_input = learning_rate*hidden_error*(y+(1-y))
        gradient2 = gradient_input.reshape((H,1))
        i = inputs[p]
        i = i.reshape((1, I))
        adjust2 = np.dot(gradient2, i)
        h_weight += -adjust2
        h_bias += -gradient_input

    
se = np.array(SSE)
se = se.astype(int)

with open("Values.csv", mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows([[value] for value in se])  # Each value is a row of its own """

count = 0
for j in range(9000):
    y = Sigmoid(np.dot(h_weight, inputs[j]) + h_bias)
    o = Sigmoid(np.dot(o_weight, y) + o_bias)
    test = Beantype(o)
    if(test == data[j+3000, 16]):
        count = count + 1

percentage_accurate = (count/9000)*100   
print(count," out of 9000 beans correctly classified: ")
print(percentage_accurate, " percent accurate for training data: ", N)

count = 0
for j in range(3000):
    y = Sigmoid(np.dot(h_weight, test_inputs[j]) + h_bias)
    o = Sigmoid(np.dot(o_weight, y) + o_bias)
    
    test = Beantype(o)
    if(test == data[j, 16]):
        count = count + 1 

percentage_accurate = (count/3000)*100   
print(count," out of 3000 beans correctly classified: ")
print(percentage_accurate, " percent accurate for validation data: ", N)

print("h_weights: ")
print(h_weight)
print("h_bais: ")
print(h_bias)
print("o_weight: ")
print(o_weight)
print("o_bias: ")
print(o_bias)


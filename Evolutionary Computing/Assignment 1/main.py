import numpy as np #Used for creating and manipulating arrays easily
import csv #Used for reading and writing to csv files

cities = [] #Initialize array, array will be converted to numpy array after it is populated
data = np.loadtxt('Properties.csv', delimiter=',', dtype=str) #Read data from Properties file and load into numpy array for type string, headers are manually removed

for row in range(500):
    for col in range(8): # manual one hot encoding of values in properties
        if(data[row,col] == 'Durban'):
            cities.append([1,0,0,0])
        if(data[row,col] == 'Cape Town'):
            cities.append([0,1,0,0])
        if(data[row,col] == 'Johannesburg'):
            cities.append([0,0,1,0])
        if(data[row,col] == 'Gqeberha'):
            cities.append([0,0,0,1])
        if(data[row, col] == 'No'):
            data[row, col] = 0
        elif(data[row,col] == 'Yes'):
            data[row, col] = 1

cities2 = np.array(cities) #Convert cities to numpy array
array1 = data[np.arange(0,500)][:, np.arange(1,10)] #Create new array without cities column
result_data = np.hstack((cities2, array1)) #add one hot encoded cities to result data
result_data = result_data.astype(float) #Convert array data type to float

#Training Data
training_inputs = result_data[np.arange(0,350)][:, np.arange(0,12)]
training_outputs = result_data[np.arange(0,350)][:, 12]

#Testing Data
test_inputs = result_data[np.arange(350, 500)][:, np.arange(0,12)]
test_outputs =result_data[np.arange(350, 500)][:, 12] 

#Demo Data, could've probably done both in a single for loop
cities = []#Reset cities
data = np.loadtxt('Demo.csv', delimiter=',', dtype=str) #Load data from Demo.csv file
for row in range(50):
    for col in range(8): #Manual one hot encoding
        if(data[row,col] == 'Durban'):
            cities.append([1,0,0,0])
        if(data[row,col] == 'Cape Town'):
            cities.append([0,1,0,0])
        if(data[row,col] == 'Johannesburg'):
            cities.append([0,0,1,0])
        if(data[row,col] == 'Gqeberha'):
            cities.append([0,0,0,1])
        if(data[row, col] == 'No'):
            data[row, col] = 0
        elif(data[row,col] == 'Yes'):
            data[row, col] = 1

#There are much better ways to handle and sort the data
cities2 = np.array(cities) #Convert cities to numpy array
array1 = data[np.arange(0,50)][:, np.arange(1,9)] #Create new array without cities column
result_data = np.hstack((cities2,array1)) #add one hot encoded cities to result data
result_data = result_data.astype(float) #Convert data to float

#Demo inputs
demo_inputs = result_data[np.arange(0,50)][:, np.arange(0,12)]

weights = (np.random.rand(12) - 0.5)/2 #Initialize weights to values between -0.25 and 0.25
bias_weight = np.random.rand(1)/4 #Initialize bias weight to value between 0 and 0.25
print("Weights before training: ") #Always good to compare weights before and after training
print(weights)

learning_rate = 0.000001 #Optimal value in my instance, other students had similar results with a much higher learning rate
#Training loop
Squared_error = [] #Create array for squared error, this data is written to excel and used for data visualization
error = 0
sse = 0
for j in range(30000): # Iterates 30000 times
    print(j) #Shows what the current iteration is
    for i in range(350):
        output = np.dot(weights, training_inputs[i]) + bias_weight*(1) #Calculate the output
        error = training_outputs[i] - output # Calculate the error
        bias_adjust = -learning_rate*2*error*(1) 
        bias_weight = bias_weight - bias_adjust #Update the bias weight
        for k in range(12):
            adjust = -learning_rate*2*error*(training_inputs[i,k]) 
            weights[k] = weights[k] - adjust #Update input weights
        sse += error*error #Calculate SSE
    Squared_error.append(sse) #Add SSE to array
    sse = 0 #Reset the SSE 

SQ = np.array(Squared_error) #Convert to a numpy array
SQ = SQ.astype(int) #Convert datatype to integer

print("Weights after training: ", weights) 
print("The bias weight is: ", bias_weight)

with open('Outputs.csv', mode='w', newline='') as file: # Write data to the CSV file
    writer = csv.writer(file)
    writer.writerows(SQ)

Squared_error = [] #Reset array
training_errors = [] #Reset array

for l in range(349):
    output = np.dot(weights, training_inputs[l]) + bias_weight
    error = abs(training_outputs[l] - output)
    training_errors.append(error)
    Squared_error.append(error*error)
#SSE on training
print("Training set SSE: ", sum(Squared_error))
#Average error on training values
avg = sum(training_errors)/len(training_errors)*1000
print("The average price difference for the training set is R", avg)
errors = []
Squared_error = []
for l in range(144): 
    output = np.dot(weights, test_inputs[l]) + bias_weight
    error = abs(test_outputs[l] - output)
    errors.append(error)
    Squared_error.append(error*error)
#SSE on test set
print("Test set SSE: ", sum(Squared_error))

#Average calculation on test set
average = (sum(errors)/len(errors)*1000)
print("The average price difference for the test set is R", average)

for i in range(50): # Calculate demo outputs
    output = np.dot(weights, demo_inputs[i]) + bias_weight
    print(output)
import pandas as pd
import numpy as np
from math import exp
import random
import timeit



start = timeit.timeit()
# Fix random seed for reproducibility
np.random.seed(7)

# Load dataset
df = pd.read_csv('ecoli\\ecoli.data', delim_whitespace=True, header=None)
# Remove redundant rows
dataset = df[df[8].isin(['cp', 'im'])]

# Convert to numpy array and shuffle
dataset = dataset.values
dataset[:, 8] = np.where(dataset[:, 8] == "im", 1, np.where(dataset[:, 8] == "cp", 0, dataset[:, 8]))
np.random.shuffle(dataset)

# Split dataset
split_ratio = 0.9
n_samples = len(dataset)
split_index = int(n_samples * split_ratio)

X_train = dataset[:split_index, 1:8]
X_test = dataset[split_index:, 1:8]
Y_train = dataset[:split_index, 8]
Y_test = dataset[split_index:, 8]

# Define activation functions
def relu(x):
    return np.maximum(0, x)
def sigmoid(z):
   return 1.0/(1+exp(-z))
def step(x):
    return 1.0 if x>= 0.0 else 0.0 

#Define weights
input_num=7
hidden_num=2
num_weights = input_num*hidden_num
weights = [random.uniform(0, 1) for j in range(num_weights)]
accuracy=0
#bias=np.random.uniform(low=-1, high=1.0, size=2)

def firstLayer(X,weights):
      activation_1 = weights[0]
      activation_1 += weights[1]*X[0]
      activation_1 += weights[2]*X[1]
      activation_1 += weights[3]*X[2]
      activation_1 += weights[4]*X[3]
      

      activation_2 = weights[5]
      activation_2 += weights[6]*X[4]
      activation_2 += weights[7]*X[5]
      activation_2 += weights[8]*X[6]

      #return relu(activation_1),relu(activation_2)
      return sigmoid(activation_1),sigmoid(activation_2)


            #for idx, x_i in enumerate(X):
                # Forward propagation
                #hidden_layer1_input = np.dot(x_i, weights) + bias
                #hidden_layer1_output = sigmoid(hidden_layer1_input)
                #return hidden_layer1_output
            
def secondLayer(row,weights):
    activation_3 = weights[9]
    activation_3 += weights[10]*row[0]
    activation_3 += weights[11]*row[1]
    #return sigmoid(activation_3)
    return 1 if activation_3>= 0 else 0.0  

def predict(row,weights):
    input_layer = row
    first_layer = firstLayer(input_layer,weights)
    second_layer = secondLayer(first_layer,weights)
    return second_layer,first_layer

n_samples=len(X_train)
output=np.zeros(n_samples)
for d in range(len(X_train)):
    output[d]=predict(X_train[d],weights)[0]
    print(predict(X_train[d],weights)[0]) 
  #Prints y_hat and y
print('**************')
for k in range(len(Y_train)):
    print(Y_train[k])

#Acuuracy without training
for j in range(len(Y_train)):
     if(output[j]==Y_train[j]):
            accuracy += 1
accuracy = accuracy/len(X_train)
print('Accurary before training',accuracy)


def train_weights(x_train,y_train,learningrate,epochs):
    for epoch in range(epochs):
        sum_error = 0.0
        for l in range(len(x_train)):
            prediction,first_layer = predict(x_train[l],weights)
            error = y_train[l]-prediction
            sum_error += error
            #First layer
            weights[0] = weights[0] + learningrate*error
            weights[5] = weights[5] + learningrate*error

            weights[1] = weights[1] + learningrate*error*x_train[l][0]
            weights[2] = weights[2] + learningrate*error*x_train[l][1]
            weights[3] = weights[3] + learningrate*error*x_train[l][2]
            weights[4] = weights[3] + learningrate*error*x_train[l][3]

            weights[6] = weights[6] + learningrate*error*x_train[l][4]
            weights[7] = weights[7] + learningrate*error*x_train[l][5]
            weights[8] = weights[8] + learningrate*error*x_train[l][6]

            #Second layer
            weights[9] = weights[9] + learningrate*error
            weights[10] = weights[10] + learningrate*error*first_layer[0]
            weights[11] = weights[11] + learningrate*error*first_layer[1]
        if((epoch%100==0) or (last_error != sum_error)):
            print("Epoch "+str(epoch) + " Learning rate " + str(learningrate) + " Error " + str(sum_error))
        last_error = sum_error
    return prediction,weights

learningrate = 0.001
epochs = 1000
train_weights = train_weights(X_train,Y_train,learningrate,epochs)
print(train_weights)

#Accuracy of Y_train after training
#new_weights=train_weights[1]
#n_samples=len(X_train)
#output_new=np.zeros(n_samples)
#for d in range(len(X_train)):
    #output_new[d]=predict(X_train[d],new_weights)[0]
    #print(predict(X_train[d],new_weights)[0])
#for r in range(len(Y_train)):
     #if(output_new[r]==Y_train[r]):
            #accuracy += 1
#accuracy_new = accuracy/len(X_train)
#print('Accurary after training',accuracy_new)

#Evalution on X_test:
new_weights=train_weights[1]
n_samples=len(X_test)
output_test=np.zeros(n_samples)
for d in range(len(X_test)):
    output_test[d]=predict(X_test[d],new_weights)[0]
    print(predict(X_test[d],new_weights)[0])

output_test=np.zeros(n_samples)
accuracy_test=0
for r in range(len(Y_test)):
     if(output_test[r]==Y_test[r]):
            accuracy_test += 1
accuracy_test = accuracy_test/len(Y_test)
print('Accurary on test data after training',accuracy_test)
end = timeit.timeit()
print(end - start)
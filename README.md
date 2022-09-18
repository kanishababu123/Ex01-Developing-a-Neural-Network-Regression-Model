# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains nine neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model.
Data is the key for the working of neural network and we need to process it before feeding to the neural network. In the first step, we will visualize data which will help us to gain insight into the data.We need a neural network model. This means we need to specify the number of hidden layers in the neural network and their size, the input and output size.Now we need to define the loss function according to our task. We also need to specify the optimizer to use with learning rate.Fitting is the training step of the neural network. Here we need to define the number of epochs for which we need to train the neural network.After fitting model, we can test it on test data to check whether the case of overfitting. 

## Neural Network Model

![image](https://user-images.githubusercontent.com/75235813/187114675-e9679f90-de93-4b36-a2cb-815e97911343.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```
#Developed by B.Kavya
# Register number: 212220230007

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data=pd.read_csv("Data_dl.csv")
data.head()
x=data[['input']].values
x
y=data[['output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)

scaler=MinMaxScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train1=scaler.transform(x_train)
x_test1=scaler.transform(x_test)

AI=Sequential([
    Dense(9,activation='relu'),
    Dense(7,activation='relu'),
    Dense(1)
])
AI.compile(optimizer='rmsprop',loss='mse')
AI.fit(xtrain1,ytrain,epochs=2000)
loss_df=pd.DataFrame(AI.history.history)
loss_df.plot()
AI.evaluate(x_test1,y_test)

x_n1=[[29]]
x_n1_1=Scaler.transform(x_n1)
AI.predict(x_n1_1)

```
## Dataset Information

![image](https://user-images.githubusercontent.com/75235813/187087298-16ab0800-937c-4d20-9578-22151a980242.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235813/187087335-510a7106-fc7b-40a5-814e-97a5fd44577f.png)


### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/75235813/187087356-cca7270f-cbd3-436f-a5fd-d3d9266b3de0.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235813/187087388-6d1e7142-b432-486c-a4a9-665dbeeb2bf4.png)


## RESULT

Thus,the neural network regression model for the given dataset is developed.

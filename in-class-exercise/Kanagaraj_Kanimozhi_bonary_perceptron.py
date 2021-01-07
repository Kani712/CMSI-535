#Multi-class perceptron (three class dataset)

import numpy as np
import sklearn.datasets as skdata

#loading dataset

wine_data = skdata.load_wine()
x = wine_data.data
y = wine_data.target

#splitting data into 80% train, 10% validation,10% testing

idx = np.random.permutation(x.shape[0])
train_split = int(0.80 * x.shape[0])
val_split = int(0.90 * x.shape[0])

train_idx = idx[:train_split]
val_idx = idx[train_split : val_split]
test_idx = idx[val_split:]

#select the samples to construct training , validationa and testing sets
x_train = x[train_idx, :]
y_train = y[train_idx]
x_val = x[val_idx, :]
y_val = y[val_idx]
x_test = x[test_idx,:]
y_test = y[test_idx]

#Training a perceptron model

from sklearn.linear_model import Perceptron
#train - val loop
models = []
scores = []

#Evaluate model on validation set: 
for tol in [0.001, 0.005, 0.010, 0.050, 0.100, 0.500, 1.000]: 
    model = Perceptron(penalty = None , alpha = 0.0 , tol = tol)
    model.fit(x_train, y_train) 
    models.append(model)
    
#predict class    
    predictions_val = model.predict(x_val)
    
#check accuracy   
    score = model.score(x_val , y_val)
    scores.append(score)
    print('\nValidation accuracy:',score)

#Evaluate model on Testing set
#predict the class
predictions_test = model.predict(x_test)

#check accuracy
scores_test = np.where(predictions_test == y_test, 1 , 0) # if correct then 1 else 0
mean_accuracy_test = np.mean(scores_test) #average of the values
print('\nTesting accuracy:',mean_accuracy_test)
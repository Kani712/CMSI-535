#Boston hosuing data
#Loading data

import numpy as np
import sklearn.datasets as skdata

boston_housing_data = skdata.load_boston()

#Loading data as numpy arrays

feature_names = boston_housing_data.feature_names
x = boston_housing_data.data
y = boston_housing_data.target

#Looking at the shape of the data

print('\n',x.shape)
print('\n',y.shape)

#Selecting 1st sample and printing its name and value

for name in range(len(feature_names)):
    if len(feature_names) == 0 :
        continue
    print('\n',feature_names[name],' :',x[0,name])
print('\nLabel: ',y[0])

#splitting the data

idx = np.random.permutation(x.shape[0])
#print(idx)

train_split_idx = int(0.80 * x.shape[0]) #train(80%)
val_split_idx = int(0.90 * x.shape[0]) #validate(10%)
train_idx = idx[:train_split_idx]
val_idx = idx[train_split_idx : val_split_idx]
test_idx = idx[val_split_idx:]


print('\nIndices:',train_idx.shape,val_idx.shape,test_idx.shape)

x_train , y_train = x[train_idx, :] , y[train_idx]
x_val , y_val = x[val_idx, :] , y[val_idx]
x_test , y_test = x[test_idx, : ] , y[test_idx]

print('\nTraining')
print(x_train.shape , y_train.shape)

print('\nValidating')
print(x_val.shape , y_val.shape)

print('\nTesting')
print(x_test.shape , y_test.shape)



print('\n\t\t---------- Breast cancer datasets --------')

breast_cancer_data = skdata.load_breast_cancer()

#Loading data

feature_names = breast_cancer_data.feature_names
x1 = breast_cancer_data.data
y1 = breast_cancer_data.target


print('\n',x1.shape)
print('\n',y1.shape)

for name , value in zip(feature_names , x1[0,...]) :
    print('\n{} : {}'.format(name , value))
print('\nLabel: {}'.format(y1[0]))

#splitting data

indices = np.random.permutation(x1.shape[0])

train_split = int(0.90 * x1.shape[0])
val_split = int(0.95 * x1.shape[0])

train_indices = indices[:train_split]
val_indices = indices[train_split : val_split]
test_indices = indices[val_split:]

print('\nIndicies:',train_indices.shape,val_indices.shape,test_indices.shape)

x1_train , y1_train = x1[train_indices,:] , y1[train_indices]
x1_val , y1_val = x1[val_indices,:] , y1[val_indices]
x1_test , y1_test = x1[test_indices,:] , y1[test_indices]

print('\nTraining_data: ') 
print('\n',x1_train.shape , y1_train.shape)

print('\nValidating_data: ') 
print('\n', x1_val.shape , y1_val.shape)

print('\nTesting_data: ') 
print('\n' ,x1_test.shape , y1_test.shape)








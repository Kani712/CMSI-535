import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:

You should answer the questions:
1) Implemented Multi-class Perceptron algorithm
2) iniailized the weights with respect to class to 0 in fit function
predicted the maximum class in predict function
computed the loss if loss is >2.0 and updated the weights of predicted class and weights of actual class until convergence in update function
Trained(80%) , validated(10%) and tested(10%) the loop in main function
3) constants are T=100, tols in fit function; hyperparameters are (tols,train_steps) in main function

Error in predciting:
Algorithm always predicts class 0, I couldn't configure out where I am going wrong

Scores:

Results on the iris dataset using scikit-learn Perceptron model
Training set mean accuracy: 0.8512
Validation set mean accuracy: 0.7333
Testing set mean accuracy: 0.9286
Results on the iris dataset using our Perceptron model trained with 60 steps and tolerance of 0.01
Training set mean accuracy: 0.3306
Validation set mean accuracy: 0.3333
Results on the iris dataset using our Perceptron model trained with 100 steps and tolerance of 0.01
Training set mean accuracy: 0.3306
Validation set mean accuracy: 0.3333
Results on the iris dataset using our Perceptron model trained with 200 steps and tolerance of 0.01
Training set mean accuracy: 0.3306
Validation set mean accuracy: 0.3333
Using best model trained with 200 steps and tolerance of 0.01
Testing set mean accuracy: 0.3571
Results on the wine dataset using scikit-learn Perceptron model
Training set mean accuracy: 0.5625
Validation set mean accuracy: 0.4118
Testing set mean accuracy: 0.4706
Results on the wine dataset using our Perceptron model trained with 60 steps and tolerance of 1
Training set mean accuracy: 0.3889
Validation set mean accuracy: 0.4706
Results on the wine dataset using our Perceptron model trained with 80 steps and tolerance of 1
Training set mean accuracy: 0.3889
Validation set mean accuracy: 0.4706
Results on the wine dataset using our Perceptron model trained with 100 steps and tolerance of 1
Training set mean accuracy: 0.3889
Validation set mean accuracy: 0.4706
Using best model trained with 100 steps and tolerance of 1
Testing set mean accuracy: 0.4118
'''

'''
Implementation of Perceptron for multi-class classification
'''
class PerceptronMultiClass(object):

    def __init__(self):
        # Define private variables, weights and number of classes
        self.__weights = None
        self.__n_class = 3

    def __predict_label_n_class(self, x_n):
        w_c = [np.expand_dims(self.__weights[:, c], axis=-1) for c in range(self.__n_class)]
        predict_class = [np.matmul(w.T, x_n) for w in w_c]

        max_val = max(predict_class)
        predictions = predict_class.index(max_val)

        return predictions
    def __update(self, x, y):
        '''
        Update the weight vector during each training iteration

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
        '''
        # TODO: Implement the member update function
        #for c in range(self.__n_class):
        #weights_c = np.expand_dims(self.__weights[:,c],axis =0)
        threshold = 1.0/self.__n_class * np.ones([1, x.shape[1]])
        x = np.concatenate([threshold, x], axis = 0)
        for n in range(x.shape[1]):
                x_n = np.expand_dims(x[:, n],axis=-1)
                predictions = self.__predict_label_n_class(x_n)

                if predictions != y[n]:
                    self.__weights[:, predictions] = self.__weights[:, predictions] - np.squeeze(x_n, axis=-1) 
                    self.__weights[:, y[n]] = self.__weights[:, y[n]] + np.squeeze(x_n, axis=-1) 

    def fit(self, x, y, T=100, tol=1e-3):
        '''
        Fits the model to x and y by updating the weight vector
        based on mis-classified examples for t iterations until convergence

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            t : int
                number of iterations to optimize perceptron
            tol : float
                change of loss tolerance, if greater than loss + tolerance, then stop
        '''
        # TODO: Implement the fit function
        #Initialize the weights to zero
        self.__n_class= len(np.unique(y)) #number of classes
        #print(x.shape[0],x.shape[1],self.__n_class) #(d+1,C)

        self.__weights = np.zeros([x.shape[0]+1, self.__n_class])
        self.__weights[0, :] = -1.0
        #print(self.__weights)
        #print(self.__weights.shape[0],self.__weights.shape[1])
        #finding misclassified examples
        # c_hat = h(x^n(t)) , c_star = y^n ---> unique values determine the __n_class
        #Initialize loss and weights
        prev_loss = 2.0
        pre_weights = np.copy(self.__weights)

        for t in range(T):
            predictions = self.predict(x)
            #loss = 1/N sum n^N
            loss = np.mean(np.where(predictions !=y, 1.0, 0.0))
            #stopping convergence
            if loss == 0.0:
                break

            elif loss > prev_loss + tol and t > 2:
                self.__weights = pre_weights
                break
            prev_loss = loss
            pre_weights = np.copy(self.__weights)
            #updating weight vector and class
            self.__update(x,y)
    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : 1 x N label vector
        '''
        # TODO: Implement the predict function
        #compute weights (d+1,N)
        #threshold shape is (1,N)
        threshold = 1.0/self.__n_class * np.ones([1, x.shape[1]])
        #print('threshold',threshold.shape)
        #x is (d,N), thus concatenate threshold and # X
        x = np.concatenate([threshold,x],axis=0) #--> (d+1,N)
        #print('Size of x',x.shape)
        #predict w^T(d+1,N)^T . (d+1,N) --> (1,N)
        predictions = np.zeros([1, x.shape[1]])
        for n in range(x.shape[1]):
            x_n = np.expand_dims(x[:, n], axis=-1)
            predictions[0, n] = self.__predict_label_n_class(x_n)
            
        return predictions
    def score(self, x, y):
        '''
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label

        Returns:
            float : mean accuracy
        '''
        # TODO: Implement the score function
        predcitions = self.predict(x)

        #accuracy score
        scores = np.where(predcitions == y, 1.0, 0.0)
        return np.mean(scores)


def split_dataset(x, y, n_sample_train_to_val_test=8):
    '''
    Helper function to splits dataset into training, validation and testing sets

    Args:
        x : numpy
            d x N feature vector
        y : numpy
            1 x N ground-truth label
        n_sample_train_to_val_test : int
            number of training samples for every validation, testing sample

    Returns:
        x_train : numpy
            d x n feature vector
        y_train : numpy
            1 x n ground-truth label
        x_val : numpy
            d x m feature vector
        y_val : numpy
            1 x m ground-truth label
        x_test : numpy
            d x m feature vector
        y_test : numpy
            1 x m ground-truth label
    '''
    n_sample_interval = n_sample_train_to_val_test + 2

    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % n_sample_interval == (n_sample_interval - 1):
            val_idx.append(idx)
        elif idx and idx % n_sample_interval == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':

    iris_data = skdata.load_iris()
    wine_data = skdata.load_wine()

    datasets = [iris_data, wine_data]
    tags = ['iris', 'wine']

    # TODO: Experiment with 3 different max training steps (T) for each dataset
    train_steps_iris = [50,500,1000]
    train_steps_wine = [60, 500, 2000]

    train_steps = [train_steps_iris, train_steps_wine]

    # TODO: Set a tolerance for each dataset
    tol_iris = 1.0
    tol_wine = 1.0

    tols = [tol_iris, tol_wine]

    for dataset, steps, tol, tag in zip(datasets, train_steps, tols, tags):
        # Split dataset into 80 training, 10 validation, 10 testing
        x = dataset.data
        y = dataset.target
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(
            x=x,
            y=y,
            n_sample_train_to_val_test=8)

        '''
        Trains and tests Perceptron model from scikit-learn
        '''
        model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
        # Trains scikit-learn Perceptron model
        model.fit(x_train, y_train)

        print('Results on the {} dataset using scikit-learn Perceptron model'.format(tag))

        # Test model on training set
        scores_train = model.score(x_train, y_train)
        print('Training set mean accuracy: {:.4f}'.format(scores_train))

        # Test model on validation set
        scores_val = model.score(x_val, y_val)
        print('Validation set mean accuracy: {:.4f}'.format(scores_val))

        # Test model on testing set
        scores_test = model.score(x_test, y_test)
        print('Testing set mean accuracy: {:.4f}'.format(scores_test))

        '''
        Trains, validates, and tests our Perceptron model for multi-class classification
        '''
        # TODO: obtain dataset in correct shape (d x N)
        x_train = np.transpose(x_train, axes=(1, 0))
        x_val = np.transpose(x_val, axes=(1, 0))
        x_test = np.transpose(x_test, axes=(1, 0))
        # Initialize empty lists to hold models and scores
        models = []
        scores = []
        for T in steps:
            # TODO: Initialize PerceptronMultiClass model
            model = PerceptronMultiClass()
            print('Results on the {} dataset using our Perceptron model trained with {} steps and tolerance of {}'.format(tag, T, tol))
            # TODO: Train model on training set
            model.fit(x_train, y_train)

            # TODO: Test model on training set
            scores_train = model.score(x_train, y_train)
            print('Training set mean accuracy: {:.4f}'.format(scores_train))

            # TODO: Test model on validation set
            scores_val = model.score(x_val, y_val)
            print('Validation set mean accuracy: {:.4f}'.format(scores_val))

            # TODO: Save the model and its score
            models.append(model)
            scores.append(scores_val)

        # TODO: Select the best performing model on the validation set
        #iterate the validation scores
        max_score = max(scores)
        best_idx = scores.index(max_score)
        
        print('Using best model trained with {} steps and tolerance of {}'.format(steps[best_idx], tol))

        # TODO: Test model on testing set
        scores_test = model.score(x_test, y_test)
        print('Testing set mean accuracy: {:.4f}'.format(scores_test))

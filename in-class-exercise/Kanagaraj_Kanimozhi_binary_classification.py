import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


'''
Name: Kanagaraj , Kanimozhi

Summary:
Results using scikit-learn Perceptron model
Training set mean accuracy: 0.7943
Validation set mean accuracy: 0.7321
Testing set mean accuracy: 0.7857
Results using our Perceptron model trained with 10 steps
Training set mean accuracy: 0.8775
Validation set mean accuracy: 0.8393
Results using our Perceptron model trained with 20 steps
Training set mean accuracy: 0.8775
Validation set mean accuracy: 0.8393
Results using our Perceptron model trained with 60 steps
Training set mean accuracy: 0.8775
Validation set mean accuracy: 0.8393
Using best model trained with 10 steps
Testing set mean accuracy: 0.8929
'''

'''
Implementation of Perceptron for binary classification
'''
class PerceptronBinary(object):

    def __init__(self):
        # Define private variables
        self.__weights = None

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
        #dimension of x (d , N)
        #argument the feature vector (dx N) thrshold(1,N)
        threshold = 0.5 * np.ones([1,x.shape[1]])# (1,N)
        #print(range(x.shape[1]))
        #print(threshold)
        x =np.concatenate([threshold,x],axis= 0) #(d+1,N)
        #print(x)
        #update for all incorrect prection
        #w^(t+1) = w^(t) + y^n x^n


        #iterate
        #walk through every example
        for n in range(x.shape[1]):
            # x is (d+1 ,N) , so shape is (d +1), weights is (d+1,1)
            #we use np.reshape(x[:,n],(-1,1)) for reshape -1 means all elements
            # np.expand_dims(x,axis=-1 for (a,b,c,d,e,f) --> (a,b,c,d,e,f,1)
            # np.expand_dims(x,axis=0 for (a,b,c,d,e,f) --> (1,a,b,c,d,e,f)
            x_n = np.expand_dims(x[:, n], axis=-1) #shape of (d+1,1)
            #print(x_n)
            #predict label for x_n
            
            prediction = np.sign(np.matmul(self.__weights.T,x_n))
            #print(np.matmul(self.__weights.T,x_n))
            #if prediction is equal to ground truth
            if prediction != y[n]:
                #print(prediction)
                #w^(t+! = w^(t) + (y_n * x_n)
                #shape: (d+1,1) = (d+1) + (1) * (d+1,1
                #print(self.__weights)
                
                self.__weights = self.__weights + (y[n] * x_n)
                
    def fit(self, x, y, T=100, tol=1.0):
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

        #intialize weight (d+1, 1)
        #[w0,w1,w2,w3,.....] = [0,0,0,,...0]
        #print(x.shape[0])
        #print('ground',y.shape)
        self.__weights = np.zeros([x.shape[0]+1, 1])
   
        self.__weights[0,0] = -1.0
        #print(self.__weights.shape[0],self.__weights.shape[1],self.__weights)
        #intinalize loss and weight
        prev_loss = 2.0
        pre_weights = np.copy(self.__weights)
        for t in range(T):
            #compute the loss
            predictions = self.predict(x)
            #l = 1/N \ sum_n^N
            loss = np.mean(np.where(predictions !=y, 1.0, 0.0))
            #print('t={} loss={}'.format(t + 1,loss))

            #stopping convergence
            if loss == 0.0:
                break
            elif loss > prev_loss + tol and t > 2:
                #3if our loss from t =0.1 , t+1 = 0.5, taks weights of previous time step
                self.__weights = pre_weights
                break

            #update previous loss and previous weights
            prev_loss = loss
            pre_weights = np.copy(self.__weights)
            
            #updates our weight vector
            self.__update(x,y)


    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : d x 1 label vector
        '''
        # TODO: Implement the predict function
        #[w0,w1,w2,w3,w4...] (d+1,N)
        #[x1,x2,x3,x4,...xd](d, N)
        #shape of threshold (1 x N)
        threshold = 0.5 * np.ones((1, x.shape[1]))

        #argument x with threshold
        x = np.concatenate([threshold,x], axis=0) # x is (d+1,N)
        #predict using w^T: (d+1, 1)^T times (d+1 , N) = 1, N)
        predictions = np.matmul(self.__weights.T,x)
        
        #print(predictions.shape[0],predictions.shape[1])
        #sign of predictions
        #h(x) = sign(w^Tx)
        #print(np.sign(predictions))
        return np.sign(predictions)

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
        predictions = self.predict(x) #(1 x N) or {-1 ,+}
        #check the accuracy
        scores = np.where(predictions == y,1.0,0.0)
        #
        return np.mean(scores)
if __name__ == '__main__':

    breast_cancer_data = skdata.load_breast_cancer()
    x = breast_cancer_data.data
    y = breast_cancer_data.target

    # 80 percent train, 10 percent validation, 10 percent test split
    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % 10 == 9:
            val_idx.append(idx)
        elif idx and idx % 10 == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    '''
    Trains and tests Perceptron model from scikit-learn
    '''
    model = Perceptron(penalty=None, alpha=0.0, tol=1)
    # Trains scikit-learn Perceptron model
    model.fit(x_train, y_train)

    print('Results using scikit-learn Perceptron model')

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
    Trains and tests our Perceptron model for binary classification
    '''
    # TODO: obtain dataset in correct shape (d x N) , previously (n x d)
    x_train = np.transpose(x_train, axes=(1 , 0))
    x_val = np.transpose(x_val, axes=(1 , 0))
    x_test = np.transpose(x_test, axes=(1,0))

    # TODO: obtain labels in {+1, -1} format
    y_train = np.where(y_train == 0 ,-1, 1)
    y_val = np.where(y_val == 0 ,-1, 1)
    y_test = np.where(y_test == 0 , -1, 1)
    # TODO: Initialize model, train model, score model on train, val and test sets

    # Train 3 PerceptronBinary models using 10, 50, and 60 steps with tolerance of 1
    models = []
    scores = []
    steps = [10, 50, 60]
    for T in steps:
        # Initialize PerceptronBinary model
        model = PerceptronBinary()
        print('Results using our Perceptron model trained with {} steps'.format(T))
        # Train model on training set
        model.fit(x_train,y_train)

        # Test model on training set
        scores_train = model.score(x_train , y_train)
        print('Training set mean accuracy: {:.4f}'.format(scores_train))

        # Test model on validation set
        scores_val =  model.score(x_val,y_val)
        print('Validation set mean accuracy: {:.4f}'.format(scores_val))

        # Save the model and its score
        models.append(model)
        scores.append(scores_val)
    # Select the best performing model on the validation set
    sort = np.sort(scores)
    for i, val in enumerate(sort):
        best_idx = i
    print('Using best model trained with {} steps'.format(steps[best_idx]))

    # Test model on testing set
    scores_test = model.score(x_test,y_test)

    print('Testing set mean accuracy: {:.4f}'.format(scores_test))

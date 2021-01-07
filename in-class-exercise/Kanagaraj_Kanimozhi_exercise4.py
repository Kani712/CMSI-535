import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
from sklearn.linear_model import LinearRegression


'''
Name: Kanagaraj, Kanimozhi


Summary:


Results using scikit-learn LinearRegression model
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results using our linear regression model trained with normal_equation
Training set mean squared error: 25.9360
Training set r-squared scores: 0.7015
Validation set mean squared error: 18.4747
Validation set r-squared scores: 0.7365
Testing set mean squared error: 18.1262
Testing set r-squared scores: 0.7679
Results using our linear regression model trained with pseudoinverse
Training set mean squared error: 25.9360
Training set r-squared scores: 0.7015
Validation set mean squared error: 18.4747
Validation set r-squared scores: 0.7365
Testing set mean squared error: 18.1262
Testing set r-squared scores: 0.7679
'''

'''
Implementation of linear regression by directly solving normal equation or pseudoinverse
'''
class LinearRegressionClosedForm(object):

    def __init__(self):
        # Define private variables
        self.__weights = None

    def __fit_normal_equation(self, X, y):
        '''
        Fits the model to X and y via normal equation

        Args:
            X : numpy
                N X d feature vector
            y : numpy
                1 X N ground-truth label
        '''
        # TODO: Implement the __fit_normal_equation function

        #compute w* = (X^T X)^-1. X^Ty
        #1. X is (N,d) , where X^TX is (d,d)
        #2. X^T.y
        inverse = np.linalg.inv(np.matmul(X.T, X)) #taking ib=nverse of X^TX
        second_term = np.matmul(X.T, y)
        self.__weights = np.matmul(inverse, second_term)        
    
    def __fit_pseudoinverse(self, X, y):
        '''
        Fits the model to X and y via pseudoinverse

        Args:
            X : numpy
                N X d feature vector
            y : numpy
                1 X N ground-truth label
        '''
        # TODO: Implement the __fit_pseudoinverse function
        
        #to get pseufo inverse of X
        #compute u,s,v_t from svd
        #take reciporcal of s and transpose of result
        #,ul v s+ u.T to X+
        
        # X is (n , d)
        #computr SVD give us u (N , N) , s(N ,d), V_t(d,d)
        U, S, V_t = np.linalg.svd(X) #--> (d,d)
        
        #u(N ,N) , s(d) , V-t(d,d)
        # conver s to (n ,d )
        
        #to get a diagonal matric ie sqare matriX from  a vector 
        #we can use numpy.diag() , steps to compute reciporcal ie s for evry sigma take 1/s
        S_diag = np.diag(1.0 / S) #(d,d)
        
        # turn it into (n,d ) matriX
        #we know s should be zero evrywhere else 
        # N-d along 0th dimention
        #and d along the 1-st dimension
        #s.shape gives N s.shape 0 gives d
        padding = np.zeros([U.shape[0] - S.shape[0], S.shape[0]])
        S_pseudo = np.concatenate([S_diag, padding], aXis= 0)
        
        #tranpose sigma+
        S_pseudo = S_pseudo.T
        
        #X+ = v.s+U.T
        #given v transpose take v trranspose
        X_pesudo = np.matmul(np.matmul(V_t.T, S_pseudo), U.T)
        #w+ = X+ y
        self.__weights = np.matmul(X_pesudo, y)
        
        
    def fit(self, X, y, solver ='normal_equation'):
        '''
        Fits the model to X and y by solving the ordinary least squares
        using normal equation or pseudoinverse (SVD)

        Args:
            X : numpy
                d X N feature vector
            y : numpy
                1 X N ground-truth label
            solver : str
                solver types: normal_equation, pseudoinverse
        '''
        # TODO: Implement the fit function
        
        #turn from (d,N) to (n,d) by taking its transpose
        X = X.T
        
        if solver == 'normal_equation':
            self.__fit_normal_equation(X, y)
        elif solver == 'pseudoinverse':
            self.__fit_pseudoinverse(X, y)
        else:
            raise ValueError('Encountred unsupported solver: {}'.format(solver))

        

    def predict(self, X):
        '''
        Predicts the label for each feature vector X

        Args:
            X : numpy
                d X N feature vector

        Returns:
            numpy : d X 1 label vector
        '''
        # TODO: Implement the predict function
        predcitions = np.matmul(self.__weights.T, X)
        print(X.shape,predcitions.shape)
        return predcitions
    
    def __score_r_squared(self, y_hat, y):
        '''
        Measures the r-squared score from groundtruth y

        Args:
            y_hat : numpy
                1 X N predictions
            y : numpy
                1 X N ground-truth label

        Returns:
            float : r-squared score
        '''
        # TODO: Implement the __score_r_squared function
        
        #uneXplained variation u : sum(y_hat - y)^2
        sum_squared_errors = np.sum((y_hat - y) ** 2)
    
        #Total variance (v) : sum(y-y_mean)^2
        sum_variance = np.sum((y - np.mean(y)) ** 2)
        
        return 1.0 - (sum_squared_errors / sum_variance)

    def __score_mean_squared_error(self, y_hat, y):
        '''
        Measures the mean squared error (distance) from groundtruth y

        Args:
            y_hat : numpy
                1 X N predictions
            y : numpy
                1 X N ground-truth label

        Returns:
            float : mean squared error (mse)
        '''
        # TODO: Implement the __score_mean_squared_error function
        
        #mean(y_hat: - y)^2
        
        return np.mean((y_hat - y) ** 2)

    def score(self, X, y, scoring_func='r_squared'):
        '''
        Predicts real values from X and measures the mean squared error (distance)
        or r-squared from groundtruth y

        Args:
            X : numpy
                d X N feature vector
            y : numpy
                1 X N ground-truth label
            scoring_func : str
                scoring function: r_squared, mean_squared_error

        Returns:
            float : mean squared error (mse)
        '''
        # TODO: Implement the score function

        predictions = self.predict(X)
        if scoring_func == 'r_squared':      
            return self.__score_r_squared(predictions, y)
        elif scoring_func == 'mean_squared_error':
            return self.__score_mean_squared_error(predictions, y)
        else:
            raise ValueError('Encountred unsupported scoring_func: {}'.format(scoring_func))
        


if __name__ == '__main__':

    boston_housing_data = skdata.load_boston()
    X = boston_housing_data.data
    y = boston_housing_data.target

    # 80 percent train, 10 percent validation, 10 percent test split
    train_idX = []
    val_idX = []
    test_idX = []
    for idX in range(X.shape[0]):
        if idX and idX % 10 == 9:
            val_idX.append(idX)
        elif idX and idX % 10 == 0:
            test_idX.append(idX)
        else:
            train_idX.append(idX)

    X_train, X_val, X_test = X[train_idX, :], X[val_idX, :], X[test_idX, :]
    y_train, y_val, y_test = y[train_idX], y[val_idX], y[test_idX]

    '''
    Trains and tests Linear regression from scikit-learn
    '''
    # TODO: Initialize scikit-learn linear regression model
    model = LinearRegression()

    # TODO: Trains scikit-learn linear regression model
    model.fit(X_train,y_train)
    
    print('Results using scikit-learn LinearRegression model')

    # TODO: Test model on training set
    predictions_train = model.predict(X_train)
    #print(np.maX(y_train))
    scores_mse_train = skmetrics.mean_squared_error(predictions_train,y_train) # take sqrt to check the values
    print('Training set mean squared error: {:.4f}'.format(scores_mse_train))

    scores_r2_train = model.score(X_train,y_train)
    print('Training set r-squared scores: {:.4f}'.format(scores_r2_train))

    # TODO: Test model on validation set
    predictions_val = model.predict(X_val)
    
    scores_mse_val = skmetrics.mean_squared_error(predictions_val,y_val)
    print('Validation set mean squared error: {:.4f}'.format(scores_mse_val))

    scores_r2_val = model.score(X_val,y_val)
    print('Validation set r-squared scores: {:.4f}'.format(scores_r2_val))

    # TODO: Test model on testing set
    predictions_test = model.predict(X_test)
    
    scores_mse_test = skmetrics.mean_squared_error(predictions_test, y_test)
    print('Testing set mean squared error: {:.4f}'.format(scores_mse_test))

    scores_r2_test = model.score(X_test, y_test)
    print('Testing set r-squared scores: {:.4f}'.format(scores_r2_test))

    '''
    Trains and tests our linear regression model using different solvers
    '''
    # TODO: obtain dataset in correct shape (d X N) previously (N X d)
    X_train = np.transpose(X_train , aXes=(1, 0))
    X_val = np.transpose(X_val, aXes=(1, 0))
    X_test = np.transpose(X_test, aXes=(1, 0))
    
    # Train 2 LinearRegressionClosedForm models using normal equation and pseudoinverse
    solvers = ['normal_equation', 'pseudoinverse']
    for solver in solvers:
        # TODO: Initialize Linear Regression model
        model = LinearRegressionClosedForm()
        print('Results using our linear regression model trained with {}'.format(solver))
        # TODO: Train model on training set
        model.fit(X_train,y_train)
        # TODO: Test model on training set using mean squared error and r-squared
        predictions_train = model.predict(X_train)
        scores_mse_train = skmetrics.mean_squared_error(predictions_train,y_train)
        print('Training set mean squared error: {:.4f}'.format(scores_mse_train))

        scores_r2_train =model.score(X_train,y_train)
        print('Training set r-squared scores: {:.4f}'.format(scores_r2_train))

        # TODO: Test model on validation set using mean squared error and r-squared
        predictions_val = model.predict(X_val)
        scores_mse_val = skmetrics.mean_squared_error(predictions_val,y_val)
        print('Validation set mean squared error: {:.4f}'.format(scores_mse_val))

        scores_r2_val = model.score(X_val,y_val)
        print('Validation set r-squared scores: {:.4f}'.format(scores_r2_val))

        # TODO: Test model on testing set using mean squared error and r-squared
        predictions_test = model.predict(X_test)
        scores_mse_test = skmetrics.mean_squared_error(predictions_test,y_test)
        print('Testing set mean squared error: {:.4f}'.format(scores_mse_test))

        scores_r2_test = model.score(X_test,y_test)
        print('Testing set r-squared scores: {:.4f}'.format(scores_r2_test))

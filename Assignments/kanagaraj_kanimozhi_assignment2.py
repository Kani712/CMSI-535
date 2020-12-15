import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression
from matplotlib import pyplot as plt


'''
Name: kanagaraj_kanimozhi


Summary:
Ridge regression is a linear regssion with regualrization term added to it

1. Plotted the graph for scikit-learn RidgeRegrssion model
    a. assigned the x values as log scale of alpha 
    b. assigned the y values as score of MSe for MSE plot and scores of R-squared for r-squared plot
    c. assigned the x_limits from 0  to maximum of x values + 1
    d. assigned the y_limits from 0 10 40 for MSE and 0 to 1 for R-squared
2. Implemented Ridge regression model
    a. took the transpose of dataset to get the shape (d,N)
    b. intialized the class 
    c. Trained hte model for evry alpha for a dgree of polynomial 2
    d. trained the model on training set using fit function
3. Implemented the fit function
    a. initialized the weights w* = (Z^T.Z + alpha(lambda)*I)^-1.Z^T.y,
        inorder to uniformaly apply lambda(alpha)[scalar] to all weights we multiply with identity matrix
        the inverse gives the solution of w*, if alpha is small the w* is big(more weights used) and if alpha is big w* is small(fewer weights used)
    b. Compute the loss until it statisfies both data fidelity and regularizrion condition
        loss = 1/N((Z.w-y)^T(Z.w-y) + alpha(lambda)w^T.w)
4. Calculate the mean_squared error and r-sqaured scores for training, validation and testing set
5. implented predict, score , mean squeared and r-swaured functions
6. Plotted the MSE and R-sqaure graph for own ridge regession model

Report your scores here. For example,

Results for scikit-learn RidgeRegression model with alpha=1.0
Training set mean squared error: 6.3724
Training set r-squared scores: 0.9267
Validation set mean squared error: 9.6293
Validation set r-squared scores: 0.8626
Testing set mean squared error: 19.2863
Testing set r-squared scores: 0.7531
Results for scikit-learn RidgeRegression model with alpha=10.0 
Training set mean squared error: 6.9915
Training set r-squared scores: 0.9195
Validation set mean squared error: 10.5660
Validation set r-squared scores: 0.8493
Testing set mean squared error: 18.0993
Testing set r-squared scores: 0.7683
Results for scikit-learn RidgeRegression model with alpha=100.0
Training set mean squared error: 7.8843
Training set r-squared scores: 0.9093
Validation set mean squared error: 11.9197
Validation set r-squared scores: 0.8300
Testing set mean squared error: 18.5883
Testing set r-squared scores: 0.7620
Results for scikit-learn RidgeRegression model with alpha=1000.0
Training set mean squared error: 8.8610
Training set r-squared scores: 0.8980
Validation set mean squared error: 11.7491
Validation set r-squared scores: 0.8324
Testing set mean squared error: 15.2857
Testing set r-squared scores: 0.8043
Results for scikit-learn RidgeRegression model with alpha=10000.0
Training set mean squared error: 10.0741
Training set r-squared scores: 0.8841
Validation set mean squared error: 11.7167
Validation set r-squared scores: 0.8329
Testing set mean squared error: 13.5444
Testing set r-squared scores: 0.8266
Results for scikit-learn RidgeRegression model with alpha=100000.0
Training set mean squared error: 11.4729
Training set r-squared scores: 0.8680
Validation set mean squared error: 12.5270
Validation set r-squared scores: 0.8213
Testing set mean squared error: 10.8895
Testing set r-squared scores: 0.8606
Results for our RidgeRegression model with alpha=1.0
Training Loss: 6.664
Data Fidelity Loss: 6.413  Regularization Loss: 0.252
Training set mean squared error: 6.4127
Training set r-squared scores: 0.9262
Validation set mean squared error: 8.9723
Validation set r-squared scores: 0.8720
Testing set mean squared error: 18.4835
Testing set r-squared scores: 0.7633
Results for our RidgeRegression model with alpha=10.0
Training Loss: 7.415
Data Fidelity Loss: 7.026  Regularization Loss: 0.389
Training set mean squared error: 7.0258
Training set r-squared scores: 0.9191
Validation set mean squared error: 9.5386
Validation set r-squared scores: 0.8639
Testing set mean squared error: 16.1997
Testing set r-squared scores: 0.7926
Results for our RidgeRegression model with alpha=100.0
Training Loss: 8.347
Data Fidelity Loss: 7.930  Regularization Loss: 0.417
Training set mean squared error: 7.9301
Training set r-squared scores: 0.9087
Validation set mean squared error: 10.6471
Validation set r-squared scores: 0.8481
Testing set mean squared error: 16.3874
Testing set r-squared scores: 0.7902
Results for our RidgeRegression model with alpha=1000.0
Training Loss: 9.429
Data Fidelity Loss: 8.911  Regularization Loss: 0.517
Training set mean squared error: 8.9114
Training set r-squared scores: 0.8974
Validation set mean squared error: 11.2366
Validation set r-squared scores: 0.8397
Testing set mean squared error: 14.5313
Testing set r-squared scores: 0.8139
Results for our RidgeRegression model with alpha=10000.0
Training Loss: 10.707
Data Fidelity Loss: 10.042  Regularization Loss: 0.665
Training set mean squared error: 10.0420
Training set r-squared scores: 0.8844
Validation set mean squared error: 11.8909
Validation set r-squared scores: 0.8304
Testing set mean squared error: 13.8512
Testing set r-squared scores: 0.8226
Results for our RidgeRegression model with alpha=100000.0
Training Loss: 12.984
Data Fidelity Loss: 11.598  Regularization Loss: 1.385
Training set mean squared error: 11.5984
Training set r-squared scores: 0.8665
Validation set mean squared error: 13.1313
Validation set r-squared scores: 0.8127
Testing set mean squared error: 11.8234
Testing set r-squared scores: 0.8486
'''

'''
Implementation of ridge regression
'''
class RidgeRegressionClosedForm(object):

    def __init__(self,alpha):

        # Define private variables
        self.__weights = None
        self.__alpha = alpha

    def fit(self, z, y, alpha =0.0):
        '''
        Fits the model to x and y using closed form solution

        Args:
            z : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            alpha : float
                weight (lambda) of regularization term
        '''

        # TODO: Implement the fit function
        #computing the weights
        # convert from (d,N) to (N,d) by taking its transpose
        Z = z.T
        #print(z.shape,Z.shape,y.shape)
        I = np.identity(Z.shape[1])
        N = np.size(y) #no of features
        #print(I.shape)
        #w* = (Z^T.Z + alpha * I)^-1.Z^T.y 
        inverse = np.linalg.inv(np.matmul(Z.T,Z) + self.__alpha * I) # inverse of (Z^T.Z + alpha*I)
        second_term = np.matmul(Z.T,y)
        self.__weights = np.matmul(inverse, second_term)
        
        # TODO: Compute loss
        # loss = 1/N((Z.w-y)^T(Z.w-y) + alpha(lambda)w^T.w)
        #wehre , loss of data fidelity = 1/N(Z.w-y)^T(Z.w-y)
        #loss of regularization = 1/N alpha(lambda)W^T.w
        loss_data_fidelity = 1/N * (np.matmul(np.transpose(np.matmul(Z, self.__weights)- y),(np.matmul(Z, self.__weights)- y)))
        loss_regularization = 1/N * (np.matmul((self.__alpha * self.__weights.T), self.__weights))
        
        loss = loss_data_fidelity + loss_regularization
        
        print('Training Loss: {:.3f}'.format(loss))
        print('Data Fidelity Loss: {:.3f}  Regularization Loss: {:.3f}'.format(
            loss_data_fidelity, loss_regularization))

    def predict(self, z):
        '''
        Predicts the label for each feature vector x

        Args:
            z : numpy
                d x N feature vector

        Returns:
            numpy : d x 1 label vector
        '''

        # TODO: Implement the predict function
        
        predictions = np.matmul(self.__weights.T,z)

        return predictions

    def __score_r_squared(self, y_hat, y):
        '''
        Measures the r-squared score from groundtruth y

        Args:
            y_hat : numpy
                1 x N predictions
            y : numpy
                1 x N ground-truth label

        Returns:
            float : r-squared score
        '''

        # TODO: Implement the __score_r_squared function
        # u : sum(y_hat - y)^2
        sum_squared_errors = np.sum((y_hat - y) ** 2)

        #v total variance : sum(y - mean(y))^2
        sum_variance = np.sum((y - np.mean(y)) ** 2)
        
        
        return 1.0 - (sum_squared_errors / sum_variance)

    def __score_mean_squared_error(self, y_hat, y):
        '''
        Measures the mean squared error (distance) from groundtruth y

        Args:
            y_hat : numpy
                1 x N predictions
            y : numpy
                1 x N ground-truth label

        Returns:
            float : mean squared error (mse)
        '''

        # TODO: Implement the __score_mean_squared_error function

        return np.mean((y_hat - y) ** 2)

    def score(self, x, y, scoring_func='r_squared'):
        '''
        Predicts real values from x and measures the mean squared error (distance)
        or r-squared from groundtruth y

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            scoring_func : str
                scoring function: r_squared, mean_squared_error

        Returns:
            float : mean squared error (mse)
        '''

        # TODO: Implement the score function
        predictions = self.predict(x)
        if scoring_func == 'r_squared':
            return self.__score_r_squared(predictions, y)
        elif scoring_func == 'mean_squared_error':
            return self.__score_mean_squared_error(predictions, y)
        else:
            raise ValueError('Encountered unsupported scoring_func: {}'.format(scoring_func))


'''
Utility functions to compute error and plot
'''
def score_mean_squared_error(model, x, y):
    '''
    Scores the model on mean squared error metric

    Args:
        model : object
            trained model, assumes predict function returns N x d predictions
        x : numpy
            N x d numpy array of features
        y : numpy
            N x 1 groundtruth vector
    Returns:
        float : mean squared error
    '''

    # Implement the score mean squared error function
    predictions = model.predict(x)
    mse = skmetrics.mean_squared_error(predictions, y)
    return mse

def plot_results(axis,
                 x_values,
                 y_values,
                 labels,
                 colors,
                 x_limits,
                 y_limits,
                 x_label,
                 y_label):
    '''
    Plots x and y values using line plot with labels and colors

    Args:
        axis :  pyplot.ax
            matplotlib subplot axis
        x_values : list[numpy]
            list of numpy array of x values
        y_values : list[numpy]
            list of numpy array of y values
        labels : str
            list of names for legend
        colors : str
            colors for each line
        x_limits : list[float]
            min and max values of x axis
        y_limits : list[float]
            min and max values of y axis
        x_label : list[float]
            name of x axis
        y_label : list[float]
            name of y axis
    '''

    # Iterate through x_values, y_values, labels, and colors and plot them
    # with associated legend
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        axis.plot(x, y, marker='o', color=color, label=label)
        axis.legend(loc='best')

    # Set x and y limits
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    # Set x and y labels
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)


if __name__ == '__main__':

    boston_housing_data = skdata.load_boston()
    x = boston_housing_data.data
    y = boston_housing_data.target
    
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
    Trains and tests Ridge regression model from scikit-learn
    '''
    # Initialize polynomial expansion of degree 2
    poly_transform = skpreprocess.PolynomialFeatures(degree=2)

    # Compute the polynomial terms needed for the data
    poly_transform.fit(x_train)

    # Transform the data by nonlinear mapping
    x_poly_train = poly_transform.transform(x_train)
    x_poly_val = poly_transform.transform(x_val)
    x_poly_test = poly_transform.transform(x_test)

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_ridge_scikit_train = []
    scores_r2_ridge_scikit_train = []
    scores_mse_ridge_scikit_val = []
    scores_r2_ridge_scikit_val = []
    scores_mse_ridge_scikit_test = []
    scores_r2_ridge_scikit_test = []

    alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    for alpha in alphas:

        # Initialize scikit-learn ridge regression model
        model_ridge_scikit = RidgeRegression(alpha=alpha)

        # Trains scikit-learn ridge regression model
        model_ridge_scikit.fit(x_poly_train, y_train)

        print('Results for scikit-learn RidgeRegression model with alpha={}'.format(alpha))

        # Test model on training set
        score_mse_ridge_scikit_train = score_mean_squared_error(model_ridge_scikit, x_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_scikit_train))

        score_r2_ridge_scikit_train = model_ridge_scikit.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_scikit_train))

        # Save MSE and R-squared training scores
        scores_mse_ridge_scikit_train.append(score_mse_ridge_scikit_train)
        scores_r2_ridge_scikit_train.append(score_r2_ridge_scikit_train)

        # Test model on validation set
        score_mse_ridge_scikit_val = score_mean_squared_error(model_ridge_scikit, x_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_scikit_val))

        score_r2_ridge_scikit_val = model_ridge_scikit.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_scikit_val))

        # Save MSE and R-squared validation scores
        scores_mse_ridge_scikit_val.append(score_mse_ridge_scikit_val)
        scores_r2_ridge_scikit_val.append(score_r2_ridge_scikit_val)

        # Test model on testing set
        score_mse_ridge_scikit_test = score_mean_squared_error(model_ridge_scikit, x_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_scikit_test))

        score_r2_ridge_scikit_test = model_ridge_scikit.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_scikit_test))

        # Save MSE and R-squared testing scores
        scores_mse_ridge_scikit_test.append(score_mse_ridge_scikit_test)
        scores_r2_ridge_scikit_test.append(score_r2_ridge_scikit_test)

    # Convert each scores to NumPy arrays
    scores_mse_ridge_scikit_train = np.array(scores_mse_ridge_scikit_train)
    scores_mse_ridge_scikit_val = np.array(scores_mse_ridge_scikit_val)
    scores_mse_ridge_scikit_test = np.array(scores_mse_ridge_scikit_test)
    scores_r2_ridge_scikit_train = np.array(scores_r2_ridge_scikit_train)
    scores_r2_ridge_scikit_val = np.array(scores_r2_ridge_scikit_val)
    scores_r2_ridge_scikit_test = np.array(scores_r2_ridge_scikit_test)

    # Clip each set of MSE scores between 0 and 40
    scores_mse_ridge_scikit_train = np.clip(scores_mse_ridge_scikit_train, 0.0, 40.0)
    scores_mse_ridge_scikit_val = np.clip(scores_mse_ridge_scikit_val, 0.0, 40.0)
    scores_mse_ridge_scikit_test = np.clip(scores_mse_ridge_scikit_test, 0.0, 40.0)

    # Clip each set of R-squared scores between 0 and 1
    scores_r2_ridge_scikit_train = np.clip(scores_r2_ridge_scikit_train, 0.0, 1.0)
    scores_r2_ridge_scikit_val = np.clip(scores_r2_ridge_scikit_val, 0.0, 1.0)
    scores_r2_ridge_scikit_test = np.clip(scores_r2_ridge_scikit_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_ridge_scikit_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1)

    # TODO: Set x (alpha in log scale) and y values (MSE)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_mse_ridge_scikit_train,
        scores_mse_ridge_scikit_val,
        scores_mse_ridge_scikit_test
    ]

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to max of x_values + 1 and y limits between 0 and 40
    # Set x label to 'alpha (log scale)' and y label to 'MSE',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0,np.max(x_values) +1],
        y_limits=[0.0, 40.0],
        x_label='alpha',
        y_label='MSE')

    # Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # TODO: Set x (alpha in log scale) and y values (R-squared)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_r2_ridge_scikit_train,
        scores_r2_ridge_scikit_val,
        scores_r2_ridge_scikit_test
    ]

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to max of x_values + 1 and y limits between 0 and 1
    # Set x label to 'alpha (log scale)' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0,np.max(x_values) +1],
        y_limits=[0.0, 1.0],
        x_label='alpha',
        y_label='R-squared')
    # TODO: Create super title 'Scikit-Learn Ridge Regression on Training, Validation and Testing Sets'
    plt.suptitle('Scikit-Learn Ridge Regression on Training, Validation and Testing Sets')
    

    '''
    Trains and tests our ridge regression model using different alphas
    '''

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_ridge_ours_train = []
    scores_r2_ridge_ours_train = []
    scores_mse_ridge_ours_val = []
    scores_r2_ridge_ours_val = []
    scores_mse_ridge_ours_test = []
    scores_r2_ridge_ours_test = []

    # TODO: convert dataset (N x d) to correct shape (d x N)
    x_poly_train = np.transpose(x_poly_train, axes=(1, 0))
    x_poly_val = np.transpose(x_poly_val, axes=(1, 0))
    x_poly_test = np.transpose(x_poly_test, axes=(1, 0))

    # For each alpha, train a ridge regression model on degree 2 polynomial features
    for alpha in alphas:

        # TODO: Initialize our own ridge regression model
        model_our_ridge = RidgeRegressionClosedForm(alpha=alpha)
        print('Results for our RidgeRegression model with alpha={}'.format(alpha))

        # TODO: Train model on training set
        model_our_ridge.fit(x_poly_train, y_train)

        # TODO: Test model on training set using mean squared error and r-squared
        
        score_mse_ridge_ours_train = score_mean_squared_error(model_our_ridge, x_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_ours_train))

        score_r2_ridge_ours_train = model_our_ridge.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_train))

        # TODO: Save MSE and R-squared training scores
        scores_mse_ridge_ours_train.append(score_mse_ridge_ours_train)
        scores_r2_ridge_ours_train.append(score_r2_ridge_ours_train)

        # TODO: Test model on validation set using mean squared error and r-squared
        score_mse_ridge_ours_val = score_mean_squared_error(model_our_ridge, x_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_ours_val))

        score_r2_ridge_ours_val = model_our_ridge.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_val))

        # TODO: Save MSE and R-squared validation scores
        scores_mse_ridge_ours_val.append(score_mse_ridge_ours_val)
        scores_r2_ridge_ours_val.append(score_r2_ridge_ours_val)

        # TODO: Test model on testing set using mean squared error and r-squared
        score_mse_ridge_ours_test = score_mean_squared_error(model_our_ridge, x_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_ours_test))

        score_r2_ridge_ours_test = model_our_ridge.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_test))

        # TODO: Save MSE and R-squared testing scores
        scores_mse_ridge_ours_test.append(score_mse_ridge_ours_test)
        scores_r2_ridge_ours_test.append(score_r2_ridge_ours_test)

    # Convert each scores to NumPy arrays
    scores_mse_ridge_ours_train = np.array(scores_mse_ridge_ours_train)
    scores_mse_ridge_ours_val = np.array(scores_mse_ridge_ours_val)
    scores_mse_ridge_ours_test = np.array(scores_mse_ridge_ours_test)
    scores_r2_ridge_ours_train = np.array(scores_r2_ridge_ours_train)
    scores_r2_ridge_ours_val = np.array(scores_r2_ridge_ours_val)
    scores_r2_ridge_ours_test = np.array(scores_r2_ridge_ours_test)

    # TODO: Clip each set of MSE scores between 0 and 40
    scores_mse_ridge_ours_train = np.clip(scores_mse_ridge_ours_train, 0.0, 40.0)
    scores_mse_ridge_ours_val = np.clip(scores_mse_ridge_ours_val, 0.0, 40.0)
    scores_mse_ridge_ours_test = np.clip(scores_mse_ridge_ours_test, 0.0, 40.0)

    # TODO: Clip each set of R-squared scores between 0 and 1
    scores_r2_ridge_ours_train = np.clip(scores_r2_ridge_ours_train, 0.0, 1.0)
    scores_r2_ridge_ours_val = np.clip(scores_r2_ridge_ours_val, 0.0, 1.0)
    scores_r2_ridge_ours_test = np.clip(scores_r2_ridge_ours_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_ridge_ours_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1)

    # TODO: Set x (alpha in log scale) and y values (MSE)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_mse_ridge_ours_train,
        scores_mse_ridge_ours_val,
        scores_mse_ridge_ours_test
    ]

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to max of x_values + 1 and y limits between 0 and 40
    # Set x label to 'alpha (log scale)' and y label to 'MSE',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.max(x_values)+ 1],
        y_limits=[0.0, 40.0],
        x_label='alpha (log scale)',
        y_label='MSE'
    )

    # Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # TODO: Set x (alpha in log scale) and y values (R-squared)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_r2_ridge_ours_train,
        scores_r2_ridge_ours_val,
        scores_r2_ridge_ours_test
    ]

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to max of x_values + 1 and y limits between 0 and 1
    # Set x label to 'alpha (log scale)' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.max(x_values)+ 1],
        y_limits=[0.0, 1.0],
        x_label='alpha (log scale)',
        y_label='R-squared'
    )


    # TODO: Create super title 'Our Ridge Regression on Training, Validation and Testing Sets'
    plt.suptitle('Our Ridge Regression on Training, Validation and Testing Sets')
    plt.show()

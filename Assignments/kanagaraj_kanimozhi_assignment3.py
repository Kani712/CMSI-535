import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression
import matplotlib.pyplot as plt


'''
Name: kanagaraj, kanimozhi



Summary:
Implemented a gradient descent optimizer for ridge regression and computed gradients for ridge regression using scikit-learn to compare our results.
a.	Implementation has two classes:
1.	 GradientDescentOptimizer class  to compute gradients. 
    The class contains computer_gradient function to compute the gradient for each optimizer type, 
    cube_root_decay function to adjust the learning rate (it tends to zero) for computing momentum and 
    a update function to update the weights for gradient descent , gradient descent with momentum, stochastic gradient descent 
    and stochastic gradient descent with momentum.
2.	RidegRegressionGradientDescent class to implement ridge regression model , 
    it has fit function which fits the model to train data by updating the weights, 
    predict function that to predict the labels, 
    compute loss function to compute the mean squared loss with weight decay (lamda_weight_decay)
b.	The hyperparameters used are:
1.	Beta ( discounting factor)
2.	Learing_rate (alpha)
3.	Time_steps (t )
4.	Batch_sizes ( B)
5.	Optimizer types
Lambda_weight_decay (weight decay (lamda)) is a constant 0.1


TODO: Please answer the following questions and report your scores

1. What did you observe when using larger versus smaller momentum for
momentum gradient descent and momentum stochastic gradient descent?

Momentum gradient descent:
a.	For larger values of momentum (beta in code) I noticed that the model converges faster compared to smaller values of momentum
1.	For example for a momentum 0.05 depending on other parameters(learning rate, time steps(T) , it converges slower than for momentum at 0.9
2.	For a momentum of 0 its same as normal gradient descent 
Stochastic gradient descent:
a.	For smaller values of momentum stochastic gradient descent like gradient descent takes longer to converge, 
    sometimes does not converge (needs larger time step or change of learning rate) than larger values of momentum
b.	Smaller momentum or no momentum cause the stochastic gradient descent to jump around a lot (ie., the loss fluctuates largely).
     Larger momentum prevents the loss fluctuations and move in the right direction. 

2. What did you observe when using larger versus smaller batch size
for stochastic gradient descent?

a. I noticed, that for larger batch size(B) the model took more time for training but had more accurate gradients (ie. Converged faster).
b. For smaller batch sizes the model took less time to train but the needed larger steps to converge as the it reduced the loss very slowly at each time. 
    Sometime the loss never reduced much for 30000-time steps (ie, the loss kept fluctuating).,
    because choosing a small batch size we will never will cover the entire dataset as we select the batch samples randomly.

3. Explain the difference between gradient descent, momentum gradient descent,
stochastic gradient descent, and momentum stochastic gradient descent?

a.	Gradient decent
    Gradient descent is a first order iterative optimization algorithm, 
    where we compute the gradient at each time-step and update the weights to minimize the error. 
    Here the whole training set (all data points) is considered before updating. 
    gradient = f'(w) = 1/N sum_n^N 2 * (w.T.x^n - y^n) x^n + 2*lambda/N * w, w is the weights
    Weight update (towards negative gradient direction):
    w^(t + 1) = w^(t) - alpha * gradient (gradient of l(w ^(t)))

b.	Momentum Gradient descent
    The only difference is that it updates the weights with weighted average gradients (momentum) 
    i.e. it also considers the fraction of previous gradients, thus increasing learning process. 
    Momentum :
    v^(t) = beta * v^(t-1) + (1-beta)
    v^(t) = beta * v^(t-1) + (1-beta) * gradient , 
    here ,  beta is discounting factor and v is the momentum
    Weight update:
    w^(t+1) = w^(t) - alpha(learning rate) * v^(t)
c.	Stochastic gradient descent
    In SGD, unlike gradient descent we only consider batch size(B) data samples from the training set, 
    which chosen randomly, thus we compute the gradient of loss only for chosen batch (based on batch size) from training set (N samples). 
    We sample batch from dataset and update the weights as usual.
    Gradient = 1/|B| sum_b^B 2 * (w.T.x^b - y^b) x^b + 2*lambda/|B|* w
    Weight update:
    w^(t + 1) = w^(t) - alpha * gradient (gradient of l_b(w ^(t)))
d.	Momentum stochastic gradient descent
    It’s similar to momentum gradient descent but we only sample batch of given batch to compute the gradient of loss 
    and update the weight the weights for the batch dataset. 
    Here also we update the weights with weighted average gradients of chosen batch sample.
    Gradient(l_b(w)) = 1/|B| sum_b^B 2 * (w.T.x^b - y^b) x^b + 2*lambda/|B|* w
    Momentum SGD : v^(t) = beta * v^(t-1) + (1-beta)* gradients
    weight update : w^(t+1) = w^(t) - alpha(learning rate) * v^(t)

Report your scores here.
Results on using scikit-learn Ridge Regression model
Training set mean squared error: 2749.2155
Validation set mean squared error: 3722.5782
Testing set mean squared error: 3169.6860
Results on using Ridge Regression using gradient descent variants
Fitting with gradient_descent using learning rate=1.0E-01, t=5000
Training set mean squared error: 2801.9254
Validation set mean squared error: 3734.9551
Testing set mean squared error: 3212.4595
Fitting with momentum_gradient_descent using learning rate=1.0E+00, t=10000
Training set mean squared error: 2749.2378
Validation set mean squared error: 3722.4851
Testing set mean squared error: 3170.2778
Fitting with stochastic_gradient_descent using learning rate=1.2E+00, t=18000
Training set mean squared error: 2769.8891
Validation set mean squared error: 3684.7954
Testing set mean squared error: 3184.2250
Fitting with momentum_stochastic_gradient_descent using learning rate=9.0E-01, t=15000
Training set mean squared error: 2781.6466
Validation set mean squared error: 3738.1596
Testing set mean squared error: 3186.9302

'''


def score_mean_squared_error(model, x, y):
    '''
    Scores the model on mean squared error metric

    Args:
        model : object
            trained model, assumes predict function returns N x d predictions
        x : numpy
            d x N numpy array of features
        y : numpy
            N element groundtruth vector
    Returns:
        float : mean squared error
    '''

    # Computes the mean squared error
    predictions = model.predict(x)
    mse = skmetrics.mean_squared_error(predictions, y)
    return mse


'''
Implementation of our gradient descent optimizer for ridge regression
'''
class GradientDescentOptimizer(object):

    def __init__(self, learning_rate):
        self.__momentum = None
        self.__learning_rate = learning_rate

    def __compute_gradients(self, w, x, y, lambda_weight_decay):
        '''
        Returns the gradient of the mean squared loss with weight decay

        Args:
            w : numpy
                d x 1 weight vector
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            lambda_weight_decay : float
                weight of weight decay

        Returns:
            numpy : 1 x d gradients
        '''

        # TODO: Implements the __compute_gradients function
        #add bias
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)

        gradients = np.zeros(x.shape)
        

        for n in range(x.shape[1]):
            x_n = np.expand_dims(x[:, n], axis=1)

            prediction = np.matmul(w.T, x_n)
            # f(w) = 1/N sum_n^N (w.T.x^n - y^n)^2 + lambda/N * w.T.w
            #f'(w) = 1/N sum_n^N 2 * (w.T.x^n - y^n) \nabla (w^T x^n - y^n) + 2 * lambda/N .w
            #f'(w) = 1/N sum_n^N 2 * (w.T.x^n - y^n) x^n + 2*lambda/N * w

            gradient = 2 *(prediction - y[n]) * x_n 
            gradients[:, n] = np.squeeze(gradient)

        #lamdaterm : 2*lambda/N * w
        lambda_term = 2 * lambda_weight_decay / x.shape[1] * w
        gradient = np.mean(gradients, axis=1, keepdims=True) + lambda_term

        return gradient

    def __cube_root_decay(self, time_step):
        '''
        Computes the cube root polynomial decay factor t^{-1/3}

        Args:
            time_step : int
                current step in optimization

        Returns:
            float : cube root decay factor to adjust learning rate
        '''

        # TODO: Implement cube root polynomial decay factor to adjust learning rate
        cube_poly = time_step ** (-1.0 / 3.0)
       
        return cube_poly

    def update(self,
               w,
               x,
               y,
               optimizer_type,
               lambda_weight_decay,
               beta,
               batch_size,
               time_step):
        '''
        Updates the weight vector based on

        Args:
            w : numpy
                1 x d weight vector
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            optimizer_type : str
                'gradient_descent',
                'momentum_gradient_descent',
                'stochastic_gradient_descent',
                'momentum_stochastic_gradient_descent'
            lambda_weight_decay : float
                weight of weight decay
            beta : str
                momentum discount rate
            batch_size : int
                batch size for stochastic and momentum stochastic gradient descent
            time_step : int
                current step in optimization

        Returns:
            numpy : 1 x d weights
        '''

        # TODO: Implement the optimizer update function

        if self.__momentum is None:
            self.__momentum = np.zeros_like(w)

        if optimizer_type == 'gradient_descent':

            # TODO: Compute gradients
            
            gradients = self.__compute_gradients(w, x, y, lambda_weight_decay)

            # TODO: Update weights
            # w^(t + 1) = w^(t) - alpha \nabla loss(w^(t))
            w = w - self.__learning_rate * gradients

            return w

        elif optimizer_type == 'momentum_gradient_descent':

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x, y, lambda_weight_decay)
            
            # TODO: Compute momentum

            # v^(t) = beta * v^(t-1) + (1-beta)
            # v^(t) = beta * v^(t-1) + (1-beta) * gradients
            #here v(self.__momentum) is the momentum
            self.__momentum = beta * self.__momentum + (1- beta) * gradients

            # TODO: Update weights

            #w^(t+1) = w^(t) - alpha(learning rate) * v^(t) 
            w = w - (self.__learning_rate * self.__momentum)

            return w

        elif optimizer_type == 'stochastic_gradient_descent':

            # TODO: Implement stochastic gradient descent
            N = x.shape[1]
            # TODO: Sample batch from dataset
            idx = np.random.randint(0, N, batch_size)
            x_b = x[:, idx]
            y_b = y[idx]
            
            # TODO: Compute gradients
            #1/|B| sum_b^B 2 * (w.T.x^b - y^b) x^b + 2*lambda/|B|* w
            gradients = self.__compute_gradients(w, x_b, y_b, lambda_weight_decay)

            # TODO: Compute cube root decay factor and multiply by learning rate
            cube_decay_factor = self.__cube_root_decay(time_step)

            eta = cube_decay_factor * self.__learning_rate

            # TODO: Update weights
            #w^(t + 1) = w^(t) - alpha * gradient (gradient of l_b(w ^(t)))
            w = w - (eta * gradients)

            return w

        elif optimizer_type == 'momentum_stochastic_gradient_descent':

            # TODO: Implement momentum stochastic gradient descent
            N = x.shape[1]

            # TODO: Sample batch from dataset
            idx = np.random.randint(0, N, batch_size)
            x_b = x[:, idx]
            y_b = y[idx]

            # TODO: Compute gradients
            #1/|B| sum_b^B 2 * (w.T.x^b - y^b) x^b + 2*lambda/|B|* w
            gradients = self.__compute_gradients(w, x_b, y_b, lambda_weight_decay)

            # TODO: Compute momentum
            
            # v^(t) = beta * v^(t-1) + (1-beta) gradients
            self.__momentum = beta * self.__momentum + (1- beta) * gradients

            # TODO: Compute cube root decay factor and multiply by learning rate
            cube_decay_factor = self.__cube_root_decay(time_step)
            eta = cube_decay_factor * self.__learning_rate

            # TODO: Update weights
            w = w - (eta * gradients)

            return w


'''
Implementation of our Ridge Regression model trained using gradient descent variants
'''
class RidgeRegressionGradientDescent(object):

    def __init__(self):
        # Define private variables
        self.__weights = None
        self.__optimizer = None
    
    def plot_loss_optimizer(self, x, y, x_label, y_label, color, optimizer_type):
        '''
        Plots the losses over time steps
        Args:
            x : float
            steps
            y : float
            losses
            optimizer_type : str
                'gradient_descent',
                'momentum_gradient_descent',
                'stochastic_gradient_descent',
                'momentum_stochastic_gradient_descent'
            x_label : str
                time steps
            y_label: str
                losses
            color : str
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.suptitle('Losses of {} over time steps'.format(optimizer_type))
        

    def fit(self,
            x,
            y,
            optimizer_type,
            learning_rate,
            t,
            lambda_weight_decay,
            beta,
            batch_size):
        '''
        Fits the model to x and y by updating the weight vector
        using gradient descent variants

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            optimizer_type : str
                'gradient_descent',
                'momentum_gradient_descent',
                'stochastic_gradient_descent',
                'momentum_stochastic_gradient_descent'
            learning_rate : float
                learning rate
            t : int
                number of iterations to train
            lambda_weight_decay : float
                weight of weight decay
            beta : str
                momentum discount rate
            batch_size : int
                batch size for stochastic and momentum stochastic gradient descent
        '''

        # TODO: Implement the fit function

        # TODO: Initialize weights
        #intialize the weights (d +1 , 1)
        self.__weights = np.zeros([x.shape[0] +1, 1])
        self.__weights[0] = 1.0

        # TODO: Initialize optimizer
        self.__optimizer = GradientDescentOptimizer(learning_rate)
         
        #lists to store the time steps and losses
        losses = []
        steps = []

        for time_step in range(1, t + 1):

            # TODO: Compute loss function
            loss, loss_data_fidelity, loss_regularization = self.__compute_loss(x, y, lambda_weight_decay)

            if (time_step % 500) == 0:
                print('Step={:5}  Loss={:.4f}  Data Fidelity={:.4f}  Regularization={:.4f}'.format(
                    time_step, loss, loss_data_fidelity, loss_regularization))
                losses.append(loss)
                steps.append(time_step)

            # TODO: Update weights
            w_t = self.__optimizer.update(
                self.__weights,
                x,
                y,
                optimizer_type,
                lambda_weight_decay,
                beta,
                batch_size,
                time_step)
            self.__weights = w_t   

        #plot the graphs for loss of each optimizer 
        self.plot_loss_optimizer(
            x=steps,
            y=losses,
            x_label='Time Steps',
            y_label='Losses',
            color='green',
            optimizer_type=optimizer_type)

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : N element vector
        '''
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)

        predictions = np.zeros([x.shape[1]])

        # TODO: Implements the predict function
        for n in range(x.shape[1]):
            x_n = np.expand_dims(x[:, n], axis=1)
            prediction = np.matmul(self.__weights.T, x_n)
            predictions[n] = prediction

        return predictions

    def __compute_loss(self, x, y, lambda_weight_decay):
        '''
        Returns the gradient of the mean squared loss with weight decay

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            lambda_weight_decay : float
                weight of weight decay

        Returns:
            float : loss
            float : loss data fidelity
            float : loss regularization
        '''

        # TODO: Implements the __compute_loss function
        N = x.shape[1]
        
        #Add a bias term to achieve (d +1, N) from (d, N)
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)
        
        #loss of Ridge Regression
        # l(w) = 1/N sum_n^N (w.T.x^n - y^n)^2 + lambda/N w.T.w
        #loss fidelity = 1/N sum_n^N (w.T.x^n - y^n)^2
        #loss regularization = lambda/N w.T.w

        df_loss = [] #stores the data_fidelity_losses
        
        for n in range(N):
            x_n = np.expand_dims(x[:, n], axis=1)
            prediction = np.matmul(self.__weights.T, x_n)
            data_fidelity = (prediction - y[n]) ** 2
            df_loss.append(data_fidelity)

        
        loss_data_fidelity = np.mean(df_loss)
        loss_regularization = np.squeeze((lambda_weight_decay / N) * np.matmul(self.__weights.T, self.__weights))

        loss = loss_data_fidelity + loss_regularization
        
            
        return loss, loss_data_fidelity, loss_regularization


if __name__ == '__main__':

    # Loads dataset with 80% training, 10% validation, 10% testing split
    data = skdata.load_diabetes()
    x = data.data
    y = data.target

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

    x_train, x_val, x_test = \
        x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = \
        y[train_idx], y[val_idx], y[test_idx]

    # Initialize polynomial expansion

    poly_transform = skpreprocess.PolynomialFeatures(degree=2)

    # Compute the polynomial terms needed for the data
    poly_transform.fit(x_train)

    # Transform the data by nonlinear mapping
    x_train = poly_transform.transform(x_train)
    x_val = poly_transform.transform(x_val)
    x_test = poly_transform.transform(x_test)

    lambda_weight_decay = 0.1

    '''
    Trains and tests Ridge Regression model from scikit-learn
    '''

    # Trains scikit-learn Ridge Regression model on diabetes data
    ridge_scikit = RidgeRegression(alpha=lambda_weight_decay)
    ridge_scikit.fit(x_train, y_train)
    print('Results on using scikit-learn Ridge Regression model')
    # Test model on training set
    scores_mse_train_scikit = score_mean_squared_error(
        ridge_scikit, x_train, y_train)
    print('Training set mean squared error: {:.4f}'.format(scores_mse_train_scikit))

    # Test model on validation set
    scores_mse_val_scikit = score_mean_squared_error(
        ridge_scikit, x_val, y_val)
    print('Validation set mean squared error: {:.4f}'.format(scores_mse_val_scikit))

    # Test model on testing set
    scores_mse_test_scikit = score_mean_squared_error(
        ridge_scikit, x_test, y_test)
    print('Testing set mean squared error: {:.4f}'.format(scores_mse_test_scikit))

    '''
    Trains and tests our Ridge Regression model trained using gradient descent variants
    '''

    # Optimization types to use
    optimizer_types = [
        'gradient_descent',
        'momentum_gradient_descent',
        'stochastic_gradient_descent',
        'momentum_stochastic_gradient_descent'
    ]

    # TODO: Select learning rates for each optimizer
    learning_rates = [0.1, 1.0, 1.2, 0.9]

    # TODO: Select number of steps (t) to train
    T = [5000, 10000, 18000, 15000]

    # TODO: Select beta for momentum (do not replace None)
    betas = [None, 0.9, None, 0.7]

    # TODO: Select batch sizes for stochastic and momentum stochastic gradient descent (do not replace None)
    batch_sizes = [None, None, 312, 290]

    # TODO: Convert dataset (N x d) to correct shape (d x N)
    x_train = np.transpose(x_train , axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1, 0))
    x_test = np.transpose(x_test, axes=(1, 0))

    print('Results on using Ridge Regression using gradient descent variants'.format())

    hyper_parameters = \
        zip(optimizer_types, learning_rates, T, betas, batch_sizes)

    for optimizer_type, learning_rate, t, beta, batch_size in hyper_parameters:

        # Conditions on batch size and beta
        if batch_size is not None:
            assert batch_size <= 0.90 * x_train.shape[1]

        if beta is not None:
            assert beta >= 0.05

        # TODO: Initialize ridge regression trained with gradient descent variants
        ridge_grad_var = RidgeRegressionGradientDescent()

        print('Fitting with {} using learning rate={:.1E}, t={}'.format(optimizer_type, learning_rate, t))

        # TODO: Train ridge regression using gradient descent variants
        ridge_grad_var.fit(
            x=x_train,
            y=y_train,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            t=t,
            lambda_weight_decay=lambda_weight_decay,
            beta=beta,
            batch_size=batch_size)

        # TODO: Test model on training set
        score_mse_grad_descent_train = score_mean_squared_error(ridge_grad_var, x_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_grad_descent_train))

        # TODO: Test model on validation set
        score_mse_grad_descent_val = score_mean_squared_error(ridge_grad_var, x_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_grad_descent_val))

        # TODO: Test model on testing set
        score_mse_grad_descent_test = score_mean_squared_error(ridge_grad_var, x_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_grad_descent_test))
    plt.show()
   
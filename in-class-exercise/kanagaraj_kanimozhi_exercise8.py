import numpy as np
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets as skdata


'''
Name: kanagaraj, kanimozhi

Summary:

1. Intialized visualized the data without PCA(principle component analysis) by truncating the last dimension
2. Intialized the Principle component analysis class for k =3 and k = 2
    a. Visualized the data after PCA in 3 and 2 diemsions, to check how PCA helps in visualizing the data easily
3. Intialized the rinciple component analysis class for k = 4,3,2,1
4. under the project_to_subsapce function
    a. we center the data by subtracting the mean (B=X-mean)
    b. calculate the covariance with center data (C = 1/N-1.B.T.B)
    c. we get the weights from the fetch weight function:
        a. where we claculate the eigen-vector and eigen-values using lingalg.eg of covariance
        b. sort the eigen-values (decending order)
        c. we sort eigen-vectors based on eigen-values
        d. which gives us W and we intialize our weights as W
    d.we get the subscape Z = BW
5. We get the data back by Z.w.T+mean (here, W.T is same as taking inverse of W)
6. calculate the MSE between the original data and data we got back after PCA
7. plot the mean for reconstructed data.


'''


def plot_scatters(X, colors, labels, markers, title, axis_names, plot_3d=False):
    '''
    Creates scatter plot

    Args:
        X : list[numpy]
            list of numpy arrays (must have 3 dimensions for 3d plot)
        colors : list[str]
            list of colors to use
        labels : list[str]
            list of labels for legends
        markers : list[str]
            list of markers to use
        axis_names : list[str]
            names of each axis
        title : str
            title of plot
        plot_3d : bool
            if set, creates 3d plot, requires 3d data
    '''

    # Make sure data matches colors, labels, markers
    assert len(X) == len(colors)
    assert len(X) == len(labels)
    assert len(X) == len(markers)

    # Make sure we have right data type and number of axis names
    if plot_3d:
        assert X[0].shape[1] == 3
        assert len(axis_names) == 3
    else:
        assert X[0].shape[1] == 2
        assert len(axis_names) == 2

    fig = plt.figure()
    fig.suptitle(title)

    if plot_3d:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.set_zlabel(axis_names[2])

        for x, c, l, m in zip(X, colors, labels, markers):
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c, label=l, marker=m)
            ax.legend(loc='best')
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])

        for x, c, l, m in zip(X, colors, labels, markers):
            ax.scatter(x[:, 0], x[:, 1], c=c, label=l, marker=m)
            ax.legend(loc='best')


'''
Implementation of Principal Component Analysis (PCA) for dimensionality reduction
'''
class PrincipalComponentAnalysis(object):

    def __init__(self, k):
        # Number of eigenvectors to keep
        self.__k = k

        # Mean of the dataset
        self.__mean = None

        # Linear weights or transformation to project to lower subspace
        self.__weights = None

    def __center(self, X):
        '''
        Centers the data to zero-mean

        Args:
            X : numpy
                N x d feature vector

        Returns:
            numpy : N x d centered feature vector
        '''

        # TODO: Center the data
        
        # B = X - mu
        #other way to calculate mean is
        '''x = X.T
        for n in range(x.shape[1]):
            x_n = np.expand_dims(x[:, n], axis=1)
            self.__mean = np.mean(x_n,axis=1)'''
        
        self.__mean = np.mean(X.T, axis=1)
        
        B = X - self.__mean
        
        return B

    def __covariance_matrix(self, X):
        '''
        Computes the covariance matrix of a feature vector

        Args:
            X : numpy
                N x d feature vector

        Returns:
            numpy : d x d covariance matrix
        '''

        # TODO: Compute the covariance matrix
        # C = 1/N-1BT B
        B = self.__center(X)
        C = np.cov(B.T)

        return C

    def __fetch_weights(self, C):
        '''
        Obtains the top k eigenvectors (weights) from a covariance matrix C

        Args:
            C : numpy
                d x d covariance matrix

        Returns:
            numpy : d x k eigenvectors
        '''

        # TODO: Obtain the top k eigenvectors

        # Make sure that k is lesser or eqaul to d
        assert self.__k <= C.shape[0]

        #Eigen decomposition : V^{-1} C V = \sigma(S)
        #d is D i slides and k is small d
        S, V = np.linalg.eig(C)

        #S is singular values
        # sort them in decending order
        #and we care about the position if new ordering
        #we will use np.argsort in ascending order
        #we want decending order so we reverse it
        #orde is indice sof eigen values, which coreesponds to eigenvectors
        order = np.argsort(S)[::-1]

        #select the top k eigen vectors from V (dxd)
        #V[:, order] rearrages V from largest to smallest based on S
        # now W is d x k , this is our latent vaible that we want to learn
        W = V[:, order][:, 0:self.__k]

        return W

    def project_to_subspace(self, X):
        '''
        Project data X to lower dimension subspace using the top k eigenvectors

        Args:
            X : numpy
                N x d covariance matrix
            k : int
                number of eigenvectors to keep

        Returns:
            numpy : N x k feature vector
        '''

        # TODO: Computes transformation to lower dimension and project to subspace

        # 1. Center your data
        B = self.__center(X)

        # 2. Computer co-variance matrix
        C = self.__covariance_matrix(X)

        #3. Find the weights that can take us from d to k dimensions
        #ans set them to self.__weights memebe variable
        self.__weights = self.__fetch_weights(C)

        #4. Project X down to k dimensions using weights (W) to yield Z
        Z = np.matmul(B, self.__weights)
       
        # 5. Return Z
        #print(Z.shape)
        return Z

    def reconstruct_from_subspace(self, Z):
        '''
        Reconstruct the original feature vector from the latent vector

        Args:
            Z : numpy
                N x k latent vector

        Returns:
            numpy : N x d feature vector
        '''

        # TODO: Reconstruct the original feature vector
        #print(Z.shape)
        
        Z = np.matmul(Z, self.__weights.T) + self.__mean
        #print(Z.shape)
        return Z


if __name__ == '__main__':

    # Load the iris dataset 150 samples of 4 dimensions
    iris_dataset = skdata.load_iris()
    X_iris = iris_dataset.data
    y_iris = iris_dataset.target

    # Initialize plotting colors, labels and markers for iris dataset
    colors_iris = ('blue', 'red', 'green')
    labels_iris = ('Setosa', 'Versicolour', 'Virginica')
    markers_iris = ('o', '^', '+')

    #TODO: Visualize iris dataset by truncating the last dimension
    
    #Iris dataset is(150, 4), so this will yield the last dimension (150, 3)
    X_iris_trunc = X_iris[:, 0:3]
    
    #Find evry sample of X that belogs to class 0, 1, 2 seprately
    X_iris_trunc_class_split = [
        #This will grab (N_class_0, 3)
        X_iris_trunc[np.where(y_iris == 0)[0], :],
        #This will grab (N_class_1, 3)
        X_iris_trunc[np.where(y_iris == 1)[0], :],
        #This will grab (N_class_2, 3)
        X_iris_trunc[np.where(y_iris == 2)[0], :]
    ]

    #Together N_class_0, N_class_1, N_class_2 = N

    plot_scatters(
        X=X_iris_trunc_class_split,
        colors=colors_iris,
        labels=labels_iris,
        markers=markers_iris,
        title='Iris Dataset truncated by last dimension',
        axis_names=['x1', 'x2', 'x3'],
        plot_3d=True
    )

    
    # TODO: Initialize Principal Component Analysis instance for k = 3

    pca_3 = PrincipalComponentAnalysis(k=3)
    
    Z = pca_3.project_to_subspace(X_iris)

    # TODO: Visualize iris dataset in 3 dimension
    z_iris_3d = Z[:, 0:3]
    z_iris_3d_class_split = [
        z_iris_3d[np.where(y_iris == 0)[0], :],
        z_iris_3d[np.where(y_iris == 1)[0], :],
        z_iris_3d[np.where(y_iris == 2)[0], :]
    ]

    plot_scatters(
        X=z_iris_3d_class_split,
        colors=colors_iris,
        labels=labels_iris,
        markers=markers_iris,
        title='Iris Dataset Projected to 3D',
        axis_names=['PC1', 'PC2', 'PC3'],
        plot_3d=True
    )
    
    # TODO: Initialize Principal Component Analysis instance for k = 2
    pca_2 = PrincipalComponentAnalysis(k=2)
    Z = pca_2.project_to_subspace(X_iris)

    # TODO: Visualize iris dataset in 2 dimensions
    z_iris_2d = Z[:, 0:2]
    z_iris_2d_class_split = [
        z_iris_2d[np.where(y_iris == 0)[0], :],
        z_iris_2d[np.where(y_iris == 1)[0], :],
        z_iris_2d[np.where(y_iris == 2)[0], :]
    ]

    plot_scatters(
        X=z_iris_2d_class_split,
        colors=colors_iris,
        labels=labels_iris,
        markers=markers_iris,
        title='Iris Dataset Projected to 2D',
        axis_names=['PC1', 'PC2'],
        plot_3d=False
    )
    

    # Possible number of eigenvectors to keep
    K = [4, 3, 2, 1]

    # MSE scores to keep track of loss from compression
    mse_scores = []

    for k in K:
        # TODO: Initialize PrincipalComponentAnalysis instance for k
        pca = PrincipalComponentAnalysis(k=k)

        # TODO: Project the data to subspace
        sub_space = pca.project_to_subspace(X_iris)

        # TODO: Reconstruct the original data
        reconstruct_origin_data = pca.reconstruct_from_subspace(sub_space)

        # TODO: Measures mean squared error between original data and reconstructed data
        mse_score = skmetrics.mean_squared_error(X_iris, reconstruct_origin_data)

        # Save MSE score
        mse_scores.append(mse_score)

    # Creat plot for MSE for reconstruction
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('Iris Dataset Reconstruction Loss')

    ax.plot(K, mse_scores, marker='o', color='b', label='MSE')
    ax.legend(loc='best')
    ax.set_xlabel('k')
    ax.set_ylabel('MSE')

    # Show plots
    plt.show()

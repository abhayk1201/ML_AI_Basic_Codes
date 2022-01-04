# Abhay Kumar (kumar95)
# CS540 HW5: Linear Regression 
import numpy as np
from matplotlib import pyplot as plt
import csv


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    INPUT: 
        filename - a string representing the path to the csv file.
    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    with open(filename, 'r') as f:
        bodyfat_csv = list(csv.reader(f))
    header = bodyfat_csv[0]
    bodyfat_data = bodyfat_csv[1:]
    for rows in bodyfat_data:
        rows.pop(0)  # ignore the "IDNO" column 
        for i in range(len(header)-1):
            rows[i] = float(rows[i])
    dataset = np.array(bodyfat_data)
    return dataset



def print_stats(dataset, col):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on.  For example, 1 refers to density.
    takes the dataset as produced by the previous function and prints several statistics about a 
    column of the dataset; does not return anything
    RETURNS: None
    """
    colm_data = dataset.transpose()[col]
    num_points = len(colm_data)
    col_mean  = sum(colm_data)/num_points
    col_std = (1 / (num_points - 1) * sum((colm_data - col_mean)**2) )**(0.5)
    print(num_points)
    print('{:.2f}'.format(col_mean))
    print('{:.2f}'.format(col_std))


def regression(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        mse of the regression model
    """
    pred = np.zeros(len(dataset))
    for i in range(len(cols)):
        prod = dataset.transpose()[cols[i]] * betas[i+1]
        pred += prod
    pred += betas[0]
    mse = (1/len(pred)) * sum((pred - dataset.transpose()[0])**2)
    return mse


def gradient_descent(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
    RETURNS:
        An 1D array of gradients
    """
    n = len(dataset)
    # will return sum(beta_i * x_i)
    temp = np.dot((dataset.transpose()[cols]).transpose(), betas[1:]) +  betas[0] - dataset.transpose()[0].transpose() 
    grads = np.zeros(len(betas))
    for i in range(len(betas)):
        if i == 0:
            grads[0] += (2/n)*  sum(temp)
        else:
            grads[i] += (2/n)*  sum(temp* dataset.transpose()[cols[i-1]].transpose())
    return  grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
   INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate
    RETURNS:
        None   # order: T, mse, beta0, beta1, beta8
    """
    for i in range(1, T+1):
        print(i, end = ' ')
        grad = gradient_descent(dataset, cols, betas)
        betas = betas - eta * grad
        mse = regression(dataset, cols, betas)
        print('{:.2f}'.format(mse), end = ' ')
        for i in range(len(betas)):
            if i < len(betas)-1:
                print('{:.2f}'.format(betas[i]), end = ' ') 
            else:
                print('{:.2f}'.format(betas[i]), end = '\n') 


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    a = np.ones((len(dataset),1))
    b = dataset.transpose()[cols].transpose()
    X = np.concatenate((a, b), axis=1)
    #print(X[100])
    y = dataset.transpose()[0].transpose()
    betas =  np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)),X.transpose()), y)
    mse = regression(dataset, cols, np.array(betas))
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = np.array(compute_betas(dataset, cols)[1:])  #remove mse
    result = np.dot(betas, np.array([1]+features)) #add 1 to features
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise
    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    noise = np.random.normal(0, sigma, X.shape[0])
    
    data_1 = np.zeros((X.shape[0],2))
    data_2 = np.zeros((X.shape[0],2))
    for i in range(len(X)):
        data_1[i][0] = betas[0] + np.dot(betas[1:], X[i]) + np.random.normal(0, sigma)
        data_1[i][1] = X[i]
        data_2[i][0] = alphas[0] + np.dot(alphas[1:], np.square(X[i])) + np.random.normal(0, sigma)
        data_2[i][1] = X[i]
    return data_1, data_2


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = np.random.randint(-100,101,(1000,1))
    betas = np.array([10,15])
    alphas = np.array([1,1])
    sigma_range = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
    mse_lin = []
    mse_quad = []
    #plt.figure()
    for sigma in sigma_range:
        syth_data_lin, syth_data_quad =  synthetic_datasets(betas, alphas, X, sigma)
        mse_lin.append(compute_betas(syth_data_lin, [1])[0])
        mse_quad.append(compute_betas(syth_data_quad, [1])[0])

    plt.title('Linear regression for linear and quadratic synthetic data ')
    plt.yscale('log')    
    plt.xscale('log')
    plt.plot(sigma_range, mse_lin, '-o', label='Linear synthetic Dataset')
    plt.plot(sigma_range, mse_quad, '-o', label='Quadratic synthetic Dataset')
    plt.ylabel('MSE (Log scale)')
    plt.xlabel('Different settings of sigma (Log scale)')
    plt.legend();
    plt.savefig('mse.pdf')
    #plt.show() 


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()

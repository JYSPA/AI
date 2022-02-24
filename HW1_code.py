import numpy as np
import matplotlib.pyplot as plt
# 1.prepare data
data = np.loadtxt(fname='dataset/data.txt', skiprows=19, usecols=range(1, 17))


# 2. Feature normalization

## Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


## Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


## Calculate min and max for each column
minmax = dataset_minmax(data)
# Normalize columns
normalize_dataset(data, minmax)
# print(np.mean(data[:,0]))
## split the data into training data and testing data
train = data[:48, :]
x_train = train[:, :-1]
y_train = train[:, -1].reshape(48, 1)
test = data[48:, :]
x_test = test[:, :-1]
y_test = test[:, -1].reshape(12, 1)
print(f"The shape of training data is{train.shape}, and the shape of testing data is {test.shape}.")

# 2. Feature appending
one = np.ones([len(x_train), 1], dtype=float)
x_train = np.append(x_train, one, axis=1)
one_test = np.ones([len(x_test), 1], dtype=float)
x_test = np.append(x_test, one_test, axis=1)


# Hypothetical function  h(x)
def predict(X, W):
    return np.dot(X, W)


# loss function
def loss(X, Y, W, l2_penality):
    Y_pred = predict(X, W)
    # print(len(Y_pred - Y),len(W))
    obj = np.dot((Y_pred - Y).T, (Y_pred - Y)) / (2 * len(Y_pred - Y)) + l2_penality * np.dot(W.T, W) / (2 * len(W))
    return obj


def calc_full_gradient(X, Y, W, l2_penality):
    Y_pred = predict(X, W).reshape(48, 1)
    # calculate gradients
    # a = (l2_penality/16) * W.reshape(16,1)
    dW = 1 / len(Y) * X.T.dot(Y_pred - Y) + (l2_penality / len(W)) * W.reshape(16, 1)
    # print(len(Y),len(W))
    return dW


def RidgeRegression(X, x_test, Y, y_test, W, l2_penality, alpha, eplo):
    '''
    Gradient descent for solving logistic regression
    Inputs:
        x: a (n,d+1) ndarray
        y: a (n,) ndarray
        alpha: a scalar
        max_epoch: an integer, the maximal epochs
    Return:
        w: a (d+1,) ndarray
        objvals: a list of objective values for all epochs
    '''

    objvals = []  # store the objective values
    mse_test = []
    k = 0
    criteria = 1000
    while criteria > eplo:
        J_k1 = loss(X, Y, W, l2_penality)
        W -= alpha * calc_full_gradient(X, Y, W, l2_penality)
        J_k = loss(X, Y, W, l2_penality)
        criteria = np.abs(J_k1 - J_k) * 100 / J_k1
        # print(J_k1,J_k,criteria)
        objvals.append(loss(X, Y, W, l2_penality))
        Y_pred = predict(x_test, W)
        mse = np.dot((y_test - Y_pred).T, (y_test - Y_pred)) / (2 * len(Y_pred))
        # print(len(Y_pred))
        mse_test.append(mse)
        k = k + 1
    print(k)

    return w, objvals, mse_test


# Model training
# Initiate w
d = x_train.shape[1]
w = np.zeros(d).reshape(16, 1)

w_gd, objvals_gd, mse_ridge_test = RidgeRegression(x_train, x_test, y_train, y_test, w, 2, 0.01, 0.001)
a = np.array(objvals_gd).reshape(-1, 1)
b = np.array(mse_ridge_test).reshape(-1, 1)


# loss function
def loss_lasso(X, Y, W, l2_penality):
    Y_pred = predict(X, W)
    # print(len(Y_pred - Y),len(W))
    obj = np.dot((Y_pred - Y).T, (Y_pred - Y)) / (2 * len(Y_pred - Y)) + l2_penality * sum(np.abs(W)) / (2 * len(W))
    return obj


def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1


vec_sign = np.vectorize(sign)


def calc_full_gradient_lasso(X, Y, W, l2_penality):
    Y_pred = predict(X, W).reshape(48, 1)
    # calculate gradients
    # a = (l2_penality/16) * W.reshape(16,1)

    # W_list = []
    # for i in range(len(W)):
    # W[i] = sign(W[i])

    dW = 1 / len(Y) * X.T.dot(Y_pred - Y) + (l2_penality / (2 * len(W))) * vec_sign(W)
    # print(vec_sign(W))
    # print(X.T.dot(Y_pred - Y),(l2_penality/(2*len(W))))
    return dW


def LassoRegression(X, x_test, Y, y_test, W, l2_penality, alpha, eplo):
    '''
    Gradient descent for solving logistic regression
    Inputs:
        x: a (n,d+1) ndarray
        y: a (n,) ndarray
        alpha: a scalar
        max_epoch: an integer, the maximal epochs
    Return:
        w: a (d+1,) ndarray
        objvals: a list of objective values for all epochs
    '''

    objvals = []  # store the objective values
    mse_test = []
    k = 0
    criteria = 1000
    while criteria > eplo:
        J_k1 = loss_lasso(X, Y, W, l2_penality)
        W -= alpha * calc_full_gradient_lasso(X, Y, W, l2_penality)
        J_k = loss_lasso(X, Y, W, l2_penality)
        criteria = np.abs(J_k1 - J_k) * 100 / J_k1
        # print(J_k1,J_k,criteria)
        objvals.append(loss_lasso(X, Y, W, l2_penality))
        Y_pred = predict(x_test, W)
        mse = np.dot((y_test - Y_pred).T, (y_test - Y_pred)) / (2 * len(Y_pred))
        # print(len(Y_pred))
        mse_test.append(mse)
        k = k + 1

    return w, objvals, mse_test


# Model training
# Initiate w
d = x_train.shape[1]
w = np.zeros(d).reshape(16, 1)

w_gd_lasso, objvals_gd_lasso, mse_test_lasso = LassoRegression(x_train, x_test, y_train, y_test, w, 0.3, 0.01, 0.001)



def mse_test(w, x, y):
    loss = np.dot(x, w) - y
    outcome = np.dot(loss.T, loss) / (2 * len(loss))
    # print(len(loss))
    return outcome[0][0]

# Visualize the objective values
def plot_fig(input1, input2, label):
    plt.plot(np.array(input1).reshape(-1,1), color = 'blue', label = label)
    plt.plot(np.array(input2).reshape(-1,1), color = 'red', label ='MSE test')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


plot_fig(objvals_gd, mse_ridge_test, "Ridge Regression")

plot_fig(objvals_gd_lasso, mse_test_lasso, "Lasso Regression")


ridge_mse = mse_test(w_gd, x_test, y_test)
lasso_mse = mse_test(w_gd_lasso, x_test, y_test)
print(ridge_mse)
print(f"The squared loss on the test data for Section 1.3 is {ridge_mse}.")
print(f"The squared loss on the test data for Section 1.4 is {lasso_mse}.")

print(
    f"The number of elements in w whose absoluate value is smaller than 0.01 for Section 1.3 is {(sum(np.abs(w_gd) < 0.01))[0]}.")
print(
    f"The number of elements in w whose absoluate value is smaller than 0.01 for Section 1.4 is {(sum(np.abs(w_gd_lasso) < 0.01))[0]}.")







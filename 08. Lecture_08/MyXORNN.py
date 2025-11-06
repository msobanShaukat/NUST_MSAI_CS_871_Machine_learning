# In this code I am going to write my first NN in Python. I am going to use the code from Internet for this.


import numpy as np
#import time

# The following is a function definition of the sigmoid function, which is the type of non-linearity
# chosen for this neural net. It is not the only type of non-linearity that can be chosen, but is has nice
# analytical features and is easy to teach with. In practice, large-scale deep learning systems
# use piecewise-linear functions because they are much less expensive to evaluate. These functions are called
# REctified Linear Units (ReLU).
def sigmoid(x, deriv=False): 
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))


# We are going to generate the input data, and remember the x0 column

#input data
X = np.array([  [0, 0],
                [0, 1],
				[1, 0],
				[1, 1]])

# Appending a vector of 1s for X0
X = np.append(np.ones([4,1]), X, 1)

# The output of the exclusive OR function follows. 
y = np.array([[0],
             [1],
             [1],
             [0]])

# The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.

np.random.seed(1)

# This is the alpha value that we will use
learning_rate = 1.0

# Now we intialize the weights to random values. Theta1 are the weights between the input layer (l1) and the hidden layer (l2).  It is a 3x3 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=3). 
# Theta2 are the weights between the hidden layer (l2) and the output layer (l3). It is a 3x1 matrix because there are 3 nodes in the hidden layer and one output. Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value.
Theta1 = 2*np.random.random((3,3)) - 1  # 3x3 matrix of weights ((2 inputs + 1 bias) x 3 nodes in the hidden layer)
Theta2 = 2*np.random.random((3,1)) - 1  # 3x1 matrix of weights. (3 nodes x 1 output) - no bias term in the hidden layer.

# Getting the current time
#t0 = time.clock()

# This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases. 
#training step
for j in range(60000):
    
    # FORWARD PROPAGATION
    a1 = X
    a2 = sigmoid(np.dot(a1, Theta1))
    a3 = sigmoid(np.dot(a2, Theta2))
    
    # BACK PROPAGATION of ERRORs using the chain rule. 
    l3_error = y - a3

    if(j % 1000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("Iteration: %d,  Error: %f " % (j, np.mean(np.abs(l3_error)),))
        #print ("Iteration: %d,  Error: %f, Time: %fs " % (j, np.mean(np.abs(l3_error)), time.clock() - t0 ))

    l3_delta = l3_error*sigmoid(a3, deriv=True)
    l2_error = l3_delta.dot(Theta2.T)
    l2_delta = l2_error * sigmoid(a2, deriv=True)
    
    #update weights
    Theta2 += a2.T.dot(l3_delta)
    Theta1 += a1.T.dot(l2_delta)
    
print("Output after training")
print(a3)


# Generating the Test Data
n_X = np.random.binomial(1, 0.5, [10, 2])
# appending the column of ones for X0
n_X = np.append(np.ones([10,1]), n_X, 1)

print("New Test Data")
print(n_X)

# FORWARD PROPAGATION
a1 = n_X
a2 = sigmoid(np.dot(a1, Theta1))
a3 = sigmoid(np.dot(a2, Theta2))

# Printing the Thresholded output
print("The predicted outputs are:")
#print(a3)
print((a3 > 0.5).astype(int))

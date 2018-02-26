import numpy as np
##change file name:
data = np.genfromtxt('file', delimiter=',')
X = data[:,0:data.shape[1]-1]
X = np.append(np.ones(X.shape[0]).reshape(X.shape[0],1),X,axis=1)
y = data[:,data.shape[1]-1]
m,n = X.shape
##constants
alpha = .3
ill = 1000
lam = 0
##change nodes:
nodes_h1 = 20
nodes_h2 = 20
nodes_output = 1

##init weights
theta1 = np.random.rand(n,nodes_h1)
theta2 = np.random.rand(nodes_h1+1,nodes_h2)
theta3 = np.random.rand(nodes_h2+1,nodes_output)

##def func
def sigmoid(z):
    return 1.0/(1.0+np.exp(-1*z))

def sigdif(z):
    return z*(1-z)

def grad(d,z):
    return np.dot(z.T,d)

def cost(X):
    a2 = sigmoid(np.dot(X,theta1))
    a2 = np.append(np.ones(a2.shape[0]).reshape(a2.shape[0], 1),a2, axis=1)
    a3 = sigmoid(np.dot(a2,theta2))
    a3 = np.append(np.ones(a3.shape[0]).reshape(a3.shape[0], 1),a3, axis=1)
    output = sigmoid(np.dot(a3,theta3))
    return - 1/m * ( np.dot(y.T,np.log(output))+np.dot((1-y.T),np.log(1-output))+ lam/(2*m) * ( np.sum(theta2[:,1:]**2)+np.sum(theta1[:,1:]**2)+np.sum(theta3[:,1:]**2) ) )

def train_NN(X,theta1,theta2,theta3):
    ##forw_prop
    a2 = sigmoid(np.dot(X,theta1))
    a2 = np.append(np.ones(a2.shape[0]).reshape(a2.shape[0], 1),a2, axis=1)
    a3 = sigmoid(np.dot(a2,theta2))
    a3 = np.append(np.ones(a3.shape[0]).reshape(a3.shape[0], 1),a3, axis=1)
    output = sigmoid(np.dot(a3,theta3))
    ##back_prob
    del4 = output - y
    del3 = np.dot(del4,theta3.T) * sigdif(a3)
    del3 = del3[:,1:]
    del2 = np.dot(del3,theta2.T) * sigdif(a2)
    del2 = del2[:,1:]
    theta1 = theta1 - (alpha/m * grad(del2,X) + lam/m * theta1)
    theta2 = theta2 - (alpha/m * grad(del3,a2) + lam/m * theta2)
    theta3 = theta3 - (alpha/m * grad(del4,a3) + lam/m * theta3)
    output = sigmoid(np.dot(a3,theta3))
    return theta1,theta2,theta3,output

##Running_algo
for i in range(ill):
    theta1,theta2,theta3,output = train_NN(X,theta1,theta2,theta3)
    print(cost(X))

import numpy as np
class NeuralNetwork:
    def __init__(self,X,y,alpha,lamb,iteration,nodes_h1,nodes_h2,nodes_output):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.lamb = lamb
        self.iteration = iteration
        self.nodes_h1 = nodes_h1
        self.nodes_h2 = nodes_h2
        self.nodes_output = nodes_output
        self.m,self.n = X.shape
        self.theta1 = np.random.rand(self.n,nodes_h1)
        self.theta2 = np.random.rand(nodes_h1+1,nodes_h2)
        self.theta3 = np.random.rand(nodes_h2+1,nodes_output)
    def Activationfunc(self,z):
        return 1.0/(1.0+np.exp(-1*z))
    def Activationfuncdiff(self,z):
        return z*(1-z)
    def gradient(self,d,z):
        return np.dot(z.T,d)
    def ForwardProp(self):
        a2 = self.Activationfunc(np.dot(self.X,self.theta1))
        a2 = np.append(np.ones(a2.shape[0]).reshape(a2.shape[0], 1),a2, axis=1)
        a3 = self.Activationfunc(np.dot(a2,self.theta2))
        a3 = np.append(np.ones(a3.shape[0]).reshape(a3.shape[0], 1),a3, axis=1)
        output = self.Activationfunc(np.dot(a3,self.theta3))
        return a2,a3,output
    def Backprob(self):
        a2,a3,output = self.ForwardProp()
        del4 = output - self.y
        del3 = np.dot(del4,self.theta3.T) * self.Activationfuncdiff(a3)
        del3 = del3[:,1:]
        del2 = np.dot(del3,self.theta2.T) * self.Activationfuncdiff(a2)
        del2 = del2[:,1:]
        return del4,del3,del2
    def Cost(self):
        _,_,output = self.ForwardProp()
        return - 1/self.m * ( np.dot(self.y.T,np.log(output))+np.dot((1-self.y.T),np.log(1-output))+ self.lamb/(2*self.m) * ( np.sum(self.theta2[:,1:]**2)+np.sum(self.theta1[:,1:]**2)+np.sum(self.theta3[:,1:]**2) ) )
    def train_NN(self):
        a2,a3,output = self.ForwardProp()
        del4,del3,del2 = self.Backprob()
        self.theta1 = self.theta1 - (self.alpha/self.m * self.gradient(del2,self.X) + self.lamb/self.m * self.theta1)
        self.theta2 = self.theta2 - (self.alpha/self.m * self.gradient(del3,a2) + self.lamb/self.m * self.theta2)
        self.theta3 = self.theta3 - (self.alpha/self.m * self.gradient(del4,a3) + self.lamb/self.m * self.theta3)
        return self.theta1,self.theta2,self.theta3
    def RunningGradientDecent(self):
        for i in range(self.iteration):
            self.theta1,self.theta2,self.theta3 = self.train_NN
        return self.theta1,self.theta2,self.theta3

class Output:
    def __init__(self,X,theta1,theta2,theta3):
        self.X = X
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
    def Activationfunc(self,z):
        return 1.0/(1.0+np.exp(-1*z))
    def ForwardProp(self):
        a2 = self.Activationfunc(np.dot(self.X,self.theta1))
        a2 = np.append(np.ones(a2.shape[0]).reshape(a2.shape[0], 1),a2, axis=1)
        a3 = self.Activationfunc(np.dot(a2,self.theta2))
        a3 = np.append(np.ones(a3.shape[0]).reshape(a3.shape[0], 1),a3, axis=1)
        output = self.Activationfunc(np.dot(a3,self.theta3))
        return output

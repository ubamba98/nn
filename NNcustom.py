import numpy as np
class NeuralNetwork:
    def __init__(self,X,y,layer=[10,10,1],iteration=100,alpha=0.01):
        self.X = X
        self.y = y
        self.layer = layer
        self.alpha = alpha
        self.iteration = iteration
        self.m,self.n = X.shape
        self.layer.insert(0,self.X.shape[1])
        self.X = np.c_[np.ones(X.shape[0]),X]
        self.theta = [(2*np.random.rand(layer[i] + 1, layer[i + 1]) - 1) for i in range(len(layer)-1)]
    def Activationfunc(self,z):
        return 1.0/(1.0+np.exp(-1*z))
    def Activationfuncdiff(self,z):
        return z*(1-z)
    def gradient(self,d,z):
        return np.dot(z.T,d)
    def forwordProp(self):
        a = []
        temp = self.Activationfunc(np.dot(self.X,self.theta[0]))
        temp= np.append(np.ones(temp.shape[0]).reshape(temp.shape[0], 1) ,temp, axis=1)
        a.append(temp)
        for i in range(len(self.layer) - 3):
            temp = self.Activationfunc(np.dot(a[i],self.theta[i+1]))
            temp= np.append(np.ones(temp.shape[0]).reshape(temp.shape[0], 1) ,temp, axis=1)
            a.append(temp)
        output = self.Activationfunc(np.dot(a[len(self.layer)-3],self.theta[len(self.layer)-2]))
        return a,output
    def Backprob(self):
        a,output = self.forwordProp()
        del_ = []
        del__ = output - self.y.T
        del_.append(del__)
        for i in range(len(self.layer)-2):
            del__ = np.dot(del_[i],self.theta[len(self.layer)-i-2].T) * self.Activationfuncdiff(a[len(self.layer)-i-3])
            del__ = del__[:,1:]
            del_.append(del__)
        return del_
    def Cost(self):
        _,output = self.forwordProp()
        return - 1/self.m * ( np.dot(self.y,np.log(output))+np.dot((1-self.y),np.log(1-output)))
    def train_NN(self):
        a,output = self.forwordProp()
        del_ = self.Backprob()
        self.theta[0]= self.theta[0]- (self.alpha/self.m * self.gradient(del_[len(self.layer)-2],self.X))
        for i in range(len(self.layer)-2):
            self.theta[i+1]= self.theta[i+1]- (self.alpha/self.m * self.gradient(del_[len(self.layer)-3-i],a[i]))
        return self.theta
    def RunningGradientDecent(self):
        for i in range(self.iteration):
            self.theta = self.train_NN()
        _,output = self.forwordProp()
        return output,self.theta
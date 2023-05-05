# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S.Sham Rathan
RegisterNumber: 212221230093 
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter=',')
X = data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
X_plot= np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta, X, y):
  h=sigmoid(np.dot(X, theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J, grad
  
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
  
def gradient(theta, X,y):
  h = sigmoid(np.dot(X, theta))
  grad = np.dot(X.T, h-y) / X.shape[0]
  return grad
  
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res=optimize.minimize(fun=cost, x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
### 1. Array Value of x
![image](https://user-images.githubusercontent.com/93587823/236507877-35c59a2a-616c-4951-8103-0d6574a47d17.png)

### 2. Array Value of y
![image](https://user-images.githubusercontent.com/93587823/236507908-af10152f-9bb2-40c3-ae24-498431486b3b.png)

### 3. Exam 1 - score graph
![image](https://user-images.githubusercontent.com/93587823/236507962-b4b9c12f-349b-4c85-9fd9-14e356f4e21c.png)

### 4. Sigmoid function graph
![image](https://user-images.githubusercontent.com/93587823/236508017-57280571-6b88-4dbb-acc9-5ba227f2f653.png)

### 5. X_train_grad value
![image](https://user-images.githubusercontent.com/93587823/236508592-80fb0071-ff43-4522-987e-1febe60662d8.png)


### 6 Y_train_grad value
![image](https://user-images.githubusercontent.com/93587823/236508544-12a2d236-d7d5-4571-8568-9ced67d33cae.png)

### 7. Print res.x
![image](https://user-images.githubusercontent.com/93587823/236508485-ca756407-267a-4b31-9b67-4c9f58d4c341.png)

### 8. Decision boundary - graph for exam score
![image](https://user-images.githubusercontent.com/93587823/236508398-f8e97649-359f-4d38-acc7-c311386730a1.png)

### 9. Proability value 
![image](https://user-images.githubusercontent.com/93587823/236511072-0eaf4b5e-2a45-4d52-8baf-c278a159d910.png)


### 10. Prediction value of mean
![image](https://user-images.githubusercontent.com/93587823/236508304-fefe13bc-91c9-4920-b700-410dd7ec589b.png)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


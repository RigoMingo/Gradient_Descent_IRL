# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:03:30 2022

@author: uhh443
"""
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

#%%
# State and Control Effort penalizing matrices, typically constants, deploid in
# the local cost function. Tells the algorithm how to go about controlling the
# states/control input.

Q = np.array([[1000, 0], [0, 1000]])
R = np.array([[100]])

numberOfStates = Q.shape[0]
numberOfControls = R.shape[1]

# Intsead of inputing a single sample to the system and looping it to make it
# learn, we add more patterns to make it learn faster so it learns how
# different samples react and adjust learning accordingly

numberOfPatterns = 10000

# Maximum magnitude of the states and control input

stateDomain = 1.5
controlDomain = 20

# DeltaT, it will be used to turn the infinite time system into a discrete time
# system and give it steps for ease of algorithms: k, k+1, k+2

deltaT = 0.01

# Discount Factor, how much will the future cost impact impact the
# instantaneous cost

gamma = 0.998

# Max number of times the loop will repeat,
# could be less if diverged/error is small enough

maxLooping = 300

# Time that will be quantized by the quatization constant,
# used for online evaluation of learned network

maxQuantizedTime = int(20/deltaT)



#%%
# Basis functions, a polynomial used, along with the weights, for
# function approximation

def phi(x1, x2, u):
    
    # Change dimension when changing the amount of basis functions
    # Type is object to apease sympy, if used for calculation purposes
    # use ".astype(float)" to change format
    
    out = np.zeros((17, np.size(x1)), dtype=object)

    x1 = np.array(x1)
    x2 = np.array(x2)
    u = np.array(u)
    
    out[0,:] = 1
    
    out[1,:] = x1
    out[2,:] = x2
    out[3,:] = u
    
    out[4,:] = x1**2
    out[5,:] = x2**2
    out[6,:] = u**2
    out[7,:] = x1 * x2
    out[8,:] = x1 * u
    out[9,:] = x2 * u
    
    out[10,:] = x1**3 
    out[11,:] = x2**3 
    # out[12,:] = x1**2 * x2 
    out[12,:] = x1**2 * u 
    out[13,:] = x2**2 * u
    # out[15,:] = x2**2 * x1 
    out[14,:] = x1 * x2 * u  
    out[15,:] = u**2 * x1 
    out[16,:] = u**2 * x2 
    # out[19,:] = u**3 
    
    # u cannot be cubic as the derivative would be squared,
    #solution could have imaginary component
    
    return out

# The drift dynamics of the system have been discretized below, F & G

def F(x):
    x1 = np.array(x[0,:])
    x2 = np.array(x[1,:])

    # outF = np.zeros((numberOfStates, np.size(x1)))
    f = np.array([
        x2, 
        ( (1-x1**2) * x2 - x1) 
    ])

    # outF[0,:] = x1 + deltaT * x2
    # outF[1,:] = x2 + deltaT * ( (1-x1**2) * x2 - x1) 

    return x + deltaT*f
    
G = deltaT * np.array([[0], [1]])

# clean shape, Python needs 2D arrays

def cs(x):
    return x.reshape(x.shape[0], 1)
#%%
# This are the coefficients for the basis functions used in function 
# approximation

numberOfNeurons = np.size(phi(1,1,1))
weights = np.random.randn(numberOfNeurons, 1)

# Random States and Controls used for training in their respective domain

xPatterns = stateDomain * (2 * np.random.rand(numberOfStates,numberOfPatterns) - 1)
uPatterns = controlDomain * (2 * np.random.rand(numberOfControls, numberOfPatterns) - 1)

# Derivative of the basis function with respect to u, used to update the policy
# and is solved through symbolically.

x1, x2, u = sp.symbols('x1,x2,u')
basisPhi = sp.Matrix(phi(x1, x2, u))
derivePhi = sp.diff(basisPhi, u)

# PHI is the value the basis functions output when xPatterns and 
# uPatterns are used as input

PHI = phi(xPatterns[0,:], xPatterns[1,:], uPatterns).astype(float)

weightHistory = np.zeros((numberOfNeurons, maxLooping))

ERROR = np.zeros((1, maxLooping))

for k in range(maxLooping):
    
    xNext = F(xPatterns) + G * uPatterns
    
    weightSym = sp.Matrix(weights)
    derEqual0 = weightSym.dot(derivePhi)
    solutionU = sp.solve(derEqual0, u)
    uNextFunction = sp.lambdify([x1,x2], solutionU, 'numpy')

    uNext = np.array(uNextFunction(xNext[0,:], xNext[1,:]))
    
    futureCost = (weights.T @ phi(xNext[0,:], xNext[1,:], uNext).astype(float)).reshape(1, numberOfPatterns)
    
    cost = (0.5 * np.sum((Q @ xPatterns**2 ), axis = 0)
            + 0.5 * np.sum((R @ uPatterns**2), axis = 0) + gamma * futureCost)

    oldWeights = weights
    weightHistory[:, k] = oldWeights.reshape(numberOfNeurons,)
    
    # Least squares method to find the next set of weights, Ax = B
    
    A = PHI @ PHI.T
    B = PHI @ cost.T
    weights = np.linalg.solve(A, B)
    
    ERROR[:, k] = np.linalg.norm(weights - oldWeights, 2)
    
    print("Iteration {} | Error = {}".format(k+1, ERROR[:, k]))
    
    if np.isnan(weights).any == True:
        print("The weights have diverged")
        break

#%%
X = np.zeros((numberOfStates, maxQuantizedTime))
X[:, 0] = np.array([[1], [1]]).reshape(2,)

U = np.zeros((numberOfControls, maxQuantizedTime-1))

for t in range(maxQuantizedTime-1):
    U[:, t] = np.array(uNextFunction(X[0,t], X[1,t]))
    X[:, t+1] = (F(cs(X[:,t])) + G * U[:,t]).reshape((2,))
    

Time = np.linspace(0, maxQuantizedTime, maxQuantizedTime) * deltaT
timeControl = np.linspace(0, maxQuantizedTime, maxQuantizedTime-1) * deltaT
iterationX = np.linspace(0, maxLooping, maxLooping)

plt.subplot(2,1,1)
plt.title('States')
plt.plot(Time, X[0,:])
plt.xlabel('Time')
plt.ylabel('Position [$x_1$]')
plt.grid(visible=True)

plt.subplot(2,1,2)
plt.plot(Time, X[1,:])
plt.xlabel('Time')
plt.ylabel('Velocity [$x_2$]')
plt.grid(visible=True)

plt.figure(num=2)
plt.title('Control')
plt.plot(timeControl, U[0,:])
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.grid(visible=True)
plt.show()

plt.figure(num=3)
plt.title('Weights')
for l in range(numberOfNeurons):
    plt.plot(iterationX, weightHistory[l,:])
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.grid(visible=True)
plt.show()

plt.figure(num=4)
plt.title('Error')
plt.plot(iterationX, ERROR[0,:])
plt.xlabel('Iteration')
plt.grid(visible=True)
plt.show()

#%%
# Saving some values for training in the notebook

x_train = np.zeros((numberOfStates, U.shape[1]))
for lol in range(U.shape[1]):
    x_train[:, lol] = X[:, lol]
u_train = U

np.savetxt('Train_States.txt', x_train, fmt='%f')
np.savetxt('Train_Controls.txt', u_train, fmt='%f')

#%%
train = np.concatenate((x_train, u_train))
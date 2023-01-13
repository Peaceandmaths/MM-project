import numpy as np
import matplotlib.pyplot as plt
from copy import copy

#Parameter setting for the model simulation
k = 100 # carrying capacity
a = 1 # birth rate of prey
b = 0.1 # death rate of prey due to predation
c = 0.02 # factor that describes how many eaten preys give birth to a new predator
d = 0.5 # natural death rate of predator
D_u = 0.1 # diffusion term for prey
D_v = 0.2 # diffusion term for predator

#Discrete time and space
dt = 0.1 #time step
end_time = 50 #total time
NT = int(end_time/dt) #number of time step
t = np.arange(0, end_time+dt, dt) # time postion

dx = 0.5 #space step
L = 100 # length of the spetial segment
NX = int(L/dx) # number of steps
x = np.arange(0, L+dx, dx) # space position

# Create prey and predator grids
n = len(x)
m = len(t)
u = np.zeros((n,m))
v = np.zeros((n,m))

# Periodic Boundary Condtion
u[0,:] = u[-1,:]
v[0,:] = v[-1,:]

# Inital Configuration
u0=20*np.exp(-0.01*(x-45)**2)
v0=10*np.exp(-0.02*(x-10)**2)

u[:,0] = u0
v[:,0] = v0

#Solve PDE with explicit method (forward in time, symmetric spatial derivative)

F1 = D_u*(dt/dx**2)
F2 = D_v*(dt/dx**2)

for j in range(1,m): #loop through time steps
    for i in range(1, n-1): #loop through space steps
        u[i,j] = (a*u[i,j-1] * (1-u[i,j-1]/k)- b*v[i,j-1]*u[i,j-1])*dt + F1*(u[i+1,j-1]-2*u[i,j-1]+u[i-1,j-1]) + u[i,j-1]
        v[i,j] = (v[i,j-1]*(c*u[i,j-1]-d))*dt + F2*(v[i+1,j-1]-2*v[i,j-1]+v[i-1,j-1]) + v[i,j-1]

plt.ion()
plt.figure()

for j in range(m):
    plt.clf()
    plt.plot(x,u[:,j],color = 'xkcd:black', label = 'Prey')
    plt.plot(x,v[:,j],color = 'xkcd:red', label = 'Predator')
    plt.xlabel('distance [m]')
    plt.ylabel('prey population')
    plt.legend()
    plt.title('Prey')
    plt.pause(0.01)
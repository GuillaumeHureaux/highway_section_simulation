import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import *
from matplotlib import animation
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from copy import deepcopy


from random import randint, uniform


ms=8


fig = plt.figure() # initialise la figure
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))
line, = ax.plot([],[],'bo',ms=8) 

vi=25

def b(v,dv,dl): #fonction qui modifie la vitesse
    return vi*dv/(np.abs(dl))

def etape(pos):
    y = randint(-1,1)
    z = randint(-1,1)
    nvpos=[y,z]
    return(nvpos)


# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image

pos = [0,0]

def init():
    line.set_data([],[])
    return line,

def animate(i): 
    t = i * dt
    pos=etape(pos)
    line.set_data([0],[y,z])
    line.set_markersize(ms)
    return line,
 
ani = animation.FuncAnimation(fig, animate, frames=100,
                              interval=150, blit=True, init_func=init)

show()
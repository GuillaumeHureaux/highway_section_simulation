import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from copy import deepcopy
from matplotlib.collections import PatchCollection

##Travail à réaliser encore
"""
Rajouter des voitures au fur et a mesure et faire disparaitres d'autres
Une deuxième voie et des critères de dépassement
Améliorer la fonction b en pouvant régler individuellement le comportement des gens (vitesse maximale désirée, intervalle minimal entre deux voitures, ...)
Pour celà, il faudra sans doute pouvoir déterminer la voiture qui se trouve devant
"""

##Varirables globales

n = 7 #nombre de voitures

L = 5 #longueur des voitures (m)
d = 50 #distance initiale entre les voitures (m)
T=2 #temps minimal de parcours de l'intervalle entre deux voitures
aa= 0.5 # acceleration naturelle
bb=3 #deceleration
v0 = 50/3.6 #vitesse désirée
delta = 1.5 #coefficient de variation de l'acceleration avec la vitesse
s0 = 1 #distance minimale entre les voitures
dt = 1/20

long = n*(L+d)

##Programmes


def b(v,dv,ecart): 
    st=s0+v*T-(v*dv)/(2*np.sqrt(aa*bb))
    return aa*(1-(v/v0)**delta-(st/ecart)**2)
    

def echelle_couleur():
    def echelle(color_begin, color_end, n_vals):
        r1, g1, b1 = color_begin[0],color_begin[1],color_begin[2]
        r2, g2, b2 = color_end[0],color_end[1],color_end[2]
        degrade = []
        etendue = n_vals - 1
        for i in range(n_vals):
            alpha = 1 - i / etendue
            beta = i / etendue
            r = r1 * alpha + r2 * beta
            g = g1 * alpha + g2 * beta
            b = b1 * alpha + b2 * beta
            degrade.append((r, g, b))
        return degrade
    
    deg1=echelle([1,0,0],[1,1,0],51)
    deg2=echelle([1,1,0],[0,1,0],50)
    deg=[]
    for i in range(len(deg1)):
        deg.append(deg1[i])
    for i in range(1,len(deg2)):
        deg.append(deg2[i])

    return [deg]

echelle=echelle_couleur()

def choisir_couleur(v):
    ind=int(99.9*v/v0)
    return echelle[0][ind]

def actu_couleurs(pos):
    vit=pos[:,2]
    coul=[]
    for i in range(n):
        coul.append(choisir_couleur(vit[i]))
    return coul

def bpremier(v):
    return (v0-v-3)/10

class Vehicules:
    """init_state est un tableau [N x 3],   N est le nombre de véhicule,
                                            la première colonne représente la position du véhicule,
                                            la deuxième colonne représente le numéro de la file du véhicule
                                            la troisième colonne représente la vitesse du véhicule,
                                            """

    def __init__(self,
                 init_state = [[1, 0, 0],
                               [-0.5, 0, 0],
                               [-1, 0, 0]],
                 bounds = [-1.5*long -1, 2*long +1, -2, 2],
                 size = 0.04):
        self.init_state = np.asarray(init_state, dtype=float)
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds


    def step(self,dt):
        self.time_elapsed += dt
        for j in range(0,n):
            if j==0: 
                vtm=self.state[j,2]
                self.state[j,2] = self.state[j,2] + dt*bpremier(self.state[j,2])
                self.state[j,0] = self.state[j,0] + dt*(self.state[j,2]+vtm)/2
            else :
                vehDevant=j-1
                ecart = self.state[vehDevant,0]-self.state[j,0]
                if ecart<-long/2:
                    ecart=ecart + long
                vtm=self.state[j,2]
                self.state[j,2] = self.state[j,2] + dt*b(self.state[j,2],self.state[vehDevant,2]-self.state[j,2],ecart)
                self.state[j,0] = self.state[j,0] + dt*(self.state[j,2]+vtm)/2
        #for j in range(0,n):
            #if self.state[j,0]>long:
                #self.state[j,0]=-self.state[j,0]



##Animation
z = np.arange(0,-(n-1)*(L+d)-1,-(L+d)) #Positions initiales des voitures
z[0]=+45 #Perturbation initiale

init_state=np.zeros((n,3)) #Matrice [N*3] comme décrite dans l'entête de la class Vehicules
init_state[:,0]=z

veh = Vehicules(init_state, size=0.04)

fig = plt.figure() # initialise la figure

ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                   xlim=(-long-1, 5*long+1), ylim=(-50,50))

x=np.array((0))
y=np.array((0))
cl=choisir_couleur(0)
pos = ax.scatter([],[],animated=True)
m=['o']*n

def init():
    global veh, dt, ax, fig
    pos.set_offsets([[],[]])
    return pos,
    
def animate(i):
    global veh, dt, ax, fig
    veh.step(dt)
    #pos.set_offsets(veh.state[:,:2])
    #pos.set_array(actu_couleurs(veh.state))
    pos = ax.scatter(veh.state[:,0],veh.state[:,1],c=actu_couleurs(veh.state),marker='s')
    return pos,


ani = animation.FuncAnimation(fig, animate, frames=1,
                              interval=15, blit=True, init_func=init)


plt.show()
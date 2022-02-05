import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from copy import deepcopy
from matplotlib.collections import PatchCollection

##Travail à réaliser encore
"""
Associer à chaque point une couleur qui dépend de sa vitesse
Rajouter des voitures au fur et a mesure et faire disparaitres d'autres
Une deuxième voie et des critères de dépassement
Améliorer la fonction b en pouvant régler individuellement le comportement des gens (vitesse maximale désirée, intervalle minimal entre deux voitures, ...)
Pour celà, il faudra sans doute pouvoir déterminer la voiture qui se trouve devant
"""

##Varirables globales

n = 7 #nombre de voitures
tmax = 100 #durée de la simulation (s)
dt = 1/10 #intervalle d'actualisation (s)


L = 5 #longueur des voitures (m)
d = 50 #distance initiale entre les voitures (m)
T=5 #temps minimal de parcours de l'intervalle entre deux voitures
aa= 0.5 # acceleration naturelle
bb=3 #deceleration
v0 = 90/3.6 #♠vitesse désirée
delta = 1.5 #coefficient de variation de l'acceleration avec la vitesse
s0 = 0.01 #distance minimale entre les voitures

k=round(tmax/dt) #le nombre de valeurs de temps
tmax=k*dt

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
            degrade.append([r, g, b])
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
    return echelle[i]




class Vehicules:
    """init_state est un tableau [N x 2],   N est le nombre de véhicule,
                                            la première colonne représente la position du véhicule,
                                            la deuxième colonne représente la vitesse du véhicule
                                            la troisième collonne représente la couleur du véhicule"""

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
                vehDevant=n-1
            else :
                vehDevant=j-1
            ecart = self.state[vehDevant,0]-self.state[j,0]
            if ecart<-long/2:
                ecart=ecart + long
            vtm=self.state[j,1]
            self.state[j,1] = self.state[j,1] + dt*b(self.state[j,1],self.state[vehDevant,1]-self.state[j,1],ecart)
            self.state[j,0] = self.state[j,0] + dt*(self.state[j,1]+vtm)/2
        for j in range(0,n):
            if self.state[j,0]>long:
                self.state[j,0]=-self.state[j,0]
                self.state[j,2]= choisir_couleur(self.state[j,1])



##Animation
np.random.seed(0)

z = np.arange(0,-(n-1)*(L+d)-1,-(L+d)) #Positions initiales des voitures
z[0]=-45 #Perturbation initiale

init_state=np.zeros((n,3)) #Matrice [N*3] comme décrite dans l'entête de la class Vehicules
init_state[:,0]=z

veh = Vehicules(init_state, size=0.04)

fig, ax = plt.subplots() # initialise la figure

#ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
#                    xlim=(-long-1, long+1), ylim=(-50,50))

x=np.array((0))
y=np.array((0))
pos, = ax.scatter(x,y,c=['k']*n,animated=True)


def init():
    global veh, dt, ax, fig
    pos.set_offset([],[])
    return pos,
    
def animate(i):
    global veh, dt, ax, fig
    veh.step(dt)
    pos.set_offset(veh.state[:,0],[0]*n)
    pos.set_array(veh.state[:,3])
    return pos,


ani = animation.FuncAnimation(fig, animate, frames=100,
                              interval=15, blit=True, init_func=init)


plt.show()
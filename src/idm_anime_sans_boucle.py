import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from copy import deepcopy
from matplotlib.collections import PatchCollection
import random

##Travail à réaliser encore
"""
Conditions de dépassement et de rabattement qui dépendent de la vitesse !
Améliorer la fonction b en pouvant régler individuellement le comportement des gens (vitesse maximale désirée, intervalle minimal entre deux voitures, ...)
Ajouter un compteur du flux de la route : nbdevoit/sec, vitesse moyenne, vitesse moyenne par file etc ...
"""

##Varirables globales

n = 3 #nombre de voitures au départ

L = 5 #longueur des voitures (m)
d = 50 #distance initiale entre les voitures (m)
T= 2 #temps minimal de parcours de l'intervalle entre deux voitures
aa= 5 # acceleration naturelle
bb= 8 #deceleration
vi = 130/3.6 #vitesse désirée
delta = 1.5 #coefficient de variation de l'acceleration avec la vitesse
s0 = 40 #distance minimale entre les voitures
dt = 1/20
nf=3 # nombre de files
freq= 25 #frequence d'apparition

long = n*(L+d)

xm= 7*long



##Programmes

##Variation de la vitesse
def b(v,dv,ecart,vv):
    if abs(ecart)<25:
        ecart=25
    st=s0+v*T-(v*dv)/(2*np.sqrt(aa*bb))
    return aa*(1-(v/vv)**delta-(st/ecart)**2)

def bpremier(v,vv):
    ecart=500
    dv=1
    st=s0+v*T-(v*dv)/(2*np.sqrt(aa*bb))
    return aa*(1-(v/vv)**delta-(st/ecart)**2)

def varvitesse(pos,i,vdv):
    ecart = abs(pos[vdv,0]-pos[i,0])
    if i==vdv:
        dv=bpremier(pos[i,2],pos[i,3])
    else:
        dv=b(pos[i,2],pos[vdv,2]-pos[i,2],ecart,pos[i,3])
    return dv
    
##Couleurs
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
    res=[0,0,1]
    if v<=vi:
        ind=int(99*v/vi)
        res=echelle[0][ind]
    return res

def actu_couleurs(pos):
    vit=pos[:,2]
    coul=[]
    for i in range(len(vit)):
        coul.append(choisir_couleur(vit[i]))
    return coul

##Faire avancer les véhicules
def max_liste(liste):
    ind=0
    for j in range(len(liste)):
        if liste[j]>liste[ind]:
            ind=j
    return ind

def min_liste(liste):
    ind=0
    for j in range(len(liste)):
        if liste[j]<liste[ind]:
            ind=j
    return ind

def veh_alentours(pos,i):
    posi=pos[i,0]
    voix=pos[i,1]
    indvdv=i
    indvdr1=i
    indvdv1=i
    indvdv0=i
    indvdr0=i
    for j in range(len(pos)):
        if pos[j,1]==voix and pos[i,0]<pos[j,0]:
            if pos[j,0]<=pos[indvdv,0] or indvdv==i: 
                indvdv=j
        if pos[j,1]==voix+1:
            if pos[j,0]<posi:
                if pos[j,0]>=pos[indvdr1,0] or indvdr1==i:
                    indvdr1=j
            if pos[i,0]-pos[j,0]<0:
                if pos[j,0]<=pos[indvdv1,0] or indvdv1==i:
                    indvdv1=j
        if pos[j,1]==voix-1:
            if pos[j,0]<posi:
                if pos[j,0]>=pos[indvdr0,0] or indvdr0==i:
                    indvdr0=j
            if pos[i,0]-pos[j,0]<0:
                if pos[j,0]<=pos[indvdv0,0] or indvdv0==i:
                    indvdv0=j
            
    return indvdv, indvdv1, indvdr1, indvdv0, indvdr0


def doubler(pos,i,vdv,vdr1,vdv1,vdv0,vdr0):
    """La décision de doubler se prend si l'écart avec la voiture de devant et assez faible, ci-celle ci ne roule pas assez vite, et si la file de gauche est libre)"""
    ecart = abs(pos[vdv,0]-pos[i,0])
    ecart1dr = abs(pos[i,0]-pos[vdr1,0])
    ecart1dv = abs(pos[i,0]-pos[vdv1,0])
    ecart0dr = abs(pos[i,0]-pos[vdr0,0])
    ecart0dv = abs(pos[i,0]-pos[vdv0,0])
    res=0
    if pos[i,4]!=0:
        pos[i,4]-=1
    if pos[i,1]+1<nf and pos[vdv,2]<pos[i,3] and abs(pos[i,2]-pos[vdv,2])<0.15*pos[vdv,2] and ecart < 6*pos[i,2] and vdv!=i and pos[i,4]==0:
        if ecart1dr > pos[vdr1,2]*T or vdr1==i:
            if ecart1dv > pos[vdv1,2]*0.75*T or vdv1==i:
                res=+1
                pos[i,4]=50
    elif pos[i,1]>0 and pos[i,4]==0:
        if ecart0dr > pos[vdr0,2]*T or vdr0==i: #distance correspondant à 2 sec a la vitesse 
            if ecart0dv > pos[vdv0,2]*0.5*T or vdv0==i:
                if pos[i,2]+2<=pos[vdv0,2] or vdv0==i:
                    res=-1
                    pos[i,4]=50
    return res




class Vehicules:
    """init_state est un tableau [N x 3],   N est le nombre de véhicule,
                                            la première colonne représente la position du véhicule,
                                            la deuxième colonne représente le numéro de la file du véhicule
                                            la troisième colonne représente la vitesse du véhicule,
                                            la quatrième colonne repésente la vitesse voulue par le véhicule,
                                            la cinquième colonne représente un compteur qui l'empeche de doubler tant qu'il n'est pas à 0 (utile pour éviter les oscillations de voix)
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


    def step(self,dt,t):
#        print(self.state)
        self.time_elapsed += 1
        if self.time_elapsed%freq==freq-1 and self.state[min_liste(self.state[:,0]),0]>-long+d:
            print(self.state[min_liste(self.state[:,0]),0])
            self.state.resize(len(self.state)+1,len(self.state[0]))
            self.state[-1]=[-n*(L+d),0,attribueVitesse(1),attribueVitesse(1),0]
            print(vitesses_moyennes(self.state))
#            print(self.state)

        indadelete=[]
        
        for j in range(0,len(self.state)):
            if self.state[j,0] > 0.95*xm :
                indadelete.append(j)
            vdv, vdv1, vdr1, vdv0, vdr0=veh_alentours(self.state[:,:2],j)
#            print(j,vdv)
#            if self.time_elapsed%200==1:
#                print("véhicule",j,"sur voix ", int(self.state[j,1]),"|Directement devant :", vdv,"|A sa gauche ; devant :", vdv1,"derrière :", vdr1,"|A sa droite ; devant:", vdv0,"derrière :", vdr0)

            self.state[j,1]+=doubler(self.state,j,vdv,vdr1,vdv1,vdv0,vdr0)
#            if vdv == j:
#                vtm=self.state[j,2]
#                self.state[j,2] = abs(self.state[j,2] + dt*bpremier(self.state[j,2]))
#                self.state[j,0] = self.state[j,0] + dt*(self.state[j,2]+vtm)/2

#            else :
            vtm=self.state[j,2]
            self.state[j,2] = abs(self.state[j,2] + dt*varvitesse(self.state,j,vdv))
            self.state[j,0] = self.state[j,0] + dt*(self.state[j,2]+vtm)/2
#            print(j,self.state[j,2])
#            if self.state[j,1]!=0 and rabattement(self.state,j,vdv,vdv0,vdr0)==True and double==False:
#                self.state[j,1] -= 1
        self.state = np.delete(self.state,indadelete,axis=0)
        return self.state

##Calculs stats

def vitesse_triee(pos):
    vit=[]
    for i in range(nf):
        vit.append([])
    for i in range(len(pos)):
        vit[int(pos[i,1])]+=[pos[i,2]]
    return vit

def moyenne_liste(l):
    s=0
    if len(l)==0:
        return 0
    else :
        for i in l:
            s+=i
        return 3.6*s/len(l)

def vitesses_moyennes(pos):
    vit = vitesse_triee(pos)
    vit_moy = [0]*(len(vit)+1)
    for i in range(len(vit_moy)-1):
        vit_moy[i]=moyenne_liste(vit[i])
    vit_moy[-1]=moyenne_liste(pos[:,2])
    return vit_moy


##Animation
z = np.arange(0,-(n-1)*(L+d)-1,-(L+d)) #Positions initiales des voitures
z[0]=-25 #Perturbation initiale

init_state=np.zeros((n,5)) #Matrice [N*3] comme décrite dans l'entête de la class Vehicules
init_state[:,0]=z

def attribueVitesse(d):
    a=[]
    if d>1:
        for i in range(d):
            c=random.randint(85,105)
            a.append(c*vi/100)
        return a
    elif d==1:
        c=random.randint(85,105)
        return c*vi/100

init_state[:,3]=attribueVitesse(n)
init_state[:,2]=attribueVitesse(n)

veh = Vehicules(init_state, size=0.04)




fig = plt.figure() # initialise la figure


ax = fig.add_subplot(111, autoscale_on=False,
                   xlim=(-long-1, xm), ylim=(-0.3,nf-0.7))

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
    global pos
    veh.step(dt,i)
    #pos.set_offsets(veh.state[:,:2])
    #pos.set_array(actu_couleurs(veh.state))
    pos = ax.scatter(veh.state[:,0],veh.state[:,1],c=actu_couleurs(veh.state),marker='s')
    return pos,


ani = animation.FuncAnimation(fig, animate, frames=1,
                              interval=30, blit=True, init_func=init, save_count=1)


plt.show()
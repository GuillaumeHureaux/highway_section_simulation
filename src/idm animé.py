## simulation d'une file de voiture sur une route circulaire (périphérique)
import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import *
from matplotlib import animation


#Variation d'acceleration donné par fonction b

##Varirables globales

n =15 #nombre de voitures
tmax = 100 #durée de la simulation (s)
dt = 1/10 #intervalle d'actualisation (s)


L = 5 #longueur des voitures (m)
d = 50 #distance initiale entre les voitures (m)
T=5 #temps minimal de parcours de l'intervalle entre deux voitures
aa= 0.5 # acceleration naturelle
bb=3 #deceleration
v0 = 90/3.6 #♠vitesse désirée
delta = 1.5 #coefficient de variation de l'acceleration avec la vitesse
s0 = 2 #distance minimale entre les voitures

k=round(tmax/dt) #le nombre de valeurs de temps
tmax=k*dt

##Initialisations des liste utiles par la suite

fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

z = np.arange(0,-(n-1)*(L+d)-1,-(L+d)) #Positions initiales des voitures
long = n*(L+d)

z[0]=0


pos=np.zeros((k,n)) #Matrice de k lignes et n colonnes répertoriant toutes les positions des voitures au fur du temps
pos[0]=z #On met dans la première ligne les position des voitures initiales

vel=np.zeros((k,n)) #Matrice repertoriant les vitesses de toutes les voitures

crash = False #Accident



##Fonction de déceleration
def b(v,dv,ecart): 
    st=s0+v*T-(v*dv)/(2*np.sqrt(aa*bb))
    return aa*(1-(v/v0)**delta-(st/ecart)**2)


##Coeur du programme : boucle qui fait avancer les tutures

    pos[i]=pos[i-1]
    for j in range(0,n):
        if j==0: 
            vehDevant=n-1
        else :
            vehDevant=j-1
        ecart = pos[i-1][vehDevant]-pos[i-1][j]
        if ecart<-long/2:
            ecart=ecart + long
        vel[i][j] = vel[i-1][j] + dt*b(vel[i-1][j],vel[i-1][vehDevant]-vel[i-1][j],ecart)
        pos[i][j] = pos[i-1][j]+dt*(vel[i][j]+vel[i-1][j])/2
        ecart = pos[i][vehDevant] - pos[i][j]
        if ecart<-long/2:
            ecart=ecart + long
        if  ecart <= L :
                crash = True
                veh1=j
                veh2=vehDevant
                temps=i
                print("Accident entre",veh1,veh2,"au temps",temps)
                break
    if crash :
        break
        
        
        
        
## simulation d'une file de voiture sur une route circulaire (périphérique)
import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import *
from matplotlib import animation


#Variation d'acceleration donné par fonction b

##Varirables globales

n = 6 #nombre de voitures
tmax = 700 #durée de la simulation (s)
dt = 1/10 #intervalle d'actualisation (s)

L = 5 #longueur des voitures (m)
d = 50 #distance initiale entre les voitures (m)
T = 2 #temps minimal de parcours de l'intervalle entre deux voitures
aa = 0.10 # acceleration naturelle
bb = 20 #deceleration
v0 = 50/3.6 #♠vitesse désirée
delta = 10 #coefficient de variation de l'acceleration avec la vitesse
s0 = 10 #distance minimale entre les voitures

k=round(tmax/dt) #le nombre de valeurs de temps
tmax=k*dt

##Initialisations des liste utiles par la suite

t = np.arange(0,tmax,dt) #liste des temps
z = np.arange(0,-(n-1)*(L+d)-1,-(L+d)) #Positions initiales des voitures
long = n*(L+d)

z[0]=-20

pos=np.zeros((k,n)) #Matrice de k lignes et n colonnes répertoriant toutes les positions des voitures au fur du temps

pos[0]=z #On met dans la première ligne les position des voitures initiales

vel=np.zeros((k,n)) #Matrice repertoriant les vitesses de toutes les voitures

crash = False #Accident



##Fonction de déceleration
def b(v,dv,ecart):
    st=s0+v*T-(v*dv)/(2*np.sqrt(aa*bb))
    return aa*(1-(v/v0)**delta-(st/ecart)**2)


##Coeur du programme : boucle qui remplie les matrices
for i in range(1,k):
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
        if vel[i][j]<0:
            vel[i][j]=0.03
        pos[i][j] = pos[i-1][j]+dt*(vel[i][j]+vel[i-1][j])/2
        ecart = pos[i][vehDevant] - pos[i][j]
        if ecart<-long/2:
            ecart=ecart + long
    #     if  ecart <= L :
    #             crash = True
    #             veh1=j
    #             veh2=vehDevant
    #             temps=i
    #             print("Accident entre",veh1,veh2,"au temps",temps)
    #             break
    # if crash :
    #     break

t = dt*t
pos = pos/10
vel = 3.6*vel
lim=4000

vitmoy=0
count=0

for i in range(lim,len(vel)):
    for j in range(len(vel[i])):
        count+=1
        vitmoy+=vel[i][j]

vitmoy/=count
print(vitmoy)

plt.figure(1)
for i in range(n):
    plt.plot(t,pos[:,i],label='Voiture '+str(i+1))
plt.title('Vitesse limite : '+str(int(v0*3.6)), fontsize = 25)
plt.legend(loc=0)
plt.xlabel("Temps (s)")
plt.ylabel("Position (m)")
    
plt.figure(2)
for i in range(n):
    plt.plot(t,vel[:,i], label='Voiture '+str(i+1))
plt.title('Vitesse limite : '+str(int(v0*3.6)), fontsize = 25)
plt.plot(t[lim:],[vitmoy]*len(t[lim:]), label = 'Vitesse moy ('+str(round(vitmoy,1))+')', linestyle='-', c='black', linewidth=3)
plt.plot(t[:lim],[vitmoy]*len(t[:lim]), linestyle='--', c='black', linewidth=3)
plt.legend(loc=1)
plt.text(-5,vitmoy,str(round(vitmoy,1)), fontsize=16)
plt.xlabel("Temps (s)")
plt.ylabel("Vitesse (km.h)")

"""
plt.figure(3)
for i in range(n):
    plt.xlabel("position")
    plt.ylabel("vitesse")
    plt.plot(pos[:,i],vel[:,i])"""
    

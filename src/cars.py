## simulation d'une file de voiture sur une route circulaire (périphérique)
import numpy as np
import matplotlib.pyplot as plt
import math


#Variation d'acceleration donné par fonction b

#Varirables globales

n = 6 #nombre de voitures
tmax = 5 #durée de la simulation (s)
dt = 1/20 #intervalle d'actualisation (s)
sig = 2 #temps de réaction en nombre de pas de temps
vi = 90/3.6 #vitesse maximale autorisée (m.s-1)
D = 0.5 #reduction de vitesse de la première voiture
L = 5 #longueur des voitures (m)
d = 15 #distance initiale entre les voitures (m)

k=round(tmax/dt) #le nombre de valeurs de temps
tmax=k*dt

#Initialisations des liste utiles par la suite

t = np.arange(0,tmax,dt) #liste des temps
z = np.arange(0,-(n-1)*(L+d)-1,-(L+d)) #Positions initiales des voitures

z1 = vi*(t+D*((t+1)*np.exp(-t)-1)*np.exp(1))
v1=vi*(1-D*t*np.exp(1-t))


pos=np.zeros((k,n)) #Matrice de k lignes et n colonnes répertoriant toutes les positions des voitures au fur du temps
pos[0]=z #On met dans la première ligne les position des voitures initiales
pos[:,0]=z1 #On met dans la première colonne la position de la voiture 1

vel=vi*np.ones((k,n)) #Matrice repertoriant les vitesses de toutes les voitures
vel[:,0]=v1 #on met dans la première colonne la vitesse de v1

crash = False #Accident

#On remplie les matrices

def b(v,dv,dl): #fonction qui modifie la vitesse
    return vi*dv/(np.abs(dl))

for i in range(1,k):
    for j in range(1,n):
        if ((j-1)*sig>=(i-1)): #si le conducteur n'a pas encore commencé a freiner
            pos[i][j]=pos[i-1][j]+(i-1)*dt*vi # il continue d'avancer et garde une vitesse constante
        else:
            vel[i][j] = vel[i-1][j] + dt*b(vel[i-1-sig][j],vel[i-1-sig][j-1]-vel[i-1-sig][j],pos[i-1-sig][j-1]-pos[i-1-sig][j]) #modification de sa vitesse dans b
            pos[i][j] = pos[i-1][j]+dt*(vel[i][j]+vel[i-1][j])/2 #modification de sa position en prenant une vitesse moyenne
            if  pos[i][j] >= (pos[i][j-1]-L):
                crash = True
                veh1=j
                veh2=j-1
                temps=i
                print("Accident entre",veh1,veh2,"au temps",temps)
                break
    if crash :
        break


plt.figure(1)
for i in range(n):
    plt.plot(t,pos[:,i])
    
plt.figure(2)
for i in range(n):
    plt.plot(t,vel[:,i])

plt.figure(3)
for i in range(n):
    plt.plot(pos[:,i],vel[:,i])
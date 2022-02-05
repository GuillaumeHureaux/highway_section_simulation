import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from matplotlib.collections import PatchCollection
import random
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Portable Guillaume\\Documents\\ffmpeg-20160318-git-a7b8a6e-win64-static\\bin\\ffmpeg.exe' 

##Travail à réaliser encore
"""
Conditions de dépassement et de rabattement qui dépendent de la vitesse !
Améliorer la fonction b en pouvant régler individuellement le comportement des gens (vitesse maximale désirée, intervalle minimal entre deux voitures, ...)
Ajouter un compteur du flux de la route : nbdevoit/sec, vitesse moyenne, vitesse moyenne par file etc ...
"""
"""http://www.blog.kamisphere.fr/recensement-de-la-circulation-sur-les-routes-nationales-autoroutes/"""


##Varirables globales

n = 3 #nombre de voitures au départ

L = 5 #longueur des voitures (m)
d = 50 #distance initiale entre les voitures (m)
T= 2 #temps minimal de parcours de l'intervalle entre deux voitures
aa= 5 # acceleration naturelle
bb= 8 #deceleration
#vi = 130/3.6 #vitesse désirée
delta = 1.5 #coefficient de variation de l'acceleration avec la vitesse
s0 = 10 #distance minimale entre les voitures
dt = 1/20
#nf = 4 # nombre de files
#freq = 20 #période d'apparition en intervalle élémentaire de temps (dt)
time=0

long = n*(L+d)

xm= 1100
xmini= -n*(L+d)


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

def choisir_couleur(v,vi):
    res=[0,0,1]
    if v<=vi:
        ind=int(99*v/vi)
        res=echelle[0][ind]
    return res

def actu_couleurs(pos,v):
    vit=pos[:,2]
    coul=[]
    for i in range(len(vit)):
        coul.append(choisir_couleur(vit[i],v))
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


def doubler(pos,i,vdv,vdr1,vdv1,vdv0,vdr0,nf):
    """La décision de doubler se prend si l'écart avec la voiture de devant et assez faible, ci-celle ci ne roule pas assez vite, et si la file de gauche est libre)"""
    ecart = abs(pos[vdv,0]-pos[i,0])
    ecart1dr = abs(pos[i,0]-pos[vdr1,0])
    ecart1dv = abs(pos[i,0]-pos[vdv1,0])
    ecart0dr = abs(pos[i,0]-pos[vdr0,0])
    ecart0dv = abs(pos[i,0]-pos[vdv0,0])
    res=0
    if pos[i,4]!=0:
        pos[i,4]-=1
    if pos[i,1]+1<nf and pos[vdv,2]<pos[i,3] and abs(pos[i,2]-pos[vdv,2])<0.5*pos[vdv,2] and ecart < 6*pos[i,2] and vdv!=i and pos[i,4]==0:
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





    """init_state est un tableau [N x 5],   N est le nombre de véhicule,
                                            la première colonne représente la position du véhicule,
                                            la deuxième colonne représente le numéro de la file du véhicule
                                            la troisième colonne représente la vitesse du véhicule,
                                            la quatrième colonne repésente la vitesse voulue par le véhicule,
                                            la cinquième colonne représente un compteur qui l'empeche de doubler tant qu'il n'est pas à 0 (utile pour éviter les oscillations de voix)
                                            """



def etape(veh,dt,t,freq,nf,v):
#        print(veh)
    global time, fig
    time += 1
    vehimpl=0

    """
    if time%freq==freq-1 :
        lindmin=[-1]*nf
        for i in range(nf):
            lposmin=[]
            lposminind=[]
            for j in range(len(veh)):
                if veh[j,1]==i:
                    lposmin.append(veh[j,0])
                    lposminind.append(j)
            if len(lposminind)!=0:
                lindmin[i]=lposminind[min_liste(lposmin)]
    
        maxdesmin=[]

        for k in range(nf):
            maxdesmin.append(veh[lindmin[k],0])
        
        indmaxdesmin=max_liste(maxdesmin)
        for i in range(len(maxdesmin)):
            if maxdesmin[i]==-1:
                indmaxdesmin=i
        if veh[indmaxdesmin,0]>-(long-d):
            veh = np.vstack((veh, np.array([-n*(L+d),indmaxdesmin,attribueVitesse(1,v),attribueVitesse(1,v),0])))
            vehimpl=1
        """
    if time%freq==freq-1 and veh[min_liste(veh[:,0]),0]>-long+d:
        #veh = np.resize(len(veh)+1,len(veh[0]))
        veh = np.vstack((veh, np.array([-n*(L+d),0,attribueVitesse(1,v),attribueVitesse(1,v),0])))
    
    indadelete=[]
    
    for j in range(0,len(veh)):
        if veh[j,0] > 0.95*xm :
            indadelete.append(j)
        vdv, vdv1, vdr1, vdv0, vdr0=veh_alentours(veh[:,:2],j)
#            print(j,vdv)
#            if self.time_elapsed%200==1:
#                print("véhicule",j,"sur voix ", int(veh[j,1]),"|Directement devant :", vdv,"|A sa gauche ; devant :", vdv1,"derrière :", vdr1,"|A sa droite ; devant:", vdv0,"derrière :", vdr0)

        veh[j,1]+=doubler(veh,j,vdv,vdr1,vdv1,vdv0,vdr0,nf)
#            if vdv == j:
#                vtm=veh[j,2]
#                veh[j,2] = abs(veh[j,2] + dt*bpremier(veh[j,2]))
#                veh[j,0] = veh[j,0] + dt*(veh[j,2]+vtm)/2

#            else :
        vtm=veh[j,2]
        veh[j,2] = abs(veh[j,2] + dt*varvitesse(veh,j,vdv))
        veh[j,0] = veh[j,0] + dt*(veh[j,2]+vtm)/2
#            print(j,veh[j,2])
#            if veh[j,1]!=0 and rabattement(veh,j,vdv,vdv0,vdr0)==True and double==False:
#                veh[j,1] -= 1
    veh = np.delete(veh,indadelete,axis=0)
    return veh, len(indadelete), vehimpl

##Calculs stats

def vitesse_triee(pos,nf):
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

def vitesses_moyennes(pos,nf):
    vit = vitesse_triee(pos,nf)
    vit_moy = [0]*(nf+1)
    for i in range(len(vit_moy)-1):
        vit_moy[i]=moyenne_liste(vit[i])
    vit_moy[-1]=moyenne_liste(pos[:,2])
    flux = len(pos)*(vit_moy[-1]/3.6)/(pos[max_liste(pos[:,0]),0]-pos[min_liste(pos[:,0]),0])
    nbVhpKm = len(pos)/(pos[max_liste(pos[:,0]),0]-pos[min_liste(pos[:,0]),0])*1000
    #flux = len(pos)*(vit_moy[-1]/3.6)/(xm-xmini)
    return vit_moy, flux, nbVhpKm


##Animation

def attribueVitesse(d,v):
    a=[]
    if d>1:
        for i in range(d):
            c=random.randint(85,105)
            a.append(c*v/100)
        return a
    elif d==1:
        c=random.randint(85,105)
        return (c*v)/100


def etat_initial(v):
    z = np.arange(0,-(n-1)*(L+d)-1,-(L+d)) #Positions initiales des voitures
    #z[0]+=250 #Perturbation initiale
    #z[1]+=220
    init_state=np.zeros((n,5)) #Matrice [N*3] comme décrite dans l'entête de la class Vehicules
    init_state[:,0]=z
    #init_state[:,2]=[35/3.6,70/3.6,130/3.6]
    #init_state[:,3]=[35/3.6,70/3.6,130/3.6]
    init_state[:,2]=attribueVitesse(n,v)
    init_state[:,3]=attribueVitesse(n,v)
    return deepcopy(init_state)

veh = etat_initial(130/3.6)
voit=0

def simulation_animee(freq,nf,v):
    global veh, time, fig, voit
#    time = 0
#    veh = etat_initial()
    print(veh)

    fig = plt.figure() # initialise la figure
 
    fig.clf()
    
    ax = fig.add_subplot(111, autoscale_on=False,
                    xlim=(-long-1, xm), ylim=(-0.3,nf-0.7))

    
    x=np.array((0))
    y=np.array((0))
    cl=choisir_couleur(0,v)
    pos = ax.scatter([],[],animated=True)
    m=['o']*n
    
    
    
    def init():
        global veh, dt, ax, fig
        pos.set_offsets([[],[]])
        return pos,
        
    def animate(i):
        global pos, veh, voit
        veh, plus, impl = etape(veh,dt,i,freq,nf,v)
        #pos.set_offsets(veh.state[:,:2])
        #pos.set_array(actu_couleurs(veh.state))
        pos = ax.scatter(veh[:,0],veh[:,1],c=actu_couleurs(veh,v),marker='s')
        voit+=plus
        return pos,


    ani = animation.FuncAnimation(fig, animate, frames=500,
                                interval=20, blit=True, init_func=init, save_count=1)
    #mywriter = animation.FFMpegWriter()
    #ani.save('simulation2.mp4', writer=mywriter, fps=1, dpi=None, codec='mpeg4', bitrate=None, extra_args=None, metadata=None, extra_anim=None, savefig_kwargs=None)
    plt.show()

    return

def simulation_animee2(freq,nf,v):
    global veh, time, fig, voit
#    time = 0
#    veh = etat_initial()
    print(veh)

    fig = plt.figure() # initialise la figure
 
    fig.clf()
    
    ax = fig.add_subplot(111, autoscale_on=False,
                    xlim=(-long-1, xm), ylim=(-0.3,nf-0.7))

    
    x=np.array((0))
    y=np.array((0))
    cl=choisir_couleur(0,v)
    pos = ax.scatter([],[],animated=True)
    m=['o']*n
    
    
    
    def init():
        global veh, dt, ax, fig
        pos.set_offsets([[],[]])
        return pos,
        
    def animate(i,freq,nf,v):
        global pos, veh, voit
        veh, plus, impl = etape(veh,dt,i,freq,nf)
        #pos.set_offsets(veh.state[:,:2])
        #pos.set_array(actu_couleurs(veh.state))
        pos = ax.scatter(veh[:,0],veh[:,1],c=actu_couleurs(veh,v),marker='s')
        voit+=plus
        return pos,


    ani = animation.FuncAnimation(fig, animate, frames=500,
                                interval=20, blit=True, init_func=init, save_count=1)
    #mywriter = animation.FFMpegWriter()
    #ani.save('simulation2.mp4', writer=mywriter, fps=1, dpi=None, codec='mpeg4', bitrate=None, extra_args=None, metadata=None, extra_anim=None, savefig_kwargs=None)
    plt.show()

    return

def principal(nb,freq,nf,v):
    global veh, time
    time = 0
    veh = etat_initial(v)
    stat = []
    count=0
    vitesse=0
    fs=0
    h=0
    voit=0
    tutu=0
    nbVhpKmMoy=0
    for i in range(nb):
        veh, plus, pp = etape(veh,dt,i,freq,nf,v)
        if i%(1/dt)==0 and i>1000:
            a,f,nbVhpKm=vitesses_moyennes(veh,nf)
            fs+=f
            nbVhpKmMoy += nbVhpKm
            vitesse+=a[-1]
            count+=1
        if i>1000:
            voit+=plus
            tutu+=pp
            h+=1
        #print(voit, i)

    if count!=0:
        flux=3600*fs/count
        nbVhpKmMoy = nbVhpKmMoy/count
        vitesse=vitesse/count
    else:
        flux=None
    vph= 3600*voit/((h)*dt)
    return flux, vph, vitesse, nbVhpKmMoy

def experience(nb):
    lfreq=[200,100,60,40,20,10]
    lnf=[1,2,3,4]
    #lv=[130/3.6,90/3.6,50/3.6]
    v=36
    #lfreq=[200,100]
    #lnf=[1,2]
    #lv=[130/3.6]
    res=[]
    for nf in lnf:
        res.append([])
        for freq in lfreq:
            print(freq,nf,v)
            a,b,c,d=principal(nb,freq,nf,v)
            print(b,c,d)
            res[nf-1].append(c)
    return res
    
def afficher_resultats():
    exp=experience(3000)
    t=[360,720,1200,1600,3200,7200]
    plt.figure(1)
    print(len(exp))
    for i in range(len(exp)):
        plt.xlabel("flux désiré")
        plt.ylabel("densité de véhicules")
        plt.plot(t,exp[i])
    return

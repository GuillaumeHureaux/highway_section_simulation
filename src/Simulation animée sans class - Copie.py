import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from matplotlib.collections import PatchCollection
import random
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Portable Guillaume\\Documents\\ffmpeg-20160318-git-a7b8a6e-win64-static\\bin\\ffmpeg.exe' 

##Travail à réaliser encore
"""
Dans la fonction etape, faire en sorte que la position de la voiture qui se trouve le moins loin de chaque file soit enregistrée à chaque étape, et ainsi gagner énormément en complexité ;) 
"""
"""http://www.blog.kamisphere.fr/recensement-de-la-circulation-sur-les-routes-nationales-autoroutes/"""


##Varirables globales

n = 3 #nombre de voitures au départ

L = 5 #longueur des voitures (m)
d = 50 #distance initiale entre les voitures (m)
T= 1.5 #temps minimal de parcours de l'intervalle entre deux voitures
aa= 5 # acceleration naturelle
bb= 8 #deceleration
delta = 1.5 #coefficient de variation de l'acceleration avec la vitesse
s0 = 30 #distance minimale entre les voitures
dt = 1/20
time=0

long = n*(L+d)

xm= 1100
xmini= -n*(L+d)


##Programmes

##Variation de la vitesse
def varvitesse(pos,i,vdv):
    
    def b(v,dv,ecart,vd):
        if abs(ecart)<25:
            ecart=25
        st=s0+v*T-(v*dv)/(2*np.sqrt(aa*bb))
        return aa*(1-(v/vd)**delta-(st/ecart)**2)
    
    def bpremier(v,vv):
        return aa*(1-(v/vv)**delta)
    
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

def actu_couleurs(pos,v):
    
    def choisir_couleur(v,vi):
        res=[0,0,1]
        if v<=vi:
            ind=int(99*v/vi)
            res=echelle[0][ind]
        return res
    
    vit=pos[:,2]
    coul=[]
    for i in range(len(vit)):
        coul.append(choisir_couleur(vit[i],v))
    return coul

##Faire avancer les véhicules
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
        if ecart1dr > pos[vdr1,2]*2*T or vdr1==i:
            if ecart1dv > pos[vdv1,2]*2*T or vdv1==i:
                res=+1
                pos[i,4]=15/dt
    elif pos[i,1]>0 and pos[i,4]==0:
        if ecart0dr > pos[vdr0,2]*2*T or vdr0==i: #distance correspondant à 2 sec a la vitesse 
            if ecart0dv > pos[vdv0,2]*2*T or vdv0==i:
                if pos[i,2]+0.3<=pos[vdv0,2] or vdv0==i:
                    res=-1
                    pos[i,4]=15/dt
    return res





    """init_state est un tableau [N x 5],   N est le nombre de véhicule,
                                            la première colonne représente la position du véhicule,
                                            la deuxième colonne représente le numéro de la file du véhicule
                                            la troisième colonne représente la vitesse du véhicule,
                                            la quatrième colonne repésente la vitesse voulue par le véhicule,
                                            la cinquième colonne représente un compteur qui l'empeche de doubler tant qu'il n'est pas à 0 (utile pour éviter les oscillations de voix)
                                            """



def etape(veh,dt,t,freq,nf,v,posmin):
    """Renvoie la liste des véhicules actualisée et le nombre de voitures qui ont été supprimées"""
    global time, fig
    time += 1
    
    indadelete=[]
    
    for j in range(0,len(veh)):
        if veh[j,0] > 0.95*xm :
            indadelete.append(j)
            if posmin[veh[j,1]][1]==j:
                posmin[veh[j,1]]=[100000,0]
                #print(posmin,' ;\n\n',veh,'\n\n-----------------------\n\n')
        vdv, vdv1, vdr1, vdv0, vdr0=veh_alentours(veh[:,:2],j)
        double=doubler(veh,j,vdv,vdr1,vdv1,vdv0,vdr0,nf)
        vtm=veh[j,2]
        veh[j,2] = abs(veh[j,2] + dt*varvitesse(veh,j,vdv))
        veh[j,0] = veh[j,0] + dt*(veh[j,2]+vtm)/2
        if double!=0:
            posmin[veh[j,1]]=[veh[vdv,0],vdv]
        else:
            posmin[veh[j,1]]=[veh[j,0],j]
        veh[j,1]+=double
        if veh[j,0]<posmin[veh[j,1]][0]:
            posmin[veh[j,1]]=[veh[j,0],j]

    
    if time%freq==freq-1 : #Test si on peut ajouter un véhicule
        maxdesmin=[posmin[0,0],posmin[0,1],0] #position de la voiture, indice de la voiture, file où elle se trouve

        for k in range(nf):
            if posmin[k,0]>posmin[maxdesmin[2],0]:
                maxdesmin=[posmin[k][0],posmin[k,1],k]
        if maxdesmin[1]>=len(veh):
            print(maxdesmin,indadelete,posmin,veh)
        if veh[maxdesmin[1],0] > -(long-d):
            vit=attribueVitesse(1,veh[maxdesmin[1],2])
            veh = np.vstack((veh, np.array([-n*(L+d), maxdesmin[2], vit, attribueVitesse(1,v), 0])))
            posmin[maxdesmin[2]]=[veh[-1,0],maxdesmin[2]]

    veh = np.delete(veh,indadelete,axis=0)
    
    return veh, len(indadelete)

##Calculs stats
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

def range_vitesse_par_file(pos,nf):
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
    """Renvoie la vitesse moyenne par voix (liste), le flux de voitures en veh.sec, et le nombre de vehicules par kilomètre"""
    vit = range_vitesse_par_file(pos,nf)
    vit_moy = [0]*(nf+1) #liste des vitesses moyenne par file, le dernier représente la vitesse moyenne globale
    for i in range(len(vit_moy)-1):
        vit_moy[i]=moyenne_liste(vit[i])
    vit_moy[-1]=moyenne_liste(pos[:,2])
    flux = len(pos)*(vit_moy[-1]/3.6)/(0.95*xm-(-n*(L+d)))
    nbVhpKm = len(pos)/(0.95*xm-(-n*(L+d)))*1000
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


def etat_initial(v,nf):
    z = np.arange(0,-(n-1)*(L+d)-1,-(L+d)) #Positions initiales des voitures
    init_state=np.zeros((n,5)) #Matrice [N*5] comme décrite dans l'entête de la class Vehicules
    init_state[:,0]=z
    init_state[:,2]=attribueVitesse(n,v)
    init_state[:,3]=attribueVitesse(n,v)
    posmin = np.array([[10000,0]]*nf)
    posmin[0]=[init_state[n-1,0],n-1]
    return deepcopy(init_state),posmin

veh, posmin = etat_initial(130/3.6,2)


def simulation_animee(freq,nf,v):
    global veh, time, fig, voit
    time = 0
    veh, posmin = etat_initial(v,nf)
    #print(veh)

    fig = plt.figure() # initialise la figure
 
    fig.clf()
    
    ax = fig.add_subplot(111, autoscale_on=False,
                    xlim=(-long-1, xm), ylim=(-0.3,nf))
    
    x=np.array((0))
    y=np.array((0))

    pos = ax.scatter([],[],animated=True)
    stat = ax.text(0.02,0.87,'',transform=ax.transAxes)
    
    def init():
        global veh, dt, ax, fig
        pos.set_offsets([[],[]])
        stat.set_text('')
        return pos, stat
        
    def animate(i):
        global pos, veh, voit
        veh, plus = etape(veh,dt,i,freq,nf,v,posmin)
        pos = ax.scatter(veh[:,0],veh[:,1],c=actu_couleurs(veh,v),marker='s')
        vitmoy,flx,osef=vitesses_moyennes(veh,nf)
        #anotationy.set_yticks([i for i in range(nf)], ['Voie '+str(i+1)+'\nVit moy : '+str(int(vitmoy[i])) for i in range(nf)], size='medium')
        stat.set_text('Temps : '+str(int(time*dt))+' s\nVit moy : '+str(int(vitmoy[-1]))+' km.h\nFlux :'+str(int(3600*flx))+' veh.h')
        return pos, stat


    ani = animation.FuncAnimation(fig, animate, frames=500, 
                                interval=50, blit=True, init_func=init)
    """mywriter = animation.FFMpegWriter()
    ani.save('simulation2.mp4', writer=mywriter, fps=1, dpi=None, codec='mpeg4', bitrate=None, extra_args=None, metadata=None, extra_anim=None, savefig_kwargs=None)"""
    chaine='Flux entrant : '+str(int(3600/(freq*dt)))+'\nVitesse max : '+str(v*3.6)+' km.h'+'\nNombre de voie(s) : '+str(nf)
    plt.text(0.63,0.87,chaine,transform=ax.transAxes,bbox=dict(ec='black',fc='white'))
    plt.yticks([i for i in range(nf)], ['Voie '+str(i+1) for i in range(nf)], size='medium')
    plt.show()

    return


def principal(nb,freq,nf,v):
    """Renvoie le flux moyen, le flux total, [vitesse moyenne par file + vit moy globale], nbveh.km"""
    global veh, time
    time = 0
    veh,posmin = etat_initial(v,nf)
    stat = []
    compteur=0 #compte le nombre de fois qu'une donnée a été écrite
    vitMoy=0 #vitesse moyenne globale
    flxMoy=0 #flux moyen
    nbiter=0 #nombre d'iteration (nf-1000 à la fin mais bon ...)
    nbvoittot=0 #le nombre de voiture qui ont franchi les 1 km
    nbVhpKmMoy=0 #le nombre de voiture par km moyen
    taillemax=0
    k=0
    
    
    for i in range(nb):
        veh, voitplus = etape(veh,dt,i,freq,nf,v,posmin)
        vit,flux,nbVhpKm=vitesses_moyennes(veh,nf) #list [vitesse moy par file + vitesse moy global], flux, veh.km-1
        if i%(1/dt)==0 and i>1000: #1 fois par seconde et une fois le régime permanent établit
            flxMoy+=flux
            nbVhpKmMoy += nbVhpKm
            vitMoy+=vit[-1]
            compteur+=1
        if i>1000:
            nbvoittot+=voitplus
            nbiter+=1
        
        if i % 50==0:
            fig = plt.figure(k)
            k+=1
            ax = fig.add_subplot(111, autoscale_on=False,
                    xlim=(-long-1, xm), ylim=(-0.3,nf))
            chaine='Flux entrant : '+str(int(3600/(freq*dt)))+'\nVitesse max : '+str(v*3.6)+' km.h'+'\nNombre de voie(s) : '+str(nf)
            plt.text(0.63,0.87,chaine,transform=ax.transAxes,bbox=dict(ec='black',fc='white'))
            plt.text(0.02,0.87,'Temps : '+str(round(time*dt-0.05,2))+' s\nVit moy : '+str(int(vit[-1]))+' km.h\nFlux :'+str(int(3600*flux))+' veh.h',transform=ax.transAxes)
            plt.yticks([i for i in range(nf)], ['Voie '+str(i+1) for i in range(nf)], size='medium')
            ax.scatter(veh[:,0],veh[:,1],c=actu_couleurs(veh,v),marker='s')
        
    if compteur==0 or nbiter==0:
        print('Pas assez ditération. Merci de rentrer une valeur de nb supérieure à 2000')
        return None
    else:
        flxMoy = 3600*flxMoy/compteur
        nbVhpKmMoy = nbVhpKmMoy/compteur
        vitMoy = vitMoy/compteur
        vph = 3600*nbvoittot/((nbiter)*dt)
        return flxMoy, vph, vitMoy, nbVhpKmMoy


##Exploitation des données

def graph_voie(nb,v):
    lflx=[360,720,1800,3600,7200] #flux en veh.h-1
    #lflx=[360,720]
    #lflx=[(360+45*i) for i in range(16)]+[(1080+i*120) for i in range(6)]+[(1800+360*i) for i in range(5)]+[(3600+720*i) for i in range(6)]
    lfreq=[int(3600/(i*dt)) for i in lflx] #interval d'actualisation
    print(lfreq)
    lnf=[1,2,3,4,5]
    resflx=[]
    resvit=[]
    for nf in lnf:
        resflx.append([])
        resvit.append([])
        for freq in lfreq:
            print(freq,nf,v)
            a,b,c,d=principal(nb,freq,nf,v) #a flux moy, b flux global, c vit, d veh.km-1
            print((a+b)/2,c)
            resflx[nf-1].append((a+b)/2)
            resvit[nf-1].append(c)
    
    plt.figure(1)
    for i in range(len(lnf)):
        plt.xlabel("flux désiré")
        plt.ylabel("flux réel")
        plt.plot(lflx,resflx[i])
    
    plt.figure(2)
    for i in range(len(lnf)):
        plt.xlabel("flux réel")
        plt.ylabel("vitesse moyenne")
        plt.plot(resflx[i],resvit[i])
    
    plt.figure(3)
    for i in range(len(lnf)):
        plt.xlabel("flux déssiré")
        plt.ylabel("vitesse moyenne")
        plt.plot(lflx,resvit[i])
    
    return None

def resultats_fct(nb,v):
    #lflx=[360,720,1200,1600,2000,2400,2800,3200,4000,4800,5600,6400,7200] #flux en veh.h-1
    #lflx=[1500]
    #lflx=[(360+45*i) for i in range(16)]+[(1080+i*120) for i in range(6)]+[(1800+360*i) for i in range(5)]+[(3600+720*i) for i in range(6)]
    lfreq=[int(3600/(i*dt)) for i in lflx] #interval d'actualisation
    print(lfreq)
    lnf=[1,2,3,4,5]
    resflx=[]
    resvit=[]
    for nf in lnf:
        resflx.append([])
        resvit.append([])
        for freq in lfreq:
            print(freq,nf,v)
            a,b,c,d=principal(nb,freq,nf,v) #a flux moy, b flux global, c vit, d veh.km-1
            print((a+b)/2,c)
            resflx[nf-1].append((a+b)/2)
            resvit[nf-1].append(c)
    return resflx, resvit, lflx

def resultats_fct_vit(nb,v):
    flx=1500
    freq=int(3600/(flx*dt)) #interval d'actualisation
    nf=3
    a,b,c,d=principal(nb,freq,nf,v) #a flux moy, b flux global, c vit, d veh.km-1
    return a,b

def ecrire_fichier(k,flxr,vit,flxd):
    cflxr, cvit, cflxd = '', '', ''
    for i in range(len(flxd)):
        cflxd+=(str(flxd[i])+';')
    for i in range(len(flxr)):
        for j in range(len(flxr[0])):
            cflxr+=(str(flxr[i][j])+';')
            cvit+=(str(vit[i][j])+';')
        cflxr+='//'
        cvit+='//'
    
    ajoute=cflxd+'\n'+cflxr+'\n'+cvit+'\n\n'
    resultats = open("resultats"+str(k)+".txt","w")
    resultats.write(ajoute)
    resultats.close()
    return None

def lecture_fichier():
    resultats = open("resultats.txt","r")
    le=resultats.read()
    be = le.split('\n')
    a,b,c=be[0],be[1],be[2]
    ap=a.split(';')
    bp=b.split('//')
    cp=c.split('//')
    res0=[]
    res1=[]
    res2=[]
    for i in range(len(ap)-1):
        res0.append(float(ap[i]))
    for i in range(len(bp)-1):
        x=bp[i].split(';')
        res1.append([])
        for j in range(len(x)-1):
            res1[i].append(float(x[j]))
    for i in range(len(bp)-1):
        x=cp[i].split(';')
        res2.append([])
        for j in range(len(x)-1):
            res2[i].append(float(x[j]))
    resultats.close()
    
    return res0, res1, res2
    
def tracer():
    lflx,resflx,resvit = lecture_fichier()
    
    plt.figure(4)
    for i in range(len(resflx)):
        plt.xlabel("flux désiré")
        plt.ylabel("flux réel")
        plt.plot(lflx,resflx[i])
    
    plt.figure(5)
    for i in range(len(resflx)):
        plt.xlabel("flux réel")
        plt.ylabel("vitesse moyenne")
        plt.plot(resflx[i],resvit[i])
    
    plt.figure(6)
    for i in range(len(resflx)):
        plt.xlabel("flux désiré")
        plt.ylabel("vitesse moyenne")
        plt.plot(lflx,resvit[i])
    
    return None

nb=2500


#a,b=resultats_fct_vit(nb,130/3.6)

#print(a,b)


# cette ligne est necessaire en Python 2 uniquement
from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from copy import deepcopy

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

plt.imshow(echelle_couleur())
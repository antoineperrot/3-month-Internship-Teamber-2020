import numpy as np
from tqdm import tqdm

def longueur(trajet, DMatrix):    #durée ou km
    d = 0
    for i in range(len(trajet)-1):
        d  += DMatrix[trajet[i],trajet[i+1]]
    return d

def modification(x):
    k = np.random.randint(1,len(x)-2)
    l = np.random.randint(k+1,len(x)-1)
    y = np.hstack([x[:k], np.flip(x[k:l+1]), x[l+1:] ])
    return y

def trajet_optimal(ordre,DM,ITERATIONS_PAR_LIEU = 800):
    if len(ordre) <= 3:
        return ordre.astype(int)
    TEMPERATURE_INITIALE = 1
    TEMPERATURE_MIN = 1e-3
    POURCENTAGE_AVANT_TMIN = 1
    N_iterations = ITERATIONS_PAR_LIEU * len(ordre)
    ordre_min = ordre
    dum = DM/DM[np.ix_(ordre,ordre)].max()
    Proba = np.zeros(N_iterations)
    E = np.zeros(N_iterations)
    E[0] = longueur(ordre, dum)
    Emin = np.copy(E)
    T = TEMPERATURE_INITIALE
    decay = 1 - np.exp(np.log(TEMPERATURE_MIN/TEMPERATURE_INITIALE)/POURCENTAGE_AVANT_TMIN/N_iterations)
    for i in tqdm(range(1,N_iterations)):
            T = T*(1-decay) if T> TEMPERATURE_MIN else TEMPERATURE_MIN
            ordre_voisin = modification(ordre)
            Evoisin = longueur(ordre_voisin, dum)
            DELTA = Evoisin - E[i-1]
            Proba[i] = min(1,np.exp( - DELTA / T))
            if np.random.rand() < Proba[i]:
                E[i] = Evoisin
                ordre = np.copy(ordre_voisin)
            else :
                E[i] = E[i-1]
            if E[i] < Emin[i-1]:
                Emin[i] = E[i]
                ordre_min = ordre
            else :
                Emin[i] = Emin[i-1]
    return ordre
import pandas as pd
import matplotlib.pyplot as plt

from TOfunctions import *

df = pd.read_excel('ADRESSES_COMPLETES.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop_duplicates(subset='Adresse',keep='first',inplace=True)
df = df.loc[df['Latitude']>40].reset_index(drop=True)
plt.scatter(df['Longitude'],df['Latitude'],marker='+');

region1 = [65,32,31,82,81,12,46,9,11,34,66,30,48]  #Occitanie
region2 = [84,13,4,83,6,5]                         #PACA
region3 = [15,43,7,26,38,73,74,1,69,42,3,63]       #Auvergne Rhones-Alpes
region4 = [64,40,47,33,24,19,23,87,16,17,79]       #Nouvelle Aquitaine
secteur = region1

sel = df.loc[df['Département'].isin(secteur),].reset_index(drop=True)
sel.reset_index(drop=True,inplace=True)
print(len(sel),"adresses potentielles")

X = sel[['Latitude','Longitude']].values
DM = np.zeros((X.shape[0],X.shape[0]))
for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        DM[i,j] = np.linalg.norm(X[i,:]-X[j,:])
        
ordre = np.hstack((np.arange(X.shape[0]),np.array(0)))

ordre_optimal = trajet_optimal(ordre, DM)

plt.scatter(X[:,1],X[:,0],marker='+')
plt.plot(X[ordre_optimal,1],X[ordre_optimal,0])
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
#import gmplot
import googlemaps
API_key = 'AIzaSyBrpQ4wnAQLv-i7-YLnqxvKVjF9j83pHUo'
gmaps = googlemaps.Client(key=API_key)

def dist_vol_doiseau(LatA,LonA,LatB,LonB):
    latA = LatA*np.pi/180
    lonA = LonA*np.pi/180
    latB = LatB*np.pi/180
    lonB = LonB*np.pi/180
    RAYON_TERRE = 6_378_000 #en m   car non la terre n'est pas plate
    dist = np.arccos(np.cos(latA)*np.cos(lonA)*np.cos(latB)*np.cos(lonB) + \
    np.cos(latA)*np.sin(lonA)*np.cos(latB)*np.sin(lonB) + \
    np.sin(latA)*np.sin(latB)) * RAYON_TERRE  #
    return int(dist)

def temps_dispo(row, debut_journee = DEBUT_JOURNEE, fin_journee = FIN_JOURNEE):
    current_day = row['Date fin rdv Origine'].date()
    if current_day  == row['Date début rdv Destination'].date():
        t_dispo = row['Date début rdv Destination'] - row['Date fin rdv Origine']
    else :
        a = pd.to_datetime(str(current_day) + ' '+ str(fin_journee))
        t_dispo = a - row['Date fin rdv Origine'] if a - row['Date fin rdv Origine'] >= datetime.timedelta(0) else datetime.timedelta(0)
        current_day += datetime.timedelta(days=1)
        while current_day != row['Date début rdv Destination'].date() :
            t_dispo += pd.to_datetime(fin_journee) - pd.to_datetime(debut_journee)
            current_day += datetime.timedelta(days=1)
        b = pd.to_datetime(str(current_day) + ' '+ str(debut_journee))
        t_dispo += row['Date début rdv Destination'] - b if row['Date début rdv Destination'] - b > datetime.timedelta(0) else datetime.timedelta(0)
    return t_dispo


# Paramètres programme :
DEBUT_JOURNEE = datetime.time(8,30)
FIN_JOURNEE = datetime.time(17,30)
DUREE_CLIENTS = datetime.timedelta(hours=2)
DUREE_PROSPECTS = datetime.timedelta(hours=0.5)
MARGE_STOP = datetime.timedelta(hours=.5)
MODE_SELECTION = 1  #1,2 ou 3
RAYON_SELECTION = 8000

# Import des contacts :
df = pd.read_excel('fichier contact.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)


# Sélection adresses candidates :
if MODE_SELECTION == 1 :
    sel = df.loc[(df["Type"]=='Prospect') &
            (df['Espérance concrétisation'] > .7) & 
            (datetime.datetime.now() - df['Dernier contact'] > datetime.timedelta(days= 30*3))
              ,].reset_index(drop=True)
if MODE_SELECTION == 2 :
    sel = df.loc[(df["Type"]=='Prospect') &
            (df['Espérance concrétisation'] > .7) & 
            (datetime.datetime.now() - df['Dernier contact'] > datetime.timedelta(days= 30*3))
              ,].reset_index(drop=True)
if MODE_SELECTION == 3 :
    sel = df.loc[(datetime.datetime.now() - df['Dernier contact'] > datetime.timedelta(days= 4*30))
              ,].reset_index(drop=True)

#sel = sel.loc[sel['Département'].isin(SECTEUR),].reset_index(drop=True)
sel['Ajouté'] = False
sel['Valide'] = False

print('Mode sélection :',MODE_SELECTION)
print(len(sel),'adresses candidates')

# Import des rendez-vous fermes :
df_rdv = pd.read_excel('RDV fermes.xlsx')
df_rdv.drop('Unnamed: 0',axis=1, inplace=True)

#Geocoding des rdv fermes si nécessaire :
'''
import googlemaps
API_key = 'AIzaSyBrpQ4wnAQLv-i7-YLnqxvKVjF9j83pHUo'
gmaps = googlemaps.Client(key=API_key)  

geocodes_rdv = []
for adresse in df_rdv['Adresse'].values:
    geocode_rdv = gmaps.geocode(adresse)
    geocodes_rdv.append(geocode_rdv)

lats, lons = [], []
for geo in geocodes_rdv :
    lats.append(geo[0]['geometry']['location']['lat'])
    lons.append(geo[0]['geometry']['location']['lng'])
    
df_rdv['Latitude'] = lats
df_rdv['Longitude'] = lons
'''


# Obtention de points le long de la route :
adresses_rdv = list(df_rdv['Adresse'].values)
resultats = []
for i in range(len(adresses_rdv)-1):
    resultats.append(gmaps.directions(origin = adresses_rdv[i],destination = adresses_rdv[i+1]))

lats_circles = []
lons_circles = []
for res in resultats :
    for i in range(len(res[0]['legs'][0]['steps'])):
        lats_circles.append(res[0]['legs'][0]['steps'][i]['start_location']["lat"])
        lons_circles.append(res[0]['legs'][0]['steps'][i]['start_location']["lng"])


# Raffinage des adresses grâce aux cercles :
valides = []
dist_mins = []
for index,row in sel.iterrows():
    lat1,lon1 = sel.loc[index,['Latitude','Longitude']].values
    valide = False
    dist_min = np.inf
    for i in range(len(lons_circles)):
        dist = dist_vol_doiseau(lat1,lon1,lats_circles[i],lons_circles[i])
        if dist < dist_min:
            dist_min = dist
        if dist < RAYON_SELECTION:
            valide = True
    if valide:
        valides.append(index)
    dist_mins.append(dist_min)

sel['Distance min cercle'] = dist_mins
sel2 = pd.DataFrame.copy(sel.loc[valides]).reset_index(drop=True)
print('Avant raffinage :',len(sel))
print('Après raffinage :',len(sel2))
print('pour des rayons de',RAYON_SELECTION,'m.')
sel_avoid = sel.drop(valides,axis=0)

# Concaténation RDV fermes + sélection :
data = pd.concat([df_rdv[['Nom',"Adresse",'Type','Latitude','Longitude']],
                         sel2[['Nom','Adresse','Type','Latitude','Longitude']]]).reset_index(drop=True)

# Obtention de la distance/duration matrix :
print('Obtention de la distance/duration matrix')
resultats = []
for i in tqdm(range(len(data))):
    res = gmaps.distance_matrix(origins = [data.loc[i,'Adresse']],
                     destinations = data['Adresse'],
                      mode="driving")
    resultats.append(res)

# Remplissage distance matrix
print('Remplissage distance matrix')
DuM = np.zeros((len(data),len(data)))   #Duration Matrix
DiM = np.zeros((len(data),len(data)))   #Distance Matrix

for i in range(len(resultats)):
    res = resultats[i]
    for j in range(len(res['rows'][0]['elements'])):
        if res['rows'][0]['elements'][j]["status"] == 'OK':           
            DuM[i,j] = res['rows'][0]['elements'][j]['duration']['value'] / 60  #en minutes
            DiM[i,j] = res['rows'][0]['elements'][j]['distance']['value'] / 1000 #en km
        else :
            DuM[i,j] = np.inf
            DiM[i,j] = np.inf

DuM = np.round(DuM).astype(int)
DiM = np.round(DiM).astype(int)

# Préparation d'un dataframe "InterRDV" 
print('Préparation dataframe InterRDV')
InterRDV = pd.DataFrame()
InterRDV['Origine'] = df_rdv.loc[df_rdv.index[:-1],'Nom'].values
InterRDV['Origine Id'] = df_rdv.index[:-1]
InterRDV['Destination'] = df_rdv.loc[df_rdv.index[1:],'Nom'].values
InterRDV['Destination Id'] = df_rdv.index[1:]

InterRDV['Date fin rdv Origine'] = df_rdv.loc[df_rdv.index[:-1],'Date fin'].values
InterRDV['Date début rdv Destination'] = df_rdv.loc[df_rdv.index[1:],'Date début'].values

for index, row in InterRDV.iterrows():
    InterRDV.loc[index,'Temps trajet initial'] = DuM[row['Origine Id'],row['Destination Id']]
    InterRDV.loc[index,'Distance trajet initial (km)'] = int(DiM[row['Origine Id'],row['Destination Id']])
    InterRDV.loc[index,'Temps dispo interRDV'] = temps_dispo(row)

InterRDV['Temps trajet initial'] = pd.to_timedelta(InterRDV['Temps trajet initial'],unit='m')
InterRDV['Temps trajet supplémentaire'] = pd.to_timedelta(0,unit='h')
InterRDV['Visites'] =''
InterRDV['Prospects visités'] = 0
InterRDV['Clients visités'] = 0
InterRDV['Temps restant'] = InterRDV['Temps dispo interRDV'] \
                        - InterRDV['Temps trajet initial']\
                        - InterRDV['Temps trajet supplémentaire']\
                        - DUREE_CLIENTS * InterRDV['Clients visités'] \
                        - DUREE_PROSPECTS * InterRDV['Prospects visités']  


# Optimisation de la tournée :
print('Optimisation de la tournée :')

lieux_potentiels_restant = np.array(sel2.loc[sel["Ajouté"]==False,].index) # + len(df_rdv)
categorie_lieux_potentiels_restant = np.array((sel2['Type'] =='Prospect' )* 1)

trajets = []
lieux_ajoutes = []
for index, row in InterRDV.iterrows():
    arg_org  = InterRDV.loc[index,"Origine Id"]
    arg_dest = InterRDV.loc[index,"Destination Id"]
    trajet = [arg_org, arg_dest]   #dans la Duration matrix
    temps_restant = InterRDV.loc[index,'Temps restant']
    stop  = temps_restant < MARGE_STOP or len(lieux_potentiels_restant) == 0
    while not stop :
        devmin = np.inf
        argmin = None
        etape = None
        t_surplacemin = None
        imin = 0
        existe_candidat = False
        for i in range(len(lieux_potentiels_restant)):
            arg = lieux_potentiels_restant[i] + len(df_rdv)
            for t in range(len(trajet)-1) :
                dev = DuM[trajet[t],arg] + DuM[arg,trajet[t+1]] - DuM[trajet[t],trajet[t+1]]
                t_surplace = DUREE_PROSPECTS if categorie_lieux_potentiels_restant[i] else DUREE_CLIENTS
                possible = InterRDV.loc[index,'Temps restant'] >= datetime.timedelta(hours=dev / 60) +  t_surplace
                if possible and dev < devmin :
                    existe_candidat = True
                    argmin = lieux_potentiels_restant[i] + len(df_rdv)    #argmin dans la Duration Matrix
                    imin = i
                    etape = t
                    devmin = dev
                    categorie = 'Client' if categorie_lieux_potentiels_restant[i] == 0 else 'Prospect'
                    t_surplacemin = t_surplace
        if existe_candidat :
            lieux_ajoutes.append(imin)
            sel2.loc[imin,'Ajouté'] = True
            trajet.insert(etape+1,argmin) 
            InterRDV.loc[index,'Temps trajet supplémentaire'] += datetime.timedelta(hours=devmin / 60)
            InterRDV.loc[index,categorie+'s visités'] += 1
            InterRDV.loc[index,'Temps restant'] -= datetime.timedelta(hours=devmin / 60) + t_surplacemin
            lieux_potentiels_restant = np.delete(lieux_potentiels_restant,imin)
            categorie_lieux_potentiels_restant = np.delete(categorie_lieux_potentiels_restant,imin)
            #temps_restant = row['Temps restant']
        stop  = InterRDV.loc[index,'Temps restant'] < MARGE_STOP or len(lieux_potentiels_restant) == 0 or not existe_candidat
    if index == 0:
        trajets = trajet
    if index > 0:
        trajets = trajets + trajet[1:]
    InterRDV.loc[index,'Visites'] = str(trajet[1:-1])[1:-1]
    
InterRDV['Temps prospects'] = InterRDV['Prospects visités'] * DUREE_PROSPECTS
InterRDV['Temps clients'] = InterRDV['Clients visités'] * DUREE_CLIENTS


# Calcul des horaires de passages pour la tournée
print("Calcul des horaires de passages pour la tournée")
tournee = data.loc[trajets].reset_index(drop=True)
tournee['Id DiM'] = trajets

for index,row in tournee.iterrows():
    if row['Type'] =='RDV':
        tournee.loc[index,'Date début'] = df_rdv.loc[df_rdv['Nom']==row['Nom'],'Date début'].values
        tournee.loc[index,'Date fin'] = df_rdv.loc[df_rdv['Nom']==row['Nom'],'Date fin'].values
    else :
        duree_trajet = (np.timedelta64(DuM[tournee.loc[index-1,'Id DiM'],tournee.loc[index,'Id DiM']],'m'))
        if tournee.loc[index -1,'Date fin'].time() > FIN_JOURNEE :
            dt = tournee.loc[index -1,'Date fin'] + datetime.timedelta(days=1)
            dt = pd.to_datetime(str(dt)[:11] + str(DEBUT_JOURNEE))
            tournee.loc[index,'Date début'] = dt + duree_trajet
            t_surplace = DUREE_CLIENTS if row['Type'] =='Client' else DUREE_PROSPECTS
            tournee.loc[index,'Date fin'] = tournee.loc[index,'Date début'] + t_surplace
        elif (tournee.loc[index -1,'Date fin'] + duree_trajet).time() > FIN_JOURNEE :
            dt = tournee.loc[index -1,'Date fin']
            fj = pd.to_datetime(str(dt)[:11] + str(FIN_JOURNEE))
            duree_trajet_restant = duree_trajet
            duree_trajet_restant -= fj - dt
            dt += datetime.timedelta(days=1)
            dt = pd.to_datetime(str(dt)[:11] + str(DEBUT_JOURNEE))
            tournee.loc[index,'Date début'] = dt + duree_trajet_restant
            t_surplace = DUREE_CLIENTS if row['Type'] =='Client' else DUREE_PROSPECTS
            tournee.loc[index,'Date fin'] = tournee.loc[index,'Date début'] + t_surplace
            
        else : 
            tournee.loc[index,'Date début'] = tournee.loc[index -1,'Date fin'] + duree_trajet
            t_surplace = DUREE_CLIENTS if row['Type'] =='Client' else DUREE_PROSPECTS
            tournee.loc[index,'Date fin'] = tournee.loc[index,'Date début'] + t_surplace
tournee.to_excel('TOURNEE147.xlsx')

## Dessin de la tournée dans gmaps :
print('Dessin de la tournée')
gmap = gmplot.GoogleMapPlotter((np.max(data['Latitude']) + np.min(data['Latitude']))/2,
                               (np.max(data['Longitude']) + np.min(data['Longitude']))/2
                               ,8.5,apikey=API_key)

##En bleu clair, les adresses des clients :
gmap.scatter(data.loc[data['Type']=='Client','Latitude'].values,
             data.loc[data['Type']=='Client','Longitude'].values,
           '#3399FF', edge_width = 8,marker=False,
            symbol='+')

##En bleu foncé, les adresses des prospects :
gmap.scatter(data.loc[data['Type']=='Prospect','Latitude'].values,
             data.loc[data['Type']=='Prospect','Longitude'].values, 
           '#0000CC', edge_width = 8,marker=False,
            symbol='+')

#En rouge, les adresses des rendez-vous fermes :
#gmap.scatter(lats_rdv, longs_rdv,  
#           'r', edge_width =5,
#             label=['1dwfgwxvc','2xvSVDwvx','wfbwdfhbDF3','4sfgsdf','5cqsdcWD'])

i = 0
done = False

origin = None
destination = None
while not done :
    way_points = []
    current_row = tournee.loc[i,]
    origin = tuple([current_row['Latitude'],current_row['Longitude']])
    i += 1
    current_row = tournee.loc[i,]
    while current_row['Type'] != 'RDV':
        way_points.append( tuple([current_row['Latitude'],current_row['Longitude']]) )
        i += 1
        current_row = tournee.loc[i,]
    destination = tuple([current_row['Latitude'],current_row['Longitude']])
    
    if len(way_points) == 0 :
        gmap.directions(origin=origin,
                   destination=destination)
    else :
        gmap.directions(origin=origin,
                   destination=destination,
                       waypoints=way_points)

        
    done = i == len(tournee) - 1 

gmap.draw("lol19.html") 






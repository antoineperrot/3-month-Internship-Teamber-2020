import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Planning:
    # Création de l'objet
    def __init__(self, nom_utilisateur, id_utilisateur, plagehoraire_filename):
        self.nom_utilisateur = str(nom_utilisateur)
        self.id = int(id_utilisateur)
        self.plagehoraire_filename = plagehoraire_filename
        self.longueur_min_plage = 0.5
        self.longueur_max_plage = 3
        self.penalties = np.array([1/6,2/3,1/6])
        self.imperatifs =[]
        self.has_tasks = False
        self.has_base = False
        self.has_scores = False
        self.initialised = False
    
    # Pour changer les paramètres d'optimisation :
    def setPenalties(self, new_penalties):
        if np.sum(new_penalties) == 0 :
            self.penalties = np.array([1/3,1/3,1/3])
        else :
            self.penalties = np.array(new_penalties)/np.sum(new_penalties)
        if self.has_tasks:
            self.UpdateScores()
    
    # Ajout des tâches à planifier            
    def addTasks(self, df_tasks):
        self.tasks = df_tasks[['id morceau','Durée',"Priorité",'id tache']].values.T
        self.df_tasks = df_tasks.set_index('id morceau')
        self.total_time_tasks = self.tasks[1,:].sum()
        self.has_tasks = True
        self.min_tasks = self.tasks# Calcul de la base de planning et des potentiels initiaux

    #Créé la base d'une planning et initialise les scores si des tâches ont été ajoutées.
    def initialise(self, DATE_DEBUT, DATE_FIN):
        self.DATE_DEBUT = DATE_DEBUT
        self.DATE_FIN   = DATE_FIN
        
        self.base = CalculBasePlanning(self.plagehoraire_filename,
                                    DATE_DEBUT, DATE_FIN)

        self.temps_total_dispo = self.base['Longueur'].sum()
        if self.temps_total_dispo == 0 :
            print(f"{self.nom_proprio} n'a aucune disponibilité à la période de projection indiquée.")
        
        self.base_array = self.base[['index','Longueur']].values.T
        
        if self.has_tasks:
            self.UpdateScores()
            self.min_chronologie = self.chronologie
        
        #self.prop_temps_perdu, self.chronologie = V_TempsPerdu(self.tasks, self.base_array)
        #self.score_priorites = V_Priorites(self.tasks)

        self.temps_total_dispo = self.base['Longueur'].sum()
        self.initialised = True
        self.has_new_imp = False

    # Ajout d'impératifs à prendre en compte
    def addImperatifs(self, df_imperatifs):
        self.base = addImperatifs(self.base, df_imperatifs)
        self.imperatifs.append(df_imperatifs)
        self.base_array = self.base[['index','Longueur']].values.T
        self.temps_total_dispo = self.base['Longueur'].sum()
        if self.has_tasks :
            self.UpdateScores()

    # (maths) Met à jour les potentiels s'il y eu des changements
    def UpdateScores(self):
        self.score_temps_perdu, self.chronologie = V_TempsPerdu(self.tasks, self.base_array)
        self.score_priorites = V_Priorites(self.tasks)
        self.score_dispersion = V_Dispersion(self.tasks)
        self.score = np.vdot(self.penalties,
            [self.score_temps_perdu, self.score_priorites, self.score_dispersion])

    #Affiche les scores pour les différents indicateurs :
    def showScores(self):
        if self.has_tasks :
            print('Pourcentage temps perdu :',self.score_temps_perdu)
            print('Non respect priorités :',self.score_priorites)
            print('Score dispersion :',self.score_dispersion)
        else:
            print("Impossible d'afficher les scores, veuillez ajouter des tâches au planning.")
    
    # (maths) Propose un arrangement voisin des taches et stocke ses potentiels
    def buildVoisin(self):
        self.next_tasks = permuteTasks(self.tasks)
        self.next_score_temps_perdu, self.next_chronologie = V_TempsPerdu(self.next_tasks, self.base_array)
        self.next_score_priorites = V_Priorites(self.next_tasks)
        self.next_score_dispersion = V_Dispersion(self.next_tasks)
        self.next_score = np.vdot(self.penalties,
            [self.next_score_temps_perdu, self.next_score_priorites, self.next_score_dispersion])
    # (maths) Remplace l'arrangement courant par l'arrangement voisin
    def replace(self):
        self.tasks = self.next_tasks
        self.score_temps_perdu = self.next_score_temps_perdu
        self.chronologie = self.next_chronologie
        self.score_priorites = self.next_score_priorites
        self.score_dispersion = self.next_score_dispersion
        self.score = self.next_score
    # (maths) Stocke l'arrangement conduisant à l'énergie totale la plus faible
    def replaceMin(self):
        self.min_tasks = self.tasks
        self.min_chronologie = self.chronologie
    # (maths) Remplace l'arrangement courant par l'arrangement "minimal" rencontré
    def applyMin(self):
        self.tasks = self.min_tasks
        self.chronologie = self.chronologie
        self.UpdateScores()
    
    # (maths) Renvoie les potentiels de l'arrangement courant ou voisin selon la valeur
    # du paramètre voisin
    def getPotentiels(self, voisin=False):
        if not voisin :
            return [self.score_temps_perdu, self.score_priorites, self.score_dispersion]
        else :
            return [self.next_score_temps_perdu, self.next_score_priorites, self.next_score_dispersion]
    # Exporte le planning en fichier excel et retourne le dataframe correspondant
    def makePlanning(self,to_file=False):
        df = pd.DataFrame()
        if self.has_tasks:
            self.applyMin()
        
            plages = np.unique(self.chronologie[1,:])
            for plage in plages :

                tmp2 = np.array([self.chronologie[0,i] if self.chronologie[1,i] == plage else -1 for i in range(self.chronologie.shape[1])])
                tmp2 = tmp2[tmp2>=0]

                p = self.base.loc[[plage]]
                t = self.df_tasks.loc[tmp2]

                
                d = p['Date début'].values[0]
                

                t['Durée cumulée'] = pd.to_timedelta(t['Durée'].cumsum(),unit='h')
                t["Durée"] = pd.to_timedelta(t['Durée'],unit='h')

                t['Date fin'] = d + t['Durée cumulée']

                t['Date début'] = t['Date fin'] - t["Durée"]
                t.drop(['Durée','Durée cumulée'],axis=1,inplace=True)
                df = pd.concat([df,t])
            

            df = df[['Objet','Priorité','Date début','Date fin']]
        
        imp = pd.concat(self.imperatifs)
        
        DF = pd.concat([df,imp])
        DF.sort_values(by='Date début',inplace=True)
        DF.reset_index(inplace=True)
        DF = DF[['Objet','Priorité','Date début',"Date fin"]]
        
        ### ON RECOLLE LES MORCEAUX DE TACHES SI POSSIBLE :
        df = DF.loc[[0],]
        index_df = 0
        index_DF = 1
        while index_DF < len(DF):
            while DF.loc[index_DF,'Date début'] == df.loc[index_df,'Date fin'] and \
                  DF.loc[index_DF,'Objet']      == df.loc[index_df,'Objet']:
                
                df.loc[index_df,'Date fin'] = DF.loc[index_DF,'Date fin']
                index_DF +=1
            if index_DF < len(DF):
                df = df.append(DF.loc[[index_DF],]).reset_index(drop=True)
                index_df +=1
                index_DF +=1

        ###EXPORT :

        if to_file :
            df.to_excel(f"EDT {self.nom_utilisateur}.xlsx")
        self.result = df
        return df

    # (maths) Optimise le planning
    def Optimise(self, ITERATIONS_PAR_TACHE = 150,TEMPERATURE_INITIALE = 1,TEMPERATURE_MIN = 1e-5,show=False):
        N_ITERATIONS = self.tasks.shape[1] * ITERATIONS_PAR_TACHE
        Proba = np.zeros(N_ITERATIONS)
        E = np.zeros(N_ITERATIONS)
        E[0] = self.score
        Emin = np.copy(E)
        Vp  = np.zeros(N_ITERATIONS)
        Vtp = np.zeros(N_ITERATIONS)
        Vd  = np.zeros(N_ITERATIONS)
        Vp[0], Vtp[0], Vd[0] = self.score_priorites, self.score_temps_perdu, self.score_dispersion
        T = TEMPERATURE_INITIALE
        decay = 1 - np.exp(np.log(TEMPERATURE_MIN/TEMPERATURE_INITIALE)/N_ITERATIONS)
        for i in range(1,N_ITERATIONS):
            T = T*(1-decay) if T> TEMPERATURE_MIN else TEMPERATURE_MIN
            self.buildVoisin()
            Ey = self.next_score
            DELTA = Ey - E[i-1]
            Proba[i] = min(1, np.exp( - DELTA / T))
            if np.random.rand() < Proba[i] :
                self.replace()
                E[i] = Ey
            else :
                E[i] = E[i-1] 

            if E[i] < Emin[i-1] :
                self.replaceMin() 
                Emin[i] = E[i]
            else :
                Emin[i] = Emin[i-1]
            Vp[i], Vtp[i], Vd[i] = self.score_priorites, self.score_temps_perdu, self.score_dispersion
        if show :
            plt.figure(figsize=(16,7))
            col = 'cornflowerblue'

            ax1 = plt.subplot(121, title= "Energie totale")
            ax1.grid()
            ax1.plot(range(N_ITERATIONS),Emin  ,c='magenta',label='Emin')
            ax1.plot(range(N_ITERATIONS),E ,label='Etotale' ,c=col)
            ax1.legend(loc='upper right')
            ax1.set_xlabel('Itérations')

            '''
            ax2 = plt.subplot(122, title='Evolution probabilité acceptation (moving average)')
            ax2.grid()
            ax2.scatter(range(N_ITERATIONS),Proba,marker='.',linewidth=.5,c=col)
            ax2.set_xlabel('Itérations')
            plt.show()
            '''
            ax3 = plt.subplot(122, title='Différents potentiels')
            ax3.plot(range(N_ITERATIONS),Vp ,label='Priorités')
            ax3.plot(range(N_ITERATIONS),Vtp,label='Temps perdu')
            ax3.plot(range(N_ITERATIONS),Vd ,label='Dispersion')
            ax3.legend()
            ax3.grid()
            plt.show()
        print('DONE :',self.nom_utilisateur)

    # Fait l'inventaire des tâches non planifiées
    def notScheduled(self):
        tasks = self.tasks[0,:]
        c = self.chronologie[0,:]
        not_scheduled = []
        for t in tasks :
            if t not in c :
                not_scheduled.append(int(t))
        if len(not_scheduled) > 0 :
            notScheduledTasks = self.df_tasks.loc[not_scheduled]
            return notScheduledTasks
        else :
            print('Toutes les tâches ont pu être planifiées.')
            return
    

# CALCULS DE POTENTIELS
# Calcule le pourcentage de temps perdu sur les plages horaires utilisées
def V_TempsPerdu(taches, base_planning_array):
    temps_perdu = 0

    temps_total_dispo = 0

    index_creneau, index_tache = 0,0
    
    creneaux = base_planning_array[1,:]
    durees = taches[1,:]
    
    taches_traitees, creneaux_taches = [], []
    
    nb_taches, nb_creneaux = taches.shape[1], base_planning_array.shape[1]
    
    while index_tache < nb_taches and index_creneau < nb_creneaux:
        current_filled = 0
        j = index_tache
        filled = False
        temps_total_dispo  += creneaux[index_creneau] 
        while j < nb_taches and not filled :
            if durees[j] + current_filled <= creneaux[index_creneau] :
                taches_traitees.append(taches[0,j])
                creneaux_taches.append(base_planning_array[0,index_creneau])    
                current_filled += durees[j]                
                j += 1
            else :
                filled = True

        index_tache = j

        
        temps_total_dispo  += creneaux[index_creneau]
        temps_perdu  += max(creneaux[index_creneau] - current_filled, 0)
        index_creneau += 1
        
    chronologie = np.array([taches_traitees,creneaux_taches])

    prop_temps_perdu   = temps_perdu / temps_total_dispo        #proportion de temps perdu par rapport au temps total dispo
    
    return prop_temps_perdu, chronologie
# Calcule un potentiel correspondant au non respect des priorités
def V_Priorites(taches):
    prio = taches[2,:]
    target = np.sort(prio)
    
    return np.linalg.norm(target - prio, ord= 1) / len(np.unique(prio)) / len(prio) #(prio != target).sum()/len(prio)
# Calcule un potentiel correspondant au taux de dispersion moyen de chaque tâche
def V_Dispersion(taches):
    taches = taches[3,:]
    list_taches = np.unique(taches)
    pen = 0
    c = 0
    for t in list_taches :
        indexs = np.where(taches == t)[0]
        if len(indexs) > 1 :
            c+=1
            pen_local = 0
            for i in range(len(indexs) -1):
                if indexs[i+1] > indexs[i] + 1:
                    pen_local += 1
            pen_local /= len(indexs)
            pen += pen_local
    return pen/c if c > 0 else 0


# TACHES MERE FILLE
# (maths)
def proj1(t,d):
    n = len(t)
    proj = np.copy(t)
    for i in range(1,n):
        if  proj[i]  < proj[i-1]+ d[i-1]:
            proj[i] = proj[i-1] + d[i-1]
    return proj
# (maths)
def proj2(t,Is):
    n = len(t)
    proj = np.copy(t)
    for i in range(n):
        if t[i]< 0:
            proj[i] == 0
        if t[i] > Is[i].max():
            proj[i] = Is[i].max()
        else:
            indexa = np.where(Is[i][:,0]<= t[i])[0].max()
            indexb = np.where(t[i]<=Is[i][:,1])[0].min()
            if indexa != indexb :
                x, y = np.where( np.abs(Is[i]-t[i]) == np.abs(Is[i]-t[i]).min())
                x = x[0]
                y = y[0]
                proj[i] = Is[i][x,y]
    return proj
# (maths)
def h(t,Is):
    n = len(t)
    proj = np.copy(t)
    penalty = 0
    for i in range(n):
        if t[i]< 0:
            penalty += np.abs(t[i])
        if t[i] > Is[i].max():
            penalty += np.abs(Is[i].max()-t[i])
        else:
            indexa = np.where(Is[i][:,0]<= t[i])[0].max()
            indexb = np.where(t[i]<=Is[i][:,1])[0].min()
            if indexa != indexb :
                penalty += np.abs(Is[i]-t[i]).min()
                
    return penalty
#Planifie les tâches multi-utilisateurs à l'aide des 3 fonctions ci-dessus
def planifieTMF(TC,  DATE_DEBUT, DATE_FIN, plannings, maxiter = 100):
    N_utilisateurs = len(plannings)
    duree_sprint = pd.to_timedelta(pd.to_datetime(DATE_FIN) - pd.to_datetime(DATE_DEBUT)) \
                / np.timedelta64(1, 'h')
    new_TC = pd.DataFrame.copy(TC)
    debut_MUT_ideal = 0#int(new_TC['Priorité'].min() / T['Priorité'].max() * duree_sprint)
    fin_MUT_ideal   =  duree_sprint#int(new_TC['Priorité'].max() / T['Priorité'].max() * duree_sprint)
    Is = []
    d = []
    INDEX = []
    for index, row in new_TC.iterrows():
        duree = row["Durée"]

        a = plannings[row['id utilisateur']].base['Temps écoulé'].values
        b = a + plannings[row['id utilisateur']].base['Longueur'].values
        
        indexValides = np.where(b - (a + duree) >= 0)[0]
        if not len(indexValides) == 0:
            a = a[indexValides]
            b = b[indexValides] - duree
            d.append(duree)
            
            I = np.array([a,b]).T 
            Is.append(I)
            INDEX.append(index)
        else :
            print(f'La tâche {row["Objet"]} ne rentre dans aucun créneau et ne sera pas planifiée.')
        
        

    score_min = np.inf

    i = 0
    while i < maxiter:
        if i > 0.95 * maxiter and score_min/N_utilisateurs > 2:
            debut_MUT = debut_MUT_ideal + (i/maxiter)**3 * (0 - debut_MUT_ideal)
            fin_MUT   = fin_MUT_ideal + (i/maxiter)**3 * (duree_sprint - fin_MUT_ideal)
        else :
            debut_MUT = debut_MUT_ideal
            fin_MUT   = fin_MUT_ideal

        t = np.sort(np.random.randint(debut_MUT,fin_MUT,len(d))) /1.0
        t = proj2(t,Is)
        t = proj1(t,d)

        if h(t,Is) < score_min :
            score_min = h(t,Is)
            tmin = t
            if score_min == 0:
                break
        i += 1
    print("Non respect des horaires en heures pour la planification des tâches mere-fille :",score_min)
    new_TC.loc[INDEX,"Date début"] = pd.to_datetime(DATE_DEBUT) + pd.to_timedelta(tmin,unit='h')
    new_TC.loc[INDEX,"Date fin"] = new_TC.loc[INDEX,"Date début"] + pd.to_timedelta(new_TC['Durée'],unit='h')
    new_TC.loc[INDEX,'Priorité'] = 'TMF '
    return new_TC.loc[INDEX,['Objet','Priorité','Date début',"Date fin",'id utilisateur']]

#AUTRES FONCTIONS :

#Calcule la base du planning, supprime les plages horaires < LONGUEUR MIN
def CalculBasePlanning(plagehoraire_filename, DATE_DEBUT, DATE_FIN, LONGUEUR_MIN = 0.5):

    DATE_DEBUT = pd.to_datetime(DATE_DEBUT)
    DATE_FIN = pd.to_datetime(DATE_FIN)

    #imp = pd.read_excel(imperatif_filename)

    #ENLEVE LES EVENEMENT HORS DE LA PERIODE DE PROJECTION :
    #imp = imp[DATE_DEBUT < imp['Date fin']]
    #imp = imp[DATE_FIN   > imp['Date début']]

    #TRONQUE LES EVENEMENTS AUX BORDS DE LA PERIODE DE PROJECTION :
    #imp.loc[imp['Date fin'] > DATE_FIN,'Date fin'] = DATE_FIN
    #imp.loc[imp['Date début'] < DATE_DEBUT,'Date début'] = DATE_DEBUT

    #IMPORT DES PLAGES HORAIRES :
    ph  = pd.read_excel(plagehoraire_filename)


    #PREMIER JOUR DE LA PROJECTION :
    day0 = DATE_DEBUT.weekday()
    current_date = datetime.date(DATE_DEBUT.year,
        DATE_DEBUT.month,
        DATE_DEBUT.day)

    #ON SE SERT DANS LES PLAGES HORAIRES TANT QU'ON A PAS ATTEINT
    #LA DATE DE FIN DE PROJECTION :
    
    DONE = False
    while not DONE :
        try :
            df = ph.loc[ph['Jour'] == day0].head(1)
            current_weekday = df['Jour'].values[0]
            DONE = True
        except :
            day0 = (day0 + 1) % 7


    max_id = ph['Nom'].max()

    next_id = df['Nom'].values[-1] +1 % (max_id )

    df['Date'] = current_date

    tm = df['Heure début'].values[-1]
    current_datetime_debut = datetime.datetime.combine(current_date,tm)
    df['Date début'] = current_datetime_debut

    tm = df['Heure fin'].values[-1]
    current_datetime_fin = datetime.datetime.combine(current_date,tm)
    df['Date fin'] = current_datetime_fin

    while current_datetime_fin < DATE_FIN :
        last_jour = df['Jour'].values[-1]
        tmp = pd.DataFrame.copy(ph[ph['Nom'] == next_id])
        jour_present = tmp['Jour'].values[-1]

        
        if last_jour != jour_present :
            n_days = jour_present - last_jour if jour_present > last_jour else (7 - last_jour + jour_present)
            current_date = current_date + datetime.timedelta(days=int(n_days))

        tmp["Date"] = current_date
        
        tm = tmp['Heure début'].values[-1]
        current_datetime_debut = datetime.datetime.combine(current_date,tm)
        tmp['Date début'] = current_datetime_debut

        tm = tmp['Heure fin'].values[-1]
        current_datetime_fin = datetime.datetime.combine(current_date,tm)
        tmp['Date fin'] = current_datetime_fin
        
        dt = tmp['Date fin'].values[-1]
        
        df = pd.concat([df,tmp])
        next_id = (next_id + 1)  % (max_id + 1)
    #CA Y EST ON A ATTEINT LA DATE DE FIN DE PROJECTION MAINTENANT
    # SI CA DEPASSE AU DEBUT OU A LA FIN ON TRONQUE :
    df = df.loc[df["Date fin"] > DATE_DEBUT]
    df.loc[df['Date début'] < DATE_DEBUT, 'Date début'] = DATE_DEBUT
    df.loc[df['Date fin']   > DATE_FIN  , 'Date fin'  ] = DATE_FIN

    #Conversions au format datetime
    df['Date début'] =  pd.to_datetime(df['Date début'])
    df['Date fin'] =  pd.to_datetime(df['Date fin'])

    #RESET INDEX
    df.reset_index(inplace=True)

    #SUPPRESSION DES COLONNES DESORMAIS INUTILES
    df.drop(['Heure début','index','Heure fin','Date','Jour','Nom'],axis =1,inplace=True)

    #Réordonnement du dataframe par ordre chronologique :
    df.sort_values(by='Date début',inplace=True)

    #calcul des longueurs des ph :
    df['Longueur'] = df['Date fin'] - df['Date début']
    df['Longueur'] = df['Longueur'].astype('timedelta64[m]')/60
    df['Temps écoulé'] = (df['Date début'] - DATE_DEBUT)/np.timedelta64(1,'h')

    #Suppression des ph trop courtes :
    indexNames = df.loc[df['Longueur'] < LONGUEUR_MIN ].index
    df.drop(indexNames,inplace=True)

    #pour faire joli :
    df = df.round(2)
    df.reset_index(inplace=True)
    df['index'] = df.index
    return df

#Reconstruit la base en prenant en compte des impératifs
def addImperatifs(base, imp, LONGUEUR_MIN = 0.5):
    df = pd.DataFrame.copy(base)
    for index, row in imp.iterrows():
        
        di = row["Date début"]  #début impératif
        fi = row['Date fin']    #fin impératif

        ## un impératif recouvre entièrement une plage horaire :
        #  on la supprime alors
        indexNames = df.loc[(df['Date début'] >= di) & (df['Date fin'] <=  fi)].index
        if len(indexNames) > 0 :

            df.drop(indexNames,inplace = True)
            df.reset_index(inplace=True,drop=True)

        # un impératif est à cheval sur le début d'une plage horaire :
        # on décale alors le début de cette ph et la marge à la fin de cet
        # impératif
        indexNames = df.loc[(df['Date début'] < fi) & (fi < df['Date fin'] ) &
                            (di <= df['Date début'])].index
        if len(indexNames) > 0 :
            df.loc[indexNames,'Date début'] = fi

            df.reset_index(inplace=True,drop=True)



        #idem : à cheval fin ph
        indexNames = df.loc[(df['Date début'] < di) & (di < df['Date fin'] ) &
                           (fi >= df["Date fin"])].index
        if len(indexNames) > 0 :
            df.loc[indexNames,'Date fin'] = di



        # Un impératif tombe au milieu d'une ph : on la transforme en
        # deux ph, une avant et une après
        indexNames = df.loc[(df['Date début'] < di) & (df['Date fin'] >  fi)].index
        if len(indexNames) > 0 :
            tmp1 = pd.DataFrame.copy(df.loc[indexNames])
            tmp2 = pd.DataFrame.copy(df.loc[indexNames])
            df.drop(indexNames,inplace=True)
            tmp1['Date fin'] = di
            tmp2['Date début'] = fi
            df = pd.DataFrame.copy(pd.concat([df,tmp1,tmp2]))
            df.reset_index(inplace=True,drop=True)
            #df = pd.DataFrame.copy(pd.concat([df,tmp2]))

        #Réordonnement du dataframe par ordre chronologique :
        df.sort_values(by='Date début',inplace=True)

        #calcul des longueurs avec/sans marges des ph :
        df['Longueur'] = df['Date fin'] - df['Date début']
        df['Longueur'] = df['Longueur'].astype('timedelta64[m]')/60
        
        #Suppression des ph trop courtes :
        indexNames = df.loc[df['Longueur'] < LONGUEUR_MIN ].index
        df.drop(indexNames,inplace=True)
        

        df = df.round(2)
        df.reset_index(inplace=True,drop=True)
        df['index'] = df.index
    
    return df

# (maths) Permute deux taches dans une liste de tache
def permuteTasks(tasks):
    y = np.copy(tasks)
    i = np.random.randint(tasks.shape[1])
    j = np.random.randint(tasks.shape[1])
    y[:,i] = tasks[:,j]
    y[:,j] = tasks[:,i]
    return  y  

# Découpe des tâches en morceaux n'excédant pas une durée choisie (mod_lenght en heure)
# ----> Avec mod_lenght = 1, une tâche de 3h30 est découpée
#       en 3 morceaux d'une heure et 1 d'une demi-heure
def split_tasks(Tasks, mod_lenght = 1):    
    id_morceau = 0
    row = Tasks.loc[0]
    current_lenght = row['Durée']
    duree = mod_lenght if current_lenght > mod_lenght else current_lenght
    new_T = pd.DataFrame({'id morceau':[id_morceau],'Objet':row['Objet'],
                               'Durée':[duree],
                               'Priorité':row['Priorité'],
                               'id tache':row['id tache']})
    current_lenght -= mod_lenght
    id_morceau += 1
    while current_lenght > 0:
        duree = mod_lenght if current_lenght > mod_lenght else current_lenght
        tmp = pd.DataFrame({'id morceau':[id_morceau],'Objet':row['Objet'],
                           'Durée':[duree],
                           'Priorité':row['Priorité'],
                           'id tache':row['id tache']})
        current_lenght -= mod_lenght
        id_morceau += 1
        new_T = pd.concat([new_T,tmp])

    for index, row in Tasks.iterrows():
        if index > 0 :
            current_lenght = row['Durée']
            while current_lenght > 0 :
                duree = mod_lenght if current_lenght >= mod_lenght else current_lenght % mod_lenght
                tmp = pd.DataFrame({'id morceau':[id_morceau],'Objet':row['Objet'],
                                   'Durée':[duree],
                                   'Priorité':row['Priorité'],
                                   'id tache':row['id tache']})
                current_lenght -= mod_lenght
                id_morceau += 1
                new_T = pd.concat([new_T,tmp])

    return new_T.reset_index(drop=True)

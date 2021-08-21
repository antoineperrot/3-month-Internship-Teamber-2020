Dans ce dossier on trouve :

Trois jupyter notebook (fichiers .ipynb) et un dossier ".ipynb_checkpoints":
- "Programme d'entraînement du modèle" qui explique comment entraîner un modèle à la classification de mails
- "Programme de prédiction" qui permet d'utiliser le modèle créé dans le programme précédent pour faire
 des prédictions sur des nouveaux mails
- "Classement de mails" qui m'a permis d'élaborer une methode de prédiction en essayant différentes approches mais
qui n'a sans doute aucun intérêt pour vous.


Un fichier excel "DATA" qui contient environ 130 000 mails échangés chez Teamber et qui va nous permettre 
d'entraîner un modèle

- un fichier "MailClassifier", le modèle que l'on aura entraîné et que l'on viendra charger pour faire de nouvelles prédictions.

3 autres fichiers : TfidfVectorizerDestinataire, TfidfVectorizerExpéditeur, TfidfVectorizerObjet.
Ces objets permettent la transformation des éléments textuels (le destinataire, l'expéditeur et l'objet) du mail
en des matrices, interprétables par le modèle.

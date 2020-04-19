# Neokmeans
@author : gouth
v0.4


implementation of the neokmeans in python, libre à vous de l'améliorer, le reprendre etc etc

lien vers le papier : http://www.cs.utexas.edu/~inderjit/public_papers/neo_kmeans_sdm15.pdf


quelques exemples :

![Ground truth 2 clusters](https://user-images.githubusercontent.com/31772947/79688471-d6bd1f80-824e-11ea-9b24-df488938ed4c.png)


![Output Neokmeans alpha(0.5) beta(0.2)](https://user-images.githubusercontent.com/31772947/79688468-cf961180-824e-11ea-96b4-750393f1beea.png)

et

![Ground truth 3 clusters](https://user-images.githubusercontent.com/31772947/79688477-df155a80-824e-11ea-9429-c0ae8169b3d7.png)

![Output Neokmeans alpha(0.15) beta(0.2)](https://user-images.githubusercontent.com/31772947/79688476-dcb30080-824e-11ea-8bd9-5d928506def6.png)




(18/04/20) update v0.2: La version n'est pas encore parfaite mais j'arrive a un resultat.
C'est une version probabiliste qui cherche les coef alpha et beta selon un critère (au moins 20 de population dans les clusters)
Je vais améliorer au fur et a mesure afin qu'il soit parfait pour le dialogue auquel il est prédestiné !

(19/04/20) update v0.3 : j'ai changé completmeent le déroulement de l'algo pour qu'il colle plus à l'implementation matlab
avant je suivais le papier ligne par ligne, mais l'auteur du papier a fait une version matalab plus opti.
j'ai :
* des clusters avec du recouvrement (des points appartiennent à plusieurs clusters)
* un parametre "J" qui me permet de ne pas systematiquement faire tmax tours, je converge jusqu'à une precision d'epsilon

je n'ai pas :
* les extremités, j'y travaille (les points qui sont en marge des clusters)

(19/04/20) update v0.4
j'ai :
* les points extremes, même si ça semble un peu éloigné du ground truth
* testé sur 3 clusters générés a partir de mon script csv_gen 
* j'arrive à avoir l'intersection des trois clusters => portabilité infinie sur le nb de cluster via comment fonctionne la matrice U ;)

je n'ai pas :
* l'intersection pour tout cluster (Ci,Cj) tel que i!=j
* automatisé l'affichage en fonction du nombre de cluster 

voici mon environnement python :

Package         Version
--------------- ------------
-cikit-learn    0.22.2.post1
kiwisolver      1.1.0
matplotlib      3.2.0
numpy           1.18.2
pyparsing       2.4.6
python-dateutil 2.8.1
scikit-learn    0.22.2.post1
scipy           1.4.1
six             1.14.0


j'ai eu quelques problèmes d'installation notamment avec scikit-learn;
une solution trouvée en ligne c'était pour les utilisateurs de windows, il faut modifier 
une clé de registre pour autorisé les chemins longs (un tel chemin était utilisé lors de l'installation de la librairie)



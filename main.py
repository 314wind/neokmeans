import sys
import numpy as np
import numpy.matlib as matlib
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as km
import random
import math

np.set_printoptions(threshold=sys.maxsize) #pour afficher les tableaux en entier sur la sortie standard
"""
parse le fichier dans une matrice X 
X[i,:] : les coordonnées de l'utilisateur i dans toutes les dimensions
X[:,j] : les coordonnées sur la dimension j (l'item j)
"""
def parse(filename):
    try:
        X = np.genfromtxt(filename+'.csv', delimiter=',')
        return X

    except IOError:
        print("abort..." , IOError.strerror)
        exit(IOError.errno)


"""
X : data
k : nb clusters
U : indices 
return the matrix M wich is the means of all centroid in data X
"""
def centroids(X, U, k):
	length = X.shape[1] #nb dimensions données
	M = np.zeros((length, k))
	for i in range (0,k):#parcours de tous les clusters
		ind = np.argwhere(U[:,i]==1) #extrait les points qui appartiennt au cluster i
		for j in range(0,length): #on somme colonne par colonne de la partie k
			tc = X[ind,j].shape[0] #taille du cluster
			sum_x = np.sum(X[ind, j])
			print("i:",i)
			print("j:",j)
			print(M.shape)
			print("=======")
			M[j,i] = sum_x/tc  ##j'avais inversé les indices i et j ça donnait de la m****



	return M

"""
calcul la distance entre chaque point dans X et claque cluster 
dont les moyennes sont dans M
k : le nombre de clusters
"""
def distance(X, M, k):
	N = len(X) #taille des données
	D = np.zeros((N, k)) #matrice distance
	dim = len(X[0]) #dimensions
	for i in range(0,k):
		diff = X - matlib.repmat(M[:,i].T, N, 1) #X - le point moyen de chaque cluester (lignei M = clusteri)
		D[:,i] = np.sum(np.power(diff,2), 1) #sum (diff²) selon l'axe des lignes

	return D

"""
X : data
k : number of clusters
alpha : coef of points that could be in two cliques
beta : coef of points that couldn't be in one clique
tmax : iteration max 
initU : for indices in X

RETURN : k cliques
"""
def neoKMeans(X, k, alpha, beta, tmax, initU):
	#init
	C = [[]for i in range (0,k)] #OUTPUT : k clusters
	N = len(X) #taille des données
	J = float('inf')
	oldJ = 0
	epsilon = 0 #on veut une réponse précision 100%
	length = X.shape[0] 
	# ==> va servir pour converger
	alphaN = math.floor(alpha*N) #tours en rab pour mettre un point dans une intersection
	betaN = math.floor(beta*N)#tours en moins pour ne pas traiter les extremes

	U = initU 
	t = 0 #current iteration
	#tq on a pas convergé et qu'on a pas atteint l'itération max
	while(abs(oldJ-J)>epsilon and t < tmax):
		oldJ = J
		J = 0

		#compute cluster means
		M = centroids(X, U, k)
		#print("M:", M)
		
		#compute cluster distance
		D = distance(X, M, k)
		#print("D:", D)
	

		###initialize###
		T = []
		S = []
		p = 0
		C_c = set()  #C_chapeau => assignement dans p < N + alphaN
		C_b = set()  #C_barre => assignement dans p < N - betaN
	
		########
		#premiere contrainte faire N - betaN assignement
		########

		#DNK 
		#Distance
		#Node n
		#Cluster k

		nb_assign = N-betaN

		dist = np.min(D, axis=1)
		node = np.arange(0,N)
		ind = np.argmin(D, axis=1)
		dnk = np.zeros((N,3))
		dnk[:,0] = dist
		dnk[:,1] = node.T
		dnk[:,2] = ind

	
		#tri le tableau croissant en fonciton de la premiere column (distance)
		dnk_sorted = dnk[dnk[:,0].argsort()] 
		sorted_d = dnk_sorted[:,0]
		sorted_n = dnk_sorted[:,1]
		sorted_k = dnk_sorted[:,2]

		
		J= J+ np.sum(sorted_d[1:nb_assign])

		#changer le type des deux dernieres colonnes en int 

		#temp est une matrice qui facilite les accès par la suite
		temp = np.zeros((nb_assign, nb_assign))
	
		temp[:,0] = sorted_n[0:int(nb_assign),]
		temp[:,1] = sorted_k[0:int(nb_assign),]

		ind_no_cluster = dnk_sorted[nb_assign:N,1].astype('int')
		
		U[ind_no_cluster,] = 0

	
		#==> moralement temp[x,0] pour x variant de 0 a N-betaN va contenir le numéro des noeuds
		#qui ont la plus petite distance entre lui et un des clusters (ici temp[x,1] est le numero
		#du cluster). Le tout est ordonnée par autre croissant selon la distance. 			
		for x in range (0,N-betaN):
			#U[int(temp[x,0]), int(temp[x,1])] = 1 #assigne au noeud temp[x,0] le cluster temp[x,1]
			D[int(temp[x,0]), int(temp[x,1])] = float('inf') #on ne considere plus le point
		
		#U = sc.sparse.coo_matrix(U, (N,k)) #opti memoire, mais il faut changer bcp de code : TODO 

		########
		#deuxieme contrainte faire alphaN + betaN assignement (intersection)
		########

		n = 0
		while n < (alphaN + betaN): #on considere des tours en plus pour faire l'intersection (critere alpha)
			#compute i and j index of argmin in distance matrix of x_i
			(i,j) = np.unravel_index(D.argmin(), D.shape) #argmin 2D	
			min_d = np.min(D)
			J = J + min_d
			U[i,j] = 1
			D[i,j] = float('inf') #on ne considere plus ce poitn
			n = n+1
		#endwhile
		t = t + 1
	#endwhile

	#print("ended in ", t, "iterations")
	#print("objectif : ", abs(1-(oldJ-J)))
	return U


"""
X : les données matrice N*d 
U : la matrice d'affectation
k : le nombre de clusters
"""
def display_cluster(X,U,k):
	length = X.shape[1]
		
	#liste des points du cluster i : c[i][:,0,:]
	
	intersection = get_intersection(X,U)
	extreme = get_extreme(X,U)
 

 	#todo automatiser l'affichage en fonction du nombre de cluster
	c0 = X[np.argwhere(U[:,0]==1)]
	c1 = X[np.argwhere(U[:,1]==1)]
	if(k==3):
			c2 = X[np.argwhere(U[:,2]==1)]

	fig, ax = plt.subplots(length,length, sharex='col', sharey='row')
	for i in range(0,length):
		for j in range(0, length):
			ax[i,j].set_xlim([-7,7])
			ax[i,j].set_ylim([-7,7])
			ax[i,j].plot(c0[:,0,i], c0[:,0,j],'ro')
			ax[i,j].plot(c1[:,0,i], c1[:,0,j],'bo')
			if(k==3):
					ax[i,j].plot(c2[:,0,i], c2[:,0,j],'mo')
			ax[i,j].plot(intersection[:,i], intersection[:,j],'go')
			ax[i,j].plot(extreme[:,i], extreme[:,j],'ko')
	plt.show()

	plt.pause(1)
	r = raw_input("<Hit Any Key To Close>")
	plt.close(fig)

"""
recupere les noeuds qui appartient à tous les clusters
X : les données
U : la matrice d'affectation
"""
def get_intersection(X,U):
	return X[np.where(np.all(U[:, :]==1, axis=1))[0]]
"""
recupere les noeuds qui n'appartiennent à aucuns clusters
X : les données
U : la matrice d'affectation
"""
def get_extreme(X,U):
	return X[np.where(np.all(U[:,:]==0, axis=1))[0]]



###############
#	MAIN	  #
###############


#filename = input("nom du fichier (.csv) : ")
#parser.parse(filename)
X = parse("csv_gen/data_c3")


#parametres
#k = int(input("nb cluster :")) #number of clusters 
k = 3
alpha = random.uniform(0,(k-1)/100) #cf papier
beta = random.uniform(0,1)
tmax = 100 #max iteration

#initialisation
#predit pour chaque sample l'id du cluster ou il sera dedans
indX = km(k, random_state=0).fit_predict(X)
#print("prediction kmeans : ", indX)

initU = np.zeros((len(X), k))
for j in range (0, k):
	initU[:,j] = indX==j

#initU va servir à récuperer les indices dans X


alpha = 0.5
beta = 0.015

U = neoKMeans(X, k, alpha, beta, tmax, initU)
"""
matrice U : matrice d'assignement
tel que Uij = 1 ssi le point i appartient au cluster j
on peut avoir Uij = Ui(j+1) = 1 si le point i appartient aux deux clusters j et j+1
"""

N = X.shape[0] #taille des données
nb_inter = len(U[np.where(U[:,0]==U[:,1])])

print("condition nb_inter doit être entre ", 0.05*N," et ", 0.25*N)


nb_inter = len(get_intersection(X,U))
nb_extrem = len(get_extreme(X,U))

i_Q1 = 0.25 #5% de la pop
i_Q2 = 0.50 #25% de la pop

e_Q1 = 0.001 #0.01% de la pop
e_Q2 = 0.05 # 0.1% de la pop

b1 = (nb_inter > int(i_Q1*N)) and (nb_inter < int(i_Q2*N))
b2 = (nb_extrem > int(e_Q1*N)) and (nb_extrem < int(e_Q2*N))
print("nb_inter:", nb_inter, "got to be between ", i_Q1*N , " and ", i_Q2*N)
print("nb_extrem:", nb_extrem, "got to be between ", e_Q1*N , " and ", e_Q2*N)

oldinter=-1

b_inf = 0
b_sup = 1
#while(not(b1) and not(b2)):
"""
while(not(b1 and b2)):
	
	#if(not b1):
	alpha = random.uniform(0,1) #cf papier
	#if(not b2):
	beta = random.uniform(0,(k-1)/10)

	print("trying (",alpha,",",beta,")")
	prevX= X
	prevU = initU
	U = neoKMeans(X, k, alpha, beta, tmax, initU)
	oldinter=nb_inter
	nb_inter = len(get_intersection(X,U))
	nb_extrem = len(get_extreme(X,U))
	b1 = (nb_inter > int(i_Q1*N)) and (nb_inter < int(i_Q2*N))
	b2 = (nb_extrem > int(e_Q1*N)) and (nb_extrem < int(e_Q2*N))
	print("nb_inter:", nb_inter, "got to be between ", i_Q1*N , " and ", i_Q2*N, " b1 : ", b1)
	print("nb_extrem:", nb_extrem, "got to be between ", e_Q1*N , " and ", e_Q2*N, " b2 ; ", b2)
	#print(nb_extrem,">",e_Q1*N,"and",nb_extrem,"<",e_Q2*N,"=== ",b2)


"""
print("====END====")
print("ALPHA : " , alpha)
print("BETA : ", beta)
print("nombre dans l'intersection :", nb_inter)
print("cluster 1.", len(U[np.argwhere(U[:,0]==1)]))
print("cluster 2.", len(U[np.argwhere(U[:,1]==1)]))
print("nb_inter : ", nb_inter)
print("nb_extrem :", nb_extrem)

display_cluster(X,U,k)
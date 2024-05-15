## The goal of this file is to test the differents priors of Beta used in the MDiNE model. 
##To achieve this goal, the differents priors will be tested on simple linear regression. 

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import pymc.math
import pickle
import pandas as pd
import scipy.special
import scipy.stats
import os
import math
import seaborn as sns
import re
import json
from datetime import date

#from MDiNE_model import run_model

from extract_data_files import get_data,separate_data_in_two_groups

def create_json_file_model():

	# Construire la structure de données
	donnees = {
		"simulation_name":"ridge_lasso",
		"generated_data":True,
		"data":{
			"nb_tests":10,
			"n_individus":100,
			"j_taxa":20,
			"k_covariates":20,
			"model_generated_data":"Multinomial"
		},
		"beta_matrix": {
			"apriori": "Ridge",
			"parameters":{
				"alpha":1,
				"beta":1
			}
		},
		"precision_matrix":{
			"apriori": "Lasso",
			"parameters": {
				"lambda_init":10
			}
		},
		"sampler":"NUTS"
	}

	# Nom du fichier JSON
	folder="test_mdine/examples_json_simulation/"
	filename = folder+donnees["simulation_name"]+".json"

	# Écrire les données dans le fichier JSON
	with open(filename, "w") as fichier_json:
		json.dump(donnees, fichier_json, indent=4)

	print("Fichier JSON créé avec succès :", filename)


def test_mdine_on_generated_data():

	# Path of the parent folder
	folder_parent = 'mdine_simulations'

	# Liste pour stocker les noms des sous-dossiers
	sub_folders = []

	# Parcourir tous les éléments dans le dossier parent
	for element in os.listdir(folder_parent):
		# Vérifier si l'élément est un dossier
		complete_path = os.path.join(folder_parent, element)
		if os.path.isdir(complete_path):
			# Ajouter le nom du sous-dossier à la liste
			sub_folders.append(element)

	print(sub_folders)

	numbers_simulations = [int(re.search(r'\d+', nom).group()) for nom in sub_folders if re.search(r'\d+', nom)]
	simulation_number = max(numbers_simulations)+1 if numbers_simulations else 1

	folder_simulation=folder_parent+"/simulation_"+str(simulation_number)
	os.makedirs(folder_simulation)

    # nb_tests=2
    # for i in range (nb_tests):
    #     folder=f"results/mdine_generated_data_2/simulation_{i}/"
    #     (X_matrix,counts_matrix)=generate_counts_data_multinomial(folder)
    #     beta_matrix_choice="Ridge"
    #     precision_matrix_choice="exp_Laplace"
    #     run_model(X_matrix,counts_matrix,beta_matrix_choice,precision_matrix_choice,folder)

		


def ppf_ZINB(quantile,n,p,pi_proportion):
	i=0
	quantile_i=0
	if quantile_i<=pi_proportion:
		return 0
	else:
		while quantile_i<=quantile:
			i+=1
			quantile_i=pi_proportion+(1-pi_proportion)*scipy.stats.nbinom.cdf(i,n,p)
	
	return i

def get_info_data_file():
	(covariate_matrix_data,counts_matrix_data,Z_vector)=get_data("data/crohns.csv")
	first_group,second_group=separate_data_in_two_groups(covariate_matrix_data,counts_matrix_data,Z_vector)# covariate puis counts
	sommes = np.sum(first_group[1], axis=1)
	#variances = np.var(first_group[1], axis=1)

	print("Sommes des lignes:  ",len(sommes))

	zeros_count = 1 - np.count_nonzero(counts_matrix_data, axis=0)/counts_matrix_data.shape[0]


	list_variance=np.var(counts_matrix_data,axis=0)
	list_mean=np.mean(counts_matrix_data,axis=0)

	# plt.hist(sommes, bins=10, color='skyblue', edgecolor='black')
	# plt.xlabel('Valeurs')
	# plt.ylabel('Fréquence')
	# plt.title('Histogramme des entiers')

	# # Afficher le graphique
	# plt.show()
	return(zeros_count,list_mean,list_variance)


def extract_data_pickle():
	# Charger les données depuis le fichier pickle
	with open("data/generated_data_mdine/multinomial.pickle", "rb") as fichier:
		Beta_matrix, precision_matrix,counts_matrix,X_matrix = pickle.load(fichier)

	print("Beta_matrix: \n",Beta_matrix,"\n\n")
	print("Precision Matrix: \n\n",precision_matrix)

def generate_data_linear_regression():

	# Définir les dimensions
	n_samples = 100
	n_features = 10

	# Générer des données pour X (matrice de features)
	#X = np.random.randn(n_samples, n_features)
	X=np.random.uniform(low=0, high=2, size=(n_samples, n_features))

	# Générer des coefficients beta
	#beta = np.random.randn(n_features, 1)  
	beta=np.array([0,1.5,2.6,0,0,1,0.3,3.1,0,1.7])

	# Définir l'écart type de l'erreur epsilon
	sigma = 1.0

	# Générer des erreurs epsilon
	epsilon=np.random.normal(loc=0, scale=sigma, size=n_samples)

	# Calculer la variable cible Y
	Y = np.dot(X, beta).T + epsilon

	return Y,X,beta

def get_link_sparsity():

	list_j_taxa=[5,10,20,40]
	nb_test_par_proba=100
	list_proba=np.linspace(0, 1, num=50)

	for i, j_taxa in enumerate(list_j_taxa):

		list_sparsity_i=[]

		for proba in list_proba:

			count_zeros=0

			for _ in range(nb_test_par_proba):

				L_precision_matrix=np.zeros((j_taxa,j_taxa))

				### Generate precision matrix
				for i in range (j_taxa):
					for j in range(i+1):
						if i==j:
							L_precision_matrix[i,j]=np.random.uniform(low=1.5, high=2.5)
						else:
							if np.random.uniform()<=1-proba:
								L_precision_matrix[i,j]=np.random.uniform(low=-1.5, high=1)

				#print("L_precision: \n\n\n",L_precision_matrix)
					
				precision_matrix=np.dot(L_precision_matrix,L_precision_matrix.T)

				count_zeros+=j_taxa**2-np.count_nonzero(precision_matrix) # Que pour un triangle

				# np.set_printoptions(suppress=True)
				# print(precision_matrix)
	
			list_sparsity_i.append(count_zeros/(j_taxa*j_taxa*nb_test_par_proba))
		plt.plot(list_proba,list_sparsity_i,label=f'J={j_taxa}')

	plt.plot([0,1], [0,1], linestyle='dashed', color='black')
	plt.xlabel('Probability of zeros in the Cholesky decomposition')
	plt.ylabel('Percentage of zeros in the precision matrix')
	plt.legend()
	#plt.title(f'Link between the sparsity of the Cholesky \n decomposition and the precision matrix')
	plt.show()

def generate_counts_data_ZINB(n_individus,k_covariables,j_taxa,folder):

	L_precision_matrix=np.zeros((j_taxa,j_taxa))

	### Generate precision matrix
	for i in range (j_taxa):
		for j in range(i+1):
			if i==j:
				L_precision_matrix[i,j]=np.random.uniform(low=1.5, high=2.5)
			else:
				if np.random.uniform()<=0.40:
					L_precision_matrix[i,j]=np.random.uniform(low=-1.5, high=1)

	#print("L_precision: \n\n\n",L_precision_matrix)
		
	precision_matrix=np.dot(L_precision_matrix,L_precision_matrix.T)

	# Correlation Matrix
	correlation_matrix=np.zeros((j_taxa,j_taxa))

	for i in range (j_taxa):
		for j in range (j_taxa):
			correlation_matrix[i,j]=precision_matrix[i,j]/(math.sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

	#W_matrix=np.zeros((n_individus,j_taxa))
	W_matrix=scipy.stats.multivariate_normal.rvs(np.zeros((j_taxa)),cov=correlation_matrix,size=(math.ceil(n_individus*(j_taxa+1)/j_taxa),))

	W_matrix=np.ravel(W_matrix)[0:(j_taxa+1)*n_individus].reshape((n_individus, j_taxa+1))


	#print(W_matrix.shape)

	quantiles = np.empty_like(W_matrix, dtype=float)
	for i in range(quantiles.shape[0]):
		for j in range(quantiles.shape[1]):
			quantiles[i, j] = scipy.stats.norm.cdf(W_matrix[i, j])

	#print(quantiles)

	zeros_counts,list_mean,list_variance=get_info_data_file()

	list_p=[list_mean[i]/list_variance[i] for i in range(len(list_mean))]
	list_n=[list_mean[i]**2/(list_variance[i]-list_mean[i]) for i in range(len(list_mean))]

	

	counts_matrix=np.zeros((n_individus,j_taxa+1))
	for i in range (n_individus):
		for j in range (j_taxa+1):
			counts_matrix[i,j]=ppf_ZINB(quantiles[i,j],list_n[j],list_p[j],zeros_counts[j])

	print(counts_matrix)




	# for i in range (n_individus):
	# 	W_matrix[i]=scipy.stats.multivariate_normal.rvs(0,cov=correlation_matrix,size=(n_individus,j_taxa))
		
	ax = sns.heatmap(correlation_matrix, linewidth=0.5)
	plt.show()

def generate_counts_data_Multinomial(n_individus,k_covariables,j_taxa,folder):
	
	nb_col_bernoulli = int(0.25 * k_covariables)

    # Générer les colonnes suivant une loi de Bernoulli
	colonnes_bernoulli = np.random.binomial(1, 0.5, size=(n_individus, nb_col_bernoulli))

    # Générer les colonnes suivant une loi normale
	colonnes_normales = np.random.normal(0, 1, size=(n_individus, k_covariables - nb_col_bernoulli))

    # Concaténer les colonnes
	X_matrix = np.concatenate((colonnes_bernoulli, colonnes_normales), axis=1)

	#print(X_matrix)

	Beta_matrix=np.random.normal(0,0.4,size=(k_covariables, j_taxa))

	for i in range (k_covariables):
		for j in range(j_taxa):
			if abs(Beta_matrix[i,j])>=0.5:
				Beta_matrix[i,j]=0


	L_precision_matrix=np.zeros((j_taxa,j_taxa))

	### Generate precision matrix
	for i in range (j_taxa):
		for j in range(i+1):
			if i==j:
				L_precision_matrix[i,j]=np.random.uniform(low=1.5, high=2.5)
			else:
				if np.random.uniform()<=0.40:
					L_precision_matrix[i,j]=np.random.uniform(low=-1.5, high=1)

	#print("L_precision: \n\n\n",L_precision_matrix)
		
	precision_matrix=np.dot(L_precision_matrix,L_precision_matrix.T)

	np.set_printoptions(suppress=True)

	#print(precision_matrix)

	# Correlation Matrix
	correlation_matrix=np.zeros((j_taxa,j_taxa))

	for i in range (j_taxa):
		for j in range (j_taxa):
			correlation_matrix[i,j]=precision_matrix[i,j]/(math.sqrt(precision_matrix[i,i]*precision_matrix[j,j]))
		
	# ax = sns.heatmap(correlation_matrix, linewidth=0.5,annot=True, fmt=".1f")
	# plt.show()

	#print(precision_matrix)

	# for i in range(j_taxa):
	# 	for j in range(i):
	# 		if np.random.uniform()<=0.50:
	# 			precision_matrix[i,j]=0
	# 			precision_matrix[j,i]=0

	# print("\n\n\n precision: \n\n\n",precision_matrix)
	# print("\n\n\n")

	mean_matrix=np.dot(X_matrix,Beta_matrix)

	covariance_matrix=np.linalg.inv(precision_matrix)

	W_matrix=np.zeros((n_individus,j_taxa))

	for i in range (n_individus):
		W_matrix[i]=scipy.stats.multivariate_normal.rvs(mean=mean_matrix[i],cov=covariance_matrix)
		# print("\n\nMean i:",mean_matrix[i],'\n\n')
		# print(scipy.stats.multivariate_normal.rvs(mean=mean_matrix[i],cov=covariance_matrix,size=(j_taxa)))
	#print(W_matrix)

	#print(np.hstack((W_matrix, np.ones((n_individus, 1)))))
  
	proportions_matrix=scipy.special.softmax(np.hstack((W_matrix, np.ones((n_individus, 1)))),axis=1)

	#print(proportions_matrix)

	counts_matrix=np.zeros((n_individus,j_taxa+1))

	for i in range(n_individus):
		#sum_counts=scipy.stats.norm.rvs(10000,1000)
		sum_counts=scipy.stats.norm.rvs(100000,10000)
		counts_matrix[i]=scipy.stats.multinomial.rvs(sum_counts,proportions_matrix[i])
		#print(scipy.stats.multinomial.rvs(sum_counts,proportions_matrix[i],size=(j_taxa+1,)))

	#print(counts_matrix)
 
	if not os.path.exists(folder):
        # Créer le sous-dossier s'il n'existe pas
		os.makedirs(folder)

	# Sauvegarder les variables dans un fichier
	with open(folder+"generated_data.pkl", "wb") as fichier:
		pickle.dump((Beta_matrix, precision_matrix,counts_matrix,X_matrix), fichier)

	#np.set_printoptions(suppress=True)

	#print(counts_matrix)

	#print(X_matrix)

	return (X_matrix,counts_matrix)



def plot_histogram_from_csv(csv_file, column_name):

    # Lire le fichier CSV
	data = pd.read_csv(csv_file)
	
	#print(data.iloc[0])
    
    # Extraire les données de la colonne spécifiée
	column_data = data[column_name]
	#firstrow=data.iloc[0]
    
    # Tracer l'histogramme
	plt.hist(column_data, bins=100, color='skyblue', edgecolor='black')
    
    # Ajouter des étiquettes et un titre
	plt.xlabel(column_name)
	plt.ylabel('Fréquence')
	plt.title('Histogramme de la colonne ' + column_name)
    
    # Afficher le graphique
	plt.show()

def test_cumulative():
	# Liste de nombres
	data = [1, 2, 3, 4, 5,3,2,1,1,3,4,9,10,22,]

	# Trier les données
	sorted_data = np.sort(data)

	# Calculer les probabilités cumulatives
	cumulative_prob = np.arange(len(sorted_data)) /len(sorted_data)

	print(cumulative_prob)

	# Tracer la fonction de distribution cumulative
	plt.plot(sorted_data, cumulative_prob, marker='o', linestyle='-')
	plt.xlabel('Valeurs')
	plt.ylabel('Probabilité Cumulative')
	plt.title('Fonction de Distribution Cumulative')
	plt.grid(True)
	plt.show()

def regression_model(choice_beta_matrix,folder):
      
	Y,X,beta=generate_data_linear_regression()

	n_samples,n_features = X.shape

	with pm.Model() as regression_model:

		if choice_beta_matrix=="Ridge":
			lambda_ridge=pm.Gamma("lambda_brige",alpha=1,beta=1)
			beta_matrix=pm.Normal("beta_matrix",mu=0,tau=lambda_ridge,shape=(n_features,))

		elif choice_beta_matrix=="Horseshoe":
			tau=pm.HalfCauchy("tau",beta=2)
			lambda_horseshoe=pm.HalfCauchy("lambda_horseshoe",beta=tau,shape=(n_features,))
			
			#beta_matrix=pm.Normal("beta_matrix",mu=0,sigma=pt.sqr(tau)*pt.sqr(lambda_horseshoe))
			beta_matrix=pm.Normal("beta_matrix",mu=0,sigma=pt.sqr(lambda_horseshoe))

		elif choice_beta_matrix=="Spike_and_slab":
			proba_gamma=pm.Beta("pi_gamma",alpha=1,beta=1,shape=(n_features,))
			gamma_matrix=pm.Bernoulli("gamma",p=proba_gamma,shape=(n_features,))

			# Générer une matrice de nombres aléatoires entre 0.1 et 1 avec deux décimales
			#tau_matrix = np.random.uniform(low=0.001, high=0.01, size=(n_features,)).round(4)
			tau=0.2
			#print(tau_matrix)

			# Générer une matrice d'entiers aléatoires entre 10 et 100
			#c_matrix = np.random.uniform(low=50, high=500, size=(n_features,)).round(2)
			c=10
			#print(c_matrix)

			# Hadamard product between two tensors: A*B
			#(pymc.math.ones(k_covariates, j_taxa)-gamma_matrix)*pm.Normal.dist(mu=0, sigma=tau_matrix, shape=(k_covariates, j_taxa))+gamma_matrix*pm.Normal.dist(mu=0, sigma=tau_matrix*c_matrix, shape=(k_covariates, j_taxa))
			#beta_matrix=pm.Deterministic("beta_matrix",(pymc.math.ones((n_features,))-gamma_matrix)*pm.Normal("Beta_Spike",mu=0, sigma=pt.sqr(tau_matrix), shape=(n_features,))+gamma_matrix*pm.Normal("Beta_Slab",mu=0, sigma=pt.sqr(tau_matrix)*pt.sqr(c_matrix), shape=(n_features,)))
			beta_matrix=pm.Deterministic("beta_matrix",(pymc.math.ones((n_features,))-gamma_matrix)*pm.Normal("Beta_Spike",mu=0, sigma=tau**2, shape=(n_features,))+gamma_matrix*pm.Normal("Beta_Slab",mu=0, sigma=tau*tau*c*c, shape=(n_features,)))

		else:
			print("\n\n\n#######Choix non valide pour la matrice beta#######\n\n\n")
		sigma_y=1
		y=pm.Normal("Y_counts",mu=pt.dot(pt.as_tensor_variable(X),beta_matrix),sigma=sigma_y,observed=Y)
	
	with regression_model:
		idata = pm.sample(10000,target_accept=0.90) ## 10000 normally

	
	filename_idata=folder+"idata.pkl"

	with open(filename_idata, "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
		pickle.dump(idata, f)

	with regression_model:
		ppc=pm.sample_posterior_predictive(idata,extend_inferencedata=True)


	az.plot_ppc(ppc,kind="kde")
	plt.savefig(folder+choice_beta_matrix+"_ppc_kde.png")
	#plt.plot()

	print(az.summary(idata, kind="stats"))

	print(idata.sample_stats)

	idata.sample_stats["tree_depth"].plot(col="chain", ls="none", marker=".", alpha=0.3)
	plt.savefig(folder+choice_beta_matrix+"_sample_stats.png")

	az.plot_energy(idata, figsize=(6, 4))
	plt.savefig(folder+choice_beta_matrix+"_energy.png")

	az.plot_posterior(idata, group="sample_stats", var_names="acceptance_rate", hdi_prob="hide", kind="hist")
	plt.savefig(folder+choice_beta_matrix+"_posterior.png")


	print("Divergence? :",idata.sample_stats["diverging"].sum())


	az.plot_trace(idata, var_names=["beta_matrix"])
	plt.savefig(folder+choice_beta_matrix+"_trace.png")
	#az.plot_trace(idata, var_names=["beta_matrix","tau","lambda_horseshoe"])
	#az.plot_trace(idata, var_names=["beta_matrix","pi_gamma","gamma"])
 
	#plt.show(block=False)

	az.plot_forest(idata, var_names=["beta_matrix"], hdi_prob=0.80)
	ax = plt.gca()

	# Tracer une ligne verticale en rouge pointillée à x=0
	#plt.plot(beta, np.arange(len(beta)), 'rx', label='Vraies données')
	ax.axvline(x=0, color='red', linestyle='--')
	plt.savefig(folder+choice_beta_matrix+"_forest.png")
	#plt.show()



if __name__=="__main__":
	#test_cdf_neg_binomial()
	#generate_counts_data_multinomial("blabla")
    #plt.savefig('figure_forest.png')
	#specie="otu.counts.f__Bacteroidaceae"         
 	#specie="otu.counts.f__Ruminococcaceae"        
	# specie="otu.counts.f__Lachnospiraceae"         
	# specie="otu.counts.f__Enterobacteriaceae"    
	# specie="otu.counts.f__Pasteurellaceae"         
	# specie="otu.counts.ref"
 	#generate_counts_data_ZINB()
  
  	create_json_file_model()

 
	#get_info_crohns_data()
 
 	#get_link_sparsity()
	#generate_counts_data_multinomial("data/generated_data_mdine12/")
	#test_mdine_on_generated_data()
 	#generate_counts_data_ZINB()
 
	#test_multinomial()
 	#extract_data_pickle()
	#generate_counts_data_multinomial()

	#plot_histogram_from_csv("data/crohns.csv",specie)

	#test_cumulative()

	# beta_choice="Ridge"
	# folder="results/linear_regression/"+beta_choice+"/"
	# #beta_choice="Horseshoe"
	# regression_model(beta_choice,folder)
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import pytensor.tensor
import pytensor.tensor.shape
from scipy.special import softmax
import pytensor
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import csv
import pymc.math
from functools import partial

print(f"Running on PyMC v{pm.__version__}")

def test_softmax():
    np.set_printoptions(precision=5)
    x = np.array([[1, 0.5, 0.2, 3],

              [1,  -1,   7, 3],

              [2,  12,  13, 3]])
    m = softmax(x)

def get_data(file_name):

    # Initialiser des listes pour stocker les données des colonnes 2 à 5 et 6 à 11
    donnees_colonnes_2_5 = []
    donnees_colonnes_6_11 = []
    Z_vector=[]

    # Ouvrir le fichier CSV en mode lecture
    with open(file_name, 'r') as fichier_csv:
        # Créer un objet lecteur CSV
        lecteur_csv = csv.reader(fichier_csv)

        next(lecteur_csv) # Passe la première ligne
        
        # Lire chaque ligne du fichier CSV
        for ligne in lecteur_csv:
            # Binarisation des données 
            ligne_transformee = [valeur.replace("CD", "1").replace("no", "0").replace("female", "1").replace("male", "0").replace("false", "0").replace("true", "1") for valeur in ligne]
            
            # Convertir les éléments de la ligne en float (sauf la première colonne si elle contient des identifiants)
            ligne_float = [float(valeur) for valeur in ligne_transformee]
            
            # Stocker les données des colonnes 2 à 5 dans une liste
            donnees_colonnes_2_5.append(ligne_float[1:5])
            Z_vector.append(ligne_float[1])
            
            # Stocker les données des colonnes 6 à 11 dans une liste
            donnees_colonnes_6_11.append(ligne_float[5:12])
        
    # Convertir les listes en matrices numpy
    covariate_matrix = np.array(donnees_colonnes_2_5)
    counts_matrix = np.array(donnees_colonnes_6_11)
    Z_vector=np.array(Z_vector)

    #print("Matrice de covariates:\n",covariate_matrix,"\n\n")
    #print("Matrice de Z:\n",Z_vector,"\n\n")
    #print("Matrice Data Y:\n", counts_matrix)

    return(covariate_matrix,counts_matrix,Z_vector)

with pm.Model() as mdine_model:

    #### Data recovery

    filename="crohns.csv"
    (covariate_matrix_data,counts_matrix_data,Z_vector)=get_data(filename)

    j_taxa_plus_ref = counts_matrix_data.shape[1]
    n_individuals = counts_matrix_data.shape[0]
    k_covariates = covariate_matrix_data.shape[1]   #### Covariate Matrix Data must not contain the z column.
    j_taxa=j_taxa_plus_ref-1

    ## Matrice Beta, loi normale de parametre sigma
    sigma=100
    beta_matrix=pm.Normal("beta_matrix",mu=0,sigma=sigma,shape=(k_covariates,j_taxa_plus_ref-1))

    ## Matrice de précision
    lambda_init=10
    lambda_mdine=pm.Exponential("lambda_mdine",lambda_init)

    precision_matrix=pm.Exponential("precision_matrix",lam=lambda_mdine,shape=(j_taxa,j_taxa))


    covariate_matrix=pm.Deterministic("X_covariates",pytensor.tensor.as_tensor_variable(covariate_matrix_data))

    product_X_Beta=pm.Deterministic("Product_X_Beta",covariate_matrix@beta_matrix)

    # Matrice W, loi normale
    #w_matrix=pm.MvNormal("w_matrix",mu=pytensor.tensor.transpose(product_X_Beta),tau=precision_matrix)
    #w_matrix=pm.MvNormal("w_matrix",mu=product_X_Beta,tau=precision_matrix,shape=(n_individuals,j_taxa))
    w_matrix=pm.MvNormal("w_matrix",mu=product_X_Beta,tau=precision_matrix)
    #w_matrix=pm.Normal("w_matrix",mu=1,sigma=2,shape=(n_individuals,j_taxa)) En mettant ca tout marche


    #Matrice de proportions, Deterministic avec Softmax
    proportions_matrix=pm.Deterministic("proportions_matrix",pymc.math.softmax(pymc.math.concatenate([w_matrix,pt.zeros(shape=(n_individuals,1))],axis=1),axis=1))

    #mdine_model.debug()

    liste_sum_counts=[]
    for i in range(n_individuals):
        liste_sum_counts.append(sum(counts_matrix_data[i]))

    ###Matrice des comptes estimées, loi multinomiale

    counts_matrix=pm.Multinomial("counts_matrix",n=liste_sum_counts,p=proportions_matrix,observed=counts_matrix_data)

    #mdine_model.debug()

graph=pm.model_to_graphviz(mdine_model)
graph.render('mdine_graph2.gv')

with mdine_model:
    idata = pm.sample(100) ## 10000 normalement
    


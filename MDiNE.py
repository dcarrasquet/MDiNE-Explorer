import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import pytensor
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import csv
import pymc.math
from functools import partial

print(f"Running on PyMC v{pm.__version__}")

# fmt: off
disaster_data = pd.Series(
    [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
    3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
    2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
    1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
    0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
    3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
)
# fmt: on
years = np.arange(1851, 1962)

plt.plot(years, disaster_data, "o", markersize=8, alpha=0.4)
plt.ylabel("Disaster count")
plt.xlabel("Year")

def get_data(file_name):
    # Nom du fichier CSV
    #file_name = "crohns.csv"

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
            # Remplacer les chaînes "CD" par 1 et "no" par 0 dans chaque élément de la ligne
            ligne_transformee = [valeur.replace("CD", "1").replace("no", "0").replace("female", "1").replace("male", "0").replace("false", "0").replace("true", "1") for valeur in ligne]
            #print(ligne_transformee)
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

    # print("Matrice de covariates:\n",covariate_matrix,"\n\n")
    # print("Matrice de Z:\n",Z_vector,"\n\n")
    # print("Matrice Data Y:\n", counts_matrix)

    return(covariate_matrix,counts_matrix,Z_vector)

(covariate_matrix_data,counts_matrix_data,Z_vector_data)=get_data("crohns.csv")

with pm.Model() as mdine_model:
    # Comment gérer les matrices et les vecteurs?
    # https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_pytensor.html#what-is-going-on-behind-the-scenes 
    # https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/dimensionality.html

    j_taxa_plus_ref=counts_matrix_data.shape[1]
    n_individus=counts_matrix_data.shape[0]
    k_covariates= covariate_matrix_data.shape[1]   #### Covariate Matrix Data ne doit pas contenir la colonne contenant les z. 

    if covariate_matrix_data.shape[0]!=n_individus:
        print("Error dimensions between covariates matrix and counts matrix don't correspond ")
    
    #rng = np.random.default_rng(seed=sum(map(ord, "dimensionality")))
    #draw = partial(pm.draw, random_seed=rng)
    
    sigma=100
    beta_matrix=pm.Normal("beta_matrix",mu=0,sigma=sigma,shape=(k_covariates,j_taxa_plus_ref-1))

    lambda_init=50
    lambda_mdine=pm.Exponential("lambda_mdine",lam=lambda_init)

    j_taxa=j_taxa_plus_ref-1

    #precision_matrix= np.empty((j_taxa_plus_ref-1, j_taxa_plus_ref-1), dtype=object)
    precision_matrix=pytensor.tensor.matrix(name="precision_matrix",shape=(j_taxa,j_taxa))
    #precision_matrix=pytensor.tensor.matrix(name="precision_matrix",shape=(j_taxa_plus_ref-1, j_taxa_plus_ref-1))
    #j_taxa=precision_matrix.shape[0]

    print("Jtaxa: ",j_taxa)

    choix_precision_matrix="coef_by_coef"

    for i in range(j_taxa):
        for j in range(i+1):
            print("Couple i j ",i," ",j)
            if i == j:
                precision_matrix.set((i,j),pm.Exponential(f'diag_{i}_{j}', lam=lambda_mdine/2))
                #precision_matrix[i, j] = pm.Exponential(f'diag_{i}_{j}', lam=lambda_mdine/2)
            else:
                distrib= pm.Laplace(f'off_diag_{i}_{j}', mu=0, b=lambda_mdine)
                #precision_matrix[i, j] = distrib
                #precision_matrix[j, i] = distrib
                precision_matrix.set((i,j),distrib)
                precision_matrix.set((j,i),distrib)

    # print(precision_matrix)
    # print(type(precision_matrix))

    # precision_matrix= precision_matrix.tolist()
    # print(precision_matrix[0][0])
    # print(type(precision_matrix[0][0]))

    # #matrix = pm.Deterministic('AZ', np.array(precision_matrix))
    # inv_matrix = pm.Deterministic('inv_matrix', pymc.math.matrix_inverse(precision_matrix))

    #normal_dists = pm.math.stack([pm.Normal.dist() for _ in range(3)])
    #print("Test: ",type(normal_dists))

    #test_matrix=pytensor.tensor.matrix(name="Test_matrix",shape=(5,5))
    #test_matrix.set((0,0),pm.Laplace('off_diag_00', mu=0, b=lambda_mdine))
    #print(test_matrix)

    product_X_Beta=pm.Deterministic("Product_X_Beta",covariate_matrix_data@beta_matrix)

    #precision_matrix=pytensor.tensor.as_tensor_variable(precision_matrix,"precision_matrix")

    print("Type xbeta 0 0: ", type(product_X_Beta[0,0]))

    print("Type precision matrix: ", type(precision_matrix))
    print("Type productbeta: ", type(product_X_Beta))

    matrix_w=pytensor.tensor.matrix(name="matrix_w",shape=(n_individus,j_taxa))
    #matrix_w=np.empty((n_individus,j_taxa),dtype=object)

    for i in range (n_individus):
        string=f"W_row_{i}"
        #print("Taille W_i: ",matrix_w[i].shape[0])
        #matrix_w[i]=pm.MvNormal(string,mu=product_X_Beta[i,:],tau=precision_matrix,shape=(1,j_taxa))
        distrib=pm.MvNormal(string,mu=product_X_Beta[i,:],tau=precision_matrix,shape=(1,j_taxa))
        pytensor.tensor.subtensor.set_subtensor(matrix_w[i:], distrib)
        #matrix_w.set((i,j))

    #proportions_matrix=np.empty((n_individus,j_taxa_plus_ref),dtype=object)
    proportions_matrix=pytensor.tensor.matrix(name="proportions_matrix",shape=(n_individus,j_taxa_plus_ref))
    for i in range (n_individus):
            for j in range (j_taxa_plus_ref):
                if j==j_taxa_plus_ref-1:
                    #proportions_matrix[i,j]=pm.Deterministic(f"String_derniere_colonne{i}{j}",1/(1+sum(matrix_w[i])))
                    proportions_matrix.set((i,j),pm.Deterministic(f"String_derniere_colonne{i}{j}",1/(1+sum(matrix_w[i]))))
                else:
                    #proportions_matrix[i,j]=pm.Deterministic(f"String{i}{j}",matrix_w[i,j]/(1+sum(matrix_w[i])))
                    proportions_matrix.set((i,j),pm.Deterministic(f"String_derniere_colonne{i}{j}",matrix_w[i,j]/(1+sum(matrix_w[i]))))
                    print(matrix_w[i,j])

    #counts_matrix=np.empty((n_individus,j_taxa_plus_ref),dtype=object)
    counts_matrix=pytensor.tensor.matrix(name="counts_matrix",shape=(n_individus,j_taxa_plus_ref))
    for i in range (n_individus):
        #counts_matrix[i]=pm.Multinomial("counts_matrix",n=counts_matrix_data[i],p=proportions_matrix[i],observed=counts_matrix_data[i])
        pytensor.tensor.subtensor.set_subtensor(counts_matrix[i:], pm.Multinomial(f"counts_matrix_ligne{i}",n=counts_matrix_data[i],p=proportions_matrix[i],observed=counts_matrix_data[i,:]))


with mdine_model:
    idata = pm.sample(10000)

graph=pm.model_to_graphviz(mdine_model)
graph.render('mdine_graph.gv')

# /Users/damien/Documents/scolarité/Centrale\ Lyon/TFE/travail/
# export PATH="/opt/homebrew/bin:$PATH"
# dot -Tsvg > output.svg Test_graph.gv



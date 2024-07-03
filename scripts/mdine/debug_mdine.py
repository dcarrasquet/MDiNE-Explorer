import pymc as pm
import pytensor.tensor
import pytensor.tensor.shape
from scipy.special import softmax
import pytensor
import pytensor.tensor as pt
import pymc.math
import pickle
import math
import numpy as np
from pytensor.tensor.linalg import inv as matrix_inverse
#import scipy.stats
import json
import re
import os

import arviz as az
import matplotlib.pyplot as plt

import pandas as pd

import sys

def make_precision_matrix(coef_diag,coef_off_diag,j_taxa):

    triang_laplace=pymc.math.expand_packed_triangular(n=j_taxa-1,packed=coef_off_diag)
    premiere_etape=pymc.math.concatenate([triang_laplace,pt.zeros(shape=(j_taxa-1,1))],axis=1)
    triang_strict=pymc.math.concatenate([pt.zeros(shape=(1,j_taxa)),premiere_etape],axis=0)

    return triang_strict+pytensor.tensor.transpose(triang_strict)+pytensor.tensor.diag(coef_diag)

def model():

    with pm.Model() as mdine_model:

        filename="scripts/mdine/simulated_data_multinomial_N50_J20_K20.json"

        with open(filename, 'r') as f:
            data = json.load(f)
            i=0
            
            counts= pd.DataFrame(data[i]["Counts"])
            covariates=pd.DataFrame(data[i]["X"])
            beta=data[i]["beta"]
            prec0=data[i]["Prec0"]
            prec1=data[i]["Prec1"]
            Z=data[i]["Z"]

            proportions = counts.div(counts.sum(axis=1), axis=0)
            #print(proportions)

            print(proportions.sum(axis=1))



            # counts_0=counts[Z==0]
            # counts_1=counts[Z==1]
            # covariates_0=covariates[Z==0]
            # covariates_1=covariates[Z==1]

        j_taxa=len(prec0)

        lambda_init=100
        lambda_mdine=pm.Exponential("lambda_mdine",1/lambda_init) 
        #lambda_mdine=pm.Normal()

        diagonal_coefficients = np.diag(prec0)

        #Construction of the diagonal and off-diagonal coefficients.

        precision_matrix_coef_diag=pm.Exponential("precision_matrix_coef_diag",lam=lambda_mdine/2,shape=(j_taxa,),observed=diagonal_coefficients)
        # precision_matrix_coef_off_diag=pm.Laplace("precision_matrix_coef_off_diag",mu=0,b=lambda_mdine,shape=(j_taxa*(j_taxa-1)/2,))

        # precision_matrix=pm.Deterministic("precision_matrix",make_precision_matrix(precision_matrix_coef_diag,precision_matrix_coef_off_diag,j_taxa))

    with mdine_model:
        #mdine_model.debug()
        idata = pm.sample(1000) ## 10000 normally

    if idata!=None:
        folder="data/"
        with open(folder+"idata.pkl", "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
            pickle.dump(idata, f)

def model_mdine():
    # print("Covariates ",covariate_matrix_data)
    # print("Txaa: ",counts_matrix_data)
    # print("Simulation: ",simulation)
    simulation={
        'beta_matrix':{
            'apriori':'Normal',
            'parameters':{
                'alpha':1,
                'beta':1
            }
        },
        'precision_matrix':{
            'apriori':'Invwishart',
            'parameters':{
                'lambda_init':10
            }
        }
    }

    beta_matrix_choice=simulation["beta_matrix"]["apriori"]
    parameters_beta_matrix=simulation["beta_matrix"]["parameters"]

    precision_matrix_choice=simulation["precision_matrix"]["apriori"]
    parameters_precision_matrix=simulation["precision_matrix"]["parameters"]

    filename="scripts/mdine/simulated_data_multinomial_N50_J20_K20.json"

    with open(filename, 'r') as f:
        data = json.load(f)
        i=2
        
        counts_matrix_data= pd.DataFrame(data[i]["Counts"])
        covariate_matrix_data=pd.DataFrame(data[i]["X"])
        beta=data[i]["beta"]
        prec0=data[i]["Prec0"]
        prec1=data[i]["Prec1"]
        Z=data[i]["Z"]

        eigenvalues, eigenvectors = np.linalg.eig(prec0)

        print("Eigenvalues:",eigenvalues)

    with pm.Model() as mdine_model:

        j_taxa_plus_ref = counts_matrix_data.shape[1]
        n_individuals = counts_matrix_data.shape[0]
        k_covariates = covariate_matrix_data.shape[1]   #### Covariate Matrix Data must not contain the z column.
        j_taxa=j_taxa_plus_ref-1

        ## Matrice Beta

        if beta_matrix_choice=="Normal":
            beta_matrix=pm.Normal("beta_matrix",mu=0,sigma=1000,shape=(k_covariates,j_taxa))

        elif beta_matrix_choice=="Ridge":
            
            lambda_ridge=pm.Gamma("lambda_brige",alpha=parameters_beta_matrix["alpha"],beta=parameters_beta_matrix["beta"],shape=(k_covariates,j_taxa))
            beta_matrix=pm.Normal("beta_matrix",mu=0,tau=lambda_ridge,shape=(k_covariates,j_taxa))

        elif beta_matrix_choice=="Lasso":
            alpha_sigma=parameters_beta_matrix["alpha_sigma"]
            beta_sigma=parameters_beta_matrix["beta_sigma"]
            alpha_lambda=parameters_beta_matrix["alpha_lambda"]
            beta_lambda=parameters_beta_matrix["beta_lambda"]


            lambda_square=pm.Gamma("lambda_square",alpha=alpha_lambda,beta=beta_lambda)
            sigma_square=pm.InverseGamma("sigma_square",alpha=alpha_sigma,beta=beta_sigma)

            
            tau_matrix=pm.Exponential("tau_vector",lam=lambda_square/2,shape=(k_covariates,j_taxa))

            beta_matrix=pm.Normal("beta_matrix",mu=0,sigma=sigma_square*tau_matrix)

        elif beta_matrix_choice=="Horseshoe":
            tau=pm.HalfCauchy("tau",beta=parameters_beta_matrix["beta_tau"])
            lambda_horseshoe=pm.HalfCauchy("lambda_horseshoe",beta=tau,shape=(k_covariates,j_taxa))

            beta_matrix=pm.Normal("beta_matrix",mu=0,sigma=lambda_horseshoe)
        
        elif beta_matrix_choice=="Spike_and_slab":
            #proba_gamma=0.5 # Eventually pm.beta
            alpha_gamma=parameters_beta_matrix["alpha_gamma"]
            beta_gamma=parameters_beta_matrix["beta_gamma"]
            proba_gamma=pm.Beta("pi_gamma",alpha=alpha_gamma,beta=beta_gamma,shape=(k_covariates,j_taxa))
            gamma_matrix=pm.Bernoulli("gamma",p=proba_gamma,shape=(k_covariates,j_taxa))

            tau=parameters_beta_matrix["tau"]
            
            c=parameters_beta_matrix["c"]

            # Hadamard product between two tensors: A*B
            beta_matrix=pm.Deterministic("beta_matrix",(pymc.math.ones((k_covariates, j_taxa))-gamma_matrix)*pm.Normal("Beta_Spike",mu=0, sigma=tau, shape=(k_covariates, j_taxa))+gamma_matrix*pm.Normal("Beta_Slab",mu=0, sigma=tau*c, shape=(k_covariates, j_taxa)))

        else:
            raise ValueError(f"Invalid precision_matrix_choice: {precision_matrix_choice}")

        ## Precision Matrix

        if precision_matrix_choice=="Lasso":

            lambda_init=parameters_precision_matrix["lambda_init"]
            lambda_mdine=pm.Exponential("lambda_mdine",1/lambda_init) 

            #Construction of the diagonal and off-diagonal coefficients.

            precision_matrix_coef_diag=pm.Exponential("precision_matrix_coef_diag",lam=lambda_mdine/2,shape=(j_taxa,))
            precision_matrix_coef_off_diag=pm.Laplace("precision_matrix_coef_off_diag",mu=0,b=lambda_mdine,shape=(j_taxa*(j_taxa-1)/2,))

            precision_matrix=pm.Deterministic("precision_matrix",make_precision_matrix(precision_matrix_coef_diag,precision_matrix_coef_off_diag,j_taxa))

        elif precision_matrix_choice=="Invwishart":
            scale_matrix=np.eye(j_taxa)
            df=j_taxa+2 #Degrees of freedom
            
            covariance_matrix=pm.WishartBartlett("covariance_matrix",scale_matrix,df)
            precision_matrix=pm.Deterministic("precision_matrix",matrix_inverse(covariance_matrix))
            

        elif precision_matrix_choice=="Invwishart_penalized":

            #User Choice
            df=j_taxa+2 # degrees of freedom
            coefs_user=[1 for _ in range(j_taxa)]

            #Scale Matrix construction for Wishart distribution
            transformed_coefs=[1/(i**2) for i in coefs_user]
            coef_diag_scale_matrix=pm.InverseGamma("coef_penalisation",1/2,transformed_coefs)
            inv_coef_diag_scale_matrix=pt.as_tensor_variable([1/i for i in coef_diag_scale_matrix])
            scale_matrix=2*df*pytensor.tensor.diag(inv_coef_diag_scale_matrix)


            #####Wishart matrix construction

            ## Matrix_A construction, off diag coefficients

            coef_off_diag_A=pm.Normal("coef_off_diag",mu=0,sigma=1,shape=(j_taxa*(j_taxa-1)/2,))

            triang_normal_A=pymc.math.expand_packed_triangular(n=j_taxa-1,packed=coef_off_diag_A,lower=True)
            premiere_etape_A=pymc.math.concatenate([triang_normal_A,pt.zeros(shape=(j_taxa-1,1))],axis=1)
            triang_strict_A=pymc.math.concatenate([pt.zeros(shape=(1,j_taxa)),premiere_etape_A],axis=0)


            ## Matrix_A construction, diag coefficients

            list_degrees_freedom=[df-i-1 for i in range(j_taxa)]
            list_degrees_freedom=pt.as_tensor_variable(list_degrees_freedom)

            coef_sqr_diag_A=pm.ChiSquared("chi2_coef",nu=list_degrees_freedom) 
            coef_diag_A=pytensor.tensor.sqrt(coef_sqr_diag_A)

            matrix_A=triang_strict_A+pytensor.tensor.diag(coef_diag_A) #pytensor.tensor.transpose(triang_strict)+

            matrix_L=pytensor.tensor.linalg.cholesky(scale_matrix)

            covariance_matrix=pt.dot(matrix_L,pt.dot(pt.dot(matrix_A,pt.transpose(matrix_A)),pt.transpose(matrix_L)))   

            precision_matrix=pm.Deterministic("precision_matrix",matrix_inverse(covariance_matrix))

        else:
            raise ValueError(f"Invalid precision_matrix_choice: {precision_matrix_choice}")


    
        ## Covariate matrix, allows it to be displayed on the graph.

        covariate_matrix=pm.Deterministic("X_covariates",pytensor.tensor.as_tensor_variable(covariate_matrix_data))

        product_X_Beta=pm.Deterministic("Product_X_Beta",covariate_matrix@beta_matrix)

        # Matrix W, Normal distribution
        w_matrix=pm.MvNormal("w_matrix",mu=product_X_Beta,tau=precision_matrix)
        
        #Proportions Matrix, Deterministic with Softmax
        proportions_matrix=pm.Deterministic("proportions_matrix",pymc.math.softmax(pymc.math.concatenate([w_matrix,pt.zeros(shape=(n_individuals,1))],axis=1),axis=1))

        liste_sum_counts = counts_matrix_data.sum(axis=1).tolist()

        ## Estimated counts matrix, Multinomial distribution
        #mdine_model.debug()

        counts_matrix=pm.Multinomial("counts_matrix",n=liste_sum_counts,p=proportions_matrix,observed=counts_matrix_data)

    with mdine_model:
        #mdine_model.debug()
        idata = pm.sample(1000) ## 10000 normally

    if idata!=None:
        folder="data/"
        with open(folder+"idata.pkl", "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
            pickle.dump(idata, f)

def study_idata():
    folder="data/"
    with open(folder+"idata.pkl", "rb") as fichier:
        idata = pickle.load(fichier)

    #print(az.hdi(idata, var_names=["lambda_mdine"], hdi_prob=0.94).lambda_mdine)

    az.plot_forest(idata)
    plt.show()

    # hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix_coef_diag"], hdi_prob=0.10).precision_matrix_coef_diag
    # print(hdi_precision_matrix)

def show_exponential_distribution(rate=0.12/2, size=1000):
    # Générer des données de la distribution exponentielle
    data = np.random.exponential(scale=1/rate, size=size)
    
    # Créer un histogramme pour visualiser la distribution
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    
    # Ajouter le titre et les labels
    plt.title('Distribution Exponentielle avec un Facteur de 0.06')
    plt.xlabel('Valeur')
    plt.ylabel('Densité de Probabilité')
    
    # Afficher le graphique
    plt.show()


if __name__=="__main__":
   # show_exponential_distribution()
    model_mdine()
    study_idata()

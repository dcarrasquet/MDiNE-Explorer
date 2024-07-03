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
    
try:
    # First Try
    from mdine.extract_data_files import get_separate_data, get_df_covariates, get_df_taxa
    from mdine.verify_mdine_model import generate_counts_data_Multinomial, generate_counts_data_ZINB
except ImportError:
    #Second Try
    from extract_data_files import get_separate_data, get_df_covariates, get_df_taxa
    from verify_mdine_model import generate_counts_data_Multinomial, generate_counts_data_ZINB

#File "/Users/damien/Documents/scolarite\xcc\x81/Centrale_Lyon/TFE/travail/mdine_github/scripts/mdine/MDiNE_model.py", line 38, in <module>    from verify_mdine_model import generate_counts_data_Multinomial, generate_counts_data_ZINB  File "/Users/damien/Documents/scolarite\xcc\x81/Centrale_Lyon/TFE/travail/mdine_github/scripts/mdine/verify_mdine_model.py", line 22, in <module>    from mdine.extract_data_files import get_data,separate_data_in_two_groupsModuleNotFoundError: No module named \'mdine\''

#print(f"Running on PyMC v{pm.__version__}")

def run_simulation(path_json_file):

    # Path of the parent folder
    folder_parent = 'test_mdine'

    # Liste pour stocker les noms des sous-dossiers
    sub_folders = []

    # Parcourir tous les éléments dans le dossier parent
    for element in os.listdir(folder_parent):
        # Vérifier si l'élément est un dossier
        complete_path = os.path.join(folder_parent, element)
        if os.path.isdir(complete_path):
            # Ajouter le nom du sous-dossier à la liste
            sub_folders.append(element)

    numbers_simulations = [int(re.search(r'\d+', nom).group()) for nom in sub_folders if re.search(r'\d+', nom)]
    simulation_number = max(numbers_simulations)+1 if numbers_simulations else 1

    folder_simulation=folder_parent+"/simulation_"+str(simulation_number)+"/"
    os.makedirs(folder_simulation)
    
    with open(path_json_file, 'r') as file:
        # Chargement des données JSON
        simulation = json.load(file)

    # Écrire les données dans le fichier JSON
    with open(folder_simulation+"simulation.json", "w") as fichier_json:
        json.dump(simulation, fichier_json, indent=4)

    if simulation["generated_data"]==True:

        model_generated_data=simulation["data"]["model_generated_data"]

        if model_generated_data=="Multinomial":
            generate_data=generate_counts_data_Multinomial
        elif model_generated_data=="ZINB":
            generate_data=generate_counts_data_ZINB
        else:
            print('The template for generating data is invalid. Choose between "Multinomial" and "ZINB".')

        n_individus=simulation["data"]["n_individus"]
        j_taxa=simulation["data"]["j_taxa"]
        k_covariables=simulation["data"]["k_covariates"]

        for i in range (simulation["data"]["nb_tests"]):
            folder_test=folder_simulation+"/test_"+str(i)+"/"
            os.makedirs(folder_test)
            (X_matrix,counts_matrix)=generate_data(n_individus,k_covariables,j_taxa,folder_test)


            #print(X_matrix)
            #print(X_matrix.shape)

            # print(counts_matrix)
            # print(counts_matrix.shape)
            run_model(X_matrix,counts_matrix,simulation,folder_test)

        


    return 0


def simulation_data_R(filename):

    folder_parent = 'data_R/'

    # Liste pour stocker les noms des sous-dossiers
    sub_folders = []

    # Parcourir tous les éléments dans le dossier parent
    for element in os.listdir(folder_parent):
        # Vérifier si l'élément est un dossier
        complete_path = os.path.join(folder_parent, element)
        if os.path.isdir(complete_path):
            # Ajouter le nom du sous-dossier à la liste
            sub_folders.append(element)

    numbers_simulations = [int(re.search(r'\d+', nom).group()) for nom in sub_folders if re.search(r'\d+', nom)]
    simulation_number = max(numbers_simulations)+1 if numbers_simulations else 1

    folder_simulation=folder_parent+"/simulation_"+str(simulation_number)+"/"
    os.makedirs(folder_simulation)

    model_parameters={
    "beta_matrix": {
        "apriori": "Ridge",
        "parameters": {
            "alpha": 1,
            "beta": 1
        }
    },
    "precision_matrix": {
        "apriori": "Lasso",
        "parameters": {
            "lambda_init": 10
        }
    }}

    with open(filename, 'r') as f:
        data = json.load(f)
        for i in range (len(data)):
            counts= pd.DataFrame(data[i]["Counts"])
            covariates=pd.DataFrame(data[i]["X"])
            beta=data[i]["beta"]
            prec0=data[i]["Prec0"]
            prec1=data[i]["Prec1"]
            Z=data[i]["Z"]

            counts_0=counts[Z==0]
            counts_1=counts[Z==1]
            covariates_0=covariates[Z==0]
            covariates_1=covariates[Z==1]

            run_model(covariates_0,counts_0,model_parameters,folder_simulation+f"test_{i}_group_0")
            run_model(covariates_1,counts_1,model_parameters,folder_simulation+f"test_{i}_group_1")

def debug_MDiNE_model():
    parameters_model={
        'beta_matrix':{
            'apriori':'Normal',
            'parameters':{
                'alpha':1,
                'beta':1
            }
        },
        'precision_matrix':{
            'apriori':'Lasso',
            'parameters':{
                'lambda_init':1
            }
        }
    }
    file_path="data/crohns-numeric-tsv.tsv"
    df=pd.read_table(file_path,sep='\t')
    start_cov=2
    end_cov=5
    start_taxa=6
    end_taxa=11
    df_covariates=df.iloc[:,start_cov-1:end_cov]
    df_taxa=df.iloc[:,start_taxa-1:end_taxa]
    run_model(df_covariates,df_taxa,parameters_model,"data/debug_MDiNE")

def run_model(covariate_matrix_data,counts_matrix_data,simulation,folder):

    # print("Covariates ",covariate_matrix_data)
    # print("Txaa: ",counts_matrix_data)
    # print("Simulation: ",simulation)

    beta_matrix_choice=simulation["beta_matrix"]["apriori"]
    parameters_beta_matrix=simulation["beta_matrix"]["parameters"]

    precision_matrix_choice=simulation["precision_matrix"]["apriori"]
    parameters_precision_matrix=simulation["precision_matrix"]["parameters"]

    if not os.path.exists(folder):
        os.makedirs(folder)

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
            precision_matrix_coef_off_diag=pm.Laplace("precision_matrix_coef_off_diag",mu=0,b=1/lambda_mdine,shape=(j_taxa*(j_taxa-1)/2,))

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

        counts_matrix=pm.Multinomial("counts_matrix",n=liste_sum_counts,p=proportions_matrix,observed=counts_matrix_data)

    with mdine_model:
        #mdine_model.debug()
        idata = pm.sample(1000) ## 10000 normally

    if idata!=None:
        print("Folder: ",folder)
        with open(folder+"idata.pkl", "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
            pickle.dump(idata, f)
            print("Fichier écrit avec succès!")

def make_precision_matrix(coef_diag,coef_off_diag,j_taxa):

    triang_laplace=pymc.math.expand_packed_triangular(n=j_taxa-1,packed=coef_off_diag)
    premiere_etape=pymc.math.concatenate([triang_laplace,pt.zeros(shape=(j_taxa-1,1))],axis=1)
    triang_strict=pymc.math.concatenate([pt.zeros(shape=(1,j_taxa)),premiere_etape],axis=0)

    return triang_strict+pytensor.tensor.transpose(triang_strict)+pytensor.tensor.diag(coef_diag)

def estimate_lambda_init(covariate_matrix_data,counts_matrix_data):

    with pm.Model() as estimate_lambda_model:

        j_taxa_plus_ref = counts_matrix_data.shape[1]
        n_individuals = counts_matrix_data.shape[0]
        k_covariates = covariate_matrix_data.shape[1]   #### Covariate Matrix Data must not contain the z column.
        j_taxa=j_taxa_plus_ref-1

        ## Matrice Beta, loi normale de parametre sigma
        sigma=1000
        beta_matrix=pm.Normal("beta_matrix",mu=0,sigma=sigma,shape=(k_covariates,j_taxa_plus_ref-1))
        covariate_matrix=pm.Deterministic("X_covariates",pytensor.tensor.as_tensor_variable(covariate_matrix_data))

        w_matrix=pm.Deterministic("w_matrix",covariate_matrix@beta_matrix)

        #Matrice de proportions, Deterministic avec Softmax
        proportions_matrix=pm.Deterministic("proportions_matrix",pymc.math.softmax(pymc.math.concatenate([w_matrix,pt.zeros(shape=(n_individuals,1))],axis=1),axis=1))

        #mdine_model.debug()

        liste_sum_counts=[]
        for i in range(n_individuals):
            liste_sum_counts.append(sum(counts_matrix_data[i]))

        ###Matrice des comptes estimées, loi multinomiale

        counts_matrix=pm.Multinomial("counts_matrix",n=liste_sum_counts,p=proportions_matrix,observed=counts_matrix_data)

        graph=pm.model_to_graphviz(estimate_lambda_model)

        graph.render('mdine_graph_model/estimate_lambda_model.gv')

        with estimate_lambda_model:
            idata = pm.sample(10000) ## 10000 normally

        Beta_matrix=idata.posterior.beta_matrix.values[3][-1]

        # print("Beta Matrix:\n")
        # print(Beta_matrix)

        product_X_B=np.dot(covariate_matrix_data,Beta_matrix)

        residual_matrix=np.zeros((n_individuals,j_taxa))

        for i in range (n_individuals):
            for j in range (j_taxa):
                residual_matrix[i,j]=math.log((1+counts_matrix_data[i,j])/(1+counts_matrix_data[i,j_taxa]))-product_X_B[i,j]
        
        two_groups=True
        if two_groups==True:
            first_residual=residual_matrix ### Manque Z
            second_residual=residual_matrix ### Z Vector manquant

        first_precision_matrix=np.linalg.pinv(np.cov(first_residual))
        second_precision_matrix=np.linalg.pinv(np.cov(second_residual))

        sum_coef=0

        for i in range(j_taxa):
            sum_coef+=0.5*(first_precision_matrix[i,i]+second_precision_matrix[i,i])
            for j in range(i):
                sum_coef+=abs(first_precision_matrix[i,j])+abs(second_precision_matrix[i,j])


        lambda_init=(j_taxa+1)*j_taxa/sum_coef
        
    return lambda_init

def run_model_terminal():

    info_current_file_store=json.loads(sys.argv[1])

    if info_current_file_store["phenotype_column"]==None:
        #Only one group
        print("Start Inference")  
        df_covariates=get_df_covariates(info_current_file_store)
        df_taxa=get_df_taxa(info_current_file_store,"df_taxa")
        run_model(df_covariates,df_taxa,info_current_file_store["parameters_model"],info_current_file_store["session_folder"])

    elif info_current_file_store["phenotype_column"]!=None:

        [df_covariates_1,df_taxa_1],[df_covariates_2,df_taxa_2]=get_separate_data(info_current_file_store)
        path_first_group=os.path.join(info_current_file_store["session_folder"],"first_group/")
        path_second_group=os.path.join(info_current_file_store["session_folder"],"second_group/")

        print("Start first inference")
        run_model(df_covariates_1,df_taxa_1,info_current_file_store["parameters_model"],path_first_group)

        print("Start second inference")
        run_model(df_covariates_2,df_taxa_2,info_current_file_store["parameters_model"],path_second_group)

if __name__=="__main__":
    run_model_terminal()
    


    


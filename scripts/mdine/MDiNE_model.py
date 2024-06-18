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



from mdine.extract_data_files import get_data
from mdine.extract_data_files import separate_data_in_two_groups
from mdine.verify_mdine_model import generate_counts_data_Multinomial,generate_counts_data_ZINB


print(f"Running on PyMC v{pm.__version__}")

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

            

def terminal_run_model():
    covariate_matrix_data=sys.argv[1]
    counts_matrix_data=sys.argv[2]
    simulation=sys.argv[3]
    folder=sys.argv[4]
    run_model(covariate_matrix_data,counts_matrix_data,simulation,folder)



def run_model(covariate_matrix_data,counts_matrix_data,simulation,folder):

    beta_matrix_choice=simulation["beta_matrix"]["apriori"]
    parameters_beta_matrix=simulation["beta_matrix"]["parameters"]

    precision_matrix_choice=simulation["precision_matrix"]["apriori"]
    parameters_precision_matrix=simulation["precision_matrix"]["parameters"]

    if not os.path.exists(folder):
        # Créer le sous-dossier s'il n'existe pas
        os.makedirs(folder)

    with pm.Model() as mdine_model:

        j_taxa_plus_ref = counts_matrix_data.shape[1]
        n_individuals = counts_matrix_data.shape[0]
        k_covariates = covariate_matrix_data.shape[1]   #### Covariate Matrix Data must not contain the z column.
        j_taxa=j_taxa_plus_ref-1

        ## Matrice Beta

        #beta_matrix_choice="Normal"

        if beta_matrix_choice=="Ridge":
            
            lambda_ridge=pm.Gamma("lambda_brige",alpha=parameters_beta_matrix["alpha"],beta=parameters_beta_matrix["beta"],shape=(k_covariates,j_taxa))
            #lambda_ridge=pm.Gamma("lambda_brige",alpha=1,beta=1,shape=(k_covariates,j_taxa))

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

            # Générer une matrice de nombres aléatoires entre 0.1 et 1 avec deux décimales
            #tau_matrix = np.random.uniform(low=0.01, high=0.1, size=(k_covariates, j_taxa)).round(2)
            tau=parameters_beta_matrix["tau"]
            #print(tau_matrix)

            # Générer une matrice d'entiers aléatoires entre 10 et 100
            #c_matrix = np.random.uniform(low=0.1, high=1, size=(k_covariates, j_taxa)).round(2)
            c=parameters_beta_matrix["c"]
            #print(c_matrix)

            # Hadamard product between two tensors: A*B
            #(pymc.math.ones(k_covariates, j_taxa)-gamma_matrix)*pm.Normal.dist(mu=0, sigma=tau_matrix, shape=(k_covariates, j_taxa))+gamma_matrix*pm.Normal.dist(mu=0, sigma=tau_matrix*c_matrix, shape=(k_covariates, j_taxa))
            beta_matrix=pm.Deterministic("beta_matrix",(pymc.math.ones((k_covariates, j_taxa))-gamma_matrix)*pm.Normal("Beta_Spike",mu=0, sigma=tau, shape=(k_covariates, j_taxa))+gamma_matrix*pm.Normal("Beta_Slab",mu=0, sigma=tau*c, shape=(k_covariates, j_taxa)))

        else:
            print("The choice for the Beta matrix is incorrect")

        ## Precision Matrix
        #precision_matrix_choice="invwishart_penalized"

        if precision_matrix_choice=="Lasso":

            lambda_init=parameters_precision_matrix["lambda_init"]
            #lambda_init=10
            lambda_mdine=pm.Exponential("lambda_mdine",lambda_init)

            # Construction des coefficients diagonaux et extra-diagonaux

            precision_matrix_coef_diag=pm.Exponential("precision_matrix_coef_diag",lam=lambda_mdine/2,shape=(j_taxa,))
            precision_matrix_coef_off_diag=pm.Laplace("precision_matrix_coef_off_diag",mu=0,b=lambda_mdine,shape=(j_taxa*(j_taxa-1)/2,))

            # Assemblage de la matrice de précision à partir des coefficients ci-dessus
            precision_matrix=pm.Deterministic("precision_matrix",make_precision_matrix(precision_matrix_coef_diag,precision_matrix_coef_off_diag,j_taxa))

        elif precision_matrix_choice=="Invwishart":
            scale_matrix=np.eye(j_taxa)
            
            covariance_matrix=pm.WishartBartlett("covariance_matrix",scale_matrix,j_taxa+2)
            precision_matrix=pm.Deterministic("precision_matrix",matrix_inverse(covariance_matrix))
            

        elif precision_matrix_choice=="Invwishart_penalized":

            #User Choice
            df=j_taxa+2 # degrees of freedom
            coefs_user=[1 for _ in range(j_taxa)]

            #Scale Matrix construction for Wishart distribution
            transformed_coefs=[1/(i**2) for i in coefs_user]
            coef_diag_scale_matrix=pm.InverseGamma("coef_penalisation",1/2,transformed_coefs)
            #inv_coef_diag_scale_matrix=pytensor.tensor.as_tensor_variable([1/i for i in coef_diag_scale_matrix])
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
            print("The choice for the precision matrix is incorrect")


    
        ## Matrice de covariables, permet de l'afficher sur le graphique

        covariate_matrix=pm.Deterministic("X_covariates",pytensor.tensor.as_tensor_variable(covariate_matrix_data))

        product_X_Beta=pm.Deterministic("Product_X_Beta",covariate_matrix@beta_matrix)

        # Matrice W, loi normale

        w_matrix=pm.MvNormal("w_matrix",mu=product_X_Beta,tau=precision_matrix)
        #w_matrix=pm.MvNormal("w_matrix",mu=product_X_Beta,cov=precision_matrix)
        


        #Matrice de proportions, Deterministic avec Softmax
        proportions_matrix=pm.Deterministic("proportions_matrix",pymc.math.softmax(pymc.math.concatenate([w_matrix,pt.zeros(shape=(n_individuals,1))],axis=1),axis=1))

        #mdine_model.debug()

        # liste_sum_counts=[]
        # for i in range(n_individuals):
        #     liste_sum_counts.append(sum(counts_matrix_data[i]))

        liste_sum_counts = counts_matrix_data.sum(axis=1).tolist()

        ###Matrice des comptes estimées, loi multinomiale

        counts_matrix=pm.Multinomial("counts_matrix",n=liste_sum_counts,p=proportions_matrix,observed=counts_matrix_data)


    #graph=pm.model_to_graphviz(mdine_model) # <class 'graphviz.graphs.Digraph'>

    
    #graph.render(folder+"model_graph",format="png")

    with mdine_model:
        #mdine_model.debug()
        #idata = pm.sample(1000) ## 10000 normally
        print("Je vais faire un fit")
        idata=None
        mean_field=pm.fit()
        print("Mean field", mean_field)



    # Beta_matrix=idata.posterior.beta_matrix.values[3][-1]

    # print("Beta Matrix:\n")
    # print(Beta_matrix)

    # with mdine_model:
    #     ppc=pm.sample_posterior_predictive(idata,extend_inferencedata=True)

    # az.plot_forest(idata, var_names=["beta_matrix"], hdi_prob=0.80)
    # az.plot_forest(idata, var_names=["precision_matrix"], hdi_prob=0.80)

    # az.plot_ppc(ppc,kind="kde")
    # plt.plot()
    # plt.show()

    #### Parties pour sauvegarder toutes les figures

    save_figures=False

    if save_figures==True:

        with mdine_model:
            ppc=pm.sample_posterior_predictive(idata,extend_inferencedata=True)

        az.plot_ppc(ppc,kind="kde")
        plt.savefig(folder+"ppc_kde.png")

        az.plot_ppc(ppc,kind="cumulative")
        plt.savefig(folder+"ppc_cumulative.png")
        #plt.plot()
        #print(az.summary(idata, kind="stats"))

        #print(idata.sample_stats)
        idata.sample_stats["tree_depth"].plot(col="chain", ls="none", marker=".", alpha=0.3)
        plt.savefig(folder+"sample_stats.png")
        
        az.plot_energy(idata, figsize=(6, 4))
        plt.savefig(folder+"energy.png")

        az.plot_posterior(idata, group="sample_stats", var_names="acceptance_rate", hdi_prob="hide", kind="hist")
        plt.savefig(folder+"posterior.png")


        #print("Divergence? :",idata.sample_stats["diverging"].sum())
        az.plot_trace(idata, var_names=["beta_matrix"])
        plt.savefig(folder+"beta_trace.png")
        #az.plot_trace(idata, var_names=["beta_matrix","tau","lambda_horseshoe"])
        #az.plot_trace(idata, var_names=["beta_matrix","pi_gamma","gamma"])
    
        #plt.show(block=False)

        az.plot_forest(idata, var_names=["beta_matrix"], hdi_prob=0.80)
        ax = plt.gca()

        # Tracer une ligne verticale en rouge pointillée à x=0
        #plt.plot(beta, np.arange(len(beta)), 'rx', label='Vraies données')
        ax.axvline(x=0, color='red', linestyle='--')
        plt.savefig(folder+"beta_forest.png")

        az.plot_trace(idata, var_names=["precision_matrix"])
        plt.savefig(folder+"precision_trace.png")
        #az.plot_trace(idata, var_names=["beta_matrix","tau","lambda_horseshoe"])
        #az.plot_trace(idata, var_names=["beta_matrix","pi_gamma","gamma"])
    
        #plt.show(block=False)

        az.plot_forest(idata, var_names=["precision_matrix"], hdi_prob=0.80)
        ax = plt.gca()

        # Tracer une ligne verticale en rouge pointillée à x=0
        #plt.plot(beta, np.arange(len(beta)), 'rx', label='Vraies données')
        ax.axvline(x=0, color='red', linestyle='--')
        plt.savefig(folder+"precision_forest.png")

	    #plt.show()

    # with open(f"results_simulations/{simulation_name}.pkl", "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
    #     pickle.dump(idata, f)

    if idata!=None:
        with open(folder+"idata.pkl", "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
            pickle.dump(idata, f)
    
    if mean_field!=None:
        with open(folder+"mean_field.pkl", "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
            pickle.dump(mean_field, f)

    
    # axes_arr = az.plot_trace(idata)
    # plt.draw()
    # plt.show()

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

        print("Beta Matrix:\n")
        print(Beta_matrix)

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

if __name__=="__main__":


    #run_simulation("test_mdine/examples_json_simulation/ridge_lasso_5.json")
    #run_simulation("test_mdine/examples_json_simulation/horseshoe_lasso.json")

    #test_wishart(3)
    
    
    # filename="data/crohns.csv"
    # (covariate_matrix_data,counts_matrix_data,Z_vector)=get_data(filename)
    # first_group,second_group=separate_data_in_two_groups(covariate_matrix_data,counts_matrix_data,Z_vector)

    beta_matrix_choice="Ridge" #Spike_and_slab Normal Lasso Horseshoe
    precision_matrix_choice="exp_Laplace" #exp_Laplace ou invwishart ou invwishart_penalized
    # stringenplus="_Lambda2"

    # simulation_name_0="simulation_group_0_"+beta_matrix_choice+"_"+precision_matrix_choice+stringenplus
    # simulation_name_1="simulation_group_1_"+beta_matrix_choice+"_"+precision_matrix_choice

    # #estimate_lambda_init(covariate_matrix_data,counts_matrix_data)

    simulation_name="simulation_data_generated_"+beta_matrix_choice+"_"+precision_matrix_choice
    # run_model(first_group[0],first_group[1],beta_matrix_choice,precision_matrix_choice,simulation_name_0)
    # #run_model(second_group[0],second_group[1],beta_matrix_choice,precision_matrix_choice,simulation_name_1)
    # Sauvegarder les variables dans un fichier
    #print(generate_data_mdine())
    #(X_matrix,counts_matrix)=generate_data_mdine()
    #run_model(X_matrix,counts_matrix,beta_matrix_choice,precision_matrix_choice,simulation_name)


    


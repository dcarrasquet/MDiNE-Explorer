import json
import re
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle
import numpy as np

import pandas as pd

import pymc as pm
import pytensor.tensor
import pytensor.tensor.shape
from scipy.special import softmax
import pytensor
import pytensor.tensor as pt
import pymc.math
from pytensor.tensor.linalg import inv as matrix_inverse


def simulation_data_R(folder_parent):

    json_filename = [f for f in os.listdir(folder_parent) if f.endswith('.json')][0]

    print(json_filename)

    split_infos = json_filename.split("_") #"simulated_data_NB_N100_J50_K20.json"
    model_generated_data = split_infos[2]
    N = int(split_infos[3][1:])
    J = int(split_infos[4][1:])
    K = int(split_infos[5].split(".")[0][1:])

    print(N,J,K)

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

    summary_simulation={
        "parameters":{
            "Model": model_generated_data,
            "N":N,
            "K":K,
            "J":J,
        },
        "AUC_Prec_0": [],
        "AUC_Prec_1": [],
        "AUC_Beta_0": [],
        "AUC_Beta_1": [],
        "Time_group_0": [],
        "Time_group_1": [],
    }

    with open(os.path.join(folder_parent, json_filename), 'r') as f:
        data = json.load(f)

    for i in range (len(data)):
        counts= pd.DataFrame(data[i]["Counts"])

        #print(counts)

        #print(counts.shape)

        #covariates=data[i]["X"]
        covariates=pd.DataFrame(data[i]["X"])

        #print(covariates.shape)
        beta=data[i]["beta"]
        #print(beta.shape)
        prec0=data[i]["Prec0"]
        #print("Precision: ",len(prec0[0]),len(prec0))
        prec1=data[i]["Prec1"]
        #print(prec1.shape)
        Z=pd.DataFrame(data[i]["Z"])
        Z = Z.iloc[:, 0]

        #print(Z)


        counts_0=counts[Z==0]
        counts_1=counts[Z==1]
        covariates_0=covariates[Z==0]
        covariates_1=covariates[Z==1]

        #print(counts_0)
        #print(covariates_0)

        #counts_0=counts[Z==0].values.tolist()
        #counts_1=counts[Z==1].values.tolist()
        #covariates_0=covariates[Z==0].values.tolist()
        #covariates_1=covariates[Z==1].values.tolist()

        idata_filename_0=os.path.join(folder_parent,f"idata_test_{i}_group_0.pkl")
        idata_filename_1=os.path.join(folder_parent,f"idata_test_{i}_group_1.pkl")

        run_model(covariates_0,counts_0,model_parameters,idata_filename_0)
        summary_simulation["AUC_Prec_0"].append(get_AUC_Prec(prec0,idata_filename_0))
        summary_simulation["AUC_Beta_0"].append(get_AUC_Beta(beta,idata_filename_0))
        summary_simulation["Time_group_0"].append(get_time_simu(idata_filename_0))

        run_model(covariates_1,counts_1,model_parameters,idata_filename_1)
        summary_simulation["AUC_Prec_1"].append(get_AUC_Prec(prec1,idata_filename_1))
        summary_simulation["AUC_Beta_1"].append(get_AUC_Beta(beta,idata_filename_1))
        summary_simulation["Time_group_1"].append(get_time_simu(idata_filename_1))

        with open(os.path.join(folder_parent, "summary_simu.json"), "w") as fichier_json:
            json.dump(summary_simulation, fichier_json, indent=4)

        os.remove(idata_filename_0)
        os.remove(idata_filename_1)

    print("Fichier JSON créé avec succès :", os.path.join(folder_parent, "summary_simu.json"))

def get_time_simu(idata_filename):
    with open(idata_filename, "rb") as fichier:
        idata = pickle.load(fichier)

    return idata.sample_stats.sampling_time


def get_AUC_Beta(beta,idata_filename):
    list_hdi_prob = np.linspace(0.001, 0.999, num=50)
    list_FPR=[]
    list_TPR=[]
    for hdi_prob in list_hdi_prob:
        FPR,TPR=get_fpr_tpr_Beta(beta,idata_filename,hdi_prob)
        list_FPR.append(FPR)
        list_TPR.append(TPR)

    sorted_index = sorted(range(len(list_FPR)), key=lambda i: list_FPR[i])
    FPR_sorted = [list_FPR[i] for i in sorted_index]
    TPR_sorted = [list_TPR[i] for i in sorted_index] 

    auc_metric = np.trapz(TPR_sorted, FPR_sorted)

    return auc_metric

def get_fpr_tpr_Beta(beta_matrix,idata_filename,hdi_probability):
    #parent_folder="results/mdine_generated_data_2/"

    TP,FP,FN,TN=0,0,0,0

    np.set_printoptions(suppress=True)

    with open(idata_filename, "rb") as fichier:
        idata = pickle.load(fichier)

    hdi_beta_matrix = az.hdi(idata, var_names=["beta_matrix"], hdi_prob=hdi_probability).beta_matrix

    #print("HDI BETA MATRIX:",hdi_beta_matrix)

    k_covariates=len(beta_matrix)
    j_taxa_ref=len(beta_matrix[0])

    #print("K J: ",k_covariates,j_taxa_ref)
    

    for i in range(k_covariates):
        for j in range(j_taxa_ref):
            if beta_matrix[i][j]==0:
                ## TN or FP
                if hdi_beta_matrix[j][i][0]*hdi_beta_matrix[j][i][1]<=0:
                    # Contains 0, it's a true Negative
                    TN+=1
                else:
                    FP+=1
            else:
                # Coefficient different from 0
                if hdi_beta_matrix[j][i][0]*hdi_beta_matrix[j][i][1]>=0:
                    ## HDI Interval doesn't contain 0 and the true value of the original precision matrix is different from 0, so TP
                    TP+=1
                else:
                    FN+=1

    # print("\n\nConfusion Matrix: \n")
    # print(f" TP: {TP}    FN: {FN}")
    # print(f" FP: {FP}    TN: {TN}")

    # precision_score=round(TP/(TP+FP),3)
    # recall_score= round(TP/(TP+FN),3)
    # f1_score=round(2*precision_score*recall_score/(precision_score+recall_score),3)

    # print("\nPrecision: ",precision_score)
    # print("Recall: ",recall_score)
    # print("F1 Score: ",f1_score)

    FPR=FP/(TN+FP)
    TPR=TP/(TP+FN)

    return FPR,TPR

def get_AUC_Prec(prec,idata_filename,plot_graph=False):
    #BoxPLot https://matplotlib.org/stable/gallery/statistics/boxplot_color.html#sphx-glr-gallery-statistics-boxplot-color-py
    list_hdi_prob = np.linspace(0.001, 0.999, num=50)
    list_FPR=[]
    list_TPR=[]
    for hdi_prob in list_hdi_prob:
        FPR,TPR=get_fpr_tpr_precision(prec,idata_filename,hdi_prob)
        list_FPR.append(FPR)
        list_TPR.append(TPR)

    sorted_index = sorted(range(len(list_FPR)), key=lambda i: list_FPR[i])
    FPR_sorted = [list_FPR[i] for i in sorted_index]
    TPR_sorted = [list_TPR[i] for i in sorted_index] 

    auc_metric = np.trapz(TPR_sorted, FPR_sorted)
    
    if plot_graph==True:   
        plt.scatter(FPR_sorted, TPR_sorted, marker='o', color='b')
        plt.plot([0,1], [0,1], linestyle='dashed', color='black')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #plt.title('Histogramme des entiers')
        plt.show()
    
    return auc_metric

def get_fpr_tpr_precision(precision_matrix,idata_filename,hdi_probability):
    #parent_folder="results/mdine_generated_data_2/"

    TP,FP,FN,TN=0,0,0,0

    np.set_printoptions(suppress=True)

    with open(idata_filename, "rb") as fichier:
        idata = pickle.load(fichier)

    hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=hdi_probability).precision_matrix

    j_taxa=len(precision_matrix)

    for i in range(j_taxa):
        for j in range(i+1):
            if precision_matrix[i][j]==0:
                ## TN or FP
                if hdi_precision_matrix[i,j][0]*hdi_precision_matrix[i,j][1]<=0:
                    # Contains 0, it's a true Negative
                    TN+=1
                else:
                    FP+=1
            else:
                # Coefficient different from 0
                if hdi_precision_matrix[i,j][0]*hdi_precision_matrix[i,j][1]>=0:
                    ## HDI Interval doesn't contain 0 and the true value of the original precision matrix is different from 0, so TP
                    TP+=1
                else:
                    FN+=1

    # print("\n\nConfusion Matrix: \n")
    # print(f" TP: {TP}    FN: {FN}")
    # print(f" FP: {FP}    TN: {TN}")

    # precision_score=round(TP/(TP+FP),3)
    # recall_score= round(TP/(TP+FN),3)
    # f1_score=round(2*precision_score*recall_score/(precision_score+recall_score),3)

    # print("\nPrecision: ",precision_score)
    # print("Recall: ",recall_score)
    # print("F1 Score: ",f1_score)

    FPR=FP/(TN+FP)
    TPR=TP/(TP+FN)

    return FPR,TPR


def run_model(covariate_matrix_data,counts_matrix_data,simulation,idata_filename):

    beta_matrix_choice=simulation["beta_matrix"]["apriori"]
    parameters_beta_matrix=simulation["beta_matrix"]["parameters"]

    precision_matrix_choice=simulation["precision_matrix"]["apriori"]
    parameters_precision_matrix=simulation["precision_matrix"]["parameters"]

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

        print(w_matrix)
        


        #Matrice de proportions, Deterministic avec Softmax
        proportions_matrix=pm.Deterministic("proportions_matrix",pymc.math.softmax(pymc.math.concatenate([w_matrix,pt.zeros(shape=(n_individuals,1))],axis=1),axis=1))

        print(proportions_matrix)
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
        idata = pm.sample(1000) ## 10000 normally

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


        

	    #plt.show()

    # with open(f"results_simulations/{simulation_name}.pkl", "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
    #     pickle.dump(idata, f)

    with open(idata_filename, "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
        pickle.dump(idata, f)
    
    # axes_arr = az.plot_trace(idata)
    # plt.draw()
    # plt.show()

def make_precision_matrix(coef_diag,coef_off_diag,j_taxa):

    triang_laplace=pymc.math.expand_packed_triangular(n=j_taxa-1,packed=coef_off_diag)
    premiere_etape=pymc.math.concatenate([triang_laplace,pt.zeros(shape=(j_taxa-1,1))],axis=1)
    triang_strict=pymc.math.concatenate([pt.zeros(shape=(1,j_taxa)),premiere_etape],axis=0)

    return triang_strict+pytensor.tensor.transpose(triang_strict)+pytensor.tensor.diag(coef_diag)

if __name__=="__main__":
    #simulation_data_R("data/data_R/NB_100_10_10")
    simulation_data_R("data/data_R/test")
    hdi_beta_matrix=[[[-1.93520568, -1.93520568],
        [ 0.53755156,  0.53755156],
        [-0.40470285, -0.40470285]],

       [[-0.19917652, -0.19917652],
        [ 0.14910554,  0.14910554],
        [-0.38327245, -0.38327245]],

       [[-0.9417462 , -0.9417462 ],
        [-0.6517906 , -0.6517906 ],
        [ 0.6252778 ,  0.6252778 ]],

       [[ 0.05701331,  0.05701331],
        [-0.2498001 , -0.2498001 ],
        [-0.17089815, -0.17089815]],

       [[-0.78734268, -0.78734268],
        [ 0.10829721,  0.10829721],
        [-0.43665981, -0.43665981]]]
    
    k_covariates=3
    j_taxa_ref=5
    TP,FP,FN,TN=0,0,0,0

    beta_matrix=[
      [0.4466, 0, 0, 0, 0],
      [0, 0, 0.1708, 0, 0],
      [0, 0, 0.1914, 0.5668, 0]
    ]
    
    for i in range(k_covariates):
        for j in range(j_taxa_ref):
            if beta_matrix[i][j]==0:
                ## TN or FP
                #if hdi_beta_matrix[i][j][0]*hdi_beta_matrix[i][j][1]<=0:
                if hdi_beta_matrix[j][i][0]*hdi_beta_matrix[j][i][1]<=0:
                    # Contains 0, it's a true Negative
                    TN+=1
                else:
                    FP+=1
            else:
                # Coefficient different from 0
                if hdi_beta_matrix[j][i][0]*hdi_beta_matrix[j][i][1]>=0:
                    ## HDI Interval doesn't contain 0 and the true value of the original precision matrix is different from 0, so TP
                    TP+=1
                else:
                    FN+=1
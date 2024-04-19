import pymc as pm
import pytensor.tensor
import pytensor.tensor.shape
from scipy.special import softmax
import pytensor
import pytensor.tensor as pt
import pymc.math
import pickle

from extract_data_files import get_data
from extract_data_files import separate_data_in_two_groups

print(f"Running on PyMC v{pm.__version__}")

def make_precision_matrix(coef_diag,coef_off_diag,j_taxa):

    triang_laplace=pymc.math.expand_packed_triangular(n=j_taxa-1,packed=coef_off_diag)
    premiere_etape=pymc.math.concatenate([triang_laplace,pt.zeros(shape=(j_taxa-1,1))],axis=1)
    triang_strict=pymc.math.concatenate([pt.zeros(shape=(1,j_taxa)),premiere_etape],axis=0)

    return triang_strict+pytensor.tensor.transpose(triang_strict)+pytensor.tensor.diag(coef_diag)

def run_model(covariate_matrix_data,counts_matrix_data,simulation_name):

    with pm.Model() as mdine_model:

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

        # Construction des coefficients diagonaux et extra-diagonaux

        precision_matrix_coef_diag=pm.Exponential("precision_matrix_coef_diag",lam=lambda_mdine/2,shape=(j_taxa,))
        precision_matrix_coef_off_diag=pm.Laplace("precision_matrix_coef_off_diag",mu=0,b=lambda_mdine,shape=(j_taxa*(j_taxa-1)/2,))

        # Assemblage de la matrice de précision à partir des coefficients ci-dessus
        precision_matrix=pm.Deterministic("precision_matrix",make_precision_matrix(precision_matrix_coef_diag,precision_matrix_coef_off_diag,j_taxa))


        ## Matrice de covariables, permet de l'afficher sur le graphique

        covariate_matrix=pm.Deterministic("X_covariates",pytensor.tensor.as_tensor_variable(covariate_matrix_data))

        product_X_Beta=pm.Deterministic("Product_X_Beta",covariate_matrix@beta_matrix)

        # Matrice W, loi normale

        w_matrix=pm.MvNormal("w_matrix",mu=product_X_Beta,tau=precision_matrix)
        


        #Matrice de proportions, Deterministic avec Softmax
        proportions_matrix=pm.Deterministic("proportions_matrix",pymc.math.softmax(pymc.math.concatenate([w_matrix,pt.zeros(shape=(n_individuals,1))],axis=1),axis=1))

        #mdine_model.debug()

        liste_sum_counts=[]
        for i in range(n_individuals):
            liste_sum_counts.append(sum(counts_matrix_data[i]))

        ###Matrice des comptes estimées, loi multinomiale

        counts_matrix=pm.Multinomial("counts_matrix",n=liste_sum_counts,p=proportions_matrix,observed=counts_matrix_data)


    graph=pm.model_to_graphviz(mdine_model)

    graph.render(f'mdine_graph_model/{simulation_name}.gv')

    with mdine_model:
        idata = pm.sample(1000) ## 10000 normally

    with open(f"results_simulations/{simulation_name}.pkl", "wb") as f: #Potentialy modify the name of the file depending on the parameters of the simulation
        pickle.dump(idata, f)
    
    # axes_arr = az.plot_trace(idata)
    # plt.draw()
    # plt.show()

if __name__=="__main__":
    #run_model()
    #print(get_species_list("crohns.csv"))
    filename="data/crohns.csv"
    (covariate_matrix_data,counts_matrix_data,Z_vector)=get_data(filename)
    first_group,second_group=separate_data_in_two_groups(covariate_matrix_data,counts_matrix_data,Z_vector)
    run_model(first_group[0],first_group[1],simulation_name="simulation_group_0")
    run_model(second_group[0],second_group[1],simulation_name="simulation_group_1")

    


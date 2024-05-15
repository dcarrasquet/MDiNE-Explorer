import pickle
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx
from matplotlib.widgets import Slider
import json

from extract_data_files import get_species_list, separate_data_in_two_groups, get_data

# Create PDF Report with Reportlab

def plot_AUC_simulation(list_simulation):

    list_list_AUC=[]
    list_j_taxa=[]

    for simulation_folder in list_simulation:

        with open(simulation_folder+"/simulation.json", 'r') as file:
            # Chargement des données JSON
            simulation = json.load(file)

        simulation_data=simulation["data"]
        nb_tests=simulation_data["nb_tests"]
        n_individus=simulation_data["n_individus"]
        j_taxa=simulation_data["j_taxa"]
        k_covariates=simulation_data["k_covariates"]
        model_generated_data=simulation_data["model_generated_data"]

        list_AUC=[]
        list_j_taxa.append(j_taxa)

        for i in range (nb_tests):
            list_AUC.append(get_AUC_single(simulation_folder+"test_"+str(i)+"/"))

        list_list_AUC.append(list_AUC)

    boxplot_AUC(list_list_AUC,list_j_taxa,n_individus,k_covariates,model_generated_data)


def boxplot_AUC(list_list_AUC,list_j_taxa,n_individus,k_covariates,data_model):

    labels=[f"J={j_taxa}" for j_taxa in list_j_taxa]

    plt.boxplot(list_list_AUC,patch_artist=True,labels=labels,boxprops={"facecolor":'lightblue'})

    #boxplot["boxes"][0].set_facecolor('lightblue')

    # Add labels and title
    plt.xlabel(f'K={k_covariates} N={n_individus}, {data_model} Model')
    plt.ylabel('AUC')
    plt.ylim(0,1)
    #plt.title('Boxplot of Probabilities')

    # Show the plot
    plt.show()

def get_AUC_single(test_folder,plot_graph=False):
    #BoxPLot https://matplotlib.org/stable/gallery/statistics/boxplot_color.html#sphx-glr-gallery-statistics-boxplot-color-py
    list_hdi_prob = np.linspace(0.001, 0.999, num=50)
    list_FPR=[]
    list_TPR=[]
    for hdi_prob in list_hdi_prob:
        FPR,TPR=get_results_generated_data(test_folder,hdi_prob)
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


def get_results_generated_data(test_folder,hdi_probability):
    #parent_folder="results/mdine_generated_data_2/"

    TP,FP,FN,TN=0,0,0,0
    

    with open(test_folder+"generated_data.pkl", "rb") as fichier:
        Beta_matrix, precision_matrix,counts_matrix,X_matrix = pickle.load(fichier)

    np.set_printoptions(suppress=True)

    with open(test_folder+"idata.pkl", "rb") as fichier:
        idata = pickle.load(fichier)

    hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=hdi_probability).precision_matrix

    j_taxa=len(precision_matrix)

    for i in range(j_taxa):
        for j in range(i+1):
            if precision_matrix[i,j]==0:
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

    precision_score=round(TP/(TP+FP),3)
    recall_score= round(TP/(TP+FN),3)
    f1_score=round(2*precision_score*recall_score/(precision_score+recall_score),3)

    # print("\nPrecision: ",precision_score)
    # print("Recall: ",recall_score)
    # print("F1 Score: ",f1_score)

    FPR=FP/(TN+FP)
    TPR=TP/(TP+FN)

    return FPR,TPR

def get_info(idata):
    print(idata.sample_stats)

def get_forest(idata,list_variables):

    for variable in list_variables:
        az.plot_forest(idata, var_names=[variable], combined=True, hdi_prob=0.2) # az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.95, r_hat=True);
        plt.title(f"Forest de la variable {variable} à 80%")

        ax = plt.gca()

        # Tracer une ligne verticale en rouge pointillée à x=0
        ax.axvline(x=0, color='red', linestyle='--')

        plt.draw()
        plt.show()


def get_co_occurence_network(idata,species_list,correlation_matrix,density_pourcentage,edge_width):

    hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=density_pourcentage).precision_matrix

    # Création du graphe
    G = nx.Graph()
    for i in range(len(species_list)):
        G.add_node(species_list[i])


    for i in range (len(correlation_matrix)):
        for j in range (i):
            lower_hdi=hdi_precision_matrix[i][j][0]
            higher_hdi=hdi_precision_matrix[i][j][1]
            if lower_hdi*higher_hdi>=0: 
                #Same sign, do not contains null value. Create edge
                coefficient = correlation_matrix[i, j]
                couleur = 'green' if coefficient > 0 else 'red'
                largeur = abs(coefficient) * edge_width  #Multiply by a factor to adjust the width of edges
                G.add_edge(species_list[i], species_list[j], weight=largeur, color=couleur)
            else: 
                #Different sign, the edge isn't displayed
                pass
    return G


def get_co_occurence_network_interactive(idata,counts_matrix,species_list):

    precision_matrix=idata.posterior.precision_matrix.values[3][-1]

    #print(precision_matrix)
    # print("\n\n ########## \n\n")

    counts_mean=np.mean(counts_matrix,axis=0)[:-1] #Remove the last reference column

    j_taxa=len(counts_mean)

    # Correlation Matrix
    correlation_matrix=np.zeros((j_taxa,j_taxa))

    for i in range (j_taxa):
        for j in range (j_taxa):
            correlation_matrix[i,j]=precision_matrix[i,j]/(math.sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

    G=get_co_occurence_network(idata,species_list,correlation_matrix,density_pourcentage=0.5,edge_width=4)

    pos = nx.spring_layout(G)  # Déterminer la disposition du graph

    fig, ax = plt.subplots()
    nx.draw(G, pos, ax=ax)

    edge_colors = nx.get_edge_attributes(G, 'color').values()
    edge_widths = nx.get_edge_attributes(G, 'weight').values()

    #nx.draw(G, pos, with_labels=True,node_color="yellow",node_size=counts_mean*new_nodes_size, edge_color=edge_colors, width=list(edge_widths),ax=ax)
    nx.draw(G, pos, with_labels=True,node_color="yellow",node_size=counts_mean*2, edge_color=edge_colors, width=list(edge_widths),ax=ax)

    node_selected = None

    ######### Interactivité de graphique

    def on_press(event):
        nonlocal pos,node_selected
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            for node, (nx, ny) in pos.items():
                dist = (x - nx) ** 2 + (y - ny) ** 2
                if dist < 0.01:
                    node_selected = node
                    break

    def on_release(event):
        nonlocal pos,node_selected
        node_selected = None

    def on_motion(event):
        nonlocal pos, node_selected
        if node_selected is not None:
            x, y = event.xdata, event.ydata
            pos[node_selected] = (x, y)
            update()
    

    # Créer des curseurs pour modifier les paramètres
    ax_param_credibility = plt.axes([0.1, 0.10, 0.8, 0.03])
    ax_param_edges_width = plt.axes([0.1, 0.05, 0.8, 0.03])
    ax_param_nodes_size = plt.axes([0.1, 0.0, 0.8, 0.03])
    

    slider_param_credibility = Slider(ax=ax_param_credibility, label='Credibility', valmin=0.01, valmax=0.99, valinit=0.5)
    slider_param_edges_width = Slider(ax=ax_param_edges_width, label='Edges Width', valmin=0.5, valmax=20, valinit=4.)
    slider_param_nodes_size = Slider(ax=ax_param_nodes_size, label='Nodes Size', valmin=0.1, valmax=5, valinit=1.0)

    def update(val=None):
        new_credibility=slider_param_credibility.val
        new_edges_width=slider_param_edges_width.val
        new_nodes_size=slider_param_nodes_size.val

        ax.clear()

        G=get_co_occurence_network(idata,species_list,correlation_matrix,new_credibility,new_edges_width)

        #pos = nx.spring_layout(G)  # Déterminer la disposition du graph

        edge_colors = nx.get_edge_attributes(G, 'color').values()
        edge_widths = nx.get_edge_attributes(G, 'weight').values()

        #nx.draw(G, pos, with_labels=True,node_color="yellow",node_size=counts_mean*new_nodes_size, edge_color=edge_colors, width=list(edge_widths),ax=ax)
        nx.draw(G, pos, with_labels=True,node_color="yellow",node_size=counts_mean*new_nodes_size, edge_color=edge_colors, width=list(edge_widths),ax=ax)
        fig.canvas.draw_idle()

    # Associer la fonction de mise à jour aux curseurs
    slider_param_credibility.on_changed(update)
    slider_param_edges_width.on_changed(update)
    slider_param_nodes_size.on_changed(update)

    # Connecter la fonction de mise à jour à l'événement de clic de la souris
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show()

def plot_all_results(filename):
    # Charger la variable à partir du fichier
    with open(filename, "rb") as f:
        idata = pickle.load(f)

    get_info(idata)

    #Posteriors distributions and traces
    az.plot_trace(idata, var_names=["precision_matrix", "beta_matrix"])#trace_kwargs={"title": f"Beta: {beta_matrix_choice} et Precision: {precision_matrix_choice}"}

    # Forest
    get_forest(idata,["precision_matrix", "beta_matrix"])

    species_list=get_species_list("data/crohns.csv")

    (covariate_matrix_data,counts_matrix_data,Z_vector)=get_data("data/crohns.csv")
    first_group,second_group=separate_data_in_two_groups(covariate_matrix_data,counts_matrix_data,Z_vector)

    counts_matrix=first_group[1]

    get_co_occurence_network_interactive(idata,counts_matrix,species_list)



if __name__=="__main__":

    beta_matrix_choice="Normal" #Spike_and_slab Normal Lasso Horseshoe
    precision_matrix_choice="exp_Laplace" #exp_Laplace ou invwishart ou invwishart_penalized
    stringenplus="_Lambda2"
    simulation_name="simulation_group_0_"+beta_matrix_choice+"_"+precision_matrix_choice+stringenplus

    filename=f"results_simulations/{simulation_name}.pkl"

    #get_results_generated_data(0.999)
    #print(get_AUC(plot_graph=True))
    #list_AUC=[0.7417,0.8077,0.7973,0.7542]
    #boxplot_AUC(list_AUC)
    list_simulation=["test_mdine/simulation_2/","test_mdine/simulation_3/","test_mdine/simulation_4/"]
    #list_simulation=["test_mdine/simulation_3/","test_mdine/simulation_6/"]
    plot_AUC_simulation(list_simulation)

    #plot_all_results(filename)
import pickle
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx

from extract_data_files import get_species_list


def get_forest(idata):
    az.plot_forest(idata, var_names=["precision_matrix"], combined=True, hdi_prob=0.80) # az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.95, r_hat=True);
    plt.draw()
    plt.show()


def get_co_occurence_network(idata,counts_matrix,species_list,density_pourcentage=0.80,edge_width=5):

    precision_matrix=idata.posterior.precision_matrix.values[3][-1]

    # Correlation Matrix
    correlation_matrix=np.zeros((5,5))

    for i in range (5):
        for j in range (5):
            correlation_matrix[i,j]=precision_matrix[i,j]/(math.sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

    

    hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=density_pourcentage).precision_matrix

    # Création du graphe
    G = nx.Graph()
    for i in range(len(species_list)):
        G.add_node(species_list[i])


    for i in range (len(precision_matrix)):
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
    
    # Extraire les attributs des arêtes pour la visualisation
    edge_colors = nx.get_edge_attributes(G, 'color').values()
    edge_widths = nx.get_edge_attributes(G, 'weight').values()

    # Dessiner le graphe
    pos = nx.spring_layout(G)  # Définir la disposition des nœuds
    #nx.draw(G, pos, with_labels=True, edge_color=edge_colors, width=list(edge_widths), arrowsize=10)
    graph_plot=nx.draw(G, pos, with_labels=True, edge_color=edge_colors, width=list(edge_widths))
    plt.show()
    plt.plot()

    return graph_plot

if __name__=="__main__":

    filename="results_simulations/simulation1.pkl"
    
    # Charger la variable à partir du fichier
    with open(filename, "rb") as f:
        idata = pickle.load(f)

    species_list=get_species_list("data/crohns.csv")

    graph=get_co_occurence_network(idata,[],species_list,density_pourcentage=0.3,edge_width=10)
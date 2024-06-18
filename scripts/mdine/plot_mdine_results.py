import pickle
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx
from matplotlib.widgets import Slider
import json
from scipy.stats import gamma, norm
from scipy.stats import gaussian_kde

import plotly.graph_objects as go


#from mdine.extract_data_files import get_species_list, separate_data_in_two_groups, get_data
from mdine.extract_data_files import get_species_list, separate_data_in_two_groups, get_data


# Create PDF Report with Reportlab

def create_figure_distributions():

    shapes=[1]
    plt.figure(figsize=(10, 6))

    for shape in shapes:
        # Paramètres de la loi Gamma
        scale = 1.0  # theta

        # Générer des échantillons de lambda (qui suit une loi Gamma)
        lambda_samples = np.random.gamma(shape, scale, 1000)

        # Générer des échantillons de la loi normale avec tau = 1 / sigma = lambda
        mu = 0
        sigma_samples = 1 / lambda_samples
        normal_samples = np.random.normal(mu, sigma_samples)

        # Calculer l'estimation de densité par noyau
        kde = gaussian_kde(normal_samples)

        # Créer une série de points sur l'axe des x pour la courbe de densité
        x_values = np.linspace(min(normal_samples), max(normal_samples), 1000)
        kde_values = kde(x_values)

        # Tracer la courbe de densité
        
        plt.plot(x_values, kde_values, lw=2,label=f"Shape={shape}")
    
    normal_samples = np.random.normal(mu, 100,size=(1000,))
    # Calculer l'estimation de densité par noyau
    kde = gaussian_kde(normal_samples)

    # Créer une série de points sur l'axe des x pour la courbe de densité
    x_values = np.linspace(min(normal_samples), max(normal_samples), 1000)
    kde_values = kde(x_values)
    plt.plot(x_values, kde_values, lw=2,label=f"Normal 0 100")
    plt.title('Distribution combinée: Normal(mu=0, sigma=1/lambda), lambda ~ Gamma(shape=1, scale=1)')
    plt.xlabel('Valeur')
    plt.legend()
    plt.ylabel('Densité')
    plt.grid(True)
    plt.show()

def get_legend_element(legend_store):

    #print("Legend store: ",legend_store)

    #Legend for co-occurrence network
    legend=[{'data': {'id': 'legend_parent', 'label': 'Legend'}}]
    space_between_nodes=25

    if legend_store==[]:
        node={'data': {'id': "group_default", 'label': "Species", 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': 0},'locked': True }
        legend.append(node)
        pos_asso= {'data': {'id': 'positive_association', 'label': 'Positive Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': space_between_nodes},'locked': True}
        neg_asso= {'data': {'id': 'negative_association', 'label': 'Negative Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': 2*space_between_nodes},'locked': True}
        legend.append(pos_asso)
        legend.append(neg_asso)
    else:
        for idx,group in enumerate(legend_store):
            #print("Group Label: ",group["label"],"Index ",idx)
            node={'data': {'id': group["id"], 'label': group["label"], 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': idx*space_between_nodes},'locked': True }
            legend.append(node)

        nb_species=len(legend_store)
        pos_asso= {'data': {'id': 'positive_association', 'label': 'Positive Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': (nb_species)*space_between_nodes},'locked': True}
        neg_asso= {'data': {'id': 'negative_association', 'label': 'Negative Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': (nb_species+1)*space_between_nodes},'locked': True}
        legend.append(pos_asso)
        legend.append(neg_asso)

    #Legend for differentiel co-occurence-network
    width_edges=30
    legend.append({'data': {'id': 'legend_diff_parent', 'label': 'Legend'}, 'classes': 'diff-network'})
    green_node= {'data': {'id': 'green_diff_node', 'label': 'Higher in second group', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': 0, 'y': 0},'locked': True}
    red_node= {'data': {'id': 'red_diff_node', 'label': 'Lower in second group', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': 0, 'y': space_between_nodes},'locked': True}
    dashed_edge1= {'data': {'id': 'dashed_edge1', 'label': 'Second group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': width_edges/6-width_edges/2, 'y': 2*space_between_nodes},'locked': True}
    dashed_edge2= {'data': {'id': 'dashed_edge2', 'label': 'Second group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': -width_edges/6+width_edges/2, 'y': 2*space_between_nodes},'locked': True}
    line_full_edge= {'data': {'id': 'line_full_edge', 'label': 'First group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': 0, 'y': 3*space_between_nodes},'locked': True}

    legend.append(green_node)
    legend.append(red_node)
    legend.append(dashed_edge1)
    legend.append(dashed_edge2)
    legend.append(line_full_edge)

    return legend

# def get_legend_element2():


#     legend_nodes = [
#     {'data': {'id': 'legend_parent', 'label': 'Legend'}}, #'position':{'x': 2000, 'y': 2000}
#     {'data': {'id': 'legend_node1', 'label': ' Positive Association', 'parent': 'legend_parent'},'grabbable': False,'position': {'x': 0, 'y': 0},'locked': True },
#     {'data': {'id': 'legend_node2', 'label': ' Negative Association', 'parent': 'legend_parent'},'grabbable': False,'position': {'x': 0, 'y': 20},'locked': True},
#     {'data': {'id': 'legend_node3', 'label': ' Species', 'parent': 'legend_parent'},'grabbable': False,'position': {'x': 0, 'y': 40},'locked': True},
#     # {'data': {'id': 'legend_node1', 'label': ' Positive Association', 'parent': 'legend_parent'},'grabbable': False,},
#     # {'data': {'id': 'legend_node2', 'label': ' Negative Association', 'parent': 'legend_parent'},'grabbable': False,},
#     # {'data': {'id': 'legend_node3', 'label': ' Rien Association', 'parent': 'legend_parent'},'grabbable': False,},
#     ]
    
#     return legend_nodes

def get_legend_style_diff_network():
    return None

def get_legend_style(legend_store,network_type):
    if network_type=="diff-network":
        width_circle=20
        style_legend=[
            {
            'selector': 'node#legend_diff_parent',
            'style': {
                'shape': 'round-rectangle',
                #'background-color': '#f0f0f0',
                'border-color': '#000000',
                'border-width': 1,
                'label': 'data(label)',
                'text-valign': 'top',
                'text-halign': 'center',
                # 'width': 500000,
                # 'height': 200,
                "layout":'concentric'
            }
        }]
        legend_green_node={
                'selector': 'node[id="green_diff_node"]',
                'style': {
                    'background-color': '#00ff00',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'right',
                    'shape':'circle',
                    'width': width_circle,
                    'height': width_circle
                }}
        legend_red_node={
                'selector': 'node[id="red_diff_node"]',
                'style': {
                    'background-color': '#ff0000',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'right',
                    'shape':'circle',
                    'width': width_circle,
                    'height': width_circle
                }}
        legend_dashed_edge1={
                'selector': 'node[id="dashed_edge1"]',
                'style': {
                    'background-color': 'black',
                    # 'border-width': 1,              
                    # 'border-color': '#000000',      
                    # 'border-style': 'dashed',       
                    'shape': 'rectangle',           
                    'width': 30/3,                   
                    'height': 3                 
                }
                }
        legend_dashed_edge2={
                'selector': 'node[id="dashed_edge2"]',
                'style': {
                    'label': 'data(label)',  
                    'text-valign': 'center',
                    'text-halign': 'right',
                    # 'border-width': 1,              
                    # 'border-color': '#000000',      
                    # 'border-style': 'dashed',
                    'background-color': 'black',       
                    'shape': 'rectangle',           
                    'width': 30/3,                   
                    'height': 3                    
                }
                }
        legend_line_full_edge={
                'selector': 'node[id="line_full_edge"]',
                'style': {
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'right',  
                    # 'border-width': 1,              
                    # 'border-color': '#000000',      
                    # 'border-style': 'dashed',       
                    'background-color': 'black',
                    'shape': 'rectangle',          
                    'width': 30,                   
                    'height': 2                   
                }
                }
        
        style_legend.append(legend_green_node)
        style_legend.append(legend_red_node)
        style_legend.append(legend_dashed_edge1)
        style_legend.append(legend_dashed_edge2)
        style_legend.append(legend_line_full_edge)

        style_normal_legend={
            'selector': '.normal-network',
            'style': {
                'display': 'none',
            }
        }

        style_legend.append(style_normal_legend)


        
    elif network_type=="normal-network":
        style_legend=[
            {
            'selector': 'node#legend_parent',
            'style': {
                'shape': 'round-rectangle',
                #'background-color': '#f0f0f0',
                'border-color': '#000000',
                'border-width': 1,
                'label': 'data(label)',
                'text-valign': 'top',
                'text-halign': 'center',
                # 'width': 500000,
                # 'height': 200,
                "layout":'concentric'
            }
        }]

        width_circle=20

        if legend_store==[]:
            legend_default={
            'selector': 'node[id="group_default"]',
            'style': {
                'background-color': '#e5e5e5',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'right',
                'shape':'circle',
                'width': width_circle,
                'height': width_circle
            }}
            style_legend.append(legend_default)
        
        else:

            for group in legend_store:
                legend_group={
                'selector': f'node[id="{group["id"]}"]',
                'style': {
                    'background-color': group["color"],
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'right',
                    'shape':'circle',
                    'width': width_circle,
                    'height': width_circle
                }}
                style_legend.append(legend_group)

        pos_asso={
            'selector': 'node[id="positive_association"]',
            'style': {
                'background-color': '#00ff00',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'right',
                'shape':'round-rectangle',
                'width': 40,
                'height': 5
            }
        }
        neg_asso={
            'selector': 'node[id="negative_association"]',
            'style': {
                'background-color': '#ff0000',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'right',
                'shape':'round-rectangle',
                'width': 40,
                'height': 5
            }
        }
        style_legend.append(pos_asso)
        style_legend.append(neg_asso)

        style_diff_legend={
            'selector': '.diff-network',
            'style': {
                'display': 'none',
            }
        }

        style_legend.append(style_diff_legend)



    else:
        print("Network type not valid")
        style_legend=None

    return style_legend


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


def get_elements_co_occurence_network(df_taxa,legend_store):
    elements=[]
    taxa_list= df_taxa.columns.tolist()[:-1]
    # for i in range(len(taxa_list)):
    #     taxa=taxa_list[i]
        
    #     elements.append({'data': {'id': str(taxa), 'label': str(taxa)},'position': {'x': 0, 'y': 0}})


    for i in range (len(taxa_list)):
        #elements.append({'data': {'id': str(taxa_list[i]), 'label': str(taxa_list[i])},'position': {'x': 0, 'y': 0}})
        elements.append({'data': {'id': str(taxa_list[i]), 'label': str(taxa_list[i])}})
        for j in range (i):
            elements.append({'data': {'source': taxa_list[i], 'target': taxa_list[j]}})

    # Add legend elements
    elements_legend=get_legend_element(legend_store)
    for el in elements_legend:
        elements.append(el)
    
    return elements

def get_dict_element_color(legend_store):
    dict_elements_group={}
    for group in legend_store:
        list_elements=group["elements"]
        if list_elements!=[]:
            for element in list_elements:
                dict_elements_group[element]=group["color"]

    return dict_elements_group

def get_stylesheet_co_occurrence_network(idata,df_taxa,legend_store,hdi,node_size,edge_width,font_size):

    #print("Credibility: ",hdi)

    hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix

    precision_matrix=idata.posterior.precision_matrix.mean(dim=["chain", "draw"])
    counts_mean=df_taxa.mean(axis=0).tolist()[:-1]
    taxa_list= df_taxa.columns.tolist()[:-1]

    j_taxa=len(counts_mean)
    # Correlation Matrix
    correlation_matrix=np.zeros((j_taxa,j_taxa))

    for i in range (j_taxa):
        for j in range (j_taxa):
            correlation_matrix[i,j]=precision_matrix[i,j]/(math.sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

    dict_element_color=get_dict_element_color(legend_store)

    stylesheet=[]
    for i in range(len(taxa_list)):
        style_node={
        'selector': f'node[id = "{taxa_list[i]}"]',
        'style': {
            'background-color': dict_element_color.get(taxa_list[i],'#e5e5e5'),
            'label': 'data(label)',
            'width': node_size*counts_mean[i],
            'height': node_size*counts_mean[i],
            'color': 'black',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': f'{font_size}px'
            }
        }
        #print("Node size: ",node_size*counts_mean[i])
        #print("Font size: ",font_size)
        stylesheet.append(style_node)
        for j in range(i):
            #Update Edges properties
            lower_hdi=hdi_precision_matrix[i][j][0]
            higher_hdi=hdi_precision_matrix[i][j][1]

            if lower_hdi*higher_hdi>=0:
            
                #Same sign, do not contains null value. Create edge
                coefficient = correlation_matrix[i, j]
                #print(coefficient)
                color = '#18B532' if coefficient > 0 else '#E22020'
                #color = '#18B532'
                width = abs(coefficient) * edge_width #Same sign, do not contains null value. Create edge
                # print("Width: ",width)
                # print("Color: ",color)
                style_edge={
                    'selector': f'edge[source = "{taxa_list[i]}"][target = "{taxa_list[j]}"]',
                    'style': {
                        'line-color': color,
                        'width': width,
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'none',
                    }
                }
                #print("Width: ",width)
                
            else:
                style_edge={
                    'selector': f'edge[source = "{taxa_list[i]}"][target = "{taxa_list[j]}"]',
                    'style': {
                        'display': 'none',
                    }
                }
            stylesheet.append(style_edge)


    # Add legend style
    style_elements_legend=get_legend_style(legend_store,'normal-network')
    for el in style_elements_legend:
        stylesheet.append(el)
    
    return stylesheet

def get_stylesheet_diff_network(idata1,df_taxa1,idata2,df_taxa2,legend_store,hdi,node_size,edge_width,font_size):

    hdi_precision_matrix1 = az.hdi(idata1, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix
    hdi_precision_matrix2 = az.hdi(idata2, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix

    precision_matrix1=idata1.posterior.precision_matrix.mean(dim=["chain", "draw"])
    precision_matrix2=idata2.posterior.precision_matrix.mean(dim=["chain", "draw"])

    taxa_list= df_taxa1.columns.tolist()[:-1]

    counts_mean1=df_taxa1.mean(axis=0).tolist()[:-1]
    counts_mean2=df_taxa2.mean(axis=0).tolist()[:-1]

    total_count_mean=sum(counts_mean1 + counts_mean2) / (len(counts_mean1) + len(counts_mean2))
    
    j_taxa=len(counts_mean1)
   

    correlation_matrix1=np.zeros((j_taxa,j_taxa))
    correlation_matrix2=np.zeros((j_taxa,j_taxa))

    for i in range (j_taxa):
        for j in range (j_taxa):
            correlation_matrix1[i,j]=precision_matrix1[i,j]/(math.sqrt(precision_matrix1[i,i]*precision_matrix1[j,j]))
            correlation_matrix2[i,j]=precision_matrix2[i,j]/(math.sqrt(precision_matrix2[i,i]*precision_matrix2[j,j]))

    stylesheet=[]
    for i in range(len(taxa_list)):
        if counts_mean2[i]>=counts_mean1[i]:
            #Higher in Second Group
            #Green nodes
            color='#00ff00'
        else:
            #Red nodes
            color='#ff0000'

        style_node={
        'selector': f'node[id = "{taxa_list[i]}"]',
        'style': {
            'background-color': color,
            'label': 'data(label)',
            'width': node_size*total_count_mean,
            'height': node_size*total_count_mean,
            'color': 'black',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': f'{font_size}px'
            }
        }
        stylesheet.append(style_node)
        for j in range(i):
            #Update Edges properties
            lower_hdi1=hdi_precision_matrix1[i][j][0]
            higher_hdi1=hdi_precision_matrix1[i][j][1]
            lower_hdi2=hdi_precision_matrix2[i][j][0]
            higher_hdi2=hdi_precision_matrix2[i][j][1]

            if min(higher_hdi1,higher_hdi2)>=max(lower_hdi1,lower_hdi2):
                #Intervals overlap, so difference not significant
                style_edge={
                    'selector': f'edge[source = "{taxa_list[i]}"][target = "{taxa_list[j]}"]',
                    'style': {
                        'display': 'none',
                    }
                }
            else:
                if abs(correlation_matrix2[i,j])>=abs(correlation_matrix1[i,j]):
                    #Abs Higher for the second group
                    style_edge={
                    'selector': f'edge[source = "{taxa_list[i]}"][target = "{taxa_list[j]}"]',
                    'style': {
                        'line-color': 'black',
                        'width': edge_width,
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'none',
                        'line-style': 'dashed'
                    }
                }
                else:
                    style_edge={
                    'selector': f'edge[source = "{taxa_list[i]}"][target = "{taxa_list[j]}"]',
                    'style': {
                        'line-color': 'black',
                        'width': edge_width,
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'none',
                        'line-style': 'solid'
                    }
                }
            stylesheet.append(style_edge)
    
    # Add legend style
    style_elements_legend=get_legend_style(legend_store,'diff-network')
    for el in style_elements_legend:
        stylesheet.append(el)
    
    return stylesheet

def co_occurence_network_cytoscape(idata,df_taxa,hdi,node_size,edge_width,list_positions=None):

    elements=[]

    hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix

    precision_matrix=idata.posterior.precision_matrix.mean(dim=["chain", "draw"])
    counts_mean=df_taxa.mean(axis=0).tolist()[:-1]

    #print(counts_mean)

    taxa_list= df_taxa.columns.tolist()[:-1]


    for i in range(len(taxa_list)):
        taxa=taxa_list[i]
        
        elements.append({'data': {'id': str(taxa), 'label': str(taxa)},'position': {'x': 0, 'y': 0}})
        #elements.append({'data': {'id': str(i), 'label': str(i)},'position': {'x': 0, 'y': 0}})
        
        #elements.append({'data': {'id': taxa, 'label': taxa, 'size': counts_mean[i]*node_size},'position':{'x': 0, 'y': 0}})

    

    j_taxa=len(counts_mean)
    # Correlation Matrix
    correlation_matrix=np.zeros((j_taxa,j_taxa))

    for i in range (j_taxa):
        for j in range (j_taxa):
            correlation_matrix[i,j]=precision_matrix[i,j]/(math.sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

    #print(correlation_matrix)

    for i in range (len(correlation_matrix)):
        for j in range (i):
            lower_hdi=hdi_precision_matrix[i][j][0]
            higher_hdi=hdi_precision_matrix[i][j][1]
            if lower_hdi*higher_hdi>=0: 
                #Same sign, do not contains null value. Create edge
                coefficient = correlation_matrix[i, j]
                color = '#18B532' if coefficient > 0 else '#E22020'
                width = abs(coefficient) * edge_width  #Multiply by a factor to adjust the width of edges
                elements.append({'data': {'source': taxa_list[i], 'target': taxa_list[j], 'width': width*edge_width, 'color': color}})
                #elements.append({'data': {'source': str(i), 'target': str(j), 'width': width*edge_width, 'color': color}})
            else: 
                #Different sign, the edge isn't displayed
                pass

    return elements

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

    #precision_matrix=idata.posterior.precision_matrix.values[3][-1]
    precision_matrix=idata.posterior.precision_matrix.mean(dim=["chain", "draw"])

    #print(precision_matrix)
    # print("\n\n ########## \n\n")

    #counts_mean=np.mean(counts_matrix,axis=0)[:-1] #Remove the last reference column
    counts_mean=counts_matrix.mean(axis=0).tolist()[:-1]

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

    return fig

    #plt.show()

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

def study_idata_file():
    filename="test_mdine/simulation_2/test_0/idata.pkl"

    with open(filename, "rb") as fichier:
        idata = pickle.load(fichier)

    # print(idata.posterior.sampling_time)
    # print(idata.sample_stats.sampling_time)
    # print(idata.sample_stats)

    #print(az.summary(idata, round_to=2))

    print(idata.keys())

    print(idata.sample_stats.sampling_time)

    # list_precision_matrix=idata.posterior.precision_matrix

    # print(idata.posterior.beta_matrix.mean(dim=["chain", "draw"]))

    # print(idata.posterior.precision_matrix[3][-1])

    # nb_chains=len(list_precision_matrix)
    # nb_steps=len(list_precision_matrix[0])


    #print(len(list_precision_matrix))

    # idata.to_json("test_mdine/simulation_3/test_0/idata.json")

    # with open("test_mdine/simulation_3/test_0/idata.json", 'r') as file:
    #     data = json.load(file)

    #print(data.keys()) #dict_keys(['posterior', 'posterior_attrs', 'sample_stats', 'sample_stats_attrs', 'observed_data', 'observed_data_attrs', 'attrs'])

    #print(data["posterior"].keys()) #dict_keys(['Product_X_Beta', 'X_covariates', 'beta_matrix', 'lambda_brige', 'lambda_mdine', 'precision_matrix', 'precision_matrix_coef_diag', 'precision_matrix_coef_off_diag', 'proportions_matrix', 'w_matrix'])
    # print(len(data["posterior"]['Product_X_Beta'])) #4 pour 4 chaines
    # print(len(data["posterior"]['Product_X_Beta'][0])) # 1000 pour 1000 etapes de sampling
    # print(len(data["posterior"]['Product_X_Beta'][0][0]))  # Matrice 10 *100 10 Taxa et 100 individus

    #print(data["posterior_attrs"]) #{'created_at': '2024-05-13T17:33:40.903861+00:00', 'arviz_version': '0.18.0', 'inference_library': 'pymc', 'inference_library_version': '5.13.0', 'sampling_time': 96.9982557296753, 'tuning_steps': 1000}
    
    #print(data["sample_stats"].keys()) #dict_keys(['acceptance_rate', 'diverging', 'energy', 'energy_error', 'index_in_trajectory', 'largest_eigval', 'lp', 'max_energy_error', 'n_steps', 'perf_counter_diff', 'perf_counter_start', 'process_time_diff', 'reached_max_treedepth', 'smallest_eigval', 'step_size', 'step_size_bar', 'tree_depth'])

    #print(data["sample_stats"]["acceptance_rate"]) # 4 listes de longueur 1000
    #print(len(data["sample_stats"]["diverging"])) pareil

    #print(data["sample_stats"]["index_in_trajectory"]) #Pas compris

    #print(data["observed_data"].keys())

    #print(idata)

    #print(idata.posterior)

    #hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=hdi_probability).precision_matrix

def get_energy_figure(idata):
    fig, ax = plt.subplots(figsize=(6, 4))
    az.plot_energy(idata, ax=ax)
    data = []

    trace_names = ["Energy Marginal", "Energy Transition"]

    legend_texts = [text.get_text() for text in ax.legend().get_texts()]
    print("Legend initiale: ",legend_texts)

    i=0
    for line in ax.get_lines():
        print("Et de 1")
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        print("X: ",x_data)
        print("Y: ",y_data)
        if len(x_data)!=0 and len(y_data)!=0:
            trace = go.Scatter(x=x_data, y=y_data, mode='lines', name=trace_names[i])
            i+=1
            data.append(trace)

    plotly_fig = go.Figure(data=data)

    return plotly_fig

def get_acceptance_rate(idata):
    fig, ax = plt.subplots(figsize=(6, 4))
    az.plot_posterior(idata, group="sample_stats", var_names="acceptance_rate", hdi_prob="hide", kind="hist",ax=ax)
    data = []

    
    for patch in ax.patches:
                x_data = [patch.get_x()]
                y_data = [patch.get_height(), patch.get_height()]
                trace = go.Bar(x=x_data, y=y_data,marker_color='blue',showlegend=False)  # Supprimer la légende) #name=patch.get_label()
                data.append(trace)

    plotly_fig = go.Figure(data=data)
    plt.close(fig)

    return plotly_fig

def get_trace_beta(idata):

    try:
        # Plot trace
        axes_list = az.plot_trace(idata, var_names=["beta_matrix"], compact=True)

        #First Graph

        trace_data_y = []
        trace_data_x = []
        ax=axes_list[0][0]
            
        for line in ax.lines:
            trace_data_x.append(line.get_xdata())
            trace_data_y.append(line.get_ydata())

        # Create Plotly figure
        plotly_fig = go.Figure()

        for i in range (len(trace_data_x)):
            trace_y=trace_data_y[i]
            trace_x=trace_data_x[i]
            plotly_fig.add_trace(go.Scatter(x=trace_x,y=trace_y, mode='lines', name='beta_matrix',showlegend=False))

        #Second Graph

        trace_data_y = []
        trace_data_x = []
        ax=axes_list[0][1]
            
        for line in ax.lines:
            trace_data_x.append(line.get_xdata())
            trace_data_y.append(line.get_ydata())

        # Create Plotly figure
        plotly_fig2 = go.Figure()

        for i in range (len(trace_data_x)):
            trace_y=trace_data_y[i]
            trace_x=trace_data_x[i]
            plotly_fig2.add_trace(go.Scatter(x=trace_x,y=trace_y, mode='lines', name='beta_matrix',showlegend=False))

        #plotly_fig.update_layout(title="Trace Plot")

        return plotly_fig,plotly_fig2

    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

def get_trace_precision_matrix(idata):

    try:
        # Plot trace
        axes_list = az.plot_trace(idata, var_names=["precision_matrix"], compact=True)

        #First Graph

        trace_data_y = []
        trace_data_x = []
        ax=axes_list[0][0]
            
        for line in ax.lines:
            trace_data_x.append(line.get_xdata())
            trace_data_y.append(line.get_ydata())

        # Create Plotly figure
        plotly_fig = go.Figure()

        for i in range (len(trace_data_x)):
            trace_y=trace_data_y[i]
            trace_x=trace_data_x[i]
            plotly_fig.add_trace(go.Scatter(x=trace_x,y=trace_y, mode='lines', name='precision_matrix',showlegend=False))

        #Second Graph

        trace_data_y = []
        trace_data_x = []
        ax=axes_list[0][1]
            
        for line in ax.lines:
            trace_data_x.append(line.get_xdata())
            trace_data_y.append(line.get_ydata())

        # Create Plotly figure
        plotly_fig2 = go.Figure()

        for i in range (len(trace_data_x)):
            trace_y=trace_data_y[i]
            trace_x=trace_data_x[i]
            plotly_fig2.add_trace(go.Scatter(x=trace_x,y=trace_y, mode='lines', name='precision_matrix',showlegend=False))

        #plotly_fig.update_layout(title="Trace Plot")

        return plotly_fig,plotly_fig2

    except Exception as e:
        print(f"Error generating plot: {e}")
        return None
    


if __name__=="__main__":
    study_idata_file()

    beta_matrix_choice="Normal" #Spike_and_slab Normal Lasso Horseshoe
    precision_matrix_choice="exp_Laplace" #exp_Laplace ou invwishart ou invwishart_penalized
    stringenplus="_Lambda2"
    simulation_name="simulation_group_0_"+beta_matrix_choice+"_"+precision_matrix_choice+stringenplus

    filename=f"results_simulations/{simulation_name}.pkl"

    #study_idata_file()
    #create_figure_distributions()

    #get_results_generated_data(0.999)
    #print(get_AUC(plot_graph=True))
    #list_AUC=[0.7417,0.8077,0.7973,0.7542]
    #boxplot_AUC(list_AUC)
    list_simulation=["test_mdine/simulation_2/","test_mdine/simulation_3/","test_mdine/simulation_4/"]
    #list_simulation=["test_mdine/simulation_3/","test_mdine/simulation_6/"]
    #plot_AUC_simulation(list_simulation)

    #plot_all_results(filename)
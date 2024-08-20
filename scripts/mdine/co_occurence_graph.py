
import arviz as az
import numpy as np
from math import cos,sin,sqrt,pi
from collections import Counter, defaultdict
import networkx as nx
import os

def analyse_gml_co_occurence_network(filename_gml):
    G = nx.read_gml(filename_gml)

def get_gml_co_occurence_network(idata,df_taxa,legend_store,hdi,filename_gml):
    G = nx.Graph()

    taxa_list= df_taxa.columns.tolist()[:-1]
    j_taxa=len(taxa_list)
    counts_mean=df_taxa.mean(axis=0).tolist()[:-1]
    mean_total_counts=np.mean(counts_mean)
    correction_size=100/mean_total_counts
    dict_element_color=get_dict_element_color(legend_store)

    hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix.values
    precision_matrix=idata.posterior.precision_matrix.mean(dim=["chain", "draw"])
    correlation_matrix=np.zeros((j_taxa,j_taxa))
    for i in range (j_taxa):
        for j in range (j_taxa):
            correlation_matrix[i,j]=precision_matrix[i,j]/(sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

    for i in range (j_taxa):
        G.add_node(taxa_list[i], color=dict_element_color.get(taxa_list[i],'#e5e5e5'), size=correction_size*counts_mean[i])
        for j in range(i):
            #Update Edges properties
            lower_hdi=hdi_precision_matrix[i][j][0]
            higher_hdi=hdi_precision_matrix[i][j][1]

            if round(float(lower_hdi),3)*round(float(higher_hdi),3)>=0:
            
                #Same sign, do not contains null value. Create edge
                coefficient = correlation_matrix[i, j]
                color_edge = '#18B532' if coefficient > 0 else '#E22020'
                G.add_edge(taxa_list[i], taxa_list[j], color=color_edge, type='solid',width=abs(coefficient))
    
    nx.write_gml(G, filename_gml)

def get_gml_diff_network(idata1,df_taxa1,idata2,df_taxa2,hdi,filename_gml):
    G = nx.Graph()

    hdi_precision_matrix1 = az.hdi(idata1, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix.values
    hdi_precision_matrix2 = az.hdi(idata2, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix.values

    precision_matrix1=idata1.posterior.precision_matrix.mean(dim=["chain", "draw"])
    precision_matrix2=idata2.posterior.precision_matrix.mean(dim=["chain", "draw"])

    taxa_list= df_taxa1.columns.tolist()[:-1]

    counts_mean1=df_taxa1.mean(axis=0).tolist()[:-1]
    counts_mean2=df_taxa2.mean(axis=0).tolist()[:-1]
    
    j_taxa=len(counts_mean1)
   

    correlation_matrix1=np.zeros((j_taxa,j_taxa))
    correlation_matrix2=np.zeros((j_taxa,j_taxa))

    for i in range (j_taxa):
        for j in range (j_taxa):
            correlation_matrix1[i,j]=precision_matrix1[i,j]/(sqrt(precision_matrix1[i,i]*precision_matrix1[j,j]))
            correlation_matrix2[i,j]=precision_matrix2[i,j]/(sqrt(precision_matrix2[i,i]*precision_matrix2[j,j]))


    for i in range (j_taxa):
        if counts_mean2[i]>=counts_mean1[i]:
            #Higher in Second Group
            #Green nodes
            color_node='#00ff00'
        else:
            #Red nodes
            color_node='#ff0000'
        G.add_node(taxa_list[i], color=color_node, size=10)
        for j in range(i):
            #Update Edges properties
            lower_hdi1=hdi_precision_matrix1[i][j][0]
            higher_hdi1=hdi_precision_matrix1[i][j][1]
            lower_hdi2=hdi_precision_matrix2[i][j][0]
            higher_hdi2=hdi_precision_matrix2[i][j][1]

            if min(higher_hdi1,higher_hdi2)<max(lower_hdi1,lower_hdi2):
                #Intervals overlap, so significant difference
                if abs(correlation_matrix2[i,j])>=abs(correlation_matrix1[i,j]):
                    G.add_edge(taxa_list[i], taxa_list[j], type='dashed')
                else:
                    G.add_edge(taxa_list[i], taxa_list[j], type='solid')
    
    nx.write_gml(G, filename_gml)
            

def get_elements_co_occurence_network(df_taxa,legend_store):
    #print("On m'appelle souvent elements")
    elements=[]
    taxa_list= df_taxa.columns.tolist()[:-1]
    nb_species=len(taxa_list)

    ## Construction of the circular layout
    x_circle=0
    y_circle=0
    radius=300


    for i in range (nb_species):
        x=x_circle+radius*cos(2*pi*i/nb_species)
        y=y_circle+radius*sin(2*pi*i/nb_species)
        elements.append({'data': {'id': str(taxa_list[i]), 'label': str(taxa_list[i])},'position': {'x':x,'y':y}})
        for j in range (i):
            elements.append({'data': {'source': taxa_list[i], 'target': taxa_list[j]}})

    # Add legend elements
    elements_legend=get_legend_element(legend_store,x_circle-2.5*radius,y_circle)
    for el in elements_legend:
        elements.append(el)
    
    return elements

def get_legend_element(legend_store,x_origin,y_origin):

    #print("Legend store: ",legend_store)

    #Legend for co-occurrence network
    legend=[{'data': {'id': 'legend_parent', 'label': 'Legend'},'classes': 'normal-network',}]
    space_between_nodes=25

    if legend_store==[]:
        node={'data': {'id': "group_default", 'label': "Species", 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin}}#,'locked': True
        legend.append(node)
        pos_asso= {'data': {'id': 'positive_association', 'label': 'Positive Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin+space_between_nodes}}#,'locked': True
        neg_asso= {'data': {'id': 'negative_association', 'label': 'Negative Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin+2*space_between_nodes}}#,'locked': True
        legend.append(pos_asso)
        legend.append(neg_asso)
    else:
        for idx,group in enumerate(legend_store):
            #print("Group Label: ",group["label"],"Index ",idx)
            node={'data': {'id': group["id"], 'label': group["label"], 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin+idx*space_between_nodes}}#,'locked': True
            legend.append(node)

        nb_species=len(legend_store)
        pos_asso= {'data': {'id': 'positive_association', 'label': 'Positive Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin+(nb_species)*space_between_nodes}}#,'locked': True
        neg_asso= {'data': {'id': 'negative_association', 'label': 'Negative Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin+(nb_species+1)*space_between_nodes}}#,'locked': True
        legend.append(pos_asso)
        legend.append(neg_asso)

    #Legend for differentiel co-occurence-network
    width_edges=30
    legend.append({'data': {'id': 'legend_diff_parent', 'label': 'Legend'}, 'classes': 'diff-network'})
    green_node= {'data': {'id': 'green_diff_node', 'label': 'Higher in second group', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin}}#,'locked': True
    red_node= {'data': {'id': 'red_diff_node', 'label': 'Lower in second group', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin+space_between_nodes}}#,'locked': True
    dashed_edge1= {'data': {'id': 'dashed_edge1', 'label': 'Second group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': x_origin+width_edges/6-width_edges/2, 'y':y_origin+ 2*space_between_nodes}}#,'locked': True
    dashed_edge2= {'data': {'id': 'dashed_edge2', 'label': 'Second group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x':x_origin-width_edges/6+width_edges/2, 'y':y_origin +2*space_between_nodes}}#,'locked': True
    line_full_edge= {'data': {'id': 'line_full_edge', 'label': 'First group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': x_origin, 'y': y_origin+3*space_between_nodes}}#,'locked': True

    legend.append(green_node)
    legend.append(red_node)
    legend.append(dashed_edge1)
    legend.append(dashed_edge2)
    legend.append(line_full_edge)

    return legend

def get_stylesheet_co_occurrence_network(idata,df_taxa,legend_store,hdi,node_size,edge_width,font_size):

    #print("On m'appelle souvent stylesheet")

    #count_edges=0

    #print("Credibility: ",hdi)
    

    hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix

    precision_matrix=idata.posterior.precision_matrix.mean(dim=["chain", "draw"])
    print("Precision Matrix: \n",precision_matrix)
    counts_mean=df_taxa.mean(axis=0).tolist()[:-1]
    taxa_list= df_taxa.columns.tolist()[:-1]

    # print(len(precision_matrix))
    # print(len(counts_mean))

    #print(precision_matrix)

    mean_total_counts=np.mean(counts_mean)
    correction_size=node_size/mean_total_counts

    j_taxa=len(counts_mean)
    #print("NB Taxa: ",j_taxa)
    # Correlation Matrix
    correlation_matrix=np.zeros((j_taxa,j_taxa))

    for i in range (j_taxa):
        for j in range (j_taxa):
            correlation_matrix[i,j]=precision_matrix[i,j]/(sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

    dict_element_color=get_dict_element_color(legend_store)

    stylesheet=[]
    for i in range(len(taxa_list)):
        style_node={
        'selector': f'node[id = "{taxa_list[i]}"]',
        'style': {
            'background-color': dict_element_color.get(taxa_list[i],'#e5e5e5'),
            'label': 'data(label)',
            'width': correction_size*counts_mean[i],
            'height': correction_size*counts_mean[i],
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

            if round(float(lower_hdi),3)*round(float(higher_hdi),3)>=0:

            
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

def get_informations_diff_network(idata1,df_taxa1,idata2,df_taxa2,hdi):
    taxa_list= df_taxa1.columns.tolist()[:-1]
    hdi_precision_matrix1 = az.hdi(idata1, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix
    hdi_precision_matrix2 = az.hdi(idata2, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix

    precision_matrix1=idata1.posterior.precision_matrix.mean(dim=["chain", "draw"])
    precision_matrix2=idata2.posterior.precision_matrix.mean(dim=["chain", "draw"])

    counts_mean1=df_taxa1.mean(axis=0).tolist()[:-1]
    counts_mean2=df_taxa2.mean(axis=0).tolist()[:-1]


    j_taxa=len(taxa_list)

    nb_green_nodes=0
    nb_red_nodes=0
    dashed_edges=0
    solid_edges=0

    dict_number_color={
        "nodes":{
            1:"red", #Higher in first group
            2:"green" #Higher in second group
        },
        "edges":{
            1:"full", #Abs Higher for the first group
            2:"dashed" #Abs Higher for the second group
        }
    }

    list_color_nodes=[]
    graph_matrix=np.zeros((j_taxa, j_taxa))

    for i in range(j_taxa):
        if counts_mean2[i]>=counts_mean1[i]:
            #Higher in Second Group
            #Green nodes
            list_color_nodes.append(2)
        else:
            #Red nodes
            list_color_nodes.append(1)
        
        for j in range(i):
            #Update Edges properties
            lower_hdi1=hdi_precision_matrix1[i][j][0]
            higher_hdi1=hdi_precision_matrix1[i][j][1]
            lower_hdi2=hdi_precision_matrix2[i][j][0]
            higher_hdi2=hdi_precision_matrix2[i][j][1]

            if min(higher_hdi1,higher_hdi2)>=max(lower_hdi1,lower_hdi2):
                #Intervals overlap, so difference not significant
                pass
            else:
                if abs(precision_matrix2[i,j])>=abs(precision_matrix1[i,j]):
                    #Abs Higher for the second group
                    graph_matrix[i][j]=2
                    graph_matrix[j][i]=2
                else:
                    graph_matrix[i][j]=1
                    graph_matrix[j][i]=1

    node_colors = [dict_number_color['nodes'][color] for color in list_color_nodes]
    edge_types = {1: dict_number_color['edges'][1], 2: dict_number_color['edges'][2]}

    # Distribution des couleurs des nœuds
    color_distribution = Counter(node_colors)
    
    # Distribution des types d'arêtes
    edge_type_distribution = Counter(graph_matrix.flatten())
    edge_type_distribution.pop(0, None)  # Retirer les zéros (absence d'arête)
    edge_type_distribution = {edge_types[key]: val for key, val in edge_type_distribution.items()}
    
    # Calcul du degré des nœuds
    degree = np.sum(graph_matrix != 0, axis=1)
    
    # Assortativité des couleurs des nœuds (très simplifié)
    same_color_edges = 0
    total_edges = 0

    for i in range(j_taxa):
        for j in range(i):
            if graph_matrix[i, j] != 0:
                total_edges += 1
                if list_color_nodes[i] == list_color_nodes[j]:
                    same_color_edges += 1

    assortativity_color = same_color_edges / total_edges if total_edges > 0 else 0

    result={
        "color_distribution": color_distribution,
        "edge_type_distribution": edge_type_distribution,
        "degree_distribution": dict(Counter(degree)),
        "assortativity_color": assortativity_color
    }
    
    return result




def get_stylesheet_diff_network(idata1,df_taxa1,idata2,df_taxa2,legend_store,hdi,node_size,edge_width,font_size):


    hdi_precision_matrix1 = az.hdi(idata1, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix
    hdi_precision_matrix2 = az.hdi(idata2, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix

    precision_matrix1=idata1.posterior.precision_matrix.mean(dim=["chain", "draw"])
    precision_matrix2=idata2.posterior.precision_matrix.mean(dim=["chain", "draw"])

    taxa_list= df_taxa1.columns.tolist()[:-1]

    counts_mean1=df_taxa1.mean(axis=0).tolist()[:-1]
    counts_mean2=df_taxa2.mean(axis=0).tolist()[:-1]

    #total_count_mean=sum(counts_mean1 + counts_mean2) / (len(counts_mean1) + len(counts_mean2))
    
    j_taxa=len(counts_mean1)
   

    correlation_matrix1=np.zeros((j_taxa,j_taxa))
    correlation_matrix2=np.zeros((j_taxa,j_taxa))

    for i in range (j_taxa):
        for j in range (j_taxa):
            correlation_matrix1[i,j]=precision_matrix1[i,j]/(sqrt(precision_matrix1[i,i]*precision_matrix1[j,j]))
            correlation_matrix2[i,j]=precision_matrix2[i,j]/(sqrt(precision_matrix2[i,i]*precision_matrix2[j,j]))

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
            'width': node_size,
            'height': node_size,
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
                #'text-valign': 'top',
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

def get_dict_element_color(legend_store):
    dict_elements_group={}
    for group in legend_store:
        list_elements=group["elements"]
        if list_elements!=[]:
            for element in list_elements:
                dict_elements_group[element]=group["color"]

    return dict_elements_group


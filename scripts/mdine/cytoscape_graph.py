
import arviz as az
import numpy as np
from math import cos,sin,sqrt,pi

def get_elements_co_occurence_network(df_taxa,legend_store):
    elements=[]
    taxa_list= df_taxa.columns.tolist()[:-1]
    nb_species=len(taxa_list)

    ## Construction of the circular layout
    x_circle=700
    y_circle=250
    radius=200


    for i in range (nb_species):
        x=x_circle+radius*cos(2*pi*i/nb_species)
        y=y_circle+radius*sin(2*pi*i/nb_species)
        elements.append({'data': {'id': str(taxa_list[i]), 'label': str(taxa_list[i])},'position': {'x':x,'y':y}})
        for j in range (i):
            elements.append({'data': {'source': taxa_list[i], 'target': taxa_list[j]}})

    # Add legend elements
    elements_legend=get_legend_element(legend_store)
    for el in elements_legend:
        elements.append(el)
    
    return elements

def get_legend_element(legend_store):

    #print("Legend store: ",legend_store)

    #Legend for co-occurrence network
    legend=[{'data': {'id': 'legend_parent', 'label': 'Legend'},'classes': 'normal-network',}]
    space_between_nodes=25

    if legend_store==[]:
        node={'data': {'id': "group_default", 'label': "Species", 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': 0}}#,'locked': True
        legend.append(node)
        pos_asso= {'data': {'id': 'positive_association', 'label': 'Positive Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': space_between_nodes}}#,'locked': True
        neg_asso= {'data': {'id': 'negative_association', 'label': 'Negative Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': 2*space_between_nodes}}#,'locked': True
        legend.append(pos_asso)
        legend.append(neg_asso)
    else:
        for idx,group in enumerate(legend_store):
            #print("Group Label: ",group["label"],"Index ",idx)
            node={'data': {'id': group["id"], 'label': group["label"], 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': idx*space_between_nodes}}#,'locked': True
            legend.append(node)

        nb_species=len(legend_store)
        pos_asso= {'data': {'id': 'positive_association', 'label': 'Positive Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': (nb_species)*space_between_nodes}}#,'locked': True
        neg_asso= {'data': {'id': 'negative_association', 'label': 'Negative Association', 'parent': 'legend_parent'}, 'classes': 'normal-network','grabbable': False,'position': {'x': 0, 'y': (nb_species+1)*space_between_nodes}}#,'locked': True
        legend.append(pos_asso)
        legend.append(neg_asso)

    #Legend for differentiel co-occurence-network
    width_edges=30
    legend.append({'data': {'id': 'legend_diff_parent', 'label': 'Legend'}, 'classes': 'diff-network'})
    green_node= {'data': {'id': 'green_diff_node', 'label': 'Higher in second group', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': 0, 'y': 0}}#,'locked': True
    red_node= {'data': {'id': 'red_diff_node', 'label': 'Lower in second group', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': 0, 'y': space_between_nodes}}#,'locked': True
    dashed_edge1= {'data': {'id': 'dashed_edge1', 'label': 'Second group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': width_edges/6-width_edges/2, 'y': 2*space_between_nodes}}#,'locked': True
    dashed_edge2= {'data': {'id': 'dashed_edge2', 'label': 'Second group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': -width_edges/6+width_edges/2, 'y': 2*space_between_nodes}}#,'locked': True
    line_full_edge= {'data': {'id': 'line_full_edge', 'label': 'First group abs association stronger', 'parent': 'legend_diff_parent'}, 'classes': 'diff-network','grabbable': False,'position': {'x': 0, 'y': 3*space_between_nodes}}#,'locked': True

    legend.append(green_node)
    legend.append(red_node)
    legend.append(dashed_edge1)
    legend.append(dashed_edge2)
    legend.append(line_full_edge)

    return legend

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
            correlation_matrix[i,j]=precision_matrix[i,j]/(sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

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

def get_dict_element_color(legend_store):
    dict_elements_group={}
    for group in legend_store:
        list_elements=group["elements"]
        if list_elements!=[]:
            for element in list_elements:
                dict_elements_group[element]=group["color"]

    return dict_elements_group

# def co_occurence_network_cytoscape(idata,df_taxa,hdi,node_size,edge_width,list_positions=None):

#     elements=[]

#     hdi_precision_matrix = az.hdi(idata, var_names=["precision_matrix"], hdi_prob=hdi).precision_matrix

#     precision_matrix=idata.posterior.precision_matrix.mean(dim=["chain", "draw"])
#     counts_mean=df_taxa.mean(axis=0).tolist()[:-1]

#     #print(counts_mean)

#     taxa_list= df_taxa.columns.tolist()[:-1]


#     for i in range(len(taxa_list)):
#         taxa=taxa_list[i]
        
#         elements.append({'data': {'id': str(taxa), 'label': str(taxa)},'position': {'x': 0, 'y': 0}})
#         #elements.append({'data': {'id': str(i), 'label': str(i)},'position': {'x': 0, 'y': 0}})
        
#         #elements.append({'data': {'id': taxa, 'label': taxa, 'size': counts_mean[i]*node_size},'position':{'x': 0, 'y': 0}})

    

#     j_taxa=len(counts_mean)
#     # Correlation Matrix
#     correlation_matrix=np.zeros((j_taxa,j_taxa))

#     for i in range (j_taxa):
#         for j in range (j_taxa):
#             correlation_matrix[i,j]=precision_matrix[i,j]/(math.sqrt(precision_matrix[i,i]*precision_matrix[j,j]))

#     #print(correlation_matrix)

#     for i in range (len(correlation_matrix)):
#         for j in range (i):
#             lower_hdi=hdi_precision_matrix[i][j][0]
#             higher_hdi=hdi_precision_matrix[i][j][1]
#             if lower_hdi*higher_hdi>=0: 
#                 #Same sign, do not contains null value. Create edge
#                 coefficient = correlation_matrix[i, j]
#                 color = '#18B532' if coefficient > 0 else '#E22020'
#                 width = abs(coefficient) * edge_width  #Multiply by a factor to adjust the width of edges
#                 elements.append({'data': {'source': taxa_list[i], 'target': taxa_list[j], 'width': width*edge_width, 'color': color}})
#                 #elements.append({'data': {'source': str(i), 'target': str(j), 'width': width*edge_width, 'color': color}})
#             else: 
#                 #Different sign, the edge isn't displayed
#                 pass

#     return elements


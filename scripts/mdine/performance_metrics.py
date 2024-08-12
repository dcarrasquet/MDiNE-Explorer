import arviz as az
#import matplotlib.pyplot as plt
import numpy as np
import math
import json
from scipy.stats import gamma, norm
from scipy.stats import gaussian_kde
import pandas as pd

import plotly.graph_objects as go

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

def get_energy_figure(idata):
    fig, ax = plt.subplots(figsize=(6, 4))
    az.plot_energy(idata, ax=ax)
    data = []

    trace_names = ["Energy Marginal", "Energy Transition"]

    legend_texts = [text.get_text() for text in ax.legend().get_texts()]
    #print("Legend initiale: ",legend_texts)

    # Extract BFMI values from legend texts and add them as annotations
    annotations = []
    for text in legend_texts:
        if 'BFMI' in text:
            annotations.append(text)

    i=0
    for line in ax.get_lines():
        #print("Et de 1")
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        # print("X: ",x_data)
        # print("Y: ",y_data)
        if len(x_data)!=0 and len(y_data)!=0:
            trace = go.Scatter(x=x_data, y=y_data, mode='lines', name=trace_names[i])
            i+=1
            data.append(trace)

    # Create a layout for the plotly figure
    layout = go.Layout(
        annotations=[
            dict(
                x=1.00,
                y=1 - i * 0.1,
                xref='paper',
                yref='paper',
                showarrow=False,
                text=annotation,
                align='left'
            ) for i, annotation in enumerate(annotations)
        ],
        margin=dict(r=200)
    )

    plotly_fig = go.Figure(data=data,layout=layout)

    plt.close(fig)

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

        plt.close()

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

        plt.close()

        return plotly_fig,plotly_fig2

    except Exception as e:
        print(f"Error generating plot: {e}")
        return None
    
def get_autocorr(idata):
    result=az.plot_autocorr(idata,var_names=['precision_matrix'],combined=True)
    #result=az.plot_autocorr(idata,var_names=['lambda_mdine'],combined=True)
    print("Coucouc")
    print(type(result))
    plt.show()
    #return None

def get_list_variables_idata(idata):
    variables_posterior = idata.posterior.data_vars
    #print("Variables dans le groupe 'posterior':", list(variables_posterior))
    return list(variables_posterior)

def get_rhat(idata,list_variables):
    rhats = {}
    for var in list_variables:
        if var in idata.posterior:
            rhat_data = az.rhat(idata.posterior[var])
            rhats[var] = rhat_data
        else:
            print(f"La variable {var} n'existe pas dans les données.")
    return rhats
    
def get_rhat_and_ess(idata,List_variable):

    filename=f"data/dash_app/session_45/idata.nc"
    idata=az.from_netcdf(filename)

    #az.rhat(idata,var_names=variable_name)

    #print("Rhat:",az.rhat(idata,var_names="beta_matrix").beta_matrix)
    #print("Rhat:",az.rhat(idata,var_names="precision_matrix").precision_matrix)
    #print("Rhat:",az.rhat(idata))
    #print("ESS:",az.ess(idata,var_names="beta_matrix").beta_matrix)
    # print("Rhat:",az.rhat(idata,var_names="precision_matrix").precision_matrix)
    # print("ESS:",az.ess(idata,var_names="precision_matrix").precision_matrix)
    # Calculer les valeurs de R-hat
    rhat_values = az.rhat(idata)

    #print(rhat_values)

        # Initialiser une liste pour les paramètres problématiques
    problematic_params = []

    # Parcourir les valeurs de R-hat
    for param, rhat in rhat_values.items():
        # Convertir les valeurs en numpy array pour faciliter la comparaison
        print(type(rhat))
        rhat_array = np.asarray(rhat)
        print(param)
        print("Array: ")
        print(rhat_array)
        
        # Vérifier si toutes les valeurs de R-hat sont supérieures à 1.1
        if np.any(rhat_array > 1.1):
            problematic_params.append((param, rhat_array))

    # Afficher les paramètres problématiques
    if problematic_params:
        print("Paramètres avec R-hat > 1.1:")
        for param, rhat in problematic_params:
            print(f"Paramètre: {param}, R-hat: {rhat}")
    else:
        print("Toutes les chaînes semblent avoir convergé (R-hat <= 1.1).")

    # Convertir les résultats en DataFrame pour une manipulation plus facile
    #rhat_df = rhat_values.to_dataframe()

    #print(rhat_df)

    # # Filtrer les paramètres avec R-hat > 1.1
    # problematic_params = rhat_df[rhat_df[0] > 1.1]

    # # Afficher les paramètres problématiques
    # print("Paramètres avec R-hat > 1.1:")
    # print(problematic_params)

    # # Si tu veux juste les noms des paramètres
    # param_names = problematic_params['variable'].tolist()
    # print("Noms des paramètres avec R-hat > 1.1:")
    # print(param_names)


if __name__=="__main__":

    filename=f"data/dash_app/session_44/idata.nc"
    idata=az.from_netcdf(filename)
    #get_autocorr(idata)
    #get_rhat_and_ess()

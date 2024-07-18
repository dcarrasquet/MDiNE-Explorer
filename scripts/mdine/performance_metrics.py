import pickle
import arviz as az
#import matplotlib.pyplot as plt
import numpy as np
import math
import json
from scipy.stats import gamma, norm
from scipy.stats import gaussian_kde

import plotly.graph_objects as go

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

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
    
def get_rhat_and_ess():

    filename=f"data/dash_app/session_5/idata.pkl"
    with open(filename, "rb") as file:
        idata = pickle.load(file)
    # print("Rhat:",az.rhat(idata,var_names="beta_matrix").beta_matrix)
    # print("ESS:",az.ess(idata,var_names="beta_matrix").beta_matrix)
    print("Rhat:",az.rhat(idata,var_names="precision_matrix").precision_matrix)
    print("ESS:",az.ess(idata,var_names="precision_matrix").precision_matrix)

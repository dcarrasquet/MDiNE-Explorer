import dash
from dash import dcc, html,callback_context
from dash.dependencies import Input, Output, State
import os
import re
from dash.exceptions import PreventUpdate
import pty
import sys
import json
import psutil

import threading
import subprocess

from maindash import app,type_storage

button_hover_style = {
    'background': '#4A90E2'
}

button_style = {
    'background': '#4CAF50',
    'color': 'white',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
    'hover': button_hover_style,
    'display': 'block',
    'margin-left': 'auto',
    'margin-right': 'auto',
}

button_reset_style = {
    'background': '#4A90E2',
    'color': 'white',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
    'hover': button_hover_style,
    'display': 'block',
    #'margin-left': 'auto',
    #'margin-right': 'auto',
}

input_field_number_style = {
    'width': '75px',
    'padding': '8px',
    'border': '1px solid #ccc',
    'borderRadius': '5px',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
}

def layout_model():
    list_hyperparameters_beta_lasso=["alpha_lambda","beta_lambda","alpha_sigma","beta_sigma"]
    list_latex_expressions_beta_lasso=["\\alpha_{\\lambda}","\\beta_{\\lambda}","\\alpha_{\\sigma}","\\beta_{\\sigma}"]
    list_children_beta_lasso=[]
    list_values_beta_lasso=[0.75,1,2,1]
    for i in range(len(list_hyperparameters_beta_lasso)):
        parameter=list_hyperparameters_beta_lasso[i]
        latex_expression=list_latex_expressions_beta_lasso[i]
        input_parameter=dcc.Input(id=f'{parameter}_lasso',type='number',placeholder='Value',min=0,value=list_values_beta_lasso[i],step='any',style=input_field_number_style)
        parameter=html.Div(children=[dcc.Markdown(f"${latex_expression}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_parameter])
        list_children_beta_lasso.append(parameter)

    # list_hyperparameters_beta_spike_slab=["alpha_gamma","beta_gamma","tau","c"]
    # list_latex_expressions_beta_spike_slab=["\\alpha_{\\gamma}","\\beta_{\\gamma}","\\tau","c"]
    # list_children_beta_spike_slab=[]
    # list_values_beta_spike_slab=[1,1,1,1]
    # for i in range(len(list_hyperparameters_beta_spike_slab)):
    #     parameter=list_hyperparameters_beta_spike_slab[i]
    #     latex_expression=list_latex_expressions_beta_spike_slab[i]
    #     input_parameter=dcc.Input(id=f'{parameter}_spike_slab',type='number',placeholder='Value',min=0,value=list_values_beta_spike_slab[i],step='any',style=input_field_number_style)
    #     parameter=html.Div(children=[dcc.Markdown(f"${latex_expression}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_parameter])
    #     list_children_beta_spike_slab.append(parameter)

    list_hyperparameters_beta_spike_slab=["mu_hat","sigma_hat","tau"]
    list_latex_expressions_beta_spike_slab=["\\hat{\\mu}","\\hat{\\sigma}","\\tau"]
    list_children_beta_spike_slab=[]
    list_values_beta_spike_slab=[0,1,5]
    for i in range(len(list_hyperparameters_beta_spike_slab)):
        parameter=list_hyperparameters_beta_spike_slab[i]
        latex_expression=list_latex_expressions_beta_spike_slab[i]
        if i==0:
            input_parameter=dcc.Input(id=f'{parameter}_spike_slab',type='number',placeholder='Value',value=list_values_beta_spike_slab[i],step='any',style=input_field_number_style)
        else:
            input_parameter=dcc.Input(id=f'{parameter}_spike_slab',type='number',placeholder='Value',min=0,value=list_values_beta_spike_slab[i],step='any',style=input_field_number_style)
        parameter=html.Div(children=[dcc.Markdown(f"${latex_expression}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_parameter])
        list_children_beta_spike_slab.append(parameter)

    
    return html.Div([
            #html.H3('Model Page'),
            html.Div(style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top','clear': 'both'}, children=[

                dcc.Store(id='progressbar-info', storage_type=type_storage),

                html.Details([
                    html.Summary("Advanced model settings"),
                    html.Div([
                        html.Button('Reset values', id='reset-values-button', n_clicks=0,style=button_reset_style),
                        html.H5("Beta Matrix",style={'fontWeight': 'bold'}),
                        html.Div(children=[
                            dcc.Store(id='apriori-beta-status', storage_type=type_storage),
                            html.H5("Prior",style={'display': 'inline-block','margin-right':'20px','text-indent':'15px'}),
                            dcc.Dropdown(
                                id='apriori-beta-input',
                                options=[
                                    {'label': 'Horseshoe', 'value': 'Horseshoe'},
                                    {'label': 'Ridge', 'value': 'Ridge'},
                                    {'label': 'Lasso', 'value': 'Lasso'},
                                    {'label': 'Spike and Slab', 'value': 'Spike-and-Slab'}, #,'disabled': True
                                ],
                                placeholder="Select an option",
                                multi=False,
                                value='Horseshoe',
                                persistence=True,
                                persistence_type=type_storage,
                                #style={'width': '50%', 'margin': '20px 0','display': 'inline-block'}
                                style={'display': 'inline-block','width':'60%','vertical-align':'middle'}
                            )
                
                        ]),
                        #html.Div(children=[html.H5("Initial Data",style={'fontWeight': 'bold'}),
                        html.Div(children=[
                            html.H5("Hyperparameters",style={'text-indent':'15px'},id="div-hyperparameters"),
                            #html.Div(id="hyperparameters-beta"),
                            html.Div(id="hyper-beta-ridge",children=[
                                html.Div(children=[
                                    dcc.Markdown("$\\alpha$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),
                                    dcc.Input(id='alpha_ridge',type='number',placeholder='Value',min=0,value=3,step='any',style=input_field_number_style)]),
                                html.Div(children=[
                                    dcc.Markdown("$\\beta$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),
                                    dcc.Input(id='beta_ridge',type='number',placeholder='Value',min=0,value=1,step='any',style=input_field_number_style)])
                            ]),
                            html.Div(id="hyper-beta-lasso",children=list_children_beta_lasso),
                            html.Div(id="hyper-beta-spike-slab",children=list_children_beta_spike_slab),
                            # html.Div(id="hyper-beta-horseshoe",children=[
                            #         dcc.Markdown("$\\beta$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),
                            #         dcc.Input(id='beta_horseshoe',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
                            #         #])

                            # ]),
                
                        ]),
                        html.H5("Precision Matrix",style={'fontWeight': 'bold','margin-top':'20px'}),
                        html.Div(children=[
                            dcc.Store(id='apriori-precision-matrix-status', storage_type=type_storage),
                            html.H5("Prior",style={'display': 'inline-block','margin-right':'20px','text-indent':'15px'}),
                            dcc.Dropdown(
                                id='apriori-precision-matrix-input',
                                options=[
                                    {'label': 'Lasso', 'value': 'Lasso'},
                                    {'label': 'InvWishart', 'value': 'InvWishart','disabled': True},
                                    {'label': 'InvWishart Penalized', 'value': 'InvWishart Penalized','disabled': True},
                                    
                                ],
                                placeholder="Select an option",
                                multi=False,
                                value='Lasso',
                                persistence=True,
                                persistence_type=type_storage,
                                #style={'width': '50%', 'margin': '20px 0','display': 'inline-block'}
                                style={'display': 'inline-block','width':'60%','vertical-align':'middle'}
                            )
                
                        ]),
                        #html.Div(children=[html.H5("Initial Data",style={'fontWeight': 'bold'}),
                        html.Div(children=[
                            html.H5("Hyperparameters",style={'text-indent':'15px'}),
                            #html.Div(id="hyperparameters-precision-matrix"),
                            html.Div(id="hyper-precision-lasso",children=[
                                dcc.Markdown("$\\lambda_{init}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),
                                dcc.Input(id='lambda_init_lasso',type='number',placeholder='1',min=0.001,value=10,step='any',style=input_field_number_style)
                            ]),
                            html.Div(id="hyper-precision-inv-wishart"),
                            html.Div(id="hyper-precision-inv-wishart-penalized"),
                
                        ]),
                        html.Div(id="tune-model-parameters",children=[
                            html.H5("Sampling parameters",style={'fontWeight': 'bold','margin-top':'20px'}),
                            html.Div(id="nb-draws-sampler",children=[
                                dcc.Markdown("Samples to draw", style={"text-indent":"15px","display":"inline-block"}),
                                dcc.Input(id='input-nb-draws-sampler',type='number',placeholder='1000',min=1,value=1000,step=1,style=input_field_number_style)
                            ]),
                            html.Div(id="nb-tune-sampler",children=[
                                dcc.Markdown("Iterations to tune",style={"text-indent":"15px","display":"inline-block"}),
                                dcc.Input(id='input-nb-tune-sampler',type='number',placeholder='2000',min=1,value=2000,step=1,style=input_field_number_style)
                            ]),
                            html.Div(id="target-accept-sampler",children=[
                                dcc.Markdown("Target accept",style={"text-indent":"15px","display":"inline-block"}),
                                dcc.Input(id='input-target-accept-sampler',type='number',placeholder='0.9',min=0,max=1,value=0.9,step='any',style=input_field_number_style)
                            ]),
                        ]),
                        
                    ], id='advanced-parameters')
                ], id='details',open=False),
                html.Div([
                    html.Button('Run Model', id='run-model-button', n_clicks=0,disabled=True),
                    html.Button('Cancel Model', id='cancel-model-button', n_clicks=0),
                ]),
                
                dcc.Interval(id='interval-component',interval=2000, n_intervals=0),
                dcc.Store(id='run-model-status', storage_type=type_storage),
                html.Div(id="run-model-output"),
                # html.Div(id="output-area"),
                # html.Progress(id="progress-bar", value="0", max="100"),


            ]),
            html.Div(style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top','borderLeft': '3px solid #ccc'}, children=[

                html.Div(id="general-text-explanation",style={'margin-left':'10px','text-indent':'15px'},children=[
                    html.P('''The MDiNE model is a Bayesian Graphical model estimating differential 
                    co-occurrence network of species. Intuitively, the model allows the estimation 
                    of precision matrices between two groups. Moreover, standard Bayesian shrinkage 
                    models are implemented permitting a variable selection procedure between variables 
                    of interest and species, accounting for the compositionality induced by taxa.''',style={"margin-top":"1em"}),
                    html.P('''Thus, the model section is separated into 2 sections, the Beta matrix corresponding
                    to the part related to the covariables and the precision matrix for estimating the 
                    associations between species across a binary group. However, all models require the 
                    user to specify certain hyperparameters to ensure that methods give accurate results. 
                    Briefly, 4 models are available when evaluating the effect of covariables on taxa, 
                    the LASSO, the Ridge, the Horseshoe and the LN-CASS.''',style={"margin-top":"1em"}),
                    html.P('''Below, we are providing guidance about the choice of values of hyperparameters 
                    depending on the type of model chosen by the user.''',style={"margin-top":"1em"})
                ]),

                html.H5("Informations about Beta Matrix model",style={'fontWeight': 'bold',"margin-top":"2em",'margin-left':'10px'}),

                html.Div(id='info-beta-horseshoe',style={'margin-left':'10px','text-indent':'15px'},children=[
                    html.P('''For the horseshoe, no value must be explicitly specified by the user. We recommend using this prior 
                           as a default choice, since the horseshoe applies a local- global- shrinkage.''',style={"margin-top":"1em"}),
                           dcc.Markdown('''This is the Bayesian hierarchical writing of the Horseshoe prior:'''),
                           dcc.Markdown(r'''$$
                             \begin{aligned}
                             \tau &\sim HalfCauchy(1) \\
                            \lambda_{j} &\sim HalfCauchy(\tau) \\
                            \beta_{j} &\sim \mathcal{N}(0,\lambda_{j}^2) 
                             \end{aligned}
                             $$''', mathjax=True),
                ]),

                html.Div(id='info-beta-lasso',style={'margin-left':'10px','text-indent':'15px'},children=[
                dcc.Markdown('''For the Bayesian LASSO, the parameter $\\tau$ controls the amount of shrinkage 
                            required for obtaining a sparse model. The higher $\\tau$, the lower the coefficients 
                            will be penalized, and conversely for small values of $\\tau$. However, $\\tau$ is also 
                            dependent on $\\lambda$ fully characterized by $\\alpha_\\lambda$ and $\\beta_\\lambda$, the 
                            two hyperparameters requiring tuning. We recommend to start with values for 
                            $\\alpha_\\lambda$ between 0.5 and 1, keeping $\\beta_\\lambda$ constant at 1. If user wants to 
                            have more aggressive shrinkage, lower values for $\\alpha_\\lambda$ should be considered. 
                            For $\\alpha_\\sigma$ and $\\beta_\\sigma$, initial values of 2 and 1, respectively are recommended 
                            as default choices. This prior could be considered if the user aims to moderately 
                            shrink coefficients towards zero.
                            ''', style={"margin-top": "1em"},mathjax=True),
                            dcc.Markdown('''This is the Bayesian hierarchical writing of the LASSO prior:'''),
                            dcc.Markdown(r'''$$
                             \begin{aligned}
                                \lambda^{2} &\sim Gamma(\alpha_{\lambda},\beta_{\lambda}) \\
                                \sigma^{2}  &\sim InvGamma(\alpha_{\sigma},\beta_{\sigma}) \\
                                \tau_{j}   &\sim Exp(\frac{\lambda^{2}}{2})\\
                                \beta_{j} &\sim \mathcal{N}(0,\sigma^{2}\tau_{j})
                                \end{aligned}
                             $$''', mathjax=True),

                        ]),


                html.Div(id='info-beta-ridge',style={'margin-left':'10px','text-indent':'15px'},children=[
                    dcc.Markdown('''For the Bayesian Ridge, the amount of shrinkage is controlled by $\\lambda$, 
                           the precision parameter of beta.  To ensure proper shrinkage on coefficients, 
                           we recommend using $\\alpha$ moderately larges between 2 and 5, fixing $\\beta$ to 1. 
                           This ensures a moderate shrinkage with beta concentrated to zero with variance 
                           equal to $\\frac{1}{\\alpha}$. We do not recommend using larger values, except if a stronger
                           prior is assumed on the effect between covariables and taxa.
                           This prior could be considered if the user aims to proceed to soft shrinkage.''',
                           style={"margin-top":"1em"},mathjax=True),
                    dcc.Markdown('''This is the Bayesian hierarchical writing of the Ridge prior:'''),
                    dcc.Markdown(r'''$$
                             \begin{aligned}
                             \lambda &\sim Gamma(\alpha,\beta) \\
                                \beta_{j} &\sim \mathcal{N}(0,\frac{1}{\lambda})
                             \end{aligned}$$''', mathjax=True),
                ]),

                html.Div(id='info-beta-spike-slab',style={'margin-left':'10px','text-indent':'15px'},children=[
                    
                    #p_\gamma\sim Beta(\alpha_{\gamma},\beta_{\gamma}) \\
                    #\gamma_{ij}\sim Bernouilli(p_{\gamma}) \\
                    #\beta_{ij}=(1-\gamma_{ij})*\mathcal{N}(0,\tau)+\gamma_{ij}*\mathcal{N}(0,\tau c)
                    dcc.Markdown('''For LN-CASS (for Logit-Normal Continuous Analogue of Spike-and-Slab), the parameter $\\lambda_{ij}$ controls the amount of shrinkage. 
                                 Small values of $\\lambda_{ij}$ impose high shrinkage over the parameters. Moreover, 
                                 $\\lambda_{ij}$ is dependent on $\\hat{\\lambda_{ij}}$ through a logistic transformation of a 
                                 Normal distribution, providing a continuous equivalent of the traditional 
                                 spike-and-slab. In other words, $\\lambda_{ij}$ can be seen as an inclusion probability. 
                                 Thus, different combination of $\\hat{\\mu}$ and $\\hat{\\sigma}$ correspond to different 
                                 patterns of inclusion probabilities. For example, $\\hat{\\mu}= 10$ and 
                                 $\\hat{\\sigma}$ = 1 will provide inclusion probabilities close to 1 with a small 
                                 variance, while $\\hat{\\mu}=0$ and $\\hat{\\sigma} = 10$, will offer a bimodal 
                                 distribution. To ensure soft shrinkage on coefficients, we recommend using 
                                 $\\hat{\\mu}=0$, fixing $\\hat{\\sigma}$ to 1. Other combinations can be explored 
                                 by the user depending on the expected amount of shrinkage.''',
                           style={"margin-top":"1em"},mathjax=True),

                    dcc.Markdown('''This is the Bayesian hierarchical writing of the LN-CASS prior:'''),

                    dcc.Markdown(r'''$$
                             \begin{aligned}
                             \hat{\lambda}_{ij}&\sim \mathcal{N}(\hat{\mu},\hat{\sigma})\\
                            s_{ij}&\sim \mathcal{N}(0,1)\\
                                 \lambda_{ij}&=\frac{1}{1+e^{-\hat{\lambda}_{ij}}} \\
                            \beta_{ij}&=\tau*s_{ij}* \lambda_{ij}
                        
                             \end{aligned}$$''', mathjax=True),

                    # dcc.Markdown(r'''$$
                    #          \begin{aligned}
                    #          p_\gamma &\sim Beta(\alpha_{\gamma},\beta_{\gamma}) \\
                    #         \gamma_{ij} &\sim Bernouilli(p_{\gamma}) \\
                    #         \beta_{ij} &\sim (1-\gamma_{ij})\mathcal{N}(0,\tau)+\gamma_{ij}\mathcal{N}(0,\tau c)
                        
                    #          \end{aligned}$$''', mathjax=True),
                ]),


                

                html.H5("Informations about Precision Matrix model",style={'fontWeight': 'bold',"margin-top":"2em",'margin-left':'10px'}),

                html.Div(id="info-precision-lasso",style={'margin-left':'10px','text-indent':'15px'},children=[
                    html.P('''For the Graphical LASSO, shrinkage in precision matrices across the two groups 
                    is controlled by the parameter lambda. Intuitively, the higher lambda is, the closer 
                    off-diagonal elements (covariances) are to zero. However, values of lambda excessively 
                    large could lead to convergence issues, making the resulting precision matrices unrealistic.''',style={"margin-top":"1em"}), 
                    html.P('''Thus, we recommend starting with values between 1 and 100, suggesting low to moderate penalizations. 
                    ''')#style={"margin-top":"1em"}
                    #For small problems, with a few taxa than the number of individuals, empirical Bayes procedures could 
                    #be exploited. We provided a complete tutorial on how to estimate lambda from the data, here: .
                ])

                # html.H4("Spike and Slab",style={'fontWeight': 'bold'}),

                # dcc.Markdown(r'''$$
                #              \begin{aligned}
                #             \pi_{j} &\sim Beta(\alpha,\beta) \\
                #             \gamma_{j} &\sim Bernouilli (\pi_{j}) \\
                #             \beta_{j} &\sim (1-\gamma_{j})\mathcal{N}(0,\tau)+\mathcal{N}(0,\tau c)
                #             \end{aligned}
                #              $$''', mathjax=True),

                # ]),
                
            ])
        ])

@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
              Output("hyper-beta-ridge","style",allow_duplicate=True),
              Output("hyper-beta-lasso","style",allow_duplicate=True),
              Output("hyper-beta-spike-slab","style",allow_duplicate=True),
              Output("div-hyperparameters","style",allow_duplicate=True),
              Output("hyper-precision-lasso","style",allow_duplicate=True),
              Output("hyper-precision-inv-wishart","style",allow_duplicate=True),
              Output("hyper-precision-inv-wishart-penalized","style",allow_duplicate=True),
              Output("apriori-beta-input","value"),
              Output("apriori-precision-matrix-input","value"),
              Output("lambda_init_lasso","value"),
              Output('input-nb-draws-sampler','value'),
              Output('input-nb-tune-sampler','value'),
              Output('input-target-accept-sampler','value'),
              Input("reset-values-button","n_clicks"),
              State("info-current-file-store","data"),prevent_initial_call=True)
def reset_values(n_clicks,info_current_file_store):
    if n_clicks==None:
        raise PreventUpdate
    style_none={"display":"none"}
    lambda_init=10
    info_current_file_store["parameters_model"]={
        'beta_matrix':{
            'apriori':'Horseshoe',
        },
        'precision_matrix':{
            'apriori':'Lasso',
            'parameters':{
                'lambda_init':lambda_init
            }
        }
    }
    return info_current_file_store,style_none,style_none,style_none,style_none,None,style_none,style_none,"Horseshoe","Lasso",lambda_init,1000,2000,0.9

@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
              Input("input-nb-draws-sampler","value"),
            Input("input-nb-tune-sampler","value"),
            Input("input-target-accept-sampler","value"),
            State("info-current-file-store","data"),
            prevent_initial_call=True)
def change_sampling_parameters(nb_draws,nb_tune,target_accept,info_current_file_store):
    info_current_file_store["parameters_model"]["nb_draws"]=nb_draws
    info_current_file_store["parameters_model"]["nb_tune"]=nb_tune
    info_current_file_store["parameters_model"]["target_accept"]=target_accept
    return info_current_file_store


@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
              Output("hyper-beta-ridge","style",allow_duplicate=True),
              Output("hyper-beta-lasso","style",allow_duplicate=True),
              Output("hyper-beta-spike-slab","style",allow_duplicate=True),
              Output("hyper-precision-lasso","style",allow_duplicate=True),
              Output("hyper-precision-inv-wishart","style",allow_duplicate=True),
              Output("hyper-precision-inv-wishart-penalized","style",allow_duplicate=True),
              Output("div-hyperparameters","style"),
              Output("info-beta-horseshoe","style"),
              Output("info-beta-lasso","style"),
              Output("info-beta-ridge","style"),
              Output("info-beta-spike-slab","style"),
              Input('apriori-beta-status', 'data'),
              Input('apriori-precision-matrix-status', 'data'),
              Input("alpha_ridge","value"),
              Input("beta_ridge","value"),
              Input("alpha_lambda_lasso","value"),
              Input("beta_lambda_lasso","value"),
              Input("alpha_sigma_lasso","value"),
              Input("beta_sigma_lasso","value"),
            #   Input("alpha_gamma_spike_slab","value"),
            #     Input("beta_gamma_spike_slab","value"),
            #     Input("tau_spike_slab","value"),
            #     Input("c_spike_slab","value"),
            Input("mu_hat_spike_slab","value"),
            Input("sigma_hat_spike_slab","value"),
            Input("tau_spike_slab","value"),
              Input("lambda_init_lasso","value"),
              State("info-current-file-store","data"),prevent_initial_call=True)
def change_parameters_model(choice_beta,choice_precision,alpha_ridge,beta_ridge,alpha_l_lasso,beta_l_lasso,alpha_s_lasso,beta_s_lasso,mu_spike,sigma_spike,tau_spike,lambda_init,info_current_file_store): #alpha_g_spike,beta_g_spike,tau_spike,c_spike
    style_none={"display":"none"}

    style_info_prior={'margin-left':'10px','text-indent':'15px'}
    style_info_horseshoe=None
    style_info_lasso=None
    style_info_ridge=None
    style_info_spike_slab=None

    style_beta_ridge=None
    style_beta_lasso=None
    style_beta_spike_slab=None

    style_prec_lasso=None
    style_prec_inv_wishart=None
    style_prec_inv_wishart_penalized=None
    style_div_hyperparameters={'text-indent':'15px'}

    ##Beta Matrix
    if choice_beta["value"]=="Ridge":
        info_current_file_store["parameters_model"]["beta_matrix"]={
            'apriori':"Ridge",
            'parameters':{
                'alpha':alpha_ridge,
                'beta':beta_ridge,
            }
        }
        style_beta_lasso=style_none
        style_beta_spike_slab=style_none


        style_info_horseshoe=style_none
        style_info_lasso=style_none
        style_info_spike_slab=style_none
        style_info_ridge=style_info_prior

    elif choice_beta["value"]=="Lasso":
        info_current_file_store["parameters_model"]["beta_matrix"]={
            'apriori':"Lasso",
            'parameters':{
                'alpha_lambda':alpha_l_lasso,
                'beta_lambda':beta_l_lasso,
                'alpha_sigma':alpha_s_lasso,
                'beta_sigma':beta_s_lasso,
            }
        }
        style_beta_ridge=style_none
        style_beta_spike_slab=style_none

        style_info_horseshoe=style_none
        style_info_lasso=style_info_prior
        style_info_ridge=style_none
        style_info_spike_slab=style_none
        
    elif choice_beta["value"]=="Horseshoe":
        info_current_file_store["parameters_model"]["beta_matrix"]={
            'apriori':"Horseshoe",
        }
        style_beta_ridge=style_none
        style_beta_lasso=style_none
        style_beta_spike_slab=style_none
        style_div_hyperparameters=style_none

        style_info_horseshoe=style_info_prior
        style_info_lasso=style_none
        style_info_ridge=style_none
        style_info_spike_slab=style_none
    elif choice_beta["value"]=="Spike-and-Slab":
        info_current_file_store["parameters_model"]["beta_matrix"]={
            'apriori':"LN-CASS",
            'parameters':{
                'mu_hat':mu_spike,
                'sigma_hat':sigma_spike,
                'tau':tau_spike,
            }
        }
        style_beta_ridge=style_none
        style_beta_lasso=style_none

        style_info_horseshoe=style_none
        style_info_lasso=style_none
        style_info_ridge=style_none
        style_info_spike_slab=style_info_prior

    ##Precision Matrix 
    if choice_precision["value"]=="Lasso":
        info_current_file_store["parameters_model"]["precision_matrix"]={
            'apriori':"Lasso",
            'parameters':{
                'lambda_init':lambda_init,
            }
        }
        style_prec_inv_wishart=style_none
        style_prec_inv_wishart_penalized=style_none
    elif choice_precision["value"]=="InvWishart":
        info_current_file_store["parameters_model"]["precision_matrix"]={
            'apriori':"InvWishart"
        }
        style_prec_lasso=style_none
        style_prec_inv_wishart_penalized=style_none
    elif choice_precision["value"]=="InvWishart Penalized":
        info_current_file_store["parameters_model"]["precision_matrix"]={
            'apriori':"Invwishart_penalized"
        }
        style_prec_lasso=style_none
        style_prec_inv_wishart=style_none

    return info_current_file_store,style_beta_ridge,style_beta_lasso,style_beta_spike_slab,style_prec_lasso,style_prec_inv_wishart,style_prec_inv_wishart_penalized,style_div_hyperparameters,style_info_horseshoe,style_info_lasso,style_info_ridge,style_info_spike_slab


# add a click to the appropriate store.
@app.callback(Output('apriori-beta-status', 'data'),
                Input("apriori-beta-input","value"),
                State('apriori-beta-status', 'data'))
def on_click(value, data):
    if value is None:
        raise PreventUpdate

    # Give a default data dict with 0 clicks if there's no data.
    data = data or {'value': ""}

    data["value"]=value

    return data

# add a click to the appropriate store.
@app.callback(Output('apriori-precision-matrix-status', 'data'),
                Input("apriori-precision-matrix-input","value"),
                State('apriori-precision-matrix-status', 'data'))
def on_click(value, data):
    if value is None:
        raise PreventUpdate

    data = data or {'value': ""}

    data["value"]=value

    return data


# add a click to the appropriate store.
@app.callback(Output('run-model-status', 'data'),
                Input("run-model-button", 'n_clicks'),
                State("run-model-status","data"))
def on_click(n_clicks,data):

    data = data or {'run_model': False}

    
    if n_clicks>0:
        #print("Je passe run model à True")
        data['run_model']=True
    
    return data

# output the stored clicks in the table cell.
@app.callback(Output("info-current-file-store",'data',allow_duplicate=True),
            Output("run-model-output", 'children'),
              Output("run-model-button",'disabled'),
              Output("run-model-button","title"),
              Output("cancel-model-button",'disabled'),
              Output('status-run-model','data'),
            Input('run-model-status', 'modified_timestamp'),
            Input('interval-component', 'n_intervals'),
            State('run-model-status','data'),
            State('info-current-file-store','data'),
            State("run-model-output", 'children'),
            State('status-run-model','data'),
            prevent_initial_call=True)
def on_data(ts,n_intervals,data,info_current_file_store,model_output_children,status_run_model):

    if ts is None:
        raise PreventUpdate

    data = data or {}

    if data.get('run_model',False)==False:
        if info_current_file_store["reference_taxa"]!=None and info_current_file_store["covar_end"]!=None and info_current_file_store['phenotype_column']!='error':
            return info_current_file_store,model_output_children,False,None,True,'not-yet'
        else:
            title="At least one error in the data section, please check the errors."
            return info_current_file_store,model_output_children,True,title,True,'not-yet'
    
    ctx = dash.callback_context

    if not ctx.triggered:
        trigger = 'No input has triggered yet'
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    #print("Trigger: ",trigger)

    if trigger == 'run-model-status' and status_run_model=="not-yet":
        # 'Run Model button was clicked'
        #print("Avant le thread")
        title="You cannot run the model twice. If you want to run another simulation, open a new window. "
        
        master_fd, slave_fd = pty.openpty()
        #cmd=[sys.executable, '/app/scripts/mdine/MDiNE_model.py',json.dumps(info_current_file_store)] #Docker
        cmd=[sys.executable, 'scripts/mdine/MDiNE_model.py',json.dumps(info_current_file_store)]#Github

        
        process = subprocess.Popen(cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,text=True, close_fds=True)

        if os.path.exists(os.path.join(info_current_file_store["session_folder"],"output_model.json")):
            os.remove(os.path.join(info_current_file_store["session_folder"],"output_model.json"))
        

        #print("File path: ",os.path.join(info_current_file_store["session_folder"],"output_model.json"))
        os.close(slave_fd)
        threading.Thread(target=write_output_to_file, args=(master_fd,os.path.join(info_current_file_store["session_folder"],"output_model.json"))).start()
        #info_current_file_store["status-run-model"]="in-progress"
        info_current_file_store["process_pid"]=process.pid
          
        
        return info_current_file_store,None,True,title,False,"in-progress"
    elif trigger == 'interval-component':
        title="You cannot run the model twice. If you want to run another simulation, open a new window. "
        #'Interval component was triggered'
    
        json_path=os.path.join(info_current_file_store["session_folder"],"output_model.json")
        all_output= read_json_file(json_path)

        children=[]

        if len(all_output)!=0 and "model.debug()" in all_output[-1].get("text","Error"):
            children.append(dcc.Markdown('''The model initialization did not work. Please check that you have 
                                         at least three taxas remaining and consider revising the hyperparameterisation, 
                                         in particular the choice of $\\lambda_{init}$ for the precision matrix.''',mathjax=True))
            children.append(dcc.Markdown('''Also check that you have count data with whole numbers and not proportions.''',mathjax=True))
        else:
            for data in all_output:

                if data!=None:

                    if "text" in data:
                        children.append(html.H5(data["text"]))
                    elif "chains" in data:
                        percentage = data.get('percentage', '0')
                        chains = data.get('chains', '--')
                        divergences = data.get('divergences', '--')
                        time_remaining = data.get('time', '--:--:--')
                        sampling_info=f"Sampling {chains} chains, {divergences} divergences"

                        percentage_str=f"{percentage} %"

                        remaining_time=f"Remaining time: {time_remaining}"

                        return_div=html.Div(children=[
                            html.H5(sampling_info),
                        html.Div([html.Progress(id="progress-bar",className="progress-bar", value=str(percentage), max="100"),html.H5(percentage_str,style={'display':'inline-block','padding-left':"2%"})]),
                        html.H5(remaining_time)

                        ])

                        children.append(return_div)
                    

        # if info_current_file_store["phenotype_column"]==None:
        #     #Only one group
        #     children=[html.H5("Inference started"),
        #               html.H5(sampling_info),
        #               html.Div([html.Progress(id="progress-bar",className="progress-bar", value=str(percentage), max="100"),html.H5(percentage_str,style={'display':'inline-block','padding-left':"2%"})]),
        #               html.H5(remaining_time)]
        # elif info_current_file_store["phenotype_column"]!="error":
        #     children=[html.H5("First inference started"),
        #               html.H5(sampling_info),
        #               html.Div([html.Progress(id="first-progress-bar",className="progress-bar", value=str(percentage), max="100"),html.H5(percentage_str,style={'display':'inline-block'})]),
        #               html.H5(remaining_time)]

        disabled_cancel_button=False
        
        if check_run_finish(info_current_file_store):
            #print("Le thread est terminé.")
            #info_current_file_store["status-run-model"] = "completed"
            status_run_model="completed"
            disabled_cancel_button=True
            
        # else:
        #     print("Le thread est toujours en cours d'exécution...")

        return info_current_file_store,children,True,title,disabled_cancel_button,status_run_model
    else:
        return info_current_file_store,model_output_children,True,None,True,status_run_model
    
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data=json.load(file)
        if data!=None:
            return data
        else:
            #print('Dans else')
            return {}
    except:
        #print("Dans except")
        return {}
    
def write_output_to_file(fd,json_filename):
            with os.fdopen(fd, 'r') as output:
                for line in output:
                    #print(line)
                    new_data=extract_info(line)
                    # try:
                    #     with open(json_filename, 'a') as file:
                    #         json.dump(data, file, indent=4)
                    # except:
                    #     #The file has been deleted by the other thread
                    #     pass
                    # Si le fichier existe, lire les données
                    if os.path.exists(json_filename):
                        with open(json_filename, 'r') as f:
                            data = json.load(f)
                    else:
                        data = []


                    # Vérifier si le dernier élément est "progression_bar"
                    if data and data[-1]!=None and "chains" in data[-1]:
                        if "chains" in new_data:
                            # Remplacer le dernier élément
                            data[-1] = new_data
                        else:
                            # Ajouter à la suite
                            data.append(new_data)
                    else:
                        # Ajouter à la suite
                        data.append(new_data)

                    # Écrire les données mises à jour dans le fichier
                    with open(json_filename, 'w') as f:
                        json.dump(data, f, indent=4)


def extract_info(output):
    match = re.search(
        r"Sampling (\d+) chains, (\d{1,3}(?:,\d{3})*|\d+) divergences.*?(\d+)%.*?(\d+:\d+:\d+)",
        output
    )
    if match:
        divergences_str = match.group(2).replace(',', '')  # Delete commas if any
        return {
            "chains": int(match.group(1)),
            "divergences": int(divergences_str),
            "percentage": int(match.group(3)),
            "time": match.group(4)
        }
    else:
        if "NUTS:" in output:
            return {
                "text":"Inference started"
            }
        elif "\x1b[2KSampling" in output:
            return {"text":""}
        elif "\u001b[?25l\n" in output:
            return {"text":""}
        elif "\u001b[?25h" in output:
            return {"text":output.replace("\u001b[?25h","")}
        elif "LinAlgWarning" in output:
            return {
                "text":"LinAlgWarning: The model may not have converged."
            }
        elif "scipy.linalg.solve" in output:
            return {
                "text":""
            }
        else:
            return {
                "text":output
            }

def check_run_finish(info_current_file_store):
    if info_current_file_store["second_group"]!=None:
        file_path=os.path.join(info_current_file_store["session_folder"],"second_group","idata.nc")
    else:
        file_path=os.path.join(info_current_file_store["session_folder"],"idata.nc")
    return os.path.exists(file_path)
    
    
@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
              Output("run-model-status","data",allow_duplicate=True),
              Input("cancel-model-button","n_clicks"),
              State("info-current-file-store","data"),prevent_initial_call=True)
def kill_process(n_clicks,info_current_file_store):
    if info_current_file_store["process_pid"]==None:
        raise PreventUpdate
    try:
        pid_to_kill=info_current_file_store["process_pid"]
        process = psutil.Process(pid_to_kill)
        process.terminate()  # Terminer le processus
        #print(f"Cancel Button Processus avec PID {pid_to_kill} terminé avec succès.")
        info_current_file_store["process_pid"]=None
        info_current_file_store["status-run-model"]="not-yet"
    except psutil.NoSuchProcess:
        print(f"Processus avec PID {pid_to_kill} n'existe pas.")
    except psutil.AccessDenied:
        print(f"Accès refusé pour terminer le processus avec PID {pid_to_kill}.")
    return info_current_file_store,{'run_model': False}
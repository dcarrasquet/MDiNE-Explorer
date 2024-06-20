import dash
from dash import dcc, html,callback_context
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
import re
import base64
from dash.exceptions import PreventUpdate
import time
import queue
import pty
import sys

import threading
import io
import contextlib

from maindash import app,info_current_file,type_storage

from mdine.MDiNE_model import run_model
from mdine.extract_data_files import get_separate_data

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
    return html.Div([
            html.H3('Model Page'),
            html.Div(style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top','clear': 'both', 'borderRight': '3px solid #ccc'}, children=[

                #dcc.Store(id='details-state', storage_type=type_storage),
                html.Details([
                    html.Summary("Advanced model settings"),
                    html.Div([
                        html.Button('Reset values', id='reset-values-button', n_clicks=0,style=button_reset_style),
                        html.H5("Beta Matrix",style={'fontWeight': 'bold'}),
                        html.Div(children=[
                            dcc.Store(id='apriori-beta-status', storage_type=type_storage),
                            html.H5("Apriori",style={'display': 'inline-block','margin-right':'20px','text-indent':'15px'}),
                            dcc.Dropdown(
                                id='apriori-beta-input',
                                options=[
                                    {'label': 'Ridge', 'value': 'Ridge'},
                                    {'label': 'Lasso', 'value': 'Lasso'},
                                    {'label': 'Horseshoe', 'value': 'Horseshoe'},
                                    #{'label': 'Spike-and-Slab', 'value': 'Spike-and-Slab'},
                                ],
                                placeholder="Select an option",
                                multi=False,
                                value='Ridge',
                                persistence=True,
                                persistence_type=type_storage,
                                #style={'width': '50%', 'margin': '20px 0','display': 'inline-block'}
                                style={'display': 'inline-block','width':'60%','vertical-align':'middle'}
                            )
                
                        ]),
                        #html.Div(children=[html.H5("Initial Data",style={'fontWeight': 'bold'}),
                        html.Div(children=[
                            html.H5("Hyperparameters",style={'text-indent':'15px'}),
                            html.Div(id="hyperparameters-beta")
                
                        ]),
                        html.H5("Precision Matrix",style={'fontWeight': 'bold'}),
                        html.Div(children=[
                            dcc.Store(id='apriori-precision-matrix-status', storage_type=type_storage),
                            html.H5("Apriori",style={'display': 'inline-block','margin-right':'20px','text-indent':'15px'}),
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
                            html.Div(id="hyperparameters-precision-matrix")
                
                        ]),
                        
                    ], id='advanced-parameters')
                ], id='details',open=False),
                html.Button('Run Model', id='run-model-button', n_clicks=0,style=button_style),
                dcc.Interval(id='interval-component',interval=5 * 1000, n_intervals=0),
                dcc.Store(id='run-model-status', storage_type=type_storage),
                html.Div(id="run-model-output"),
                html.Div(id="output-area"),






            ]),
            html.Div(style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}, children=[


                html.H3('Partie Droite'),
                html.P('''The MDiNE model is a bayesian hierarchical model.
                To estimate co-occurrence networks, we need to put some priors  
                over two parameters, the beta Matrix and the precision matrix.
                '''),
                html.P('''The Beta Matrix represents the influence of covariates in the final counts table.
                Differents priors such as Lasso, Ridge, Spike and Slab, Horseshoe are available through this
                interface. In a general case the Lasso prior is used. If you want to change the differents 
                priors you can edit the basic model to adjust the priors to your model.'''),
                html.P('''The precision Matrix represents the influence of correlation between taxa on the final 
                counts table. The co-occurence networks are plot from this precision matrix. Differents priors 
                such as Lasso, Inv-Wishart and Inv-Wishart Penalized are available through this interface. 
                In general case the Lasso prior is used. If you want to change the differents priors you can edit 
                the basic model to adjust the priors to your model.'''),

                html.H2("Aprioris for Beta Matrix"),

                html.H4("Ridge",style={'fontWeight': 'bold'}),

                dcc.Markdown(r'''$$
                             \begin{aligned}
                             \lambda &\sim Gamma(\alpha,\beta) \\
                                \beta_{j} &\sim \mathcal{N}(0,\frac{1}{\lambda})
                             \end{aligned}$$''', mathjax=True),

                
                html.H4("Lasso",style={'fontWeight': 'bold'}),

                dcc.Markdown(r'''$$
                             \begin{aligned}
                                \lambda^{2} &\sim Gamma(\alpha_{\lambda},\beta_{\lambda}) \\
                                \sigma^{2}  &\sim InvGamma(\alpha_{\sigma},\beta_{\sigma}) \\
                                \tau_{j}   &\sim Exp(\frac{\lambda^{2}}{2})\\
                                \beta_{j} &\sim \mathcal{N}(0,\sigma^{2}\tau_{j})
                                \end{aligned}
                             $$''', mathjax=True),

                html.H4("Horseshoe",style={'fontWeight': 'bold'}),

                dcc.Markdown(r'''$$
                             \begin{aligned}
                             \tau &\sim HalfCauchy(\beta) \\
                            \lambda_{j} &\sim HalfCauchy(\tau) \\
                            \beta_{j} &\sim \mathcal{N}(0,\lambda_{j}) 
                             \end{aligned}
                             $$''', mathjax=True),

                html.H4("Spike and Slab",style={'fontWeight': 'bold'}),

                dcc.Markdown(r'''$$
                             \begin{aligned}
                            \pi_{j} &\sim Beta(\alpha,\beta) \\
                            \gamma_{j} &\sim Bernouilli (\pi_{j}) \\
                            \beta_{j} &\sim (1-\gamma_{j})\mathcal{N}(0,\tau)+\mathcal{N}(0,\tau c)
                            \end{aligned}
                             $$''', mathjax=True),

                
                
                
            ])
        ])




# add a click to the appropriate store.
@app.callback(Output('apriori-beta-status', 'data'),
                #Input("phenotype-column-button", 'n_clicks'),
                Input("apriori-beta-input","value"),
                State('apriori-beta-status', 'data'))
def on_click(value, data):
    # if n_clicks is None:
    #     # prevent the None callbacks is important with the store component.
    #     # you don't want to update the store for nothing.
    #     raise PreventUpdate
    if value is None:
        raise PreventUpdate

    # Give a default data dict with 0 clicks if there's no data.
    data = data or {'value': ""}

    data["value"]=value

    return data

# output the stored clicks in the table cell.
@app.callback(Output("hyperparameters-beta", 'children'),
                Input('apriori-beta-status', 'modified_timestamp'),
                State('apriori-beta-status', 'data'))
def on_data(ts, data):
    if ts is None:
        raise PreventUpdate

    data = data or {}

    if data.get('value',None)==None:
        return #html.H5('Rien')
    else:
        if data["value"]=="":
            return "Error : The field cannot be empty."
        elif data["value"]=="Ridge":
            input_alpha=dcc.Input(id='alpha_ridge',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
            alpha=html.Div(children=[dcc.Markdown("$\\alpha$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_alpha])
            input_beta=dcc.Input(id='beta_ridge',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
            beta=html.Div(children=[dcc.Markdown("$\\beta$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_beta])
            return [alpha,beta]
        
        elif data["value"]=="Lasso":
            list_hyperparameters=["alpha_lambda","beta_lambda","alpha_sigma","beta_sigma"]
            list_latex_expressions=["\\alpha_{\\lambda}","\\beta_{\\lambda}","\\alpha_{\\sigma}","\\beta_{\\sigma}"]
            list_children=[]
            for i in range(len(list_hyperparameters)):
                parameter=list_hyperparameters[i]
                latex_expression=list_latex_expressions[i]
                input_parameter=dcc.Input(id=f'{parameter}_lasso',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
                parameter=html.Div(children=[dcc.Markdown(f"${latex_expression}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_parameter])
                list_children.append(parameter)
            return list_children
        
        elif data["value"]=="Horseshoe":
            input_beta=dcc.Input(id='beta_horseshoe',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
            beta=html.Div(children=[dcc.Markdown("$\\beta$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_beta])
            return beta
        elif data["value"]=="Spike-and-Slab":
            return
        else:
            return "Error"


# add a click to the appropriate store.
@app.callback(Output('apriori-precision-matrix-status', 'data'),
                #Input("phenotype-column-button", 'n_clicks'),
                Input("apriori-precision-matrix-input","value"),
                State('apriori-precision-matrix-status', 'data'))
def on_click(value, data):
    # if n_clicks is None:
    #     # prevent the None callbacks is important with the store component.
    #     # you don't want to update the store for nothing.
    #     raise PreventUpdate
    if value is None:
        raise PreventUpdate

    # Give a default data dict with 0 clicks if there's no data.
    data = data or {'value': ""}

    data["value"]=value

    return data

# output the stored clicks in the table cell.
@app.callback(Output("hyperparameters-precision-matrix", 'children',allow_duplicate=True),
                Input('apriori-precision-matrix-status', 'modified_timestamp'),
                State('apriori-precision-matrix-status', 'data'),prevent_initial_call=True)
def on_data(ts, data):
    if ts is None:
        raise PreventUpdate

    data = data or {}

    if data.get('value',None)==None:
        return #html.H5('Rien')
    else:
        if data["value"]=="":
            return "Error : The field cannot be empty."
        elif data["value"]=="Lasso":
            input_lambda_init=dcc.Input(id='lambda_init_lasso',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
            lambda_init=html.Div(children=[dcc.Markdown("$\\lambda_{init}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_lambda_init])
            return lambda_init
        
        elif data["value"]=="InvWishart":
            list_hyperparameters=["alpha_lambda","beta_lambda","alpha_sigma","beta_sigma"]
            list_latex_expressions=["\\alpha_{\\lambda}","\\beta_{\\lambda}","\\alpha_{\\sigma}","\\beta_{\\sigma}"]
            list_children=[]
            for i in range(len(list_hyperparameters)):
                parameter=list_hyperparameters[i]
                latex_expression=list_latex_expressions[i]
                input_parameter=dcc.Input(id=f'{parameter}_lasso',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style,)
                parameter=html.Div(children=[dcc.Markdown(f"${latex_expression}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_parameter])
                list_children.append(parameter)
            return list_children
        
        elif data["value"]=="InvWishart Penalized":
            input_beta=dcc.Input(id='beta_horseshoe',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style,)
            beta=html.Div(children=[dcc.Markdown("$\\beta$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_beta])
            return beta
        elif data["value"]=="Spike-and-Slab":
            return
        else:
            return "Error"

output_queue = queue.Queue() 
def read_output(fd):
    buffer = b''
    for _ in range(20):
    # while True:
        try:
            chunk = os.read(fd, 1024)
            #print("Chunk reussi")
            #print("Chunk: ",chunk)
        except OSError:
            break
        buffer += chunk
        #buffer = chunk
        lines = buffer.splitlines()
        #print("Lines: ",lines)
        for line in lines[:-1]:
            output_queue.put(line.decode().strip())
        buffer = lines[-1]
        #print("Buffer: ",buffer)
        output_queue.put(buffer.decode().strip())
        print("Output buffer:",buffer)
    if buffer:
        print("Je suis rentre pas icii")
        output_queue.put(buffer.decode().strip())

def execute_model(choice_run_model):
    pid, fd = pty.fork()
    if pid == 0:
        # Processus enfant : exécute le script long_task_script.py
        current_working_directory = os.getcwd()
        #os.chdir(current_working_directory)  # Changer le répertoire de travail
        #os.execvp("pwd", ["pwd"])
        
        os.execvp(sys.executable, [sys.executable, 'scripts/mdine/model_run_terminal.py',choice_run_model])
    else:
        # Processus parent : lit la sortie du processus enfant
        read_output(fd)

# add a click to the appropriate store.
@app.callback(Output('run-model-status', 'data'),
                Input("run-model-button", 'n_clicks'),
                State("run-model-status","data"))
def on_click(n_clicks,data):

    data = data or {'run_model': False}

    print("Actual data run model: ",data)

    # if n_clicks==0:
    #     # prevent the None callbacks is important with the store component.
    #     # you don't want to update the store for nothing.
    #     raise PreventUpdate 
    
    if n_clicks>0:
        print("Je passe run model à True")
        data['run_model']=True
    
    return data

# output the stored clicks in the table cell.
@app.callback(Output("run-model-output", 'children'),
              Output("output-area","children"),
            Input('run-model-status', 'modified_timestamp'),
            Input('interval-component', 'n_intervals'),
            State('run-model-status','data'))
def on_data(ts,n_intervals,data):
    global info_current_file

    if ts is None:
        raise PreventUpdate

    data = data or {}

    #print("Status run model ",info_current_file["status-run-model"])

    if data.get('run_model',False)==False:
        return html.H5("Fonction pas encore lancée"),html.H5("Truc pas encore lancé")
    
    ctx = dash.callback_context

    if not ctx.triggered:
        trigger = 'No input has triggered yet'
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    print("Trigger: ",trigger)

    if trigger == 'run-model-status' and info_current_file["status-run-model"]=="not-yet":
        # 'Run Model button was clicked'
        if info_current_file["phenotype_column"]==None:
            print("J'execute le premier modele")
            threading.Thread(target=execute_model,args=("one_group",)).start()
        else:
            threading.Thread(target=execute_model,args=("two_groups",)).start()
        return None, None 
    elif trigger == 'interval-component':
        #'Interval component was triggered'
        last_output = ''
        while not output_queue.empty():
            last_output = output_queue.get()

        return html.H5("Fonction lancée!"),last_output
    else:
        return None,None
    
    # if info_current_file["status-run-model"]=="in-progress" or info_current_file["status-run-model"]=="completed":
    #     return "Tu as deja lancer le modele tu vas pas le faire deux fois quand meme",html.H5("Pas deux fois")
    # else:
    #     print("Je vais lancer l'inférence")
    #     if info_current_file["phenotype_column"]==None:
    #         threading.Thread(target=execute_model,args=("one_group")).start()
    #     else:
    #         threading.Thread(target=execute_model,args=("two_groups")).start()
    # else:
    #     print("Je vais lancer l'inférence")
    #     if info_current_file["phenotype_column"]==None:
    #         info_current_file["status-run-model"]="in-progress"
    #         try:
    #             run_model(info_current_file["df_covariates"],info_current_file["df_taxa"],info_current_file["parameters_model"],info_current_file["session_folder"])
    #             info_current_file["status-run-model"]="completed"
    #         except:
    #             info_current_file["status-run-model"]="error"
    #     else:
    #         try:
    #             info_current_file["status-run-model"]="in-progress"
    #             [df_covariates_1,df_taxa_1],[df_covariates_2,df_taxa_2]=get_separate_data(info_current_file)
    #             path_first_group=os.path.join(info_current_file["session_folder"],"first_group/")
    #             path_second_group=os.path.join(info_current_file["session_folder"],"second_group/")
    #             print("Je vais lancer le premier modele")
    #             run_model(df_covariates_1,df_taxa_1,info_current_file["parameters_model"],path_first_group)
    #             print("Je vais lancer le deuxième modele")
    #             run_model(df_covariates_2,df_taxa_2,info_current_file["parameters_model"],path_second_group)
    #             print("Les deux simualtions sont terminées")
    #             info_current_file["status-run-model"]="completed"
    #         except:
    #             info_current_file["status-run-model"]="error"
        # Récupérer la dernière ligne de la file d'attente
        # last_output = ''
        # while not output_queue.empty():
        #     last_output = output_queue.get()

        # return html.H5("Fonction lancée!"),last_output

@app.callback(
    Output("apriori-beta-input","value"),
    Output("apriori-precision-matrix-input","value"),
    Output("hyperparameters-precision-matrix", 'children',allow_duplicate=True),
    Output("hyperparameters-beta", 'children',allow_duplicate=True),

    Input("reset-values-button","n_clicks"),prevent_initial_call=True
)
def reset_model_values(n_clicks):
    if n_clicks==None:
        raise PreventUpdate
    
    input_alpha=dcc.Input(id='alpha_ridge',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style,persistence=True)
    alpha=html.Div(children=[dcc.Markdown("$\\alpha$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_alpha])
    input_beta=dcc.Input(id='beta_ridge',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style,persistence=True)
    beta=html.Div(children=[dcc.Markdown("$\\beta$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_beta])
    
    
    input_lambda_init=dcc.Input(id='lambda_init_lasso',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
    lambda_init=html.Div(children=[dcc.Markdown("$\\lambda_{init}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_lambda_init])
    

    
    return "Ridge",'Lasso',lambda_init,[alpha,beta]
    
    
    

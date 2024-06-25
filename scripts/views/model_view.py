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
import json

import threading
import io
import contextlib

from maindash import app,type_storage #info_current_file,

# from mdine.MDiNE_model import run_model
# from mdine.extract_data_files import get_separate_data

model_thread = None
thread_finished = threading.Event()

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
                html.Button('Run Model', id='run-model-button', n_clicks=0),
                dcc.Interval(id='interval-component',interval=2000, n_intervals=0),
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
    #for _ in range(20):
    while True:
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
        clean_buffer = remove_ansi_escape_sequences(buffer.decode(errors='replace').strip())
        #print("Buffer: ",buffer)
        output_queue.put(clean_buffer)
        #print("Output buffer:",clean_buffer)
    if buffer:
        #print("Je suis rentre pas icii")
        cleaned_buffer = remove_ansi_escape_sequences(buffer.decode(errors='replace').strip())
        output_queue.put(cleaned_buffer)

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def execute_model(info_current_file_store):
    global thread_finished
    pid, fd = pty.fork()
    if pid == 0:
        # Processus enfant : exécute le script long_task_script.py
        #os.chdir(current_working_directory)  # Changer le répertoire de travail
        #os.execvp("pwd", ["pwd"])

        info_current_file_store_str=json.dumps(info_current_file_store)
        
        os.execvp(sys.executable, [sys.executable, 'scripts/mdine/MDiNE_model.py',info_current_file_store_str])
    else:
        # Processus parent : lit la sortie du processus enfant
        try:
            read_output(fd)
        finally:
            os.waitpid(pid, 0)  # Attend la fin du processus enfant
            thread_finished.set()  # Signale que le thread est terminé

# add a click to the appropriate store.
@app.callback(Output('run-model-status', 'data'),
                Input("run-model-button", 'n_clicks'),
                State("run-model-status","data"))
def on_click(n_clicks,data):

    data = data or {'run_model': False}

    #print("Actual data run model: ",data)

    # if n_clicks==0:
    #     # prevent the None callbacks is important with the store component.
    #     # you don't want to update the store for nothing.
    #     raise PreventUpdate 
    
    if n_clicks>0:
        #print("Je passe run model à True")
        data['run_model']=True
    
    return data

# output the stored clicks in the table cell.
@app.callback(Output("info-current-file-store", 'data',allow_duplicate=True),
        Output("run-model-output", 'children'),
              Output("output-area","children"),
              Output("run-model-button",'disabled'),
              Output("run-model-button","title"),
            Input('run-model-status', 'modified_timestamp'),
            Input('interval-component', 'n_intervals'),
            State('run-model-status','data'),
            State('info-current-file-store','data'),prevent_initial_call=True)
def on_data(ts,n_intervals,data,info_current_file_store):

    global model_thread,thread_finished

    if ts is None:
        raise PreventUpdate

    data = data or {}

    #print("Status run model ",info_current_file["status-run-model"])

    if data.get('run_model',False)==False:
        if info_current_file_store["reference_taxa"]!=None or info_current_file_store["covar_end"]!=None:
            return info_current_file_store,html.H5("Fonction pas encore lancée"),html.H5("Truc pas encore lancé"),False,None
        else:
            title="At least one error in the data section, please check the errors."
            return info_current_file_store,html.H5("Fonction pas encore lancée"),html.H5("Truc pas encore lancé"),True,title
    
    ctx = dash.callback_context

    if not ctx.triggered:
        trigger = 'No input has triggered yet'
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    #print("Trigger: ",trigger)

    if trigger == 'run-model-status' and info_current_file_store["status-run-model"]=="not-yet":
        # 'Run Model button was clicked'
        #print("Avant le thread")
        title="You cannot run the model twice. If you want to run another simulation, open a new window. "
        model_thread=threading.Thread(target=execute_model,args=(info_current_file_store,))
        model_thread.start()
        #print("Après le lancement du thread")
        #print("Thread alive: ",model_thread.is_alive())
        info_current_file_store["status-run-model"]="in-progress"
          
        
        return info_current_file_store,None, None,True,title
    elif trigger == 'interval-component':
        title="You cannot run the model twice. If you want to run another simulation, open a new window. "
        #'Interval component was triggered'
        last_output = ''
        while not output_queue.empty():
            last_output = output_queue.get()
        
        if check_run_finish(info_current_file_store):
            #print("Le thread est terminé.")
            info_current_file_store["status-run-model"] = "completed"
            
        # else:
        #     print("Le thread est toujours en cours d'exécution...")

        return info_current_file_store,html.H5("Fonction lancée!"),last_output,True,title
    else:
        return info_current_file_store,None,None,True,None
    

def check_run_finish(info_current_file_store):
    if info_current_file_store["second_group"]!=None:
        file_path=os.path.join(info_current_file_store["session_folder"],"second_group","idata.pkl")
    else:
        file_path=os.path.join(info_current_file_store["session_folder"],"idata.pkl")
    #print("Exist file: ",os.path.exists(file_path))
    return os.path.exists(file_path)

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
    
    
    

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
    for i in range(len(list_hyperparameters_beta_lasso)):
        parameter=list_hyperparameters_beta_lasso[i]
        latex_expression=list_latex_expressions_beta_lasso[i]
        input_parameter=dcc.Input(id=f'{parameter}_lasso',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
        parameter=html.Div(children=[dcc.Markdown(f"${latex_expression}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),input_parameter])
        list_children_beta_lasso.append(parameter)
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
                            html.H5("Apriori",style={'display': 'inline-block','margin-right':'20px','text-indent':'15px'}),
                            dcc.Dropdown(
                                id='apriori-beta-input',
                                options=[
                                    {'label': 'Ridge', 'value': 'Ridge'},
                                    {'label': 'Lasso', 'value': 'Lasso'},
                                    {'label': 'Horseshoe', 'value': 'Horseshoe'},
                                    {'label': 'Spike-and-Slab', 'value': 'Spike-and-Slab','disabled': True},
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
                            #html.Div(id="hyperparameters-beta"),
                            html.Div(id="hyper-beta-ridge",children=[
                                html.Div(children=[
                                    dcc.Markdown("$\\alpha$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),
                                    dcc.Input(id='alpha_ridge',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)]),
                                html.Div(children=[
                                    dcc.Markdown("$\\beta$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),
                                    dcc.Input(id='beta_ridge',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)])
                            ]),
                            html.Div(id="hyper-beta-lasso",children=list_children_beta_lasso),
                            html.Div(id="hyper-beta-horseshoe",children=[
                                    dcc.Markdown("$\\beta$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),
                                    dcc.Input(id='beta_horseshoe',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
                                    #])

                            ]),
                
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
                            #html.Div(id="hyperparameters-precision-matrix"),
                            html.Div(id="hyper-precision-lasso",children=[
                                dcc.Markdown("$\\lambda_{init}$", mathjax=True,style={"text-indent":"30px","display":"inline-block"}),
                                dcc.Input(id='lambda_init_lasso',type='number',placeholder='1',min=0,value=1,step='any',style=input_field_number_style)
                            ]),
                            html.Div(id="hyper-precision-inv-wishart"),
                            html.Div(id="hyper-precision-inv-wishart-penalized"),
                
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

                html.Div(style={'margin-left':'10px','text-indent':'15px'},children=[
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

                ]),
                #html.H3('Partie Droite'),
                

                
                
                
            ])
        ])

@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
              Output("hyper-beta-ridge","style",allow_duplicate=True),
              Output("hyper-beta-lasso","style",allow_duplicate=True),
              Output("hyper-beta-horseshoe","style",allow_duplicate=True),
              Output("hyper-precision-lasso","style",allow_duplicate=True),
              Output("hyper-precision-inv-wishart","style",allow_duplicate=True),
              Output("hyper-precision-inv-wishart-penalized","style",allow_duplicate=True),
              Output("apriori-beta-input","value"),
              Output("apriori-precision-matrix-input","value"),
              Output("alpha_ridge","value"),
              Output("beta_ridge","value"),
              Output("lambda_init_lasso","value"),
              Input("reset-values-button","n_clicks"),
              State("info-current-file-store","data"),prevent_initial_call=True)
def reset_values(n_clicks,info_current_file_store):
    if n_clicks==None:
        raise PreventUpdate
    style_none={"display":"none"}
    alpha_ridge=1
    beta_ridge=1
    lambda_init=1
    info_current_file_store["parameters_model"]={
        'beta_matrix':{
            'apriori':'Ridge',
            'parameters':{
                'alpha':alpha_ridge,
                'beta':beta_ridge
            }
        },
        'precision_matrix':{
            'apriori':'Lasso',
            'parameters':{
                'lambda_init':lambda_init
            }
        }
    }
    return info_current_file_store,None,style_none,style_none,None,style_none,style_none,"Ridge","Lasso",alpha_ridge,beta_ridge,lambda_init


@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
              Output("hyper-beta-ridge","style",allow_duplicate=True),
              Output("hyper-beta-lasso","style",allow_duplicate=True),
              Output("hyper-beta-horseshoe","style",allow_duplicate=True),
              Output("hyper-precision-lasso","style",allow_duplicate=True),
              Output("hyper-precision-inv-wishart","style",allow_duplicate=True),
              Output("hyper-precision-inv-wishart-penalized","style",allow_duplicate=True),
              Input('apriori-beta-status', 'data'),
              Input('apriori-precision-matrix-status', 'data'),
              Input("alpha_ridge","value"),
              Input("beta_ridge","value"),
              Input("alpha_lambda_lasso","value"),
              Input("beta_lambda_lasso","value"),
              Input("alpha_sigma_lasso","value"),
              Input("beta_sigma_lasso","value"),
              Input("beta_horseshoe","value"),
              Input("lambda_init_lasso","value"),
              State("info-current-file-store","data"),prevent_initial_call=True)
def change_parameters_model(choice_beta,choice_precision,alpha_ridge,beta_ridge,alpha_l_lasso,beta_l_lasso,alpha_s_lasso,beta_s_lasso,beta_horseshoe,lambda_init,info_current_file_store):
    style_none={"display":"none"}

    style_beta_ridge=None
    style_beta_lasso=None
    style_beta_horseshoe=None
    style_prec_lasso=None
    style_prec_inv_wishart=None
    style_prec_inv_wishart_penalized=None

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
        style_beta_horseshoe=style_none
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
        style_beta_horseshoe=style_none
    elif choice_beta["value"]=="Horseshoe":
        info_current_file_store["parameters_model"]["beta_matrix"]={
            'apriori':"Horseshoe",
            'parameters':{
                'beta_tau':beta_horseshoe,
            }
        }
        style_beta_ridge=style_none
        style_beta_lasso=style_none

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

    return info_current_file_store,style_beta_ridge,style_beta_lasso,style_beta_horseshoe,style_prec_lasso,style_prec_inv_wishart,style_prec_inv_wishart_penalized


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
@app.callback(Output("info-current-file-store", 'data',allow_duplicate=True),
            Output("run-model-output", 'children'),
              Output("run-model-button",'disabled'),
              Output("run-model-button","title"),
              Output("cancel-model-button",'disabled'),
            Input('run-model-status', 'modified_timestamp'),
            Input('interval-component', 'n_intervals'),
            State('run-model-status','data'),
            State('info-current-file-store','data'),
            State("run-model-output", 'children'),
            prevent_initial_call=True)
def on_data(ts,n_intervals,data,info_current_file_store,model_output_children):

    if ts is None:
        raise PreventUpdate

    data = data or {}

    if data.get('run_model',False)==False:
        if info_current_file_store["reference_taxa"]!=None and info_current_file_store["covar_end"]!=None and info_current_file_store['phenotype_column']!='error':
            return info_current_file_store,model_output_children,False,None,True
        else:
            title="At least one error in the data section, please check the errors."
            return info_current_file_store,model_output_children,True,title,True
    
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
        
        master_fd, slave_fd = pty.openpty()
        #cmd=[sys.executable, '/app/scripts/mdine/MDiNE_model.py',json.dumps(info_current_file_store)] #Docker
        cmd=[sys.executable, 'scripts/mdine/MDiNE_model.py',json.dumps(info_current_file_store)]#Github

        
        process = subprocess.Popen(cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,text=True, close_fds=True)
        

        print("File path: ",os.path.join(info_current_file_store["session_folder"],"output_model.json"))
        os.close(slave_fd)
        threading.Thread(target=write_output_to_file, args=(master_fd,os.path.join(info_current_file_store["session_folder"],"output_model.json"))).start()
        info_current_file_store["status-run-model"]="in-progress"
        info_current_file_store["process_pid"]=process.pid

        if info_current_file_store["phenotype_column"]==None:
            #Only one group
            children=[html.H5("Inference started"),
                      html.H5("Sampling -- chains, -- divergences"),
                      html.Progress(id="progress-bar",className="progress-bar", value="0", max="100"),
                      html.H5("Remaining time: --:--:--")]
        elif info_current_file_store["phenotype_column"]!="error":
            children=[html.H5("First inference started"),
                      html.H5("Sampling -- chains, -- divergences"),
                      html.Progress(id="first-progress-bar",className="progress-bar", value="0", max="100"),
                      html.H5("Remaining time: --:--:--")]
          
        
        return info_current_file_store,children,True,title,False
    elif trigger == 'interval-component':
        title="You cannot run the model twice. If you want to run another simulation, open a new window. "
        #'Interval component was triggered'
    
        json_path=os.path.join(info_current_file_store["session_folder"],"output_model.json")
        data = read_json_file(json_path)

        if "text" in data:
            children=[html.H5(data["text"])]
        elif "chains" in data:
            percentage = data.get('percentage', '0')
            chains = data.get('chains', '--')
            divergences = data.get('divergences', '--')
            time_remaining = data.get('time', '--:--:--')
            sampling_info=f"Sampling {chains} chains, {divergences} divergences"

            percentage_str=f"{percentage} %"

            remaining_time=f"Remaining time: {time_remaining}"

            children=[html.H5(sampling_info),
            html.Div([html.Progress(id="progress-bar",className="progress-bar", value=str(percentage), max="100"),html.H5(percentage_str,style={'display':'inline-block','padding-left':"2%"})]),
            html.H5(remaining_time)]
        else:
            print("Data null je pense: ")
            print(data)
            children=html.H5("I have found nothing")

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
        
        if check_run_finish(info_current_file_store):
            #print("Le thread est terminé.")
            info_current_file_store["status-run-model"] = "completed"
            
        # else:
        #     print("Le thread est toujours en cours d'exécution...")

        return info_current_file_store,children,True,title,False
    else:
        return info_current_file_store,model_output_children,True,None,True
    
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data=json.load(file)
        if data!=None:
            return data
        else:
            print('Dans else')
            return {}
    except:
        print("Dans except")
        return {}
    
def write_output_to_file(fd,json_filename):
            with os.fdopen(fd, 'r') as output:
                for line in output:
                    data=extract_info(line)
                    try:
                        with open(json_filename, 'w') as file:
                            json.dump(data, file, indent=4)
                    except:
                        #The file has been deleted by the other thread
                        pass

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
        return {
            "text":output
        }

def check_run_finish(info_current_file_store):
    if info_current_file_store["second_group"]!=None:
        file_path=os.path.join(info_current_file_store["session_folder"],"second_group","idata.pkl")
    else:
        file_path=os.path.join(info_current_file_store["session_folder"],"idata.pkl")
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

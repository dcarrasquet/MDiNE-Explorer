from dash import dcc, html,dash_table, callback_context
from dash.dependencies import Input, Output, State
import os
import re
import base64
import pandas as pd
from dash.exceptions import PreventUpdate

from maindash import app, type_storage #info_current_file,
#from ..mdine.extract_data_files import get_infos_file
#from scripts.mdine.extract_data_files import get_infos_file
from mdine.extract_data_files import (
    get_infos_file,
    get_df_file,
    get_info_separate_groups,
    get_df_taxa,
    find_reference_taxa,
    get_list_taxa,
    get_list_binary_covariates)

#data_path="data/test_app/"

input_field_text_style = {
    'width': '20%',
    'padding': '8px',
    'border': '1px solid #ccc',
    'borderRadius': '5px',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
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
    'hover': button_hover_style
}

def layout_data():
    return html.Div([
          html.Div([
            html.H3('Import Data file',style={'display': 'inline-block'}),
            html.Span("i", id="info-icon-file", title="Only csv and tsv files are accepted.",
              style={'display': 'inline-block', 'marginLeft': '10px',
                     'width': '20px', 'height': '20px', 'borderRadius': '50%',
                     'backgroundColor': '#007BFF', 'color': 'white', 'textAlign': 'center',
                     'lineHeight': '20px', 'cursor': 'pointer'}),
        ]),
            html.Div(style={'width': '70%', 'display': 'inline-block','clear': 'both', 'borderRight': '3px solid #ccc'}, children=[
        
        # Contenu de la partie gauche
        html.Div(style={'border-bottom': '3px solid #ccc'},children=[
            # Data part
            #dcc.Store(id='upload-status', storage_type=type_storage),
        html.Div(children=[
            dcc.Upload(id='upload-data',children=html.Div(['Drag and Drop or ',html.A('Select File')]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'display': 'inline-block'
            },
            multiple=False,
            accept='.csv,.tsv',
        ),
    html.Div(id='output-data-upload',style={'display': 'inline-block','margin-bottom':'1em'})
        ]),
    html.Div(id='output-df-upload'),
    
    html.Br(),
    dcc.Store(id='all-checks-status', storage_type=type_storage),
    html.Div(style={'display': 'inline-block','vertical-align': 'middle'},children=[
        dcc.Store(id='interval-covariate-status', storage_type=type_storage),
          html.H5("Interval of covariate columns ",style={'display': 'inline-block'}),
    dcc.Input(id='interval-covariate-input', type='text',disabled=True, placeholder='ex: 2-5',persistence=True,persistence_type=type_storage,style=input_field_text_style),
    #dcc.Input(id='interval-covariate-input', type='text',disabled=True,value="2-5", placeholder='ex: 2-5',persistence=True,persistence_type=type_storage,style=input_field_text_style),
    html.Button('Confirm', id='validate-covariate-button', n_clicks=0,disabled=True,),#style=button_style
    html.Div(id='interval-covariate-output',style={'display': 'inline-block'}),
    ]),
    html.Div(style={'display': 'inline-block','vertical-align': 'middle'},children=[
        dcc.Store(id='interval-taxa-status', storage_type=type_storage),
          html.H5("Interval of taxa columns ",style={'display': 'inline-block'}),
    dcc.Input(id='interval-taxa-input', type='text',disabled=True, placeholder='ex: 6-11',persistence=True,persistence_type=type_storage,style=input_field_text_style),
    #dcc.Input(id='interval-taxa-input', type='text',disabled=True,value='6-11', placeholder='ex: 6-11',persistence=True,persistence_type=type_storage,style=input_field_text_style),
    html.Button('Confirm', id='validate-taxa-button',disabled=True, n_clicks=0,), #style=button_style
    html.Div(id='interval-taxa-output',style={'display': 'inline-block'}),
    ]),

    html.Div(children=[
        html.H5("Presence of a reference taxa",style={'display': 'inline-block','margin-right': '10px','margin-top':'1em','margin-bottom':'1em'}),
        dcc.Checklist(options=[{'label': '', 'value': 'checked',"disabled":True}],value=[],id='check-ref-taxa',persistence=True,persistence_type=type_storage,style={'display': 'inline-block'}),
        html.Span("i", id="info-icon-ref-taxa", title='''The reference taxa is a species with a low deviation/mean ratio. It will not be plotted on the final network. If no species is given by the user, the species with the lowest ratio will be chosen by default.''',
              style={'display': 'inline-block', 'marginLeft': '10px',
                     'width': '20px', 'height': '20px', 'borderRadius': '50%',
                     'backgroundColor': '#007BFF', 'color': 'white', 'textAlign': 'center',
                     'lineHeight': '20px', 'cursor': 'pointer'}),
        html.Div(id='select-ref-taxa',style={'display':'none'},children=[
            html.Div(
            style={'display': 'flex', 'align-items': 'center'},  # Utilisation de flexbox pour aligner les éléments
            children=[
                html.H5(
                    "Reference taxa",
                    style={'text-indent':'30px','margin-right': '10px','flex-shrink': 0}  # Petit espace entre le texte et le dropdown
                ),
                html.Div(children=[dcc.Dropdown(
                    id='reference-taxa-dropdown',
                    clearable=False,
                    persistence=True,
                    persistence_type=type_storage
                )],
                style={'width': '50%'}),
            ]
        ),
        dcc.Store(id='reference-taxa-status', storage_type=type_storage),
    #       html.H5("Reference taxa",style={"text-indent": '30px','display': 'inline-block'}),
    # dcc.Store(id='reference-taxa-status', storage_type=type_storage),
    # dcc.Dropdown(id='reference-taxa-dropdown',clearable=False,persistence=True,persistence_type=type_storage,style={'display': 'inline-block','width': '50%', 'margin': '20px 0'}),
    
        ]),
    ]),

    html.Div(children=[
        html.H5("Separate data in two groups",style={'display': 'inline-block','margin-right': '10px','margin-top':'1em','margin-bottom':'1em'}),
        dcc.Checklist(options=[{'label': '', 'value': 'checked',"disabled":True}],value=[],id='check-separate-data',persistence=True,persistence_type=type_storage,style={'display': 'inline-block'}),
        html.Span("i", id="info-icon-separate-data", title='''If you want to separate your data in two groups, the phenotype column must contain only 0 and 1. If not it raises an error. ''',
              style={'display': 'inline-block', 'marginLeft': '10px',
                     'width': '20px', 'height': '20px', 'borderRadius': '50%',
                     'backgroundColor': '#007BFF', 'color': 'white', 'textAlign': 'center',
                     'lineHeight': '20px', 'cursor': 'pointer'}),
        html.Div(id='select-separate-data',style={'display':'none'},children=[
            html.Div(
            style={'display': 'flex', 'align-items': 'center','margin-bottom':'1em'},
            children=[
                html.H5(
                    "Phenotype column",
                    style={'text-indent':'30px','margin-right': '10px','flex-shrink': 0}
                ),
                html.Div(children=[dcc.Dropdown(
                    id='phenotype-column-dropdown',
                    clearable=False,
                    persistence=True,
                    persistence_type=type_storage
                )],
                style={'width': '50%'}),
            ]
        ),
        dcc.Store(id='phenotype-column-status', storage_type=type_storage)]),
        

        # html.Div(id='select-separate-data',style={'display': 'none','vertical-align': 'middle'},children=[
        #     #style={'display': 'inline-block','vertical-align': 'middle'}
        #     html.H5("Phenotype column ",style={"text-indent": '30px','display': 'inline-block'}),
        #     dcc.Store(id='phenotype-column-status', storage_type=type_storage),
        #     dcc.Input(id='phenotype-column-input',type='number',persistence=True,persistence_type=type_storage,style=input_field_number_style),
        #     #min=info_current_file_store['covar_start'],max=info_current_file_store['covar_end'],step=1,value=info_current_file_store['covar_start']
        #     html.Div(id='phenotype-column-output',style={'display': 'inline-block'}),

        # ]),
    ])
    ]),

    #Filters Part

    html.H5("Filters",style={'fontWeight': 'bold','margin-top':'0.5em'}),

    html.Div(style={'margin-top':'1em'},children=[
        html.H5("Delete columns with a certain pourcent of zeros",style={'display': 'inline-block','margin-right': '10px'}),
        dcc.Checklist(options=[{'label': '', 'value': 'checked',"disabled":True}],value=[],id='check-filter-columns-zero',persistence=True,persistence_type=type_storage,style={'display': 'inline-block'}),
        #html.Div(id='select-filter-columns-zero'),
        html.Div(id='select-filter-columns-zero',style={'display': 'none','vertical-align': 'middle'},children=[
          html.H5("Select the pourcent of zeros ",style={"text-indent": '30px','display': 'inline-block'}),
          dcc.Store(id='filter-columns-zero-status', storage_type=type_storage),
          dcc.Input(id='filter-columns-zero-input',type='number',min=0,max=100,step=1,value=60,persistence=True,persistence_type=type_storage,style=input_field_number_style),
        html.Div(id='filter-columns-zero-output',style={'display': 'inline-block'}),
        ])
        
    ]),

    html.Div(style={'margin-top':'1em'},children=[
        html.H5("Keep columns with highest deviation/mean",style={'display': 'inline-block','margin-right': '10px'}),
        dcc.Checklist(options=[{'label': '', 'value': 'checked',"disabled":True}],value=[],id='check-filter-deviation-mean',persistence=True,persistence_type=type_storage,style={'display': 'inline-block'}),
        #html.Div(id='select-filter-deviation-mean'),
        html.Div(id='select-filter-deviation-mean',style={'display': 'none','vertical-align': 'middle'},children=[
            html.H5("Select the number of colums to keep ",style={"text-indent": '30px','display': 'inline-block'}),
            dcc.Store(id='filter-deviation-mean-status', storage_type=type_storage),
        dcc.Input(id='filter-deviation-mean-input',type='number',min=1,step=1,value=1,persistence=True,persistence_type=type_storage,style=input_field_number_style),
        html.Div(id='filter-deviation-mean-output',style={'display': 'inline-block'}),
        ])
    ])
    
    ]),


    html.Div(style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}, children=[
        # Contenu de la partie droite
        #html.H1("Data Informations"),
        # Ajoutez ici les composants de votre partie droite
        html.Div(children=[

            html.Div(style={'margin-top': '1em','margin-left': '20px'},children=[

                html.Div(children=[html.H5("Initial Data",style={'fontWeight': 'bold'}),

                html.Div(id="columns-info",children=[html.H5(f"Number columns: ",style={"text-indent": '15px'})]),
                html.Div(style={"text-indent": '15px'},children=[
                    html.Div(id="covariates-info",children=[html.H5("Covariates: ",style={"text-indent": '30px'})]),
                    html.Div(id="taxa-info",children=[html.H5("Taxa: ",style={"text-indent": '30px'})]),
                    html.Div(id="taxa-reference-info",children=[html.H5("Reference Taxa: ",style={"text-indent": '30px'})])
                ]),
                html.H5("Individuals: ",style={"text-indent": '15px'},id="rows-info"),
                html.Div(id="separate-groups-info",children=[
                    html.Div(id="first-group-info",children=[html.H5("First Group: ",style={"text-indent": '30px'})]),
                    html.Div(id="second-group-info",children=[html.H5("Second Group: ",style={"text-indent": '30px'})])
                ])
                

            ]),

            html.Div(style={'margin-top': '4em'},children=[

                html.H5("Filtered Data",style={'fontWeight': 'bold'}),
                html.Div(id="filtered-data-zeros"),
                html.Div(id="filtered-data-dev-mean"),
                html.Div(id="filtered-data-summary")

            ])

            
        ])
    ]),
    html.Div(style={'clear': 'both', 'borderLeft': '1px solid #ccc'}),
])

    ])
def find_smallest_missing_integer(lst):
    # Convert the list to a set for O(1) look-up times
    num_set = set(lst)
    # Start checking from the smallest positive integer
    i = 1
    # Keep incrementing i until you find a missing integer
    while i in num_set:
        i += 1
    return i

def get_new_session_folder():
    # Path of the parent folder
    folder_parent = 'data/dash_app'

    sub_folders = []

    for element in os.listdir(folder_parent):
        complete_path = os.path.join(folder_parent, element)
        if os.path.isdir(complete_path):
            sub_folders.append(element)

    numbers_simulations = [int(re.search(r'\d+', nom).group()) for nom in sub_folders if re.search(r'\d+', nom)]
    #simulation_number = max(numbers_simulations)+1 if numbers_simulations else 1
    simulation_number=find_smallest_missing_integer(numbers_simulations) if numbers_simulations else 1
    print(simulation_number)

    folder_simulation=folder_parent+"/session_"+str(simulation_number)+"/"
    os.makedirs(folder_simulation)
    return folder_simulation

###### File Upload ######
    
# add a click to the appropriate store.
@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
        #Output('upload-status', 'data'),
                Input("upload-data", 'filename'),
                Input('upload-data', 'contents'),
                #State('upload-status', 'data'),
                State("info-current-file-store","data"),prevent_initial_call=True)
def on_click(filename,contents, info_current_file_store):#data,
    if filename is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate
    if contents is not None:
        # Récupérer les données et le nom du fichier depuis les métadonnées
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if info_current_file_store["session_folder"]==None:
            session_folder=get_new_session_folder()
        else:
            session_folder=info_current_file_store["session_folder"]
            try:
                os.remove(info_current_file_store["filename"])
            except:
                pass
            info_current_file_store["filename"]==None


        with open(os.path.join(session_folder, filename), 'wb') as f:
            f.write(decoded)

        info_current_file_store["filename"]=os.path.join(session_folder, filename)
        info_current_file_store["session_folder"]=session_folder
        info_current_file_store["nb_rows"]=None
        info_current_file_store["nb_columns"]=None
        info_current_file_store["covar_start"]=None
        info_current_file_store["covar_end"]=None
        info_current_file_store["taxa_start"]=None
        info_current_file_store["taxa_end"]=None
        info_current_file_store["reference_taxa"]=None
        info_current_file_store["phenotype_column"]=None
        info_current_file_store["first_group"]=None
        info_current_file_store["filter_zeros"]=None
        info_current_file_store["filter_dev_mean"]=None

        return info_current_file_store#,data

# output the stored clicks in the table cell.
@app.callback(Output("output-data-upload", 'children'),
              Output("output-df-upload","children"),
              Output("columns-info","children"),
              Output("rows-info","children"),
              Output("interval-covariate-input", 'disabled'),
              Output("validate-covariate-button", 'disabled'),
              Output("interval-taxa-input", 'disabled'),
              Output("validate-taxa-button", 'disabled'),
              Output("upload-data","disabled"),
              Output("info-current-file-store","data",allow_duplicate=True),
                #Input('upload-status', 'modified_timestamp'),
                #State('upload-status', 'data'),
                State("output-data-upload", 'children'),
                State("output-df-upload","children"),
                State("columns-info","children"),
                State("rows-info","children"),
                Input("info-current-file-store","modified_timestamp"),
                State("info-current-file-store","data"),prevent_initial_call=True)
def on_data(output_data,output_df,columns_info,rows_info,ts,info_current_file_store): #ts, data,
    if ts is None:
        raise PreventUpdate
    
    #print('Info current afer reuplod: ',info_current_file_store)
    # data = data or {}

    #print("Data get filename: ",data.get('filename',None))

    if info_current_file_store["filename"]==None:
        raise PreventUpdate
    
    if info_current_file_store["status-run-model"]=="in-progress" or info_current_file_store["status-run-model"]=="completed":
        #Disable changes in data view
        return output_data,output_df,columns_info,rows_info,True,True,True,True,True,info_current_file_store

    if info_current_file_store["filename"]!=None and check_df_numeric(get_df_file(info_current_file_store)):
    #     return html.Div(),html.Div(),html.Div(),html.Div(),True,True,True,True,False,info_current_file_store
    # else:
        #return data.get('filename', "")
        filename_base=os.path.basename(info_current_file_store["filename"])
        valid_file= html.H5('File successfully downloaded : {}'.format(filename_base),style={'display': 'inline-block'})
        nb_rows,nb_columns=get_infos_file(info_current_file_store['filename'])
        dash_table=create_dash_table(get_df_file(info_current_file_store))
        info_current_file_store["nb_rows"]=nb_rows
        info_current_file_store["nb_columns"]=nb_columns
        infos_columns=html.H5(f"Number columns: {nb_columns}",style={"text-indent": '15px'})
        infos_rows=html.H5(f"Number rows: {nb_rows}",style={"text-indent": '15px'})
        
        return  valid_file,dash_table,infos_columns,infos_rows,False,False,False,False,False,info_current_file_store
    else:
        filename_base=os.path.basename(info_current_file_store["filename"])
        invalid_file= html.H5('ERROR file {} contains non-numerical values. The file must contain numbers only (except column names).'.format(filename_base),style={'display': 'inline-block','color':'red'})
        infos_columns=html.H5(f"Number columns: ERROR file contains non-numerical values",style={"text-indent": '15px','color':'red'})
        infos_rows=html.H5(f"Number rows: ERROR file contains non-numerical values",style={"text-indent": '15px','color':'red'})
        os.remove(info_current_file_store["filename"])
        info_current_file_store["filename"]=None
        return invalid_file,None,infos_columns,infos_rows,True,True,True,True,False,info_current_file_store
    
def check_df_numeric(df):
    #print("Dtypes: ",type(df.dtypes))
    #print(set(df.dtypes.tolist()))
    return df.map(is_numeric).all().all()

def is_numeric(value):
    try:
        pd.to_numeric(value)
        return True
    except ValueError:
        #print("Error value: ",value)
        return False

def create_dash_table(df):
    return html.Div([
        dash_table.DataTable(
            data=df.to_dict('records'),
            #columns=[{'name': i, 'id': i} for i in df.columns],
            columns = [{'name': f"{i+1}. {col}", 'id': col} for i, col in enumerate(df.columns)],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '0px', 
                #'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
        ),
        html.Hr()  # horizontal line
    ],style={'width':"90%",'margin-left': 'auto','margin-right': 'auto'})

def check_intervals(interval_cov,interval_taxa,check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean,info_current_file_store):

    # print("Interval cov: ",interval_cov)
    # print("Interval taxa: ",interval_taxa)

    info_current_file_store["covar_start"],info_current_file_store["covar_end"]=None,None
    info_current_file_store["taxa_start"],info_current_file_store["taxa_end"]=None,None


    options_check_boxes=[{'label': '', 'value': 'checked',"disabled":True}]

    match_cov = re.match(r'^(\d+)-(\d+)$', interval_cov)
    match_taxa = re.match(r'^(\d+)-(\d+)$', interval_taxa)

    values_check=[[],[],[],[]]

    # print("Match cov: ", match_cov)
    # print("Match taxa: ",match_taxa)
    if match_cov and match_taxa:
        start_taxa, end_taxa = map(int, match_taxa.groups())
        start_cov, end_cov = map(int, match_cov.groups())
        if start_taxa>=end_taxa or start_cov>=end_cov:
            info_cov=html.H5("Covariates: ERROR Covariates or Taxa field is in wrong order",style={"margin-left": '30px','color': 'red'})
            output_cov=html.H5("ERROR Covariates or Taxa field is in wrong order (must have start<end)",style={'color': 'red'})
            info_taxa=html.H5("Taxa: ERROR Covariates or Taxa field is in wrong order",style={"margin-left": '30px','color': 'red'})
            output_taxa=html.H5("ERROR Covariates or Taxa field is in wrong order (must have start<end)",style={'color': 'red'})
        elif max(end_cov,end_taxa)>info_current_file_store["nb_columns"]:
            info_cov=html.H5("Covariates: ERROR Interval upper bound too large",style={"margin-left": '30px','color': 'red'})
            output_cov=html.H5(f"ERROR Upper bound to large: {max(end_cov,end_taxa)} (only {info_current_file_store["nb_columns"]} number)",style={'color': 'red'})
            info_taxa=html.H5("Taxa: ERROR Interval upper bound too large",style={"margin-left": '30px','color': 'red'})
            output_taxa=html.H5(f"ERROR Upper bound to large: {max(end_cov,end_taxa)} (only {info_current_file_store["nb_columns"]} number)",style={'color': 'red'})
        elif min(end_cov,end_taxa)>=max(start_taxa,start_cov):
            # Intervals overlap
            info_cov=html.H5("Covariates: ERROR Intervals overlap",style={"margin-left": '30px','color': 'red'})
            output_cov=html.H5(f"ERROR Intervals Overlap: {min(end_cov,end_taxa)} (upper bound) and {max(start_taxa,start_cov)} (lower bound)",style={'color': 'red'})
            info_taxa=html.H5("Taxa: ERROR Intervals overlap",style={"margin-left": '30px','color': 'red'})
            output_taxa=html.H5(f"ERROR Intervals Overlap: {min(end_cov,end_taxa)} (upper bound) and {max(start_taxa,start_cov)} (lower bound)",style={'color': 'red'})
        else:
            info_current_file_store["covar_start"],info_current_file_store["covar_end"]=start_cov,end_cov
            info_current_file_store["taxa_start"],info_current_file_store["taxa_end"]=start_taxa,end_taxa
            output_cov= html.H5(f"Valid interval : {start_cov} - {end_cov}",style={'display': 'inline-block','vertical-align': 'middle'})
            info_cov=html.H5(f"Covariates: {end_cov-start_cov+1}",style={"margin-left": '30px'})
            output_taxa= html.H5(f"Valid interval : {start_taxa} - {end_taxa}",style={'display': 'inline-block','vertical-align': 'middle'})
            info_taxa=html.H5(f"Taxa: {end_taxa-start_taxa+1}",style={"margin-left": '30px'})
            options_check_boxes=[{'label': '', 'value': 'checked'}]

            
            #check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean
            if len(check_ref_taxa)!=0:
                values_check[0]=['checked']
            if len(check_separate_data)!=0:
                values_check[1]=['checked']
            if len(check_filter_zeros)!=0:
                values_check[2]=['checked']
            if len(check_filter_dev_mean)!=0:
                values_check[3]=['checked']
    else:
        # At least one interval is empty or invalid. 
        info_cov=html.H5(f"Covariates: ERROR Covariates or Taxa field is empty or incorrect",style={"margin-left": '30px','color': 'red'})
        output_cov=html.H5("ERROR Covariates or Taxa field is empty or incorrect",style={'color': 'red'})
        info_taxa=html.H5(f"Taxa: ERROR Covariates or Taxa field is empty or incorrect",style={"margin-left": '30px','color': 'red'})
        output_taxa=html.H5("ERROR Covariates or Taxa field is empty or incorrect",style={'color': 'red'})
    
    return info_current_file_store,output_cov,info_cov,output_taxa,info_taxa,options_check_boxes,options_check_boxes,options_check_boxes,options_check_boxes,values_check[0],values_check[1],values_check[2],values_check[3]
        



###### Interval covariate and taxa ######

# add a click to the appropriate store.
@app.callback(Output('interval-covariate-status', 'data'),
              Output('interval-taxa-status', 'data'),
                Input("validate-covariate-button", 'n_clicks'),
                Input("validate-taxa-button", 'n_clicks'),
                State("interval-covariate-input","value"),
                State("interval-taxa-input","value"),
                State('interval-covariate-status', 'data'),
                State('interval-taxa-status', 'data'))
def on_click(n_clicks_cov,n_clicks_taxa,value_cov,value_taxa,data_cov,data_taxa):
    if n_clicks_cov==None and n_clicks_taxa==None and value_cov==None and value_taxa==None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate

    
    data_cov = data_cov or {'value': ""}
    data_taxa = data_taxa or {'value': ""}
    
    if value_cov!=None:
        data_cov["value"]=value_cov
    if value_taxa!=None:
        data_taxa["value"]=value_taxa

    return data_cov,data_taxa

# output the stored clicks in the table cell.
@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
              Output("interval-covariate-output", 'children'),
              Output("covariates-info","children"),
              Output("interval-taxa-output", 'children'),
              Output("taxa-info","children"),
              Output("check-ref-taxa","options"),
              Output("check-separate-data","options"),
              Output("check-filter-columns-zero","options"),
              Output("check-filter-deviation-mean","options"),
              Output("check-ref-taxa","value"),
              Output("check-separate-data","value"),
              Output("check-filter-columns-zero","value"),
              Output("check-filter-deviation-mean","value"),
              Output("reference-taxa-dropdown","disabled"),
              Output("phenotype-column-dropdown","disabled"),
              Output("filter-columns-zero-input","disabled"),
              Output("filter-deviation-mean-input","disabled"),
                Input('interval-covariate-status', 'modified_timestamp'),
                Input('interval-taxa-status', 'modified_timestamp'),
                State('interval-covariate-status', 'data'),
                State('interval-taxa-status', 'data'),
                State("check-ref-taxa","value"),
                State("check-separate-data","value"),
                State("check-filter-columns-zero","value"),
                State("check-filter-deviation-mean","value"),
                State("interval-covariate-output", 'children'),
                State("covariates-info","children"),
                State("interval-taxa-output", 'children'),
                State("taxa-info","children"),
                Input("info-current-file-store","modified_timestamp"),
                State("info-current-file-store","data"),prevent_initial_call=True)
def on_data(ts_cov,ts_taxa,data_cov,data_taxa,check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean,children_cov,children_cov_info,children_taxa,children_taxa_info,ts_info_store,info_current_file_store):

    options_check_boxes=[{'label': '', 'value': 'checked',"disabled":True}]

    if info_current_file_store["filename"]==None or info_current_file_store["nb_columns"]==None:
        #raise PreventUpdate
        return info_current_file_store,children_cov,children_cov_info,children_taxa,children_taxa_info,options_check_boxes,options_check_boxes,options_check_boxes,options_check_boxes,check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean,True,True,True,True

    if ts_cov is None and ts_taxa is None:
        raise PreventUpdate
    
    if info_current_file_store["status-run-model"]=="in-progress" or info_current_file_store["status-run-model"]=="completed":
        return info_current_file_store,children_cov,children_cov_info,children_taxa,children_taxa_info,options_check_boxes,options_check_boxes,options_check_boxes,options_check_boxes,check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean,True,True,True,True

    data_cov = data_cov or {}
    data_taxa=data_taxa or {}

    if data_cov.get('value',None)==None or data_taxa.get('value',None)==None:
        #print("Premiere option")
        return info_current_file_store,html.Div(),html.Div(),html.Div(),html.Div(),options_check_boxes,options_check_boxes,options_check_boxes,options_check_boxes,[],[],[],[],False,False,False,False
    else:
        #print("Deuxieme option")
        # print("Liste des checks: \n")
        # print(check_ref_taxa)
        # print(check_separate_data)
        # print(check_filter_zeros)
        # print(check_filter_dev_mean)
        response=check_intervals(data_cov["value"],data_taxa["value"],check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean,info_current_file_store)
        return *response,False,False,False,False
    


###### Reference Taxa column #######

#### Update reference-taxa-status ####
@app.callback(Output('reference-taxa-status', 'data'),
                Input("reference-taxa-dropdown","value"),
                State('reference-taxa-status', 'data'))
def on_click(value, data):
    if value is None:
        raise PreventUpdate

    return value


#### Update all components related to Reference Taxa ####
@app.callback(
        Output("info-current-file-store","data",allow_duplicate=True),
        Output('select-ref-taxa', 'style'),
        Output('reference-taxa-dropdown', 'options'),
        Output('reference-taxa-dropdown', 'value'),
        Output("taxa-reference-info", 'children',allow_duplicate=True),
        Input('check-ref-taxa', 'value'),
        Input('reference-taxa-status', 'modified_timestamp'),
        State('reference-taxa-status', 'data'),
        State("info-current-file-store","data"),prevent_initial_call=True)
def update_dropdown_ref_taxa(value_check,ts_ref_taxa,value_ref_taxa,info_current_file_store):

    if len(value_check)==0:
        ## Not Checked
        message=None
        options=[]
        value_dropdown=None
        if info_current_file_store["taxa_start"]!=None:
            try:
                ref_taxa=find_reference_taxa(info_current_file_store)
                message=html.H5(f"Reference Taxa: {ref_taxa}",style={"text-indent": '30px'})
                info_current_file_store["reference_taxa"]=ref_taxa
                options={'label':ref_taxa,'value':ref_taxa}
                value_dropdown=ref_taxa
            except:
                message=html.H5("Reference Taxa: ERROR Enter Taxa interval",style={"text-indent": '30px',"color":"red"})
                info_current_file_store["reference_taxa"]=None

        else:
            message=html.H5("Reference Taxa: ERROR Enter Taxa interval",style={"text-indent": '30px',"color":"red"})
            info_current_file_store["reference_taxa"]=None
        
        return info_current_file_store,{'display':'none'},options,value_dropdown,message

    else:
        ##Checked

        if info_current_file_store["taxa_start"]==None:
            raise PreventUpdate
        
        taxa_list=get_list_taxa(info_current_file_store)
        options=[]
        for taxa_name in taxa_list:
            options.append({'label':taxa_name,'value':taxa_name})

        if value_ref_taxa==None:
            #Automatic reference taxa
            ref_taxa=find_reference_taxa(info_current_file_store)
        else:
            ref_taxa=value_ref_taxa
        info_current_file_store["reference_taxa"]=ref_taxa
        message=html.H5(f"Reference Taxa: {ref_taxa}",style={"text-indent": '30px'})
        return info_current_file_store,None,options,ref_taxa,message
        

###### Phenotype column ######

#### Update reference-taxa-status ####
@app.callback(Output('phenotype-column-status', 'data'),
                Input("phenotype-column-dropdown","value"),
                State('phenotype-column-status', 'data'))
def on_click(value, data):
    if value is None:
        raise PreventUpdate

    return value


#### Update all components related to Reference Taxa ####
@app.callback(
        Output("info-current-file-store","data",allow_duplicate=True),
        Output('select-separate-data', 'style'),
        Output('phenotype-column-dropdown', 'disabled',allow_duplicate=True),
        Output('phenotype-column-dropdown', 'options'),
        Output('phenotype-column-dropdown', 'value'),
        Output("first-group-info","children"),
        Output("second-group-info","children"),
        Input('check-separate-data', 'value'),
        Input('phenotype-column-status', 'modified_timestamp'),
        State('phenotype-column-status', 'data'),
        State("info-current-file-store","data"),prevent_initial_call=True)
def update_dropdown_phentype_column(value_check,ts_phenotype_column,value_phenotype_column,info_current_file_store):

    if len(value_check)==0:
        ## Not Checked
        info_current_file_store['phenotype_column']=None
        info_current_file_store['first_group']=None
        info_current_file_store['second_group']=None
        return info_current_file_store,{'display':'none'},False,[],None,[],[]

    else:
        if info_current_file_store["taxa_start"]==None:
            raise PreventUpdate
        ##Checked
        binary_covariates_list=get_list_binary_covariates(info_current_file_store)

        if binary_covariates_list==[]:
            first_group=html.H5("First Group: No binary covariate column",style={"text-indent": '30px','color':'red'})
            second_group=html.H5("Second Group: No binary covariate column",style={"text-indent": '30px','color':'red'})
            options=[{"label": html.Span(['No binary covariate column'], style={'color': 'red'}),"value": "error",}]
            value="error"
            info_current_file_store['phenotype_column']='error'
            info_current_file_store['first_group']='error'
            info_current_file_store['second_group']='error'
            return info_current_file_store,None,True,options,value,first_group,second_group
        else:
            options=[]
            for covar_name in binary_covariates_list:
                options.append({'label':covar_name,'value':covar_name})

            if value_phenotype_column==None:
                #Automatic reference taxa
                phenotype_column=binary_covariates_list[0]
            else:
                phenotype_column=value_phenotype_column

            disabled_dropdown=False
            if info_current_file_store["status-run-model"]!='not-yet':
                disabled_dropdown=True

            info_current_file_store["phenotype_column"]=phenotype_column
            nb_first_group,nb_second_group=get_info_separate_groups(info_current_file_store)
            info_current_file_store['first_group']=nb_first_group
            info_current_file_store['second_group']=nb_second_group
            first_group=html.H5(f"First Group: {nb_first_group}",style={"text-indent": '30px'})
            second_group=html.H5(f"Second Group: {nb_second_group}",style={"text-indent": '30px'})
            
            return info_current_file_store,None,disabled_dropdown,options,phenotype_column,first_group,second_group

# # add a click to the appropriate store.
# @app.callback(Output('info-current-file-store','data',allow_duplicate=True),
#                 Input("phenotype-column-input","value"),
#                 State('info-current-file-store','data'),prevent_initial_call=True)
# def on_click(value, info_current_file_store):
#     if value is None:
#         raise PreventUpdate
    
#     print("Velaueueu: ",value)
    
#     info_current_file_store["phenotype_column"]=value

#     return info_current_file_store

# # output the stored clicks in the table cell.
# @app.callback(Output("info-current-file-store","data"),
#                 Output("phenotype-column-output", 'children'),
#               Output("first-group-info","children"),
#               Output("second-group-info","children"),
#                 Input('info-current-file-store', 'modified_timestamp'),
#                 State("info-current-file-store","data"),
#                 State('check-separate-data', 'value'),
#                 Input("phenotype-column-input","value"),prevent_initial_call=True)
# def on_data(ts,info_current_file_store,value_check,value_input):
#     if ts is None:
#         raise PreventUpdate
    
#     print('valeu check: ',value_check)
#     print('Phen column: ',info_current_file_store["phenotype_column"])
    
#     if len(value_check)==0:
#         return info_current_file_store,[],[],[]
#     else:
#         if info_current_file_store["phenotype_column"]!="error" and info_current_file_store["phenotype_column"]!=None and check_phenotype_column(info_current_file_store):
#             nb_first_group,nb_second_group=get_info_separate_groups(info_current_file_store)
#             validation_message=html.I(className="fas fa-check-circle", style={'color': 'green'})
#             first_group=html.H5(f"First Group: {nb_first_group}",style={"text-indent": '30px'})
#             second_group=html.H5(f"Second Group: {nb_second_group}",style={"text-indent": '30px'})
#             #info_current_file_store['phenotype_column']=data["value"]
#             info_current_file_store['first_group']=nb_first_group
#             info_current_file_store['second_group']=nb_second_group

#             return info_current_file_store,validation_message,first_group,second_group
#                 #return f"Valid interval : {start} - {end}"
#         else:
#             first_group=html.H5("First Group: Error Phenotype Column isn't binary",style={"text-indent": '30px','color':'red'})
#             second_group=html.H5("Second Group: Error Phenotype Column isn't binary",style={"text-indent": '30px','color':'red'})
#             error_message=[html.I(className="fas fa-times-circle", style={'color': 'red'}),html.H5("Error : Column is not binary.",style={'display': 'inline-block'})]
#             info_current_file_store['phenotype_column']='error'
#             info_current_file_store['first_group']='error'
#             info_current_file_store['second_group']='error'
#             return info_current_file_store,error_message,first_group,second_group
        
# @app.callback(Output('info-current-file-store', 'data',allow_duplicate=True),
#             Output('select-separate-data', 'style'),
#             Output('phenotype-column-input', 'min'),
#             Output('phenotype-column-input', 'max'),
#             Output('separate-groups-info','style'),
#             Input('check-separate-data', 'value'),
#             State("info-current-file-store","data"),
#             State('phenotype-column-input', 'value'),prevent_initial_call=True)
# def update_output_data(value,info_current_file_store,value1):
#     print(value1)
#     print("Je suis appelé alors que je devrais pas")
#     if len(value)!=0:
#         min=info_current_file_store['covar_start']
#         max=info_current_file_store['covar_end']
#         #value=info_current_file_store['covar_start']
#         #style={'display': 'inline-block','vertical-align': 'middle'}
#         style={'vertical-align': 'middle'}
#         return info_current_file_store,style,min,max,None
#     else:
#         info_current_file_store['phenotype_column']=None
#         info_current_file_store['first_group']=None
#         info_current_file_store['second_group']=None
#         return info_current_file_store,{'display':'none'},None,None,{'display':'none'}
  

###### Filters ######

###### Filters Zeros 

# add a click to the appropriate store.
@app.callback(Output('filter-columns-zero-status', 'data'),
                Input("filter-columns-zero-input","value"),
                State('filter-columns-zero-status', 'data'))
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

### Filter deviation/mean
    
# add a click to the appropriate store.
@app.callback(Output('filter-deviation-mean-status', 'data'),
                Input("filter-deviation-mean-input","value"),
                State('filter-deviation-mean-status', 'data'))
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

@app.callback(Output("info-current-file-store","data",allow_duplicate=True),
              Output('select-filter-columns-zero', 'style'),
              Output('select-filter-deviation-mean', 'style'),
              Output('filter-deviation-mean-output', 'children'),
              Output('filter-columns-zero-output', 'children'),
              Output("filtered-data-zeros","children"),
              Output("filtered-data-dev-mean","children"),
              Output("filtered-data-summary","children"),
              Output("filter-deviation-mean-input","max"),
              Output("filter-deviation-mean-input","value"),
              Input("check-filter-deviation-mean","value"),
              Input("check-filter-columns-zero","value"),
              Input('filter-deviation-mean-status', 'modified_timestamp'),
              Input("filter-columns-zero-status","modified_timestamp"),
              State('filter-deviation-mean-status', 'data'),
              State('filter-columns-zero-status', 'data'),
              State('info-current-file-store','data'),prevent_initial_call=True)
def get_changes_filters(check_dev_mean,check_zeros,ts_dev_mean,ts_zeros,data_dev_mean,data_zeros,info_current_file_store):

    if info_current_file_store["taxa_start"]==None:
        raise PreventUpdate

    style_not_displayed={'display': 'none','vertical-align': 'middle'}
    style_displayed={'vertical-align': 'middle'}

    filter_zero=None
    filter_dev_mean=None

    if len(check_zeros)!=0:
        filter_zero=data_zeros["value"]

    if len(check_dev_mean)!=0:
        filter_dev_mean=data_dev_mean["value"]
    
    if filter_zero==None and filter_dev_mean==None:
        info_current_file_store["filter_zeros"]=None
        info_current_file_store["filter_dev_mean"]=None

        #print("Info filter1: ",info_filter)

        return info_current_file_store,style_not_displayed,style_not_displayed,None,None,None,None,html.H5("No filter applied",style={"text-indent": '15px'}),None,1
    elif filter_zero!=None and filter_dev_mean==None:
        info_current_file_store["filter_zeros"]=filter_zero
        info_current_file_store["filter_dev_mean"]=None
        info_filter=get_df_taxa(info_current_file_store,"info") 

        #print("Info filter2: ",info_filter)

        info_zeros=html.H5(f"Filter with pourcent of zeros: {info_filter["zeros-deleted"]} taxa deleted",style={"text-indent": '15px'})
        output_zeros=html.H5(f"{info_filter["zeros-deleted"]} taxa deleted",style={'display': 'inline-block'})
        summary_filter=html.H5(f"Summary: {info_filter["remaining-taxa"]} taxa remaining",style={'display': 'inline-block',"text-indent": '15px'})
        return info_current_file_store,style_displayed,style_not_displayed,None,output_zeros,info_zeros,None,summary_filter,None,1
    
    elif filter_zero==None and filter_dev_mean!=None:
        info_current_file_store["filter_zeros"]=filter_zero
        info_current_file_store["filter_dev_mean"]=filter_dev_mean
        info_filter=get_df_taxa(info_current_file_store,"info")

        output_dev_mean=html.H5(f"{info_filter["dev-mean-deleted"]} taxa deleted",style={'display': 'inline-block'})
        info_dev_mean=html.H5(f"Filter with deviation/mean ratio: {info_filter["dev-mean-deleted"]} taxa deleted",style={"text-indent": '15px'})
        summary_filter=html.H5(f"Summary: {info_filter["remaining-taxa"]} taxa remaining",style={'display': 'inline-block',"text-indent": '15px'})

        return info_current_file_store,style_not_displayed,style_displayed,output_dev_mean,None,None,info_dev_mean,summary_filter,info_filter["taxa-init"]-info_filter["zeros-deleted"],min(filter_dev_mean,info_filter["taxa-init"]-info_filter["dev-mean-deleted"])
    else:
        #Filter on zeros percent and dev mean activated at the same time

        info_current_file_store["filter_zeros"]=filter_zero
        info_current_file_store["filter_dev_mean"]=filter_dev_mean
        info_filter=get_df_taxa(info_current_file_store,"info")

        #print("Info filter4: ",info_filter)

        info_zeros=html.H5(f"Filter with pourcent of zeros: {info_filter["zeros-deleted"]} taxa deleted",style={"text-indent": '15px'})
        output_zeros=html.H5(f"{info_filter["zeros-deleted"]} taxa deleted",style={'display': 'inline-block'})
        output_dev_mean=html.H5(f"{info_filter["dev-mean-deleted"]} taxa deleted",style={'display': 'inline-block'})
        info_dev_mean=html.H5(f"Filter with deviation/mean ratio: {info_filter["dev-mean-deleted"]} taxa deleted",style={"text-indent": '15px'})
        summary_filter=html.H5(f"Summary: {info_filter["remaining-taxa"]} taxa remaining",style={'display': 'inline-block',"text-indent": '15px'})
        
        return info_current_file_store,style_displayed,style_displayed,output_dev_mean,output_zeros,info_zeros,info_dev_mean,summary_filter,info_filter["taxa-init"]-info_filter["zeros-deleted"],min(filter_dev_mean,info_filter["taxa-init"]-info_filter["zeros-deleted"])

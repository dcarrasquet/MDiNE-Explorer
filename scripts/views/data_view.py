import dash
from dash import dcc, html,dash_table, callback_context
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
import re
import base64
from dash.exceptions import PreventUpdate
import math

from maindash import app,info_current_file, type_storage
#from ..mdine.extract_data_files import get_infos_file
#from scripts.mdine.extract_data_files import get_infos_file
from mdine.extract_data_files import get_infos_file,filter_deviation_mean,get_df_file,check_phenotype_column,filter_percent_zeros, get_info_separate_groups,find_reference_taxa

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
            html.Span("i", id="info-icon", title="Only csv and tsv files are accepted.",
              style={'display': 'inline-block', 'marginLeft': '10px',
                     'width': '20px', 'height': '20px', 'borderRadius': '50%',
                     'backgroundColor': '#007BFF', 'color': 'white', 'textAlign': 'center',
                     'lineHeight': '20px', 'cursor': 'pointer'}),
        ]),
            html.Div(style={'width': '70%', 'display': 'inline-block','clear': 'both', 'borderRight': '3px solid #ccc'}, children=[
        
        # Contenu de la partie gauche
        html.Div(style={'border-bottom': '3px solid #ccc'},children=[
            # Data part
            dcc.Store(id='upload-status', storage_type=type_storage),
        html.Div(children=[
            dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
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
    html.Div(id='output-data-upload',style={'display': 'inline-block'})
        ]),
    html.Div(id='output-df-upload'),
    
    html.Br(),
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
        html.H5("Presence of a reference taxa",style={'display': 'inline-block','margin-right': '10px'}),
        dcc.Checklist(options=[{'label': '', 'value': 'checked',"disabled":True}],value=[],id='check-ref-taxa',persistence=True,persistence_type=type_storage,style={'display': 'inline-block'}),
        html.Span("i", id="info-icon", title='''The reference taxa is a species with a low deviation/mean ratio. It will not be plotted on the final network. If no species is given by the user, the species with the lowest ratio will be chosen by default.''',
              style={'display': 'inline-block', 'marginLeft': '10px',
                     'width': '20px', 'height': '20px', 'borderRadius': '50%',
                     'backgroundColor': '#007BFF', 'color': 'white', 'textAlign': 'center',
                     'lineHeight': '20px', 'cursor': 'pointer'}),
        html.Div(id='select-ref-taxa'),
    ]),

    html.Div(children=[
        html.H5("Separate data in two groups",style={'display': 'inline-block','margin-right': '10px'}),
        dcc.Checklist(options=[{'label': '', 'value': 'checked',"disabled":True}],value=[],id='check-separate-data',persistence=True,persistence_type=type_storage,style={'display': 'inline-block'}),
        html.Div(id='select-separate-data'),
    ])
    ]),

    #Filters Part

    html.H5("Filters",style={'fontWeight': 'bold'}),

    html.Div(children=[
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

    html.Div(children=[
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
        html.H1("Data Informations"),
        # Ajoutez ici les composants de votre partie droite
        html.Div(children=[

            html.Div(children=[

                html.Div(children=[html.H5("Initial Data",style={'fontWeight': 'bold'}),

                html.Div(id="columns-info",children=[html.H5(f"Number columns: ",style={"text-indent": '15px'})]),
                html.Div(style={"text-indent": '15px'},children=[
                    html.Div(id="covariates-info",children=[html.H5("Covariates: ",style={"text-indent": '30px'})]),
                    html.Div(id="taxa-info",children=[html.H5("Taxa: ",style={"text-indent": '30px'})]),
                    html.Div(id="taxa-reference-info",children=[html.H5("Reference Taxa: ",style={"text-indent": '30px'})])
                ]),
                html.H5("Individuals: ",style={"text-indent": '15px'},id="rows-info"),
                html.Div(id="separate-groups-info")
                

            ]),

            html.Div(children=[

                html.H5("Filtered Data",style={'fontWeight': 'bold'}),
                html.Div(id="filtered-data-zeros"),
                html.Div(id="filtered-data-dev-mean"),
                html.Div(id="filtered-data-summary")

            ])

            
        ])
    ]),
    html.Div(style={'clear': 'both', 'borderLeft': '1px solid #ccc'}),
])

    ]) ## AJouté ici

def get_new_session_folder():
    # Path of the parent folder
    folder_parent = 'data/dash_app'

    # Liste pour stocker les noms des sous-dossiers
    sub_folders = []

    # Parcourir tous les éléments dans le dossier parent
    for element in os.listdir(folder_parent):
        # Vérifier si l'élément est un dossier
        complete_path = os.path.join(folder_parent, element)
        if os.path.isdir(complete_path):
            # Ajouter le nom du sous-dossier à la liste
            sub_folders.append(element)

    numbers_simulations = [int(re.search(r'\d+', nom).group()) for nom in sub_folders if re.search(r'\d+', nom)]
    simulation_number = max(numbers_simulations)+1 if numbers_simulations else 1

    folder_simulation=folder_parent+"/session_"+str(simulation_number)+"/"
    os.makedirs(folder_simulation)
    return folder_simulation

###### File Upload ######
    
# add a click to the appropriate store.
@app.callback(Output('upload-status', 'data'),
                Input("upload-data", 'filename'),
                Input('upload-data', 'contents'),
                State('upload-status', 'data'))
def on_click(filename,contents, data):
    global info_current_file
    if filename is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate
    if contents is not None:
        # Récupérer les données et le nom du fichier depuis les métadonnées
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        #filename = 'uploaded_file.txt'  # Nom de fichier par défaut
        # for content in content_type.split(';'):
        #     print("Content: ",content)
            # if 'filename=' in content:
            #     filename = content.split('=')[1].strip('"')
        # Enregistrer le fichier sur le serveur avec le même nom que celui déposé

        if info_current_file["session_folder"]==None:
            session_folder=get_new_session_folder()
        else:
            session_folder=info_current_file["session_folder"]
            os.remove(info_current_file["filename"])
            info_current_file["filename"]==None


        with open(os.path.join(session_folder, filename), 'wb') as f:
            f.write(decoded)

        # Give a default data dict with 0 clicks if there's no data.
        data = data or {'filename': filename}

        data["filename"]=filename

        info_current_file["filename"]=os.path.join(session_folder, filename)
        info_current_file["session_folder"]=session_folder

        return data

# output the stored clicks in the table cell.
@app.callback(Output("output-data-upload", 'children'),
              Output("output-df-upload","children"),
              Output("columns-info","children"),
              Output("rows-info","children"),
              Output("interval-covariate-input", 'disabled'),
              Output("validate-covariate-button", 'disabled'),
              Output("interval-taxa-input", 'disabled'),
              Output("validate-taxa-button", 'disabled'),
                Input('upload-status', 'modified_timestamp'),
                State('upload-status', 'data'))
def on_data(ts, data):
    global info_current_file
    if ts is None:
        raise PreventUpdate

    data = data or {}

    print("Data get filename: ",data.get('filename',None))

    if info_current_file["filename"]==None:
        raise PreventUpdate

    if data.get('filename',None)==None:
        return html.Div(),html.Div(),html.Div(),html.Div(),True,True,True,True
    else:
        #return data.get('filename', "")
        valid_file= html.H5('File successfully downloaded : {}'.format(data.get('filename', "")),style={'display': 'inline-block'})
        nb_rows,nb_columns=get_infos_file(info_current_file['filename'])
        dash_table=create_dash_table(get_df_file(info_current_file))
        info_current_file["nb_rows"]=nb_rows
        info_current_file["nb_columns"]=nb_columns
        infos_columns=html.H5(f"Number columns: {nb_columns}",style={"text-indent": '15px'})
        infos_rows=html.H5(f"Number rows: {nb_rows}",style={"text-indent": '15px'})
        
        return  valid_file,dash_table,infos_columns,infos_rows,False,False,False,False
    
def create_dash_table(df):
    return html.Div([
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
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



# @app.callback(Output("interval-covariate-output", 'children'),
#               Output("covariates-info","children"),
#               Output("interval-taxa-output", 'children'),
#               Output("taxa-info","children"),
#               Output("check-ref-taxa","options"),
#               Output("check-separate-data","options"),


def check_intervals(interval_cov,interval_taxa,check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean):
    global info_current_file

    # print("Interval cov: ",interval_cov)
    # print("Interval taxa: ",interval_taxa)

    info_current_file["covar_start"],info_current_file["covar_end"]=None,None
    info_current_file["taxa_start"],info_current_file["taxa_end"]=None,None


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
            info_cov=html.H5("Covariates: Covariates or Taxa field is in wrong order",style={"text-indent": '30px'})
            output_cov=html.H5("Covariates or Taxa field is in wrong order (must have start<end)")
            info_taxa=html.H5("Taxa: Covariates or Taxa field is in wrong order",style={"text-indent": '30px'})
            output_taxa=html.H5("Covariates or Taxa field is in wrong order (must have start<end)")
        elif max(end_cov,end_taxa)>info_current_file["nb_columns"]:
            info_cov=html.H5("Covariates: Interval upper bound too large",style={"text-indent": '30px'})
            output_cov=html.H5(f"Upper bound to large: {max(end_cov,end_taxa)} (only {info_current_file["nb_columns"]} number)")
            info_taxa=html.H5("Taxa: Interval upper bound too large",style={"text-indent": '30px'})
            output_taxa=html.H5(f"Upper bound to large: {max(end_cov,end_taxa)} (only {info_current_file["nb_columns"]} number)")
        elif min(end_cov,end_taxa)>=max(start_taxa,start_cov):
            # Intervals overlap
            info_cov=html.H5("Covariates: Intervals overlap",style={"text-indent": '30px'})
            output_cov=html.H5(f"Intervals Overlap: {min(end_cov,end_taxa)} (upper bound) and {max(start_taxa,start_cov)} (lower bound)")
            info_taxa=html.H5("Taxa: Intervals overlap",style={"text-indent": '30px'})
            output_taxa=html.H5(f"Intervals Overlap: {min(end_cov,end_taxa)} (upper bound) and {max(start_taxa,start_cov)} (lower bound)")
        else:
            info_current_file["covar_start"],info_current_file["covar_end"]=start_cov,end_cov
            info_current_file["taxa_start"],info_current_file["taxa_end"]=start_taxa,end_taxa
            info_current_file["df_taxa"]=get_df_file(info_current_file).iloc[:,start_taxa-1:end_taxa]
            info_current_file["df_covariates"]=get_df_file(info_current_file).iloc[:,start_cov-1:end_cov]
            output_cov= html.H5(f"Valid interval : {start_cov} - {end_cov}",style={'display': 'inline-block','vertical-align': 'middle'})
            info_cov=html.H5(f"Covariates: {end_cov-start_cov+1}",style={"text-indent": '30px'})
            output_taxa= html.H5(f"Valid interval : {start_taxa} - {end_taxa}",style={'display': 'inline-block','vertical-align': 'middle'})
            info_taxa=html.H5(f"Taxa: {end_taxa-start_taxa+1}",style={"text-indent": '30px'})
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
        info_cov=html.H5(f"Covariates: Covariates or Taxa field is empty or incorrect",style={"text-indent": '30px'})
        output_cov=html.H5("Covariates or Taxa field is empty or incorrect")
        info_taxa=html.H5(f"Taxa: Covariates or Taxa field is empty or incorrect",style={"text-indent": '30px'})
        output_taxa=html.H5("Covariates or Taxa field is empty or incorrect")
    
    return output_cov,info_cov,output_taxa,info_taxa,options_check_boxes,options_check_boxes,options_check_boxes,options_check_boxes,values_check[0],values_check[1],values_check[2],values_check[3]
        



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
@app.callback(Output("interval-covariate-output", 'children'),
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
                Input('interval-covariate-status', 'modified_timestamp'),
                Input('interval-taxa-status', 'modified_timestamp'),
                State('interval-covariate-status', 'data'),
                State('interval-taxa-status', 'data'),
                State("check-ref-taxa","value"),
                State("check-separate-data","value"),
                State("check-filter-columns-zero","value"),
                State("check-filter-deviation-mean","value"))
def on_data(ts_cov,ts_taxa,data_cov,data_taxa,check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean):
    global info_current_file

    options_check_boxes=[{'label': '', 'value': 'checked',"disabled":True}]

    if info_current_file["filename"]==None or info_current_file["nb_columns"]==None:
        raise PreventUpdate

    if ts_cov is None and ts_taxa is None:
        raise PreventUpdate

    data_cov = data_cov or {}
    data_taxa=data_taxa or {}

    if data_cov.get('value',None)==None or data_taxa.get('value',None)==None:
        print("Premiere option")
        return html.Div(),html.Div(),html.Div(),html.Div(),options_check_boxes,options_check_boxes,options_check_boxes,options_check_boxes,[],[],[],[]
    else:
        print("Deuxieme option")
        # print("Liste des checks: \n")
        # print(check_ref_taxa)
        # print(check_separate_data)
        # print(check_filter_zeros)
        # print(check_filter_dev_mean)
        return check_intervals(data_cov["value"],data_taxa["value"],check_ref_taxa,check_separate_data,check_filter_zeros,check_filter_dev_mean)



###### Select Taxa and groups ######

@app.callback(
    Output('select-ref-taxa', 'children'),
    Output('taxa-reference-info','children',allow_duplicate=True),
    [Input('check-ref-taxa', 'value')],prevent_initial_call='initial_duplicate'
)
def update_output_taxa(value):
    if len(value)!=0:
        return html.Div(style={'display': 'inline-block','vertical-align': 'middle'},children=[
          html.H5("Reference taxa column ",style={"text-indent": '30px','display': 'inline-block'}),
    dcc.Store(id='reference-taxa-status', storage_type=type_storage),
    dcc.Input(id='reference-taxa-input',type='number',min=info_current_file['taxa_start'],max=info_current_file['taxa_end'],step=1,value=info_current_file['taxa_end'],persistence=True,persistence_type=type_storage,style=input_field_number_style),
    #dcc.Input(id='reference-taxa-input', type='text', placeholder='ex: 124',persistence=True,persistence_type=type_storage,style=input_field_style),
    #html.Button('Confirm', id='reference-taxa-button', n_clicks=0,style=button_style),
    html.Div(id='reference-taxa-output',style={'display':'inline-block'}),
    ]), None
    else:
        if info_current_file["taxa_start"]!=None:
            message=html.H5(f"Reference Taxa: {find_reference_taxa(info_current_file)}",style={"text-indent": '30px'})
            return None,message
        else:
            return None, html.H5("Reference Taxa: Error Enter Taxa interval",style={"text-indent": '30px'})

@app.callback(
    Output('select-separate-data', 'children'),
    Output('separate-groups-info','children'),
    [Input('check-separate-data', 'value')]
)
def update_output_data(value):
    global info_current_file
    if len(value)!=0:
        groups_info=[html.Div(id="first-group-info",children=[html.H5("First Group: ",style={"text-indent": '30px'})]),
        html.Div(id="second-group-info",children=[html.H5("Second Group: ",style={"text-indent": '30px'})])]
        pheno_select=html.Div(style={'display': 'inline-block','vertical-align': 'middle'},children=[
          html.H5("Phenotype column ",style={"text-indent": '30px','display': 'inline-block'}),
          dcc.Store(id='phenotype-column-status', storage_type=type_storage),
    dcc.Input(id='phenotype-column-input',type='number',min=info_current_file['covar_start'],max=info_current_file['covar_end'],step=1,value=info_current_file['covar_start'],persistence=True,persistence_type=type_storage,style=input_field_number_style),
    #dcc.Input(id='phenotype-column-input', type='text', placeholder='ex: 124',persistence=True,persistence_type=type_storage,style=input_field_style),
    #html.Button('Confirm', id='phenotype-column-button', n_clicks=0,style=button_style),
    html.Div(id='phenotype-column-output',style={'display': 'inline-block'}),
    ])
        return pheno_select,groups_info
    else:
        info_current_file['phenotype_column']=None
        info_current_file['first_group']=None
        info_current_file['second_group']=None
        return None,None


###### Reference Taxa column #######

# add a click to the appropriate store.
@app.callback(Output('reference-taxa-status', 'data'),
                Input("reference-taxa-input","value"),
                State('reference-taxa-status', 'data'))
def on_click(value, data):
    if value is None:
        raise PreventUpdate

    # Give a default data dict with 0 clicks if there's no data.
    data = data or {'value': ""}

    data["value"]=value

    return data

# output the stored clicks in the table cell.
@app.callback(Output("reference-taxa-output", 'children'),
              Output("taxa-reference-info", 'children',allow_duplicate=True),
                Input('reference-taxa-status', 'modified_timestamp'),
                State('reference-taxa-status', 'data'),prevent_initial_call='initial_duplicate')
def on_data(ts, data):
    if ts is None:
        raise PreventUpdate

    data = data or {}

    print(data)

    if data.get('value',None)==None:
        return None,None
    else:
        if data["value"]=="":
            name_reference_taxa=find_reference_taxa(info_current_file)
        else:
            reference_taxa=data["value"]
            print("Par curioisté: ",find_reference_taxa(info_current_file))
            name_reference_taxa=find_reference_taxa(info_current_file,reference_taxa)
        return html.H5(f"Reference Taxa: {name_reference_taxa}",style={'display': 'inline-block'}),html.H5(f"Reference Taxa: {name_reference_taxa}",style={'text-indent': '30px'})
  


###### Phenotype column ######

# add a click to the appropriate store.
@app.callback(Output('phenotype-column-status', 'data'),
                #Input("phenotype-column-button", 'n_clicks'),
                Input("phenotype-column-input","value"),
                State('phenotype-column-status', 'data'))
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
@app.callback(Output("phenotype-column-output", 'children'),
              Output("first-group-info","children"),
              Output("second-group-info","children"),
                Input('phenotype-column-status', 'modified_timestamp'),
                State('phenotype-column-status', 'data'))
def on_data(ts, data):
    global info_current_file
    if ts is None:
        raise PreventUpdate

    data = data or {}

    if data.get('value',None)==None:
        return [],[],[]
    else:
        if data["value"]=="":
            return "Error : The field cannot be empty."
        else:
            if check_phenotype_column(info_current_file,data["value"]):
                nb_first_group,nb_second_group=get_info_separate_groups(info_current_file,data["value"])
                validation_message=html.I(className="fas fa-check-circle", style={'color': 'green'})
                first_group=html.H5(f"First Group: {nb_first_group}",style={"text-indent": '30px'})
                second_group=html.H5(f"Second Group: {nb_second_group}",style={"text-indent": '30px'})
                info_current_file['phenotype_column']=data["value"]
                info_current_file['first_group']=nb_first_group
                info_current_file['second_group']=nb_second_group

                return validation_message,first_group,second_group
                    #return f"Valid interval : {start} - {end}"
            else:
                first_group=html.H5("First Group: Error Phenotype Column isn't binary",style={"text-indent": '30px'})
                second_group=html.H5("Second Group: Error Phenotype Column isn't binary",style={"text-indent": '30px'})
                error_message=[html.I(className="fas fa-times-circle", style={'color': 'red'}),html.H5("Error : Column is not binary.",style={'display': 'inline-block'})]
                info_current_file['phenotype_column']='error'
                info_current_file['first_group']='error'
                info_current_file['second_group']='error'
                return error_message,first_group,second_group
  

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





@app.callback(Output('select-filter-columns-zero', 'style'),
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
              )
def get_changes_filters(check_dev_mean,check_zeros,ts_dev_mean,ts_zeros,data_dev_mean,data_zeros):

    style_not_displayed={'display': 'none','vertical-align': 'middle'}
    style_displayed={'vertical-align': 'middle'}

    filter_zero=None
    filter_dev_mean=None

    if len(check_zeros)!=0:
        filter_zero=data_zeros["value"]

    if len(check_dev_mean)!=0:
        filter_dev_mean=data_dev_mean["value"]
    
    if filter_zero==None and filter_dev_mean==None:
        return style_not_displayed,style_not_displayed,None,None,None,None,html.H5("No filter applied",style={"text-indent": '15px'}),None,1
    elif filter_zero!=None and filter_dev_mean==None:
        df_filtered=filter_percent_zeros(info_current_file,filter_zero)
        info_current_file["df_taxa"]=df_filtered
        info_current_file["filter_zeros"]=filter_zero
        nb_taxa=info_current_file['taxa_end']-info_current_file['taxa_start']+1
        info_zeros=html.H5(f"Filter with pourcent of zeros: {nb_taxa-df_filtered.shape[1]} taxa deleted",style={"text-indent": '15px'})
        output_zeros=html.H5(f"{nb_taxa-df_filtered.shape[1]} taxa deleted",style={'display': 'inline-block'})
        summary_filter=html.H5(f"Summary: {df_filtered.shape[1]} taxa remaining",style={'display': 'inline-block',"text-indent": '15px'})
        return style_displayed,style_not_displayed,None,output_zeros,info_zeros,None,summary_filter,None,1
    
    elif filter_zero==None and filter_dev_mean!=None:
        info_current_file["df_taxa"]=get_df_file(info_current_file).iloc[:,info_current_file["taxa_start"]-1:info_current_file["taxa_end"]]
        nb_taxa=info_current_file['taxa_end']-info_current_file['taxa_start']+1
        df_filtered=filter_deviation_mean(info_current_file,filter_dev_mean)
        info_current_file["df_taxa"]=df_filtered
        output_dev_mean=html.H5(f"{nb_taxa-df_filtered.shape[1]} taxa deleted",style={'display': 'inline-block'})
        info_dev_mean=html.H5(f"Filter with deviation/mean ratio: {nb_taxa-df_filtered.shape[1]} taxa deleted",style={"text-indent": '15px'})
        summary_filter=html.H5(f"Summary: {df_filtered.shape[1]} taxa remaining",style={'display': 'inline-block',"text-indent": '15px'})

        return style_not_displayed,style_displayed,output_dev_mean,None,None,info_dev_mean,summary_filter,info_current_file["df_taxa"].shape[1],min(filter_dev_mean,df_filtered.shape[1])
    else:
        #Filter on zeros percent and dev mean activated at the same time
        df_filtered=filter_percent_zeros(info_current_file,filter_zero)
        nb_taxa_init=info_current_file['taxa_end']-info_current_file['taxa_start']+1
        info_current_file["df_taxa"]=df_filtered
        nb_taxa_after_zeros=info_current_file["df_taxa"].shape[1]
        info_zeros=html.H5(f"Filter with pourcent of zeros: {nb_taxa_init-nb_taxa_after_zeros} taxa deleted",style={"text-indent": '15px'})
        output_zeros=html.H5(f"{nb_taxa_init-nb_taxa_after_zeros} taxa deleted",style={'display': 'inline-block'})

        df_filtered=filter_deviation_mean(info_current_file,filter_dev_mean)
        info_current_file["df_taxa"]=df_filtered
        output_dev_mean=html.H5(f"{nb_taxa_after_zeros-df_filtered.shape[1]} taxa deleted",style={'display': 'inline-block'})
        info_dev_mean=html.H5(f"Filter with deviation/mean ratio: {nb_taxa_after_zeros-df_filtered.shape[1]} taxa deleted",style={"text-indent": '15px'})
        summary_filter=html.H5(f"Summary: {df_filtered.shape[1]} taxa remaining",style={'display': 'inline-block',"text-indent": '15px'})
        
        return style_displayed,style_displayed,output_dev_mean,output_zeros,info_zeros,info_dev_mean,summary_filter,nb_taxa_after_zeros,min(filter_dev_mean,nb_taxa_after_zeros)

    # if ts_dev_mean is None:
    #         raise PreventUpdate



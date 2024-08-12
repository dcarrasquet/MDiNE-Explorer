import dash

#import dash_bootstrap_components as dbc

# Utilisation de Bulma via un CDN
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css']
#external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',dbc.themes.BOOTSTRAP]

#dash._dash_renderer._set_react_version('18.2.0')
app = dash.Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=external_stylesheets)#,external_stylesheets=external_stylesheets

app.title = "iMDiNE"
#type_storage="memory"
type_storage="session"

direct_simu=False

direct_info_current_file_store={'monitor_thread_launched_pid': False, 
                                'monitor_thread_launched_folder': False, 
                                'process_pid': 64632, 
                                'filename': 'data/dash_app/session_0/table_phylum.tsv', 
                                'session_folder': 'data/dash_app/session_0/', 
                                'nb_rows': 118, 
                                'nb_columns': 170, 
                                'covar_start': 1, 
                                'covar_end': 6, 
                                'taxa_start': 7, 
                                'taxa_end': 170, 
                                'reference_taxa': 'Firmicutes_A', 
                                'phenotype_column': 'Study.Group', 
                                'first_group': 34, 
                                'second_group': 84, 
                                'filter_zeros': 90, 
                                'filter_dev_mean': 70, 
                                'parameters_model': 
                                {'beta_matrix': 
                                 {'apriori': 'Normal', 
                                  'parameters': 
                                  {'alpha': 1, 'beta': 1}}, 
                                  'precision_matrix': 
                                  {'apriori': 'Lasso', 'parameters': 
                                   {'lambda_init': 0.1}}}}




import dash

#import dash_bootstrap_components as dbc

# Utilisation de Bulma via un CDN
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css']
#external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',dbc.themes.BOOTSTRAP]

#dash._dash_renderer._set_react_version('18.2.0')
app = dash.Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=external_stylesheets)#,external_stylesheets=external_stylesheets

app.title = "MDiNE Explorer"
#type_storage="memory"
type_storage="session"

direct_simu=False

direct_info_current_file_store={'monitor_thread_launched_pid': False, 
                                'monitor_thread_launched_folder': False, 
                                'process_pid': 64632, 
                                'filename': 'data/dash_app/session_00/merged_table_final.tsv', 
                                'session_folder': 'data/dash_app/session_00/', 
                                'nb_rows': 44, 
                                'nb_columns': 331, 
                                'covar_start': 1, 
                                'covar_end': 10, 
                                'taxa_start': 11, 
                                'taxa_end': 331, 
                                'reference_taxa': 'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola', 
                                'phenotype_column': 'Study.Group', 
                                'first_group': 34, 
                                'second_group': 21, 
                                'filter_zeros': 23, 
                                'filter_dev_mean': 40, 
                                'status-run-model': 'not-yet', 
                                'parameters_model': {
                                    'beta_matrix': {
                                        'apriori': 'Horseshoe',
                                        'parameters': {} 
                                    },
                                    'precision_matrix': 
                                    {'apriori': 'Lasso',
                                    'parameters': {'lambda_init': 20}
                                    },
                                    'nb_draws':1000,
                                    'nb_tune':2000,
                                    'target_accept':0.9}}




import dash
from dash import CeleryManager

#import dash_bootstrap_components as dbc

# Utilisation de Bulma via un CDN
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css']
#external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.3/css/bulma.min.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',dbc.themes.BOOTSTRAP]


app = dash.Dash(__name__,external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

app.title = "MDiNE web app"
#type_storage="memory"
type_storage="session"

# info_current_file={
#     'filename':"data/dash_app/session_5/crohns-numeric-tsv.tsv",
#     'session_folder':None,
#     'nb_rows':None,
#     'nb_columns':None,
#     'covar_start':None,
#     'covar_end':None,
#     'taxa_start':None,
#     'taxa_end':None,
#     'reference_taxa':None,
#     'phenotype_column':None,
#     'first_group':None,
#     'second_group':None,
#     'filter_zeros':None,
#     'filter_dev_mean':None,
#     'df_taxa':None,
#     'df_covariates':None,
#     'parameters_model':{
#         'beta_matrix':{
#             'apriori':'Ridge',
#             'parameters':{
#                 'alpha':1,
#                 'beta':1
#             }
#         },
#         'precision_matrix':{
#             'apriori':'Lasso',
#             'parameters':{
#                 'lambda_init':10
#             }
#         }
#     }
# }

info_current_file={
    'filename':None,
    'session_folder':None,
    'nb_rows':None,
    'nb_columns':None,
    'covar_start':None,
    'covar_end':None,
    'taxa_start':None,
    'taxa_end':None,
    'reference_taxa':None,
    'phenotype_column':None,
    'first_group':None,
    'second_group':None,
    'filter_zeros':None,
    'filter_dev_mean':None,
    'df_taxa':None,
    'df_covariates':None,
    'status-run-model':'not-yet',
    'parameters_model':{
        'beta_matrix':{
            'apriori':'Ridge',
            'parameters':{
                'alpha':1,
                'beta':1
            }
        },
        'precision_matrix':{
            'apriori':'Lasso',
            'parameters':{
                'lambda_init':10
            }
        }
    }
}

elements_graph=None

# info_current_file={
#     'filename':"data/test_app/crohns-numeric-tsv.tsv",
#     'nb_rows':100,
#     'nb_columns':11,
#     'covar_start':2,
#     'covar_end':5,
#     'taxa_start':6,
#     'taxa_end':11,
#     'reference_taxa':None,
#     'separate_data':None,
#     'first_group':None,
#     'second_group':None,
#     'filter_zeros':None,
#     'filter_dev_mean':None,
#     'list_remaining_columns':None,
#     'parameters_model':{
#         'beta_matrix':{
#             'apriori':'Ridge',
#             'parameters':{
#                 'alpha':1,
#                 'beta':1
#             }
#         },
#         'precision_matrix':{
#             'apriori':'Lasso',
#             'parameters':{
#                 'lambda_init':10
#             }
#         }
#     }
# }




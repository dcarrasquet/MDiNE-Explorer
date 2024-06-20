import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from MDiNE_model import run_model
from extract_data_files import get_separate_data
from maindash import info_current_file

#from scripts.mdine.MDiNE_model import run_model
# from scripts.mdine.extract_data_files import get_separate_data
# from scripts.maindash import info_current_file


def run_model_terminal():
    global info_current_file


    choice_run_model=sys.argv[1] ## 2 choices: one_group, two_groups

    print("Choice run model: ",choice_run_model)

    print("Info current file: ",info_current_file)

    if choice_run_model=="one_group":
        print("Choice one grooooop")
        run_model(info_current_file["df_covariates"],info_current_file["df_taxa"],info_current_file["parameters_model"],info_current_file["session_folder"])
        
        info_current_file["status-run-model"]="completed"

    elif choice_run_model=="two_groups":
        [df_covariates_1,df_taxa_1],[df_covariates_2,df_taxa_2]=get_separate_data(info_current_file)
        path_first_group=os.path.join(info_current_file["session_folder"],"first_group/")
        path_second_group=os.path.join(info_current_file["session_folder"],"second_group/")
        print("Je vais lancer le premier modele")
        run_model(df_covariates_1,df_taxa_1,info_current_file["parameters_model"],path_first_group)
        print("Je vais lancer le deuxième modele")
        run_model(df_covariates_2,df_taxa_2,info_current_file["parameters_model"],path_second_group)
    else:
        print("Argument not valid")

    

    # try:

    #     choice_run_model=sys.argv[1] ## 2 choices: one_group, two_groups

    #     print("Choice run model: ",choice_run_model)

    #     if choice_run_model=="one_group":
    #         print("Choice one grooooop")
    #         run_model(info_current_file["df_covariates"],info_current_file["df_taxa"],info_current_file["parameters_model"],info_current_file["session_folder"])

    #         info_current_file["status-run-model"]="completed"

    #     elif choice_run_model=="two_groups":
    #         [df_covariates_1,df_taxa_1],[df_covariates_2,df_taxa_2]=get_separate_data(info_current_file)
    #         path_first_group=os.path.join(info_current_file["session_folder"],"first_group/")
    #         path_second_group=os.path.join(info_current_file["session_folder"],"second_group/")
    #         print("Je vais lancer le premier modele")
    #         run_model(df_covariates_1,df_taxa_1,info_current_file["parameters_model"],path_first_group)
    #         print("Je vais lancer le deuxième modele")
    #         run_model(df_covariates_2,df_taxa_2,info_current_file["parameters_model"],path_second_group)
    #     else:
    #         print("Argument not valid")

    # except:
    #     info_current_file["status-run-model"]="error"


run_model_terminal()
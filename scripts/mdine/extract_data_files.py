import csv
import numpy as np
import pandas as pd


def get_species_list(filename):

    with open(filename, newline='') as csvfile:
        lecteur_csv = csv.reader(csvfile, delimiter=',')
        premiere_ligne = next(lecteur_csv)[5:-1]  # Extraire la première ligne
        
    for i in range(len(premiere_ligne)):
        premiere_ligne[i]=premiere_ligne[i][14:]
    
    return premiere_ligne

def get_data(file_name):

    # Initialiser des listes pour stocker les données des colonnes 2 à 5 et 6 à 11
    donnees_colonnes_2_5 = []
    donnees_colonnes_6_11 = []
    Z_vector=[]

    # Ouvrir le fichier CSV en mode lecture
    with open(file_name, 'r') as fichier_csv:
        # Créer un objet lecteur CSV
        lecteur_csv = csv.reader(fichier_csv)

        next(lecteur_csv) # Passe la première ligne
        
        # Lire chaque ligne du fichier CSV
        for ligne in lecteur_csv:
            # Binarisation des données 
            ligne_transformee = [valeur.replace("CD", "1").replace("no", "0").replace("female", "1").replace("male", "0").replace("false", "0").replace("true", "1") for valeur in ligne]
            
            # Convertir les éléments de la ligne en float (sauf la première colonne si elle contient des identifiants)
            ligne_float = [float(valeur) for valeur in ligne_transformee]

            #print(ligne_float)
            
            # Stocker les données des colonnes 2 à 5 dans une liste
            donnees_colonnes_2_5.append(ligne_float[2:5])
            Z_vector.append(ligne_float[1])
            
            # Stocker les données des colonnes 6 à 11 dans une liste
            donnees_colonnes_6_11.append(ligne_float[5:12])

        #print(donnees_colonnes_2_5)
        #print(donnees_colonnes_6_11)
        
    # Convertir les listes en matrices numpy
    covariate_matrix = np.array(donnees_colonnes_2_5)
    counts_matrix = np.array(donnees_colonnes_6_11)
    Z_vector=np.array(Z_vector)

    #print("Matrice de covariates:\n",covariate_matrix,"\n\n")
    #print("Matrice de Z:\n",Z_vector,"\n\n")
    #print("Matrice Data Y:\n", counts_matrix)

    return(covariate_matrix,counts_matrix,Z_vector)

def get_separate_data(info_current_file):
    if info_current_file["phenotype_column"]==None:
        return None
    else:
        df_init=get_df_file(info_current_file)
        df_taxa=info_current_file["df_taxa"]
        df_covariates=info_current_file["df_covariates"]

        df_taxa_1=df_taxa[df_init.iloc[:,info_current_file["phenotype_column"]-1]==0]
        df_taxa_2=df_taxa[df_init.iloc[:,info_current_file["phenotype_column"]-1]==1]
        df_covariates_1=df_covariates[df_init.iloc[:,info_current_file["phenotype_column"]-1]==0]
        df_covariates_2=df_covariates[df_init.iloc[:,info_current_file["phenotype_column"]-1]==1]
        return [df_covariates_1,df_taxa_1],[df_covariates_2,df_taxa_2]





def separate_data_in_two_groups(covariate_matrix_data,counts_matrix_data,Z_vector):

    first_group=[covariate_matrix_data[Z_vector==0],counts_matrix_data[Z_vector==0]]
    second_group=[covariate_matrix_data[Z_vector==1],counts_matrix_data[Z_vector==1]]
    return (first_group,second_group)

def get_infos_file(filename):
    file_extention=filename.split(".")[-1]
    if file_extention=="csv":
        df = pd.read_table(filename,sep=",")
    elif file_extention=="tsv":
        df = pd.read_table(filename,sep='\t')
    else:
        print("Error")
    num_rows, num_columns = df.shape
    return num_rows,num_columns 

def get_df_file(info_current_file):
    filename=info_current_file["filename"]
    file_extention=filename.split(".")[-1]
    if file_extention=="csv":
        df = pd.read_table(filename,sep=",")
    elif file_extention=="tsv":
        df = pd.read_table(filename,sep='\t')
    else:
        print("Error")
    return df

def get_info_separate_groups(info_current_file,phenotype_column):

    df=get_df_file(info_current_file)
    column_name = df.columns[phenotype_column-1]
    zero_count = (df[column_name] == 0).sum()
    one_count = (df[column_name] == 1).sum()

    # df_zeros = df[df[column_name] == 0].copy()
    # df_ones = df[df[column_name] == 1].copy()

    return zero_count,one_count

def filter_deviation_mean(info_current_file,nb_columns):

    # if info_current_file['filter_zeros']==None:
    #     taxa_start=info_current_file["taxa_start"]
    #     taxa_end=info_current_file["taxa_end"]

    #     df=get_df_file(info_current_file)
    #     df_taxa=df.iloc[:,taxa_start:taxa_end+1]
    # else:
    #     df_taxa=filter_percent_zeros(info_current_file,info_current_file['filter_zeros'])

    df_taxa=info_current_file["df_taxa"]
    
    mean_values = df_taxa.mean()
    deviation_values = df_taxa.var()

    ratios=deviation_values/mean_values
    sorted_taxa = ratios.sort_values(ascending=False)
    selected_columns = sorted_taxa.head(nb_columns).index

    df_filtered = df_taxa[selected_columns]

    return df_filtered

    #print(df_filtered.shape)

    #print(df_filtered.head())

def find_reference_taxa(info_current_file,reference_column=None):

    taxa_start=info_current_file["taxa_start"]
    taxa_end=info_current_file["taxa_end"]

    df=get_df_file(info_current_file)
    df_taxa=df.iloc[:,taxa_start:taxa_end+1]

    if reference_column==None:

        mean_values = df_taxa.mean()
        deviation_values = df_taxa.var()

        ratios=deviation_values/mean_values
        sorted_taxa = ratios.sort_values(ascending=True)
        reference_column = sorted_taxa.head(1).index

        #print(reference_column[0])

        
        return reference_column[0]
    else:
        return df.columns[reference_column-1]

    print(df_filtered.columns)

    #print(df_filtered.shape)

    #print(df_filtered.head())

def check_phenotype_column(info_current_file,column_index):
    df=get_df_file(info_current_file)
    column = df.iloc[:, column_index-1]
    unique_values = column.unique()
    
    # Check whether the unique values are only 0 and 1
    if set(unique_values) == {0, 1}:
        return True
    else:
        return False
    
def filter_percent_zeros(info_current_file,percent_filtre):
    threshold = percent_filtre / 100.0  # Convertir le pourcentage en proportion
    columns_to_keep = []

    df=get_df_file(info_current_file)

    taxa_start=info_current_file["taxa_start"]-1
    taxa_end=info_current_file["taxa_end"]-1

    df=get_df_file(info_current_file)
    df_taxa=df.iloc[:,taxa_start:taxa_end+1]

    #print(df_taxa)

    for column in df_taxa.columns:
        zero_count = (df[column] == 0).sum()
        zero_percentage = zero_count / len(df)
        if zero_percentage <= threshold:
            columns_to_keep.append(column)

    df_filtered=df[columns_to_keep]

    return df_filtered



if __name__=="__main__":
    print(1+1)
    #simulation_data_R("data/data_R/simulated_data_NB_N100_J50_K20.json")
    #filter_deviation_mean("data/crohns.csv",10)
    # (covariate_matrix_data,counts_matrix_data,Z_vector)=get_data("data/crohns.csv")
    # first_group,second_group=separate_data_in_two_groups(covariate_matrix_data,counts_matrix_data,Z_vector)
    # print(first_group[0].shape,first_group[1].shape)
    # print(second_group[0].shape,second_group[1].shape)
    

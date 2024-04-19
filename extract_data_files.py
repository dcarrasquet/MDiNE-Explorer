import csv
import numpy as np

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
            
            # Stocker les données des colonnes 2 à 5 dans une liste
            donnees_colonnes_2_5.append(ligne_float[1:5])
            Z_vector.append(ligne_float[1])
            
            # Stocker les données des colonnes 6 à 11 dans une liste
            donnees_colonnes_6_11.append(ligne_float[5:12])
        
    # Convertir les listes en matrices numpy
    covariate_matrix = np.array(donnees_colonnes_2_5)
    counts_matrix = np.array(donnees_colonnes_6_11)
    Z_vector=np.array(Z_vector)

    #print("Matrice de covariates:\n",covariate_matrix,"\n\n")
    #print("Matrice de Z:\n",Z_vector,"\n\n")
    #print("Matrice Data Y:\n", counts_matrix)

    return(covariate_matrix,counts_matrix,Z_vector)

def separate_data_in_two_groups(covariate_matrix_data,counts_matrix_data,Z_vector):

    first_group=[covariate_matrix_data[Z_vector==0],counts_matrix_data[Z_vector==0]]
    second_group=[covariate_matrix_data[Z_vector==1],counts_matrix_data[Z_vector==1]]
    return (first_group,second_group)

if __name__=="__main__":
    covariate_matrix,counts_matrix,Z_vector=get_data("data/crohns.csv")
    first_group,second_group=separate_data_in_two_groups(covariate_matrix,counts_matrix,Z_vector)
    print(first_group[0].shape,first_group[1].shape)
    print(second_group[0].shape,second_group[1].shape)
    

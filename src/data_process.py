
import pandas as pd
import numpy as np
import os
def prepare_data(input_path="data/train (1).csv", output_dir="data/processed"):


    # Lire les données
    df = pd.read_csv(input_path)

    # Sélectionner uniquement les colonnes numériques
    df_num = df.select_dtypes(include=[np.number])

    # Supprimer les lignes contenant des valeurs manquantes
    df_num = df_num.dropna()

    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sauvegarder les données nettoyées
    output_path = os.path.join(output_dir, "cleaned_data.csv")
    df_num.to_csv(output_path, index=False)

    print(f"Données nettoyées sauvegardées dans : {output_path}")


if __name__ == "__main__":
prepare_data()
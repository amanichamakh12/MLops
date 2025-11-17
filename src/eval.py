    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics import mean_squared_error, r2_score
def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):

    # Charger les données nettoyées
    data_path = f"{data_dir}/cleaned_data.csv"
    df_num = pd.read_csv(data_path)

    X = df_num.drop(columns=["SalePrice"])
    y = df_num["SalePrice"]

    # Charger le modèle entraîné
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Faire des prédictions
    y_pred = model.predict(X)

    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print("\n===== Évaluation du modèle =====")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")

if __name__ == "__main__":
evaluate_model()

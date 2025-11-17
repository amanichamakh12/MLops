import pandas as pd
import numpy as np
 def train_model(data_dir="data/processed", output_dir="models"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModèle entraîné avec succès.")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calcul des métriques
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print("\n===== Scores du modèle =====")
    print(f"RMSE (train) : {rmse_train:.2f}")
    print(f"RMSE (test)  : {rmse_test:.2f}")
    print(f"R² (train)   : {r2_train:.3f}")
    print(f"R² (test)    : {r2_test:.3f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nModèle sauvegardé sous 'model.pkl'")

if __name__ == "__main__":
train_model()

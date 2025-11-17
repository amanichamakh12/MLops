import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/user/Downloads/MLops/MLops/data/train (1).csv")

print("Aperçu des données :")
display(df.head())

print("\nInformations sur le dataset :")
print(df.info())
df_num = df.select_dtypes(include=[np.number])

# Supprimer les lignes contenant des valeurs manquantes
df_num = df_num.dropna()

print("\nDimensions après nettoyage :", df_num.shape)

X = df_num.drop(columns=["SalePrice"])
y = df_num["SalePrice"]

# Stratification par binning (pour équilibrer les classes de prix)
y_binned = pd.cut(y, bins=10, labels=False)
sns.histplot(y, kde=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y_binned
)



print("\nTaille du train :", X_train.shape)
print("Taille du test :", X_test.shape)
sns.histplot(X_train, kde=True)
sns.histplot(y_train, kde=True)

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
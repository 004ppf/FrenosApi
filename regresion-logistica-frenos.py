import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Leer como antes
df_raw = pd.read_csv("falla_frenos.csv", header=None)
df_split = df_raw[0].str.split(",", expand=True)

# La primera fila son los nombres de las columnas
df_split.columns = df_split.iloc[0]  

# Quitar esa primera fila de los datos
df_split = df_split.drop(0).reset_index(drop=True)

# Convertir a numérico donde aplique
df_split = df_split.apply(pd.to_numeric, errors="ignore")

print(df_split.head())

#"kms_recorridos,años_uso,ultima_revision,temperatura_frenos,cambios_pastillas,estilo_conduccion,carga_promedio,luz_alarma_freno,falla_frenos"

# rellenar valores nulos
df_split = df_split.fillna(df_split.median())
df_split = df_split.fillna(method="ffill").fillna(method="bfill")
df_split = df_split.fillna(0)
print(df_split.isnull().sum())

# Dividir en características y etiqueta
X = df_split.drop("falla_frenos", axis=1)
y = df_split["falla_frenos"]
print(X.head())
print(y.head())

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Crear y entrenar el modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#guardar modelo
import joblib
joblib.dump(model, "modelo_frenos.pkl")
# features.pkl
joblib.dump(X.columns.tolist(), "features_frenos.pkl")

print("Modelo y características guardadas.")

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar modelo y features con rutas absolutas
model = joblib.load(os.path.join(BASE_DIR, "modelo_frenos.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features_frenos.pkl"))

# Crear instancia de la API
app = FastAPI(title="API Predicción de Ventas")
# Definir el esquema de entrada
class DatosEntrada(BaseModel):
    kms_recorridos: float
    años_uso: float
    ultima_revision: float
    temperatura_frenos: float
    cambios_pastillas: float
    estilo_conduccion: str  # "suave", "normal", "agresivo"
    carga_promedio: float
    luz_alarma_freno: int  # 0 o 1


# ruta GET para verificar que la API está funcionando
@app.get("/")
def read_root():
    return {"message": "API de predicción de fallas en frenos está funcionando"}


# Ruta POST para predicciones
@app.post("/predecir")
def predecir(data: DatosEntrada):
    # Convertir entrada en DataFrame
    df = pd.DataFrame([{
        "kms_recorridos": data.kms_recorridos,
        "años_uso": data.años_uso,
        "ultima_revision": data.ultima_revision,
        "temperatura_frenos": data.temperatura_frenos,
        "cambios_pastillas": data.cambios_pastillas,
        "estilo_conduccion": data.estilo_conduccion,
        "carga_promedio": data.carga_promedio,
        "luz_alarma_freno": data.luz_alarma_freno
    }])

    # Codificación de estilo_conduccion (dummy variables)
    df = pd.get_dummies(df, columns=["estilo_conduccion"], drop_first=True)

    # Asegurar que tenga las mismas columnas que el modelo
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    # Hacer predicción
    prediccion = model.predict(df)[0]

    if prediccion == 1:
        resultado = {
            "prediccion_falla_frenos": 1,
            "mensaje": "Se predice una posible falla en los frenos"
        }
    else:
        resultado = {
            "prediccion_falla_frenos": 0,
            "mensaje": "No se predice falla en los frenos"
        }

    return resultado


# ejemplosuso datos de entrada:
# {
#   "kms_recorridos": 15000,
#   "años_uso": 3,
#   "ultima_revision": 6,
#   "temperatura_frenos": 75,
#   "cambios_pastillas": 1,
#   "estilo_conduccion": "normal",
#   "carga_promedio": 200,
#   "luz_alarma_freno": 0
# }

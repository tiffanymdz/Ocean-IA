# Librerías

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import io


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Rutas de los modelos

RNN_MODEL_PATH = MODELS_DIR / "modelo_final.keras"
SPECIES_MODEL_PATH = MODELS_DIR / "modelo_CNN_peces.h5"
SCALER_X_PATH = MODELS_DIR / "scaler_x.pkl"
SCALER_Y_PATH = MODELS_DIR / "scaler_y.pkl"

# Configuración oceáno

HORAS_ENTRADA = 720
HORAS_SALIDA = 72


COLUMNAS_ENTRADA = [
    "wave_height",
    "wave_period",
    "wave_direction",
    "sst",
    "wind_speed",
    "pressure",
    "fase_lunar"]


# Clases modelos peces

CLASES_PECES = [
    "Atun_Aleta_amarilla_n",
    "Black Sea Sprat",
    "Gilt Head Bream",
    "Horse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout",
    "corvina_reina",
    "dorado",
    "pargo_mancha_n",
    "pez_vela_n",
    "tiburon_martillo_n",
    "tortuga_marina_n"
]


# Especies protegidas

ESPECIES_PROTEGIDAS = {
    "pez_vela_n",
    "tiburon_martillo_n",
    "tortuga_marina_n"
}


# API

app = FastAPI(
    title="OceanoIA API",
    description="API para predicción de peces y variables oceánicas"
)


# Permitir conexión con Streamlit

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


modelo_rnn = None
modelo_peces = None
scaler_x = None
scaler_y = None


# Carga de modelos

@app.on_event("startup")
def cargar_modelos():

    global modelo_rnn, modelo_peces, scaler_x, scaler_y


    modelo_rnn = load_model(RNN_MODEL_PATH)

    modelo_peces = load_model(SPECIES_MODEL_PATH)

    scaler_x = joblib.load(SCALER_X_PATH)

    scaler_y = joblib.load(SCALER_Y_PATH)


    print("Modelos cargados correctamente")


# Home

@app.get("/")
def home():

    return {
        "mensaje": "OceanoIA API funcionando correctamente"
    }



# Predicción del oceáno

@app.post("/predict/oceano")
async def predecir_oceano(file: UploadFile = File(...)):


    if modelo_rnn is None:

        raise HTTPException(
            status_code=503,
            detail="Modelo oceánico no cargado"
        )


    contenido = await file.read()


    try:

        df = pd.read_csv(
            io.BytesIO(contenido)
        )


    except:

        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser un CSV válido"
        )


    faltantes = [
        columna
        for columna in COLUMNAS_ENTRADA
        if columna not in df.columns
    ]


    if faltantes:

        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas: {faltantes}"
        )


    if len(df) < HORAS_ENTRADA:

        raise HTTPException(
            status_code=400,
            detail="El CSV necesita mínimo 720 horas de datos"
        )


    datos = df[COLUMNAS_ENTRADA].tail(HORAS_ENTRADA).values


    datos = scaler_x.transform(datos)


    datos = datos.reshape(
        1,
        HORAS_ENTRADA,
        len(COLUMNAS_ENTRADA)
    )


    prediccion = modelo_rnn.predict(datos)


    prediccion_real = scaler_y.inverse_transform(prediccion)[0]


    return {

        "horas_predichas": HORAS_SALIDA,

        "unidad": "metros",

        "prediccion_oleaje": [
            round(float(valor), 3)
            for valor in prediccion_real
        ]

    }


# Predicción de peces

@app.post("/predict/especie")
async def predecir_especie(file: UploadFile = File(...)):


    if modelo_peces is None:

        raise HTTPException(
            status_code=503,
            detail="Modelo de peces no cargado"
        )


    contenido = await file.read()


    try:

        imagen = Image.open(
            io.BytesIO(contenido)
        )


    except:

        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen válida"
        )


    imagen = imagen.convert("RGB")


    imagen = imagen.resize(
        (150,150)
    )


    imagen_array = np.array(imagen) / 255.0


    imagen_array = np.expand_dims(
        imagen_array,
        axis=0
    )


    prediccion = modelo_peces.predict(
        imagen_array
    )


    probabilidades = prediccion[0]


    top_indices = probabilidades.argsort()[-3:][::-1]


    top_3 = []


    for i in top_indices:

        especie = CLASES_PECES[i]


        top_3.append(
            {
                "especie": especie,
                "confianza": round(float(probabilidades[i]), 3),
                "protegida_o_en_veda": especie in ESPECIES_PROTEGIDAS
            }
        )


    return {
        "prediccion_principal": top_3[0],
        "alternativas": top_3
    }
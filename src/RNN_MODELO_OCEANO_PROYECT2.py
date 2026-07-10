#carga_de_Librerias

#librerias para manejar datos y numeros

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import StringIO

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

import copernicusmarine
import xarray as xr
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf

#CARGAR LOS DATOS

archivo = "open-meteo-9.71N84.87W0m.csv"

with open(archivo, "r", encoding="utf-8") as f:
    lineas = f.readlines()

bloques = []
bloque_actual = []
for linea in lineas:
    if linea.strip() == "":
        if bloque_actual:
            bloques.append(bloque_actual)
            bloque_actual = []
    else:
        bloque_actual.append(linea)
if bloque_actual:
    bloques.append(bloque_actual)


bloque_oleaje = max(bloques, key=len)
df = pd.read_csv(StringIO("".join(bloque_oleaje)))


df.columns = [c.split(" (")[0] for c in df.columns]
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")
df = df.sort_index()

print("Datos cargados:", df.shape)
print(df.head())


fecha_inicio = df.index.min().strftime("%Y-%m-%dT%H:%M:%S")
fecha_fin = df.index.max().strftime("%Y-%m-%dT%H:%M:%S")

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao"],
    minimum_longitude=-85.1,
    maximum_longitude=-84.6,
    minimum_latitude=9.5,
    maximum_latitude=10,
    start_datetime=fecha_inicio,
    end_datetime=fecha_fin,
    minimum_depth=0.49402499198913574,
    maximum_depth=0.49402499198913574,
    output_filename="sst_copernicus.nc",
    output_directory=".",
)



ds = xr.open_dataset("sst_copernicus.nc")
sst = ds["thetao"].mean(dim=["latitude", "longitude"])
if "depth" in sst.dims:
    sst = sst.isel(depth=0)

df_sst = sst.to_dataframe(name="sst").reset_index()[["time", "sst"]]
df_sst["time"] = pd.to_datetime(df_sst["time"])
df_sst = df_sst.set_index("time").sort_index()
df_sst = df_sst.resample("1H").interpolate("linear")


df = df.join(df_sst, how="left")

#limpiar nombres de columnas (quitar las unidades entre parentesis)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 9.708336,
    "longitude": -84.87499,
    "start_date": df.index.min().strftime("%Y-%m-%d"),
    "end_date": df.index.max().strftime("%Y-%m-%d"),
    "hourly": "wind_speed_10m,surface_pressure",
    "timezone": "America/Costa_Rica",
}

respuesta = requests.get(url, params=params)
datos_clima = respuesta.json()

df_clima = pd.DataFrame({
    "time": pd.to_datetime(datos_clima["hourly"]["time"]),
    "wind_speed": datos_clima["hourly"]["wind_speed_10m"],
    "pressure": datos_clima["hourly"]["surface_pressure"],
}).set_index("time")

df = df.join(df_clima, how="left")



#ber hecho una vez: copernicusmarine login

#VARIABLE DE FASE LUNAR

#aproximar la variable "marea" del proyecto. Ni Open-Meteo ni Copernicus dan un historico horario real de marea, asi que se calcula la fase lunar de forma matematica como proxy (esta relacionada con el ciclo de mareas).

def calcular_fase_lunar(fechas):
    luna_nueva = datetime(2000, 1, 6, 18, 14)
    ciclo = 29.53058867
    dias = (fechas - luna_nueva).total_seconds() / 86400
    fase = (dias % ciclo) / ciclo
    return (1 - np.cos(2 * np.pi * fase)) / 2

df["fase_lunar"] = calcular_fase_lunar(df.index)


#rellenar cualquier hueco

df = df.interpolate().ffill().bfill()

#ARMAR LAS VENTANAS (batches de 30 dias)
HORAS_ENTRADA = 30 * 24
HORAS_SALIDA = 72

columnas_entrada = list(df.columns)
columna_objetivo = "wave_height"

X = []
y = []

valores = df[columnas_entrada].values
objetivo = df[columna_objetivo].values

for i in range(0, len(df) - HORAS_ENTRADA - HORAS_SALIDA, 6):
    ventana_x = valores[i : i + HORAS_ENTRADA]
    ventana_y = objetivo[i + HORAS_ENTRADA : i + HORAS_ENTRADA + HORAS_SALIDA]
    X.append(ventana_x)
    y.append(ventana_y)

X = np.array(X)
y = np.array(y)

print("Shape de X:", X.shape)
print("Shape de y:", y.shape)



#TRAIN / TEST
horas_test = 30 * 24
n_test = horas_test // 6

X_train, X_test = X[:-n_test], X[-n_test:]
y_train, y_test = y[:-n_test], y[-n_test:]




#ESCALAR
n_features = X_train.shape[2]

scaler_x = MinMaxScaler()
scaler_x.fit(X_train.reshape(-1, n_features))

def escalar_X(datos):
    original_shape = datos.shape
    datos_2d = datos.reshape(-1, n_features)
    datos_escalados = scaler_x.transform(datos_2d)
    return datos_escalados.reshape(original_shape)

X_train_esc = escalar_X(X_train)
X_test_esc = escalar_X(X_test)

scaler_y = MinMaxScaler()
scaler_y.fit(y_train)
y_train_esc = scaler_y.transform(y_train)
y_test_esc = scaler_y.transform(y_test)

#MODELO LSTM

np.random.seed(42)
tf.random.set_seed(42)

#MODELO
modelo = Sequential()
modelo.add(Input(shape=(HORAS_ENTRADA, n_features)))
modelo.add(LSTM(100, return_sequences=True))
modelo.add(LSTM(50))
modelo.add(Dropout(0.2))
modelo.add(Dense(HORAS_SALIDA))

modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
modelo.summary()

#ENTRENAR
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

historia = modelo.fit(
    X_train_esc, y_train_esc,
    validation_split=0.1,
    epochs=80,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)


#Metrica CON RMSE Y MAE
pred_esc = modelo.predict(X_test_esc)
pred = scaler_y.inverse_transform(pred_esc)
real = scaler_y.inverse_transform(y_test_esc)

rmse = np.sqrt(mean_squared_error(real.flatten(), pred.flatten()))
mae = mean_absolute_error(real.flatten(), pred.flatten())

print("RMSE:", rmse)
print("MAE:", mae)


#Grafico
plt.figure(figsize=(10, 4))
plt.plot(real[0], label="Real")
plt.plot(pred[0], label="Predicho")
plt.title("Prediccion vs real - primeras 72h de la ventana de test")
plt.legend()
plt.savefig("prediccion_test.png")
plt.show()



#REENTRENAR
scaler_x_final = MinMaxScaler()
scaler_x_final.fit(X.reshape(-1, n_features))
X_esc_final = scaler_x_final.transform(X.reshape(-1, n_features)).reshape(X.shape)

scaler_y_final = MinMaxScaler()
scaler_y_final.fit(y)
y_esc_final = scaler_y_final.transform(y)

modelo_final = Sequential()
modelo_final.add(Input(shape=(HORAS_ENTRADA, n_features)))
modelo_final.add(LSTM(100, return_sequences=True))
modelo_final.add(LSTM(50))
modelo_final.add(Dropout(0.2))
modelo_final.add(Dense(HORAS_SALIDA))
modelo_final.compile(optimizer="adam", loss="mse", metrics=["mae"])
modelo_final.fit(X_esc_final, y_esc_final, epochs=15, batch_size=32, verbose=1)



#prediccion

ultima_ventana = df[columnas_entrada].values[-HORAS_ENTRADA:]
ultima_ventana_esc = scaler_x_final.transform(ultima_ventana)
ultima_ventana_esc = ultima_ventana_esc.reshape(1, HORAS_ENTRADA, n_features)

prediccion_futura_esc = modelo_final.predict(ultima_ventana_esc)
prediccion_futura = scaler_y_final.inverse_transform(prediccion_futura_esc)[0]

fechas_futuras = pd.date_range(
    df.index[-1] + timedelta(hours=1), periods=HORAS_SALIDA, freq="H"
)

df_prediccion = pd.DataFrame({"wave_height_predicho": prediccion_futura}, index=fechas_futuras)
df_prediccion.to_csv("prediccion_futura_72h.csv")
print(df_prediccion.head())
df_prediccion = pd.read_csv("prediccion_futura_72h.csv", index_col=0, parse_dates=True)

plt.figure(figsize=(10, 4))
plt.plot(df_prediccion.index, df_prediccion["wave_height_predicho"])
plt.title("Prediccion de oleaje - proximas 72 horas")
plt.xlabel("Fecha")
plt.ylabel("Altura de oleaje (m)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


#Guardar
modelo_final.save("modelo_final.keras")
print("Listo. RMSE =", rmse, " MAE =", mae)






















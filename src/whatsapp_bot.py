# ============================================================
# bot/whatsapp_bot.py
# ============================================================
# Sirve para: el punto extra del proyecto, version WhatsApp.
# Usa Twilio (Sandbox de WhatsApp, gratis para pruebas) para
# recibir una FOTO de un pez y responder con la especie detectada
# y si es legal pescarla en Costa Rica.
#
# DIFERENCIA CLAVE con Telegram: WhatsApp funciona por "webhook",
# no por polling. Eso significa que Twilio necesita poder mandarle
# una peticion HTTP a TU computadora desde internet. Para pruebas
# locales se usa "ngrok".
#
# COMO CORRERLO:
#   pip install flask twilio pillow numpy joblib tensorflow requests
#   python whatsapp_bot.py
#
# Luego, en OTRA terminal:
#   ngrok http 5000
#
# Esa URL + "/whatsapp" se pega en Twilio Console, en:
#   Messaging > Try it out > Sandbox settings > "When a message comes in"
# ============================================================

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import requests
import numpy as np
from PIL import Image
import io

# ------------------------------------------------------------
# PARCHE: esta version de Keras guarda un parametro nuevo
# ('quantization_config') en las capas Dense que su propio
# cargador no reconoce (bug de esta version). Este parche
# intercepta la creacion de cada Dense y descarta ese parametro
# extra antes de que truene.
# ------------------------------------------------------------
from keras.layers import Dense
from tensorflow.keras.models import load_model

_dense_init_original = Dense.__init__

def _dense_init_parcheado(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _dense_init_original(self, *args, **kwargs)

Dense.__init__ = _dense_init_parcheado

# ------------------------------------------------------------
# CONFIGURACION
# ------------------------------------------------------------
# estos dos los sacas de la consola de Twilio (Account Info)
TWILIO_ACCOUNT_SID = "AC3c6c7bf0fbbab930c62e014c14c286a1"
TWILIO_AUTH_TOKEN = "7be3ef0f5fcc5989c59e4e011f4f0eb9"

CLASES_CNN = [
    "Atun_Aleta_amarilla_n",  # 0
    "Black Sea Sprat",        # 1  (especie generica del dataset base, no de CR)
    "Gilt Head Bream",        # 2  (especie generica del dataset base, no de CR)
    "Horse Mackerel",         # 3  (especie generica del dataset base, no de CR)
    "Red Mullet",             # 4  (especie generica del dataset base, no de CR)
    "Red Sea Bream",          # 5  (especie generica del dataset base, no de CR)
    "Sea Bass",               # 6  (especie generica del dataset base, no de CR)
    "Shrimp",                 # 7  (especie generica del dataset base, no de CR)
    "Striped Red Mullet",     # 8  (especie generica del dataset base, no de CR)
    "Trout",                  # 9  (especie generica del dataset base, no de CR)
    "corvina_reina",          # 10
    "dorado",                 # 11
    "pargo_mancha_n",         # 12
    "pez_vela_n",             # 13
    "tiburon_martillo_n",     # 14
    "tortuga_marina_n",       # 15
]

# Sirve para: decidir que le contesta el bot sobre la legalidad,
# segun la especie que detecte. Solo las 7 especies del proyecto
# tienen info de legalidad para Costa Rica; las otras 9 son del
# dataset base (Kaggle, especies genericas/mediterraneas) y se
# marcan como "no evaluado" porque no aplica la legislacion de CR.
LEGALIDAD = {
    "dorado":                {"estado": "PERMITIDO", "nota": "Pesca deportiva y comercial permitida."},
    "Atun_Aleta_amarilla_n":  {"estado": "PERMITIDO", "nota": "Permitido, sujeto a cuotas de INCOPESCA."},
    "pargo_mancha_n":        {"estado": "PERMITIDO", "nota": "Permitido, respetar talla minima."},
    "corvina_reina":         {"estado": "PERMITIDO", "nota": "Permitido, respetar talla minima."},
    "pez_vela_n":            {"estado": "PROTEGIDA", "nota": "Pesca deportiva exclusiva, liberacion obligatoria (Ley 8622)."},
    "tortuga_marina_n":      {"estado": "PROTEGIDA", "nota": "Especie en peligro, captura prohibida (Ley de Vida Silvestre)."},
    "tiburon_martillo_n":    {"estado": "EN VEDA", "nota": "Especie amenazada, captura prohibida en Costa Rica."},
}

# ------------------------------------------------------------
# CARGAR EL MODELO CNN (una sola vez al iniciar)
# ------------------------------------------------------------
# Sirve para: cargar el modelo tal como lo guardaste en el notebook.
# Debe estar en la carpeta models/ del proyecto.
modelo_cnn = load_model("models/modelo_CNN_peces.h5")

app = Flask(__name__)


# ------------------------------------------------------------
# FUNCION: clasificar la especie de la foto
# ------------------------------------------------------------
def clasificar_especie(bytes_imagen):
    """Sirve para: usar el modelo CNN para identificar la especie
    en la foto que mando el usuario por WhatsApp. Usa 150x150 porque
    asi se entreno el modelo (image_shape en el notebook)."""
    imagen = Image.open(io.BytesIO(bytes_imagen)).convert("RGB")
    imagen = imagen.resize((150, 150))
    arreglo = np.array(imagen) / 255.0
    arreglo = arreglo.reshape(1, 150, 150, 3)

    prediccion = modelo_cnn.predict(arreglo, verbose=0)[0]
    indice = int(np.argmax(prediccion))
    especie = CLASES_CNN[indice]
    confianza = float(prediccion[indice])
    return especie, confianza


# ------------------------------------------------------------
# ENDPOINT: aqui Twilio manda cada mensaje que llega por WhatsApp
# ------------------------------------------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    respuesta = MessagingResponse()
    mensaje = respuesta.message()

    num_imagenes = int(request.form.get("NumMedia", 0))

    if num_imagenes == 0:
        mensaje.body(
            "Hola! Mandame una FOTO de un pez y te digo la especie "
            "y si es legal pescarla en Costa Rica."
        )
        return str(respuesta)

    # ------------------------------------------------------------
    # Sirve para: descargar la imagen. OJO: las URLs de medios de
    # Twilio requieren autenticacion (Account SID + Auth Token),
    # a diferencia de Telegram que da la imagen directo.
    # ------------------------------------------------------------
    url_imagen = request.form.get("MediaUrl0")
    resp_imagen = requests.get(
        url_imagen,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
    )
    bytes_imagen = resp_imagen.content

    # 1) clasificar la especie
    especie, confianza = clasificar_especie(bytes_imagen)
    info_legal = LEGALIDAD.get(
        especie,
        {"estado": "NO EVALUADO",
         "nota": "Esta especie es del dataset base (Kaggle) y no tiene "
                 "informacion de legalidad para Costa Rica registrada."}
    )

    # 2) armar la respuesta
    texto_final = (
        f"Especie detectada: {especie.replace('_', ' ')}\n"
        f"Confianza: {confianza*100:.1f}%\n\n"
        f"Legalidad: {info_legal['estado']}\n"
        f"{info_legal['nota']}"
    )
    mensaje.body(texto_final)

    return str(respuesta)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
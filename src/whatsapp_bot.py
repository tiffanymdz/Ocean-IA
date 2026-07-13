

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import requests
import numpy as np
from PIL import Image
import io


from keras.layers import Dense
from tensorflow.keras.models import load_model

_dense_init_original = Dense.__init__

def _dense_init_parcheado(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _dense_init_original(self, *args, **kwargs)

Dense.__init__ = _dense_init_parcheado


TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""

CLASES_CNN = [
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
    "tortuga_marina_n",
]


LEGALIDAD = {
    "dorado":                {"estado": "PERMITIDO", "nota": "Pesca deportiva y comercial permitida."},
    "Atun_Aleta_amarilla_n":  {"estado": "PERMITIDO", "nota": "Permitido, sujeto a cuotas de INCOPESCA."},
    "pargo_mancha_n":        {"estado": "PERMITIDO", "nota": "Permitido, respetar talla minima."},
    "corvina_reina":         {"estado": "PERMITIDO", "nota": "Permitido, respetar talla minima."},
    "pez_vela_n":            {"estado": "PROTEGIDA", "nota": "Pesca deportiva exclusiva, liberacion obligatoria (Ley 8622)."},
    "tortuga_marina_n":      {"estado": "PROTEGIDA", "nota": "Especie en peligro, captura prohibida (Ley de Vida Silvestre)."},
    "tiburon_martillo_n":    {"estado": "EN VEDA", "nota": "Especie amenazada, captura prohibida en Costa Rica."},
}


modelo_cnn = load_model("models/modelo_CNN_peces.h5")

app = Flask(__name__)



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
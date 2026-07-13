import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict/oceano"


st.title("🌊 Predicción Oceánica")

st.write(
    "Seleccione un archivo CSV con los datos históricos del océano."
)


archivo = st.file_uploader(
    "Seleccione un CSV",
    type=["csv"]
)


if archivo:

    st.success("Archivo cargado correctamente.")


    if st.button("🌊 Predecir oleaje"):


        with st.spinner("Procesando información..."):


            try:

                respuesta = requests.post(
                    API_URL,
                    files={
                        "file": (
                            archivo.name,
                            archivo.getvalue(),
                            "text/csv"
                        )
                    }
                )


                if respuesta.status_code == 200:


                    resultado = respuesta.json()


                    st.success(
                        "Predicción generada correctamente."
                    )


                    st.info(
                        "Predicción para las próximas 72 horas."
                    )


                    predicciones = resultado["prediccion_oleaje"]


                    st.subheader("🌊 Altura del oleaje")


                    for hora, valor in enumerate(predicciones, start=1):

                        st.write(
                            f"Hora {hora}: {valor} metros"
                        )


                else:

                    error = respuesta.json()

                    st.error(
                        error.get(
                            "detail",
                            "Error desconocido en la API"
                        )
                    )


            except requests.exceptions.ConnectionError:


                st.error(
                    "No se pudo conectar con la API. "
                    "Verifique que FastAPI esté ejecutándose."
                )
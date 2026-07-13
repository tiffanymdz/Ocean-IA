import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict/especie"


st.title("🐟 Identificación de Especies")

st.write(
    "Seleccione una fotografía de una especie marina para realizar la identificación."
)


imagen = st.file_uploader(
    "Seleccione una imagen",
    type=["jpg", "jpeg", "png"]
)


if imagen:

    st.image(
        imagen,
        width=350
    )


    if st.button("🔍 Identificar especie"):


        with st.spinner("Analizando imagen..."):


            try:

                respuesta = requests.post(
                    API_URL,
                    files={
                        "file": (
                            imagen.name,
                            imagen.getvalue(),
                            imagen.type
                        )
                    }
                )


                if respuesta.status_code == 200:


                    resultado = respuesta.json()


                    principal = resultado["prediccion_principal"]


                    especie = principal["especie"]

                    confianza = principal["confianza"]

                    protegida = principal["protegida_o_en_veda"]



                    st.success("Predicción realizada correctamente")


                    st.subheader("🐟 Resultado")


                    st.metric(
                        "Especie",
                        especie
                    )


                    st.metric(
                        "Confianza",
                        f"{confianza * 100:.1f}%"
                    )


                    if protegida:

                        st.error(
                            "🚨 Especie protegida o en veda"
                        )

                    else:

                        st.success(
                            "✅ Sin alerta de protección"
                        )



                    st.divider()


                    st.subheader("🔎 Alternativas")


                    for alternativa in resultado["alternativas"]:

                        st.write(
                            f"""
                            **{alternativa['especie']}**  
                            Confianza: {alternativa['confianza']*100:.1f}%
                            """
                        )


                else:

                    st.error(
                        "Error en la API:"
                        + respuesta.text
                    )


            except Exception as e:

                st.error(
                    f"No se pudo conectar con la API: {e}"
                )
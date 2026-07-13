# Librerías

import streamlit as st
from PIL import Image
from pathlib import Path

# Configuración

st.set_page_config(
    page_title="OceanoIA",
    page_icon="🌊",
    layout="wide"
)

# CSS

css = Path("app/styles.css").read_text()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)

# Imágenes

logo = Image.open("app/assets/logo.png")
banner = Image.open("app/assets/banner.png")

# Encabezado

col1, col2 = st.columns([1, 5])

with col1:
    st.image(logo, width=120)

with col2:
    st.title("🌊 OceanoIA")
    st.caption("Inteligencia Artificial aplicada al monitoreo marino")

# BANNER

st.image(
    banner,
    use_container_width=True
)

st.markdown("---")

# Bienvenida

st.markdown("""
## 🌊 Bienvenido

OceanoIA es una plataforma desarrollada para apoyar el monitoreo marino mediante modelos de Inteligencia Artificial.

Con esta aplicación podrá:

- 🐟 Identificar especies marinas mediante fotografías.
- 🌊 Predecir el comportamiento del oleaje para las próximas 72 horas.
- 🤖 Utilizar modelos de aprendizaje automático entrenados con datos reales.

Seleccione una opción desde el menú lateral para comenzar.
""")

# Métricas

st.markdown("---")

c1, c2, c3 = st.columns(3)

c1.metric("🐟 Especies", "16")
c2.metric("🌊 Predicción", "72 h")
c3.metric("🤖 Modelos IA", "2")

# Funcionalidades

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.info("""
### 🐟 Identificación de especies

Suba una fotografía para identificar automáticamente una especie marina mediante nuestro modelo CNN.
""")

with col2:
    st.success("""
### 🌊 Predicción oceánica

Suba un archivo CSV con datos históricos para obtener la predicción del oleaje para las próximas 72 horas.
""")

# Footer

st.markdown("---")

st.caption("""
Proyecto desarrollado para el curso de Inteligencia Artificial Aplicada.

CUC • Big Data • 2026
""")
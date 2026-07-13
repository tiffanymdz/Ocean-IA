# Ocean-IA
# 🐟 Clasificación Multiclase de Especies Marinas con Deep Learning

¡Bienvenido al repositorio del proyecto de clasificación automatizada de especies marinas! Este proyecto implementa un flujo de trabajo en Python (utilizando un Jupyter Notebook) para el preprocesamiento, análisis exploratorio y visualización de un dataset de imágenes diseñado para identificar 16 especies marinas diferentes.

## 📋 Descripción del Proyecto

El objetivo principal de este proyecto es clasificar imágenes de peces y otras especies marinas en un entorno multiclase. El conjunto de datos base se extrajo de **"A Large Scale Fish Dataset" (Kaggle)**, el cual fue complementado con 7 especies adicionales de alto interés ecológico y comercial para robustecer el modelo.

El pipeline actual incluye:
* **Carga y Organización:** Estructuración de datos en un directorio limpio clasificado por carpetas (una por especie).
* **Análisis de Distribución:** Identificación del nivel de desbalance de datos entre las clases.
* **Visualización Automatizada:** Generación de muestras representativas del dataset para control de calidad visual.

---

## 📊 El Dataset (Especies)

El conjunto de datos limpio (`NA_fish_dataset_limpio_limpio`) contiene un total de **16 clases** con una distribución variable de imágenes por especie:

| Especie / Clase | Cantidad de Imágenes | Origen / Estado |
| :--- | :---: | :--- |
| `Atun_Aleta_amarilla_n` | 76 | Especie Adicional |
| `Black Sea Sprat` | 50 | Dataset Base (Kaggle) |
| `corvina_reina` | 50 | Especie Adicional |
| `dorado` | 138 | Especie Adicional |
| `Gilt Head Bream` | 50 | Dataset Base (Kaggle) |
| `Horse Mackerel` | 50 | Dataset Base (Kaggle) |
| `pargo_mancha_n` | 41 | Especie Adicional |
| `pez_vela_n` | 52 | Especie Adicional |
| `Red Mullet` | 97 | Dataset Base (Kaggle) |
| `Red Sea Bream` | 50 | Dataset Base (Kaggle) |
| `Sea Bass` | 50 | Dataset Base (Kaggle) |
| `Shrimp` | 50 | Dataset Base (Kaggle) |
| `Striped Red Mullet` | 73 | Dataset Base (Kaggle) |
| `tiburon_martillo_n` | 69 | Especie Adicional |
| `tortuga_marina_n` | 42 | Especie Adicional |
| `Trout` | 58 | Dataset Base (Kaggle) |

> ⚠️ **Nota sobre los datos:** El dataset presenta un desbalance natural (por ejemplo, *dorado* cuenta con 138 imágenes frente a las 41 de *pargo_mancha_n*). El código incluye etapas para identificar este comportamiento, lo cual es clave para definir las métricas de evaluación (como F1-Score) y técnicas de Data Augmentation en la fase de entrenamiento.

---

## 🛠️ Tecnologías y Librerías Utilizadas

El proyecto está desarrollado completamente en **Python 3** utilizando el entorno de Jupyter Notebooks. Las librerías principales empleadas son:

* **`os`**: Manipulación del sistema de archivos y mapeo de directorios de imágenes.
* **`pandas` y `numpy`**: Estructuración y análisis de datos vectoriales.
* **`matplotlib` y `seaborn`**: Visualización de datos, gráficos estadísticos y renderizado de la matriz de imágenes muestra.

---

## 🚀 Estructura del Código

El archivo principal `clasificacion_especies_marinas.ipynb` se divide en las siguientes secciones estratégicas:

1. **Importación de librerías:** Carga del entorno de trabajo optimizado para computer vision.
2. **Importación de la fuente de datos:** Mapeo de la carpeta raíz y extracción automatizada del listado de clases.
3. **Preprocesado:**
   * **3.1 Distribución del dataset:** Conteo dinámico de archivos por carpeta para auditoría de desbalance.
   * **3.2 Visualización:** Generación de un subplot de 4 × 4 que muestra de manera aleatoria/representativa una imagen de cada una de las 16 clases para verificar la correcta lectura de los ficheros.

---

## 💻 Instalación y Uso

Si deseas replicar este análisis localmente, sigue estos pasos:

1. **Clonar el repositorio:**
```bash
   git clone https://github.com/tiffanymdz/TU_REPOSITORIO.git
   cd Ocean-IA
```

2. **Instalar las dependencias:**
```bash
   pip install pandas numpy matplotlib seaborn jupyter
```

3. **Abrir el notebook:**
```bash
   jupyter notebook clasificacion_especies_marinas.ipynb
```

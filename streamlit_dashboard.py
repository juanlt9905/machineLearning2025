import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




#Cargar Datos

datos_diabetes = pd.read_csv('datasets/diabetes_012_health_indicators_BRFSS2015.csv')
#datos_diabetes = pd.read_csv('/home/juan/machineLearning2025/datasets/diabetes_012_health_indicators_BRFSS2015.csv')
#Crear la columna diabetes_01 que unifique prediabetes con diabetes
datos_diabetes['diabetes_01'] = datos_diabetes['Diabetes_012']
datos_diabetes['diabetes_01'] = datos_diabetes['diabetes_01'].replace(2,1)

#Reparar nombres de columnas. Se usa el formato loweCamelCase para el nombre de las caracteristicas.

new_col_names = []

for name in datos_diabetes.columns:
    # Luego, pon todas las letras en minúsculas
    name_lowered_first_letter = name[0].lower() + name[1:]
    # Elimina los espacios al principio y al final
    name_stripped = name_lowered_first_letter.strip()
    # Por último, reemplaza los espacios entre palabras por guiones bajos
    name_no_spaces = name_stripped.replace(' ', '_')
    # Agrega el nuevo nombre a la lista de nuevos nombres de columna
    new_col_names.append(name_no_spaces)

datos_diabetes.columns = new_col_names

datos_diabetes = datos_diabetes.rename(columns={'bMI':'bmi'})

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard de Diabetes",
    page_icon="🩺",
    layout="wide" # 'wide' aprovecha mejor el espacio en pantalla
)


# --- Barra Lateral de Navegación ---
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ve a la sección:", 
    [
        "Introducción", 
        "Análisis Exploratorio", 
        "Preparación del Modelo",
        "Selección de características"
        #"Resultados de los Modelos", 
        #"Conclusiones"
    ]
)



# --- Título y Descripción del Dashboard ---
st.title('Análisis de pacientes con Diabetes en EU')

if page == "Introducción":

    st.write("""
    Dataset de origen

    El Sistema de Vigilancia de Factores de Riesgo en el Comportamiento (BRFSS) se encarga de realizar 
    encuestas telefónicas relacionadas con la salud de los residentes de EE.UU., relativos a sus comportamientos 
    de riesgos para su salud, como lo son enfermedades crónicas, hábitos de preveención de enfermedades y uso de 
    servicios de salud. 

    https://www.cdc.gov/brfss/annual_data/annual_data.htm
             
    Objetivo de trabajar con este dataset:


    * ¿Cuáles son las variables más importantes para porder clasificar a los pacientes respecto a riesgo de  diabetes?

    * ¿La información recopilada por la BRFSS puede clasificar pacientes con diabetes y pacientes sanos ?
    """)


elif page =="Análisis Exploratorio":

        # --- Visualizaciones ---
    st.header('')

    col_a, col_b, col_c = st.columns([1, 2, 1]) # [Espacio, Gráfico, Espacio]
    # --- Gráfica 1: Distribución por Sexo (usando tu código) ---
    with col_b:
        st.subheader("Distribución de la variable objetivo(Diabetes)")

        # Crear una figura de Matplotlib
        fig, ax = plt.subplots()

        # Contar los valores de la columna 'diabetes_01'
        target_counts = datos_diabetes['diabetes_01'].value_counts()

        # Definir etiquetas para el gráfico
        target_labels = {0.0: 'Sin Diabetes', 1.0: 'Diabetes o Pre-Diabetes'}
        target_counts.index = target_counts.index.map(target_labels)

        # Crear el gráfico de pastel
        ax.pie(
            target_counts, 
            labels=target_counts.index, 
            autopct='%1.1f%%', # Formato para mostrar porcentajes
            colors=sns.color_palette('RdYlGn_r', len(target_counts)) # Paleta de colores
        )

        ax.set_title("Distribución de la variable Objetivo")
        st.pyplot(fig)

        
        st.markdown("""
        **Observación:** Será necesario abordar técnicas de balanceo.
        """)

 

    st.subheader("Análisis por Sexo y Edad")

    
    col1, col2 = st.columns(2)

    # --- Gráfica 2: Distribución de Edad (usando tu código) ---
    with col1:
                # Creamos la figura de matplotlib PASTEL
        fig1, ax1 = plt.subplots(figsize=(8, 6)) 

        sexo_counts = datos_diabetes['sex'].value_counts() # Asumiendo que tu columna se llama 'Sex'
        sexo_labels = {1: "Masculino", 0: "Femenino"}
        
        sexo_counts.index = sexo_counts.index.map(lambda x: sexo_labels.get(x, "No especificado"))

        ax1.pie(sexo_counts, labels=sexo_counts.index, autopct='%1.1f%%', 
                colors=sns.color_palette('pastel', len(sexo_counts)),
                textprops={'fontsize': 14}) # Hacemos el texto más grande
        
        # Usamos st.pyplot() para mostrar la figura de matplotlib en Streamlit
        st.pyplot(fig1)
    with col2:
        
        # Creamos la figura de matplotlib BOXPLOT
        fig2, ax2 = plt.subplots(figsize=(8, 6)) # Creamos otra figura
        
        # Mapeamos los valores de diabetes para que las etiquetas sean claras en la leyenda
        diabetes_labels = {0: 'Sin Diabetes', 1: 'Pre-Diabetes', 2: 'Con Diabetes'}
        # Creamos una copia para no modificar el dataframe original al cambiar los labels
        datos_plot = datos_diabetes.copy()
        binary_labels = {0.0: 'Sin Diabetes', 1.0: 'Diabetes o Pre-Diabetes'}
        datos_plot['diabetes_status'] = datos_plot['diabetes_01'].map(binary_labels) #
        
        sns.boxplot(data=datos_plot, x='sex', y='age', hue='diabetes_status', ax=ax2, palette='viridis') # Asumiendo columnas 'Sex' y 'Age'
        ax2.set_xticklabels(['Femenino', 'Masculino'])
        ax2.set_xlabel("Sexo")
        ax2.set_ylabel("Edad")
        ax2.legend(title='Estado de Diabetes')

        target_labels = {0.0: 'Sin Diabetes', 1.0: 'Diabetes o Pre-Diabetes'}
        target_counts.index = target_counts.index.map(target_labels)


        # Mostramos la figura en Streamlit
        
        st.pyplot(fig2)

    # --- Mostrando los Datos ---
    #st.header('Cabeza del dataset')
    #st.dataframe(datos_diabetes.head())


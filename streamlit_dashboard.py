import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle




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

#Definicion de datos enteros.
for col in datos_diabetes.columns:
    #if datos_diabetes[col].dtype == 'float64':
    datos_diabetes[col] = datos_diabetes[col].astype(int)

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
        "Balanceo de clases",
        "Importancia de características", 
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
             

    Dataset desbalanceado. 22 variables. Variaable predictora: diabetes.

    Diabetes_012 -> 0 no diabetes, 1 prediabetes, 2 diabetes. Varliable Categórica.

    HighBP-> Hipertension. Variable Boolena.

    HighChol -> COlesterol alto. Variable Booleana.

    CholCheck-> Chequeo de colesterol los ultimos 5 años. Variable Booleana.

    BMI -> Indice de masa corporal. Variable discreta.

    Smoker ->Ha fumado al menos 100 cigarrros en su vida. Variable booleana.

    Stroke -> Derrame cerebral. Variable booleana.

    HeartDiseaseorAttack -> infarto coronario o infarto al miocardio. Variable booleana.

    PhysActivity -> Actividad fisica los ultimos 30 dias. Variable booleana.

    Fruits -> Consume al menos 1 fruta al dia. Variable booleana.

    Veggies -> Consume vegetales al menos 1 vez al dia. Variable booleana.

    HvyAlcoholConsump -> Hombres que toman mas de 14 bebidas alcoholicas por semana, mujeres mas de 7. Variable booleana.

    AnyHealthcare-> Tiene algun seguro médico. Variable booleana.

    NoDocbcCost-> En el ultimo año, no visito a un doctor debido a no poder costear los servicios médicos. Variable booleana.

    GenHlth -> Opinion de salud general, tu salud general es? Escala 1-5. Variable categórica.

    MentHlth -> Por cuantos dias durante el ultimo mes (1-30) no tuviste una salud buena?. Variable categórica. 

    PhysHlth -> Por cuantos dias durante el ultimo mes (1-30) tu salud no fue buena?. Variable categórica.

    DiffWalk -> TIenes dificultades para caminar o subir escaleras?. Variable booleana.

    Sex -> 0=Femenino, 1= Masculino. Variable boolena.

    Age ->  escala del 1-13, 1= 18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49, 7=50-54 , 8=55-59 , 9=60-64, 10= 65-69, 11= 70-74, 12=75-79 ,13=80 o mayores. Variable categórica.

    Education -> escala del 1-6, 1=nunca fue a la escuela o solo kinder, 2=grados 1-8, 3= grados 9-11, 4= grado 12 o graduados de HIgh school, 5= 1-3 de universidad, 6= 4 o mas a;os de universidad. Variable categórica.

    Income -> escala 1-8 1= menos de 10,000, 5= menos de 35,000, 8=75,000 o más. Variable categórica.
    """)


elif page =="Análisis Exploratorio":


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
        
        sns.boxplot(data=datos_plot, x='sex', y='age', hue='diabetes_status', ax=ax2, palette='viridis') 
        ax2.set_xticklabels(['Femenino', 'Masculino'])
        ax2.set_xlabel("Sexo")
        ax2.set_ylabel("Edad")
        ax2.legend(title='Estado de Diabetes')

        target_labels = {0.0: 'Sin Diabetes', 1.0: 'Diabetes o Pre-Diabetes'}
        target_counts.index = target_counts.index.map(target_labels)


        # Mostramos la figura en Streamlit
        
        st.pyplot(fig2)
        st.write("""
            El conjunto de datos de la BRFSS presenta 


            """)    


    st.subheader("Análisis de BMI")
    col1, col2 = st.columns(2)
    #BMI
    with col1:
        fig1 = plt.figure(figsize=(8, 6))
        sexo_labels = {1: "Masculino", 0: "Femenino"}

        diabetes_labels = {0.0: 'Negativo a Diabetes', 1.0: 'Positivo a Diabetes'}
        # Creamos la nueva columna 'status' usando el mapeo
        datos_diabetes['status'] = datos_diabetes['diabetes_01'].map(diabetes_labels)     
        sns.boxplot(x='sex', y='bmi', hue='status', data=datos_diabetes, palette='magma')

    # Ajustar los títulos y etiquetas

        plt.title('Distribución de BMI por Sexo y Diabetes')
        plt.xlabel('sexo')
        plt.xticks(ticks=range(len(sexo_labels)), labels=[sexo_labels[sex] for sex in sorted(sexo_labels.keys())])
        plt.ylabel('bmi')

        # Mostrar la gráfica
        plt.legend(title='diabetes')
        st.pyplot(fig1)

        st.write("""
            Independientemente del sexo, las personas con mayor BMI tienen mayor problabilidad de tener prediabetes o diabetes, según los datos.

        """)

    with col2:
        fig2 = plt.figure(figsize=(8, 6))

        df_bmi_60 = datos_diabetes[datos_diabetes['bmi']>=60]
        diabetes_labels = {0.0: 'Negativo a Diabetes', 1.0: 'Positivo a Diabetes'}
        # Creamos la nueva columna 'status' usando el mapeo
        df_bmi_60['status'] = df_bmi_60['diabetes_01'].map(diabetes_labels)           
           
        ##Grafica de frecuencias para bmi mayor a 60
        sns.histplot(data=df_bmi_60, x='bmi', hue='status', kde=False) #bins=k, ax=axes[i], multiple='stack')
            #st.pyplot(fig2)

           # st.write("""
            #Independientemente del sexo, las personas con mayor BMI tienen mayor problabilidad de tener prediabetes o diabetes, según los datos.

            #""")
        #3 columnas para los boxplot restantes


    st.subheader("Otras variables")
    col_a, col_b, col_c = st.columns(3) # [Espacio, Gráfico, Espacio]

    with col_a:
        fig1 = plt.figure(figsize=(8, 6))
        sexo_labels = {1: "Masculino", 0: "Femenino"}

        diabetes_labels = {0.0: 'Negativo a Diabetes', 1.0: 'Positivo a Diabetes'}
        # Creamos la nueva columna 'status' usando el mapeo
        datos_diabetes['status'] = datos_diabetes['diabetes_01'].map(diabetes_labels)     
        sns.boxplot(x='sex', y='education', hue='status', data=datos_diabetes, palette='viridis')

        # Ajustar los títulos y etiquetas

        plt.title('Distribución de Educación por Sexo y Diabetes')
        plt.xlabel('sexo')
        plt.xticks(ticks=range(len(sexo_labels)), labels=[sexo_labels[sex] for sex in sorted(sexo_labels.keys())])
        plt.ylabel('educación')

        # Mostrar la gráfica
        plt.legend(title='diabetes')
        st.pyplot(fig1)

    with col_b:
        fig1 = plt.figure(figsize=(8, 6))
        sexo_labels = {1: "Masculino", 0: "Femenino"}

        diabetes_labels = {0.0: 'Negativo a Diabetes', 1.0: 'Positivo a Diabetes'}
        # Creamos la nueva columna 'status' usando el mapeo
        datos_diabetes['status'] = datos_diabetes['diabetes_01'].map(diabetes_labels)     
        sns.boxplot(x='sex', y='income', hue='status', data=datos_diabetes, palette='viridis')

        # Ajustar los títulos y etiquetas

        plt.title('Distribución de Ingresos por Sexo y Diabetes')
        plt.xlabel('sexo')
        plt.xticks(ticks=range(len(sexo_labels)), labels=[sexo_labels[sex] for sex in sorted(sexo_labels.keys())])
        plt.ylabel('Ingresos')

        # Mostrar la gráfica
        plt.legend(title='diabetes')
        st.pyplot(fig1)

    with col_c:
        fig1 = plt.figure(figsize=(8, 6))
        sexo_labels = {1: "Masculino", 0: "Femenino"}

        diabetes_labels = {0.0: 'NO', 1.0: 'SI'}
        # Creamos la nueva columna 'status' usando el mapeo
        datos_diabetes['status'] = datos_diabetes['noDocbcCost'].map(diabetes_labels)     
        sns.boxplot(x='sex', y='income', hue='status', data=datos_diabetes, palette='viridis')

        # Ajustar los títulos y etiquetas

        plt.title('Distribución de Ingresos por Sexo y noDocbcCost')
        plt.xlabel('sexo')
        plt.xticks(ticks=range(len(sexo_labels)), labels=[sexo_labels[sex] for sex in sorted(sexo_labels.keys())])
        plt.ylabel('Ingresos')

        # Mostrar la gráfica
        plt.legend(title='noDocbcCost')
        st.pyplot(fig1)

        st.write("""
                *noDocbcCost: el paciente no asistio a cita medica el último mes debido a falta de dinero.
                    

        """)
    st.write("""
        * Las personas con un nivel educativo e ingresos más bajos tienden a tener mayor prevalencia de diabetes.

    """)    

    datos_diabetes=datos_diabetes.drop(columns='status')

    correlacion = datos_diabetes.corr()
    figcorr=plt.figure(figsize=(14, 12)) 
    rango_a = 0.2
    rango_b = -0.2

    # Aplicar filtro
    filtro = (correlacion >= rango_a) | (correlacion <= rango_b)
    correlacion_filtrada = correlacion.where(filtro)

    sns.heatmap(correlacion_filtrada, annot=True, cbar=True, cmap="RdYlGn")
    st.pyplot(figcorr)

    st.write("""
        Variables más correlacionadas ($correlacion > |0.3| $):

        *PhysHlth y  GenHlth: 0.52

        *PhysHlth y MentHlth: 0.34

        *PhysHlth y DiffWalk: 0.47

        *GenHlth y DiffWalk: 0.45

        *Income y Education: 0.42

        *Income y GenHlth: -0.33

        *Income y DiffWalk: -0.3

        *Age vs HighBP: 0.34

    """)

if page=="Balanceo de clases":

    st.header("Comparación de Técnicas de balanceo")
    
    metrics_unbalanced = pd.read_csv('metrics_logreg_unbalanced.csv', index_col=0)
    metrics_smote = pd.read_csv('metrics_logreg_SMOTE.csv', index_col=0)
    metrics_ros = pd.read_csv('metrics_logreg_RandomOverSampler.csv', index_col=0)

    # --- Creación de la Tabla Comparativa ---
    summary_df = pd.concat([
        metrics_unbalanced['test'],
        metrics_smote['test'],
        metrics_ros['test']
    ], axis=1)
        
    summary_df.columns = ['Datos Desbalanceados', 'SMOTE', 'RandomOverSampler']

    st.subheader("Tabla Comparativa de Rendimiento (en Test)")
    st.dataframe(summary_df)

    st.markdown("""
    **Se observa una ligera mejora en F1 con técnicas de balanceo como SMOTE Y Random OverSampler** 
    """)

    st.divider()

        # --- Graficas
    st.subheader("Visualización por Caso")

    with st.expander("Ver Gráficas para Datos Desbalanceados"):
        try:
            with open('fig_logreg_unbalanced.pkl', 'rb') as f:
                fig = pickle.load(f)
                st.pyplot(fig)
        except FileNotFoundError:
            st.warning("No se encontró el archivo 'figure_logreg_unbalanced.pkl'.")

    with st.expander("Ver Gráficas para SMOTE"):
        try:
            with open('fig_logreg_SMOTE.pkl', 'rb') as f:
                fig = pickle.load(f)
                st.pyplot(fig)
        except FileNotFoundError:
            st.warning("No se encontró el archivo 'figure_logreg_SMOTE.pkl'.")

    with st.expander("Ver Gráficas para RandomOverSampler"):
        try:
            with open('fig_logreg_RandomOverSampler.pkl', 'rb') as f:
                fig = pickle.load(f)
                st.pyplot(fig)
        except FileNotFoundError:
            st.warning("No se encontró el archivo 'fig_logreg_RandomOverSampler.pkl'.")
elif page=="Importancia de características":


    scenario_choice = st.selectbox(
        "Selecciona el escenario para analizar:",
        ("Sin Balanceo (Datos Originales)", "Con Balanceo (RandomOverSampler)")
    )

    # --- Lógica para cargar los archivos correctos ---
    if scenario_choice == "Sin Balanceo (Datos Originales)":
        features_file = 'feature_importances_extratreeclassificator.csv'
        results_file = 'results_extratreeclassificator.csv'
    else: # Con Balanceo
        features_file = 'feature_importances_extratreeclassificator_RandOverSampler.csv'
        results_file = 'results_extratreeclassificator_RandOverSampler.csv'

    try:
        # Cargar los datos desde los archivos CSV
        feature_importances = pd.read_csv(features_file)
        results_df = pd.read_csv(results_file)

        st.subheader(f"Resultados para: {scenario_choice}")

        # --- Visualización 1: Gráfico de Barras de Importancia ---
        top_features = feature_importances.head(15)

        fig1, ax1 = plt.subplots(figsize=(12, 8))
        sns.barplot(x="Importance", y="Feature", data=top_features, palette="rocket", ax=ax1)
        ax1.set_title("Top 15 Características Más Predictivas", fontsize=16)
        ax1.set_xlabel("Importancia (calculada con ExtraTrees)", fontsize=12)
        ax1.set_ylabel("Característica", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig1)
        
        st.divider()

        # --- Visualización 2: Gráfico de Líneas de Rendimiento vs. N° de Características ---
        st.subheader("Búsqueda del Número Óptimo de Características")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            # Usaremos la columna 'F1' que ya habías calculado
            metric_to_plot = 'F1'
            ax2.plot(results_df["Number of Features"], results_df[metric_to_plot], marker="o", linestyle="--")
            ax2.set_xlabel("Número de Características Utilizadas")
            ax2.set_ylabel(f"{metric_to_plot} Score")
            ax2.set_title(f"Rendimiento ({metric_to_plot}) vs. Número de Características")
            
            # Encontrar y marcar el punto óptimo basado en F1
            optimal_row = results_df.loc[results_df[metric_to_plot].idxmax()]
            optimal_num = int(optimal_row["Number of Features"])
            max_metric = optimal_row[metric_to_plot]
            
            ax2.axvline(x=optimal_num, color="r", linestyle="--", label=f"Óptimo: {optimal_num} feats.")
            ax2.legend()
            st.pyplot(fig2)

        with col2:
            st.write("") # Espacio para alinear
            st.metric(
                label=f"Número Óptimo de Características (por {metric_to_plot})",
                value=f"{optimal_num}"
            )
            st.metric(
                label=f"{metric_to_plot} Score Máximo",
                value=f"{max_metric:.4f}"
            )
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo necesario: {e.filename}")

    # --- Mostrando los Datos ---
    #st.header('Cabeza del dataset')
    #st.dataframe(datos_diabetes.head())


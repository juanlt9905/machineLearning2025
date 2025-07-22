El Sistema de Vigilancia de los Factores de Riesgo en el Comportamiento (BRFSS) es el principal sistema nacional de encuestas telefónicas relacionadas con la salud que recopila datos estatales sobre los residentes de EE.UU. relativos a sus comportamientos de riesgo relacionados con la salud, enfermedades crónicas y uso de servicios preventivos. Establecido en 1984 con 15 estados, el BRFSS recoge ahora datos en los 50 estados, así como en el Distrito de Columbia y tres territorios de EE.UU.. El BRFSS realiza más de 400.000 entrevistas a adultos cada año, lo que lo convierte en el mayor sistema de encuestas sanitarias realizadas de forma continua en el mundo.


Objetivo de trabajar con este dataset:

* ¿La información recopilada por la BRFSS puede predecir pacientes con diabetes ?

* ¿Cuáles son las variables más importantes para porder clasificar a los pacientes respecto a la diabetes?

NOTEBOOKS IMPORTANTES:

* exploracion_de_datos.ipynb: Análisis exploratorio de datos.
* balanceo_pruebas_varios_metodos.ipynb: Prueba de distintas técnicas de valanceo, como RandomOverSampler, SMOTE, SMOTETomek, NearMiss, etc.
* balanceoRandomOv_y_seleccionExRegTree.ipynb: Selección de mejores características despues de balancear.
* modelos_dashboard_dt.ipynb: Pruebas con DecisionTrees.
* modelos_dashboard_RandForest.ipynb: Pruebas con RandomForestClassifier.
* modelos_dashboard_redes.ipynb: Pruebas con Perceptron Multicapa.
* balancedRandomForest.ipynb: Pruebas con Balanced Random Forest.
* XGBoost: Pruebas con xgboost.
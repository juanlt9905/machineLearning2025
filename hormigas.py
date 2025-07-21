import numpy as np
import random

# --------------------------------------------------------------------------
# FUNCIONES AUXILIARES DEL ALGORITMO
# --------------------------------------------------------------------------

def calcular_longitud_ruta(ruta, matriz_distancias):
    """Calcula la longitud total de una ruta (lista de ciudades)."""
    longitud = 0
    # Recorre la ruta y suma las distancias entre ciudades consecutivas
    for i in range(len(ruta) - 1):
        longitud += matriz_distancias[ruta[i], ruta[i+1]]
    # Añade la distancia desde la última ciudad de vuelta a la primera
    longitud += matriz_distancias[ruta[-1], ruta[0]]
    return longitud

def elegir_siguiente_ciudad(ciudad_actual, ciudades_visitadas, feromonas, visibilidad, alpha, beta):
    """
    Elige la siguiente ciudad a visitar basándose en la probabilidad ACO.
    """
    # Obtiene los valores de feromona y visibilidad desde la ciudad actual
    valores_feromona = feromonas[ciudad_actual, :] ** alpha
    valores_visibilidad = visibilidad[ciudad_actual, :] ** beta
    
    # Calcula las probabilidades de transición
    probabilidades = valores_feromona * valores_visibilidad
    
    # Pone la probabilidad de ir a ciudades ya visitadas a cero
    probabilidades[list(ciudades_visitadas)] = 0
    
    # Normaliza las probabilidades para que sumen 1
    suma_prob = probabilidades.sum()
    if suma_prob == 0:
        # Si no hay opciones (caso raro), elige una al azar de las no visitadas
        ciudades_no_visitadas = list(set(range(len(feromonas))) - set(ciudades_visitadas))
        return random.choice(ciudades_no_visitadas)
        
    probabilidades /= suma_prob
    
    # Elige la siguiente ciudad usando la ruleta de probabilidades
    num_ciudades = len(feromonas)
    siguiente_ciudad = np.random.choice(range(num_ciudades), p=probabilidades)
    return siguiente_ciudad

def construir_rutas_hormigas(num_hormigas, num_ciudades, feromonas, visibilidad, alpha, beta):
    """
    Simula todas las hormigas para que cada una construya una ruta.
    """
    todas_las_rutas = []
    
    # Para cada hormiga...
    for _ in range(num_hormigas):
        # Elige una ciudad de inicio al azar
        ciudad_inicial = random.randint(0, num_ciudades - 1)
        ruta = [ciudad_inicial]
        ciudades_visitadas = {ciudad_inicial}
        
        # Construye el resto de la ruta
        ciudad_actual = ciudad_inicial
        while len(ruta) < num_ciudades:
            siguiente_ciudad = elegir_siguiente_ciudad(ciudad_actual, ciudades_visitadas, feromonas, visibilidad, alpha, beta)
            ruta.append(siguiente_ciudad)
            ciudades_visitadas.add(siguiente_ciudad)
            ciudad_actual = siguiente_ciudad
        
        todas_las_rutas.append(ruta)
        
    return todas_las_rutas

def actualizar_feromonas(feromonas, todas_las_rutas, matriz_distancias, rho, q):
    """
    Actualiza la matriz de feromonas: evaporación y depósito.
    """
    # 1. Evaporación de la feromona (una parte se pierde)
    feromonas *= rho
    
    # 2. Depósito de feromona por cada hormiga
    for ruta in todas_las_rutas:
        longitud_ruta = calcular_longitud_ruta(ruta, matriz_distancias)
        # La cantidad de feromona a depositar es inversamente proporcional a la longitud
        deposito_feromona = q / longitud_ruta
        
        # Añade la feromona a cada arista de la ruta
        for i in range(len(ruta) - 1):
            ciudad_inicio, ciudad_fin = ruta[i], ruta[i+1]
            feromonas[ciudad_inicio, ciudad_fin] += deposito_feromona
            feromonas[ciudad_fin, ciudad_inicio] += deposito_feromona # Para grafos simétricos
        
        # Añadir feromona al camino de vuelta al inicio
        feromonas[ruta[-1], ruta[0]] += deposito_feromona
        feromonas[ruta[0], ruta[-1]] += deposito_feromona


# --------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# --------------------------------------------------------------------------

# 1. DEFINIR EL PROBLEMA
# Un conjunto de ciudades con coordenadas (x, y)
ciudades = np.array([
    [10, 20], [30, 50], [10, 80], [70, 90], [90, 10],
    [50, 40], [60, 60], [40, 5], [25, 70], [80, 30]
])
num_ciudades = len(ciudades)

# Calcular la matriz de distancias euclidianas entre ciudades
matriz_distancias = np.zeros((num_ciudades, num_ciudades))
for i in range(num_ciudades):
    for j in range(num_ciudades):
        matriz_distancias[i, j] = np.linalg.norm(ciudades[i] - ciudades[j])


# 2. DEFINIR LOS PARÁMETROS DEL ALGORITMO
num_hormigas = 10           # Número de hormigas (m)
num_iteraciones = 100       # Número máximo de ciclos (NC_MAX)
alpha = 1.0                 # Influencia de la feromona (α)
beta = 5.0                  # Influencia de la visibilidad (β)
rho = 0.5                   # Tasa de persistencia de feromona (ρ), 1-rho es la evaporación
q = 100                     # Cantidad de feromona a depositar (Q)


# 3. INICIALIZAR EL ALGORITMO
# Matriz de feromonas (τ), todas las aristas empiezan con el mismo valor
feromonas = np.ones((num_ciudades, num_ciudades))

# Matriz de visibilidad (η = 1/d), se usa para que las hormigas prefieran caminos cortos
# Se suma un valor muy pequeño (1e-10) para evitar división por cero
visibilidad = 1 / (matriz_distancias + 1e-10)

# Variables para guardar la mejor solución encontrada
mejor_ruta_global = None
mejor_distancia_global = float('inf')


# 4. BUCLE PRINCIPAL DEL ALGORITMO
print("Iniciando búsqueda con Colonia de Hormigas...")
for i in range(num_iteraciones):
    # Cada hormiga construye su ruta
    rutas_generadas = construir_rutas_hormigas(num_hormigas, num_ciudades, feromonas, visibilidad, alpha, beta)
    
    # Se actualiza la feromona en el mapa
    actualizar_feromonas(feromonas, rutas_generadas, matriz_distancias, rho, q)
    
    # Buscar la mejor ruta de esta iteración
    for ruta in rutas_generadas:
        distancia_actual = calcular_longitud_ruta(ruta, matriz_distancias)
        # Si la ruta actual es la mejor que hemos visto hasta ahora, la guardamos
        if distancia_actual < mejor_distancia_global:
            mejor_distancia_global = distancia_actual
            mejor_ruta_global = ruta
    
    print(f"Iteración {i+1}/{num_iteraciones} | Mejor distancia encontrada: {mejor_distancia_global:.2f}")


# 5. MOSTRAR RESULTADOS
print("\n----------------------------------------")
print("Búsqueda finalizada.")
print(f"La mejor ruta encontrada es: {mejor_ruta_global}")
print(f"Distancia total: {mejor_distancia_global:.2f}")
print("----------------------------------------")
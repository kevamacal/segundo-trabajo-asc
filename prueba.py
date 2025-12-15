import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
import os  # Import necesario para gestionar carpetas

# --- CONFIGURACIÓN DEL PROBLEMA (ZDT3) ---
NUM_VARS = 30
NUM_OBJS = 2
X_L = 0.0 
X_U = 1.0 

# --- PARÁMETROS FIJOS ---
# Nota: POP_SIZE se pasa ahora como argumento para ser flexible, 
# pero por defecto usaremos 100 como tenías.
T_SIZE = 20           
DE_F = 0.5           
DE_CR = 0.5          
MUT_SIG = 20.0       
MUT_PR = 1.0 / NUM_VARS 

class Solucion:
    def __init__(self, vector=None):
        if vector is None:
            self.x = np.random.uniform(X_L, X_U, NUM_VARS)
        else:
            self.x = vector
        self.f = np.zeros(NUM_OBJS)
        self.evaluated = False

    def evaluar(self, file_handle=None):
        # Implementación ZDT3
        f1 = self.x[0]
        g = 1 + 9 * np.sum(self.x[1:]) / (NUM_VARS - 1)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        self.f = np.array([f1, f2])
        self.evaluated = True
        
        # --- GUARDAR EVALUACIÓN EN FICHERO ---
        if file_handle:
            # Formato: f1 \t f2 \t constr_violation(0.0)
            file_handle.write(f"{f1:.6e}\t{f2:.6e}\t0.000000e+00\n")

def distancia_euclidea(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))

def tchebycheff(solucion, lambdas, z_ideal):
    max_val = -float('inf')
    for i in range(NUM_OBJS):
        diff = abs(solucion.f[i] - z_ideal[i])
        weighted_diff = lambdas[i] * diff
        if lambdas[i] == 0:
            weighted_diff = 0.0001 * diff 
        if weighted_diff > max_val:
            max_val = weighted_diff
    return max_val

def domina(sol_a, sol_b):
    mejor_en_algo = False
    for i in range(NUM_OBJS):
        if sol_a.f[i] > sol_b.f[i]:
            return False
        if sol_a.f[i] < sol_b.f[i]:
            mejor_en_algo = True
    return mejor_en_algo

def ejecutar_moead(semilla, max_evaluaciones, carpeta, pop_size):
    """
    Ejecuta una instancia de MOEA/D con una semilla y configuración específica.
    """
    # 1. Fijar semilla para reproducibilidad
    random.seed(semilla)
    np.random.seed(semilla)
    
    nombre_fichero = os.path.join(carpeta, f"eval_seed_{semilla}.out")
    nombre_grafica = os.path.join(carpeta, f"grafica_seed_{semilla}.png")
    
    print(f"   -> Ejecutando Semilla {semilla}: {max_evaluaciones} evals. Guardando en {nombre_fichero}")

    with open(nombre_fichero, "w") as f_out:
        
        # --- INICIALIZACIÓN ---
        lambdas = []
        for i in range(pop_size):
            l1 = i / (pop_size - 1)
            l2 = 1.0 - l1
            lambdas.append(np.array([l1, l2]))
        lambdas = np.array(lambdas)

        vecindades = []
        for i in range(pop_size):
            distancias = []
            for j in range(pop_size):
                dist = distancia_euclidea(lambdas[i], lambdas[j])
                distancias.append((dist, j))
            distancias.sort(key=lambda x: x[0])
            indices_vecinos = [x[1] for x in distancias[:T_SIZE]]
            vecindades.append(indices_vecinos)

        poblacion = [Solucion() for _ in range(pop_size)]
        z_ideal = np.full(NUM_OBJS, float('inf'))

        evaluaciones = 0
        
        # Evaluar población inicial
        for ind in poblacion:
            ind.evaluar(file_handle=f_out)
            evaluaciones += 1
            for k in range(NUM_OBJS):
                if ind.f[k] < z_ideal[k]:
                    z_ideal[k] = ind.f[k]

        ep = [] # External Population

        # --- BUCLE PRINCIPAL ---
        generacion = 1
        while evaluaciones < max_evaluaciones:
            for i in range(pop_size):
                if evaluaciones >= max_evaluaciones:
                    break

                # Selección y Reproducción
                idxs = vecindades[i]
                seleccion = random.sample(idxs, 3)
                r1, r2, r3 = poblacion[seleccion[0]], poblacion[seleccion[1]], poblacion[seleccion[2]]

                vector_v = r1.x + DE_F * (r2.x - r3.x)
                
                vector_u = np.copy(poblacion[i].x)
                j_rand = random.randint(0, NUM_VARS - 1)
                for j in range(NUM_VARS):
                    if random.random() < DE_CR or j == j_rand:
                        vector_u[j] = vector_v[j]

                # Mutación Polinómica / Gaussiana
                for j in range(NUM_VARS):
                    if random.random() < MUT_PR:
                        sigma = (X_U - X_L) / MUT_SIG
                        vector_u[j] += random.gauss(0, sigma)

                vector_u = np.clip(vector_u, X_L, X_U)

                # Evaluar hijo
                nuevo_ind = Solucion(vector_u)
                nuevo_ind.evaluar(file_handle=f_out)
                evaluaciones += 1

                # Actualizar Z ideal
                for k in range(NUM_OBJS):
                    if nuevo_ind.f[k] < z_ideal[k]:
                        z_ideal[k] = nuevo_ind.f[k]

                # Actualizar Vecinos
                for j_vecino in idxs:
                    g_te_nuevo = tchebycheff(nuevo_ind, lambdas[j_vecino], z_ideal)
                    g_te_viejo = tchebycheff(poblacion[j_vecino], lambdas[j_vecino], z_ideal)

                    if g_te_nuevo <= g_te_viejo:
                        poblacion[j_vecino] = copy.deepcopy(nuevo_ind)

                # Actualizar EP (External Population)
                es_dominado = False
                remover = []
                for sol_ep in ep:
                    if domina(sol_ep, nuevo_ind):
                        es_dominado = True
                        break
                    if domina(nuevo_ind, sol_ep):
                        remover.append(sol_ep)
                
                if not es_dominado:
                    for rem in remover:
                        ep.remove(rem)
                    ep.append(copy.deepcopy(nuevo_ind))

            generacion += 1

    # --- GUARDAR GRÁFICA (SIN MOSTRAR VENTANA PARA NO BLOQUEAR EL BUCLE) ---
    final_set = ep if len(ep) > 0 else poblacion # Si EP está vacío por alguna razón, usar población final
    f1_vals = [sol.f[0] for sol in final_set]
    f2_vals = [sol.f[1] for sol in final_set]

    plt.figure(figsize=(10, 6))
    plt.scatter(f1_vals, f2_vals, c='blue', s=10, label=f'Seed {semilla}')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'Pareto ZDT3 - Evals: {max_evaluaciones} - Seed: {semilla}')
    plt.grid(True)
    plt.savefig(nombre_grafica)
    plt.close() # Importante cerrar la figura para liberar memoria

def main():
    # CONFIGURACIÓN DE EXPERIMENTOS
    NUM_EJECUCIONES = 10  # Número de semillas (seeds) a generar por configuración
    
    # Definimos los escenarios: (Max Evaluaciones, Carpeta Destino, Tamaño Población)
    escenarios = [
        {"evals": 4000,  "carpeta": "EVAL4000",  "pop": 100},
        {"evals": 10000, "carpeta": "EVAL10000", "pop": 100} 
    ]

    print("--- INICIANDO GENERACIÓN AUTOMÁTICA DE SEMILLAS ---")

    for escenario in escenarios:
        max_evals = escenario["evals"]
        carpeta = escenario["carpeta"]
        pop = escenario["pop"]
        
        # Crear carpeta si no existe
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
            print(f"Carpeta creada: {carpeta}")
        else:
            print(f"Usando carpeta existente: {carpeta}")

        print(f"Iniciando lote de {NUM_EJECUCIONES} ejecuciones para {max_evals} evaluaciones...")

        for i in range(NUM_EJECUCIONES):
            # La semilla 'i' asegura que cada ejecución sea diferente pero reproducible
            ejecutar_moead(semilla=i, max_evaluaciones=max_evals, carpeta=carpeta, pop_size=pop)

    print("\n--- PROCESO FINALIZADO ---")
    print(f"Se han generado {NUM_EJECUCIONES} ficheros en 'EVAL4000' y {NUM_EJECUCIONES} en 'EVAL10000'.")

if __name__ == "__main__":
    main()
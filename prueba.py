import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DEL PROBLEMA (ZDT3) ---
NUM_VARS = 30
NUM_OBJS = 2
X_L = 0.0 # Límite inferior
X_U = 1.0 # Límite superior

# --- PARÁMETROS DEL ALGORITMO ---
POP_SIZE = 100       # Tamaño de población (N)
MAX_EVALUACIONES = 4000 # Presupuesto de evaluaciones
T_SIZE = 20          # Tamaño vecindad (20% de 100)
DE_F = 0.5           # DE Factor F
DE_CR = 0.5          # DE Crossover CR
MUT_SIG = 20.0       # Parametro SIG para mutación Gaussiana
MUT_PR = 1.0 / NUM_VARS # Probabilidad mutación

# Nombre del fichero de salida para métricas (formato .out)
OUTPUT_FILENAME = "eval_moead.out"

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
        
        # --- GUARDAR EVALUACIÓN EN FICHERO (Formato exacto NSGA-II) ---
        if file_handle:
            # Formato: f1 \t f2 \t constr_violation(0.0) en notación científica
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

def main():
    print(f"Iniciando MOEA/D. Guardando evaluaciones en '{OUTPUT_FILENAME}'...")
    
    # Abrir fichero para escribir todas las evaluaciones
    with open(OUTPUT_FILENAME, "w") as f_out:
        
        # 1. INICIALIZACIÓN
        lambdas = []
        for i in range(POP_SIZE):
            l1 = i / (POP_SIZE - 1)
            l2 = 1.0 - l1
            lambdas.append(np.array([l1, l2]))
        lambdas = np.array(lambdas)

        vecindades = []
        for i in range(POP_SIZE):
            distancias = []
            for j in range(POP_SIZE):
                dist = distancia_euclidea(lambdas[i], lambdas[j])
                distancias.append((dist, j))
            distancias.sort(key=lambda x: x[0])
            indices_vecinos = [x[1] for x in distancias[:T_SIZE]]
            vecindades.append(indices_vecinos)

        poblacion = [Solucion() for _ in range(POP_SIZE)]
        z_ideal = np.full(NUM_OBJS, float('inf'))

        evaluaciones = 0
        
        # Evaluar población inicial y guardar
        for ind in poblacion:
            ind.evaluar(file_handle=f_out) # Pasamos el fichero para guardar
            evaluaciones += 1
            for k in range(NUM_OBJS):
                if ind.f[k] < z_ideal[k]:
                    z_ideal[k] = ind.f[k]

        ep = []

        # Bucle principal
        generacion = 1
        while evaluaciones < MAX_EVALUACIONES:
            for i in range(POP_SIZE):
                if evaluaciones >= MAX_EVALUACIONES:
                    break

                # Reproducción
                idxs = vecindades[i]
                seleccion = random.sample(idxs, 3)
                r1, r2, r3 = poblacion[seleccion[0]], poblacion[seleccion[1]], poblacion[seleccion[2]]

                vector_v = r1.x + DE_F * (r2.x - r3.x)
                
                vector_u = np.copy(poblacion[i].x)
                j_rand = random.randint(0, NUM_VARS - 1)
                for j in range(NUM_VARS):
                    if random.random() < DE_CR or j == j_rand:
                        vector_u[j] = vector_v[j]

                for j in range(NUM_VARS):
                    if random.random() < MUT_PR:
                        sigma = (X_U - X_L) / MUT_SIG
                        vector_u[j] += random.gauss(0, sigma)

                vector_u = np.clip(vector_u, X_L, X_U)

                # Evaluación y Guardado
                nuevo_ind = Solucion(vector_u)
                nuevo_ind.evaluar(file_handle=f_out) # AQUÍ SE GUARDA LA EVALUACIÓN
                evaluaciones += 1

                # Actualización
                for k in range(NUM_OBJS):
                    if nuevo_ind.f[k] < z_ideal[k]:
                        z_ideal[k] = nuevo_ind.f[k]

                for j_vecino in idxs:
                    g_te_nuevo = tchebycheff(nuevo_ind, lambdas[j_vecino], z_ideal)
                    g_te_viejo = tchebycheff(poblacion[j_vecino], lambdas[j_vecino], z_ideal)

                    if g_te_nuevo <= g_te_viejo:
                        poblacion[j_vecino] = copy.deepcopy(nuevo_ind)

                # EP Update
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

    # --- VISUALIZACIÓN FINAL ---
    print(f"Fin. Evaluaciones: {evaluaciones}")
    print(f"Fichero '{OUTPUT_FILENAME}' generado correctamente.")
    
    final_set = ep
    f1_vals = [sol.f[0] for sol in final_set]
    f2_vals = [sol.f[1] for sol in final_set]

    plt.figure(figsize=(10, 6))
    plt.scatter(f1_vals, f2_vals, c='red', label='MOEA/D (EP Final)')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Frente de Pareto ZDT3 - MOEA/D')
    plt.legend()
    plt.grid(True)
    plt.savefig("Grafica_MOEAD_Final.png")
    plt.show()

if __name__ == "__main__":
    main()
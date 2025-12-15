import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DEL PROBLEMA (ZDT3) ---
NUM_VARS = 30
NUM_OBJS = 2
X_L = 0.0 
X_U = 1.0 

# --- PARÁMETROS DEL ALGORITMO MEJORADOS ---
POP_SIZE = 100       
MAX_EVALUACIONES = 4000 
T_SIZE = 20          
DE_F = 0.5           
DE_CR = 0.9          # SUBIDO: 0.9 fomenta mejor exploración en ZDT3
MUT_SIG = 20.0       
MUT_PR = 1.0 / NUM_VARS 
NR_LIMIT = 2         # NUEVO: Límite de reemplazo para mantener diversidad

OUTPUT_FILENAME = "eval_moead_mejorado.out"

class Solucion:
    def __init__(self, vector=None):
        if vector is None:
            self.x = np.random.uniform(X_L, X_U, NUM_VARS)
        else:
            self.x = vector
        self.f = np.zeros(NUM_OBJS)
        
    def evaluar(self, file_handle=None):
        x = np.clip(self.x, X_L, X_U) # Asegurar límites antes de evaluar
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (NUM_VARS - 1)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        self.f = np.array([f1, f2])
        
        if file_handle:
            file_handle.write(f"{f1:.6e}\t{f2:.6e}\t0.000000e+00\n")

def distancia_euclidea(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))

def tchebycheff_normalizado(solucion, lambdas, z_ideal, z_nadir):
    """ Tchebycheff con normalización para manejar escalas dispares """
    max_val = -float('inf')
    for i in range(NUM_OBJS):
        # Normalización: (valor - ideal) / (nadir - ideal)
        denominador = z_nadir[i] - z_ideal[i]
        if denominador < 1e-6: denominador = 1e-6 # Evitar div por cero
        
        diff_norm = abs(solucion.f[i] - z_ideal[i]) / denominador
        
        weighted_diff = lambdas[i] * diff_norm
        if lambdas[i] == 0:
            weighted_diff = 0.0001 * diff_norm 
            
        if weighted_diff > max_val:
            max_val = weighted_diff
    return max_val

def domina(sol_a, sol_b):
    """ Devuelve True si sol_a domina a sol_b """
    mejor_en_algo = False
    for i in range(NUM_OBJS):
        if sol_a.f[i] > sol_b.f[i]: # Asumiendo minimización
            return False
        if sol_a.f[i] < sol_b.f[i]:
            mejor_en_algo = True
    return mejor_en_algo

def main():
    print(f"Iniciando MOEA/D Mejorado (Normalización + NR={NR_LIMIT})...")
    
    with open(OUTPUT_FILENAME, "w") as f_out:
        
        # 1. INICIALIZACIÓN
        lambdas = []
        for i in range(POP_SIZE):
            l1 = i / (POP_SIZE - 1)
            l2 = 1.0 - l1
            lambdas.append(np.array([l1, l2]))
        lambdas = np.array(lambdas)

        # Precalcular vecindades
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
        z_nadir = np.full(NUM_OBJS, -float('inf')) # Inicialización del Nadir

        evaluaciones = 0
        
        # Evaluar población inicial
        fitness_matrix = np.zeros((POP_SIZE, NUM_OBJS))
        for idx, ind in enumerate(poblacion):
            ind.evaluar(file_handle=f_out)
            evaluaciones += 1
            fitness_matrix[idx] = ind.f
            # Actualizar ideal
            for k in range(NUM_OBJS):
                if ind.f[k] < z_ideal[k]: z_ideal[k] = ind.f[k]
        
        # Inicializar Nadir con el peor de la población inicial
        z_nadir = np.max(fitness_matrix, axis=0)

        ep = [] # Archivo externo

        # Bucle principal
        while evaluaciones < MAX_EVALUACIONES:
            for i in range(POP_SIZE):
                if evaluaciones >= MAX_EVALUACIONES: break

                # --- 1. SELECCIÓN Y REPRODUCCIÓN ---
                idxs = vecindades[i]
                seleccion = random.sample(idxs, 3) # Selección de vecindario
                r1, r2, r3 = poblacion[seleccion[0]], poblacion[seleccion[1]], poblacion[seleccion[2]]

                # Operador DE/rand/1/bin
                vector_v = r1.x + DE_F * (r2.x - r3.x)
                vector_u = np.copy(poblacion[i].x)
                
                j_rand = random.randint(0, NUM_VARS - 1)
                for j in range(NUM_VARS):
                    if random.random() < DE_CR or j == j_rand:
                        vector_u[j] = vector_v[j]

                # Mutación Gaussiana
                for j in range(NUM_VARS):
                    if random.random() < MUT_PR:
                        sigma = (X_U - X_L) / MUT_SIG
                        vector_u[j] += random.gauss(0, sigma)

                vector_u = np.clip(vector_u, X_L, X_U) # Clip final imprescindible

                # --- 2. EVALUACIÓN ---
                nuevo_ind = Solucion(vector_u)
                nuevo_ind.evaluar(file_handle=f_out)
                evaluaciones += 1

                # --- 3. ACTUALIZAR REFERENCIAS (Ideal y Nadir) ---
                actualizado_z = False
                for k in range(NUM_OBJS):
                    if nuevo_ind.f[k] < z_ideal[k]:
                        z_ideal[k] = nuevo_ind.f[k]
                        actualizado_z = True
                    if nuevo_ind.f[k] > z_nadir[k]: # Actualización dinámica del Nadir
                        z_nadir[k] = nuevo_ind.f[k]
                
                # --- 4. ACTUALIZACIÓN DE VECINOS (Limitada por NR) ---
                # Mezclamos los índices para no favorecer siempre a los primeros del vecindario
                idxs_mezclados = list(idxs)
                random.shuffle(idxs_mezclados)
                
                c = 0 # Contador de reemplazos
                for j_vecino in idxs_mezclados:
                    if c >= NR_LIMIT: 
                        break # Parar si ya reemplazamos a suficientes vecinos
                    
                    # Usamos Tchebycheff Normalizado
                    g_old = tchebycheff_normalizado(poblacion[j_vecino], lambdas[j_vecino], z_ideal, z_nadir)
                    g_new = tchebycheff_normalizado(nuevo_ind, lambdas[j_vecino], z_ideal, z_nadir)

                    if g_new <= g_old:
                        poblacion[j_vecino] = copy.deepcopy(nuevo_ind)
                        c += 1

                # --- 5. ACTUALIZAR EP (Archivo Externo) ---
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

    # --- VISUALIZACIÓN FINAL ---
    print(f"Fin. Evaluaciones: {evaluaciones}")
    print(f"Rango f2 final: [{min(s.f[1] for s in ep):.4f}, {max(s.f[1] for s in ep):.4f}]")
    
    final_set = ep
    f1_vals = [sol.f[0] for sol in final_set]
    f2_vals = [sol.f[1] for sol in final_set]

    plt.figure(figsize=(10, 6))
    plt.scatter(f1_vals, f2_vals, c='blue', marker='o', label='MOEA/D Mejorado')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'Frente de Pareto ZDT3 - MOEA/D Mejorado ({evaluaciones} evals)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Grafica_MOEAD_Mejorado.png")
    print("Gráfica guardada como 'Grafica_MOEAD_Mejorado.png'")
    # plt.show() # Comentar si se ejecuta en servidor sin pantalla

if __name__ == "__main__":
    main()
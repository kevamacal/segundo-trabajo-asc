import numpy as np
import matplotlib.pyplot as plt
import os

class ZDT3:
    def __init__(self, n_vars=30):
        self.n_vars = n_vars
        self.bound_min = np.zeros(n_vars)
        self.bound_max = np.ones(n_vars)

    def evaluate(self, x):
        """Calcula f1 y f2 para ZDT3"""
        x = np.clip(x, self.bound_min, self.bound_max)
        
        f1 = x[0]
        g = 1 + 9 * np.mean(x[1:])
        # La fórmula clásica de ZDT3 genera valores negativos en f2
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])

class MOEAD_Normalized:
    def __init__(self, problem, n_pop=100, n_neighbors=20, max_gen=100, seed=1):
        self.problem = problem
        self.N = n_pop
        self.T = n_neighbors
        self.max_gen = max_gen
        self.seed = seed
        
        # --- PARÁMETROS AJUSTADOS PARA ZDT3 ---
        self.CR = 0.9      # Crossover rate alto para DE
        self.F = 0.5       # Factor de escala DE estándar
        self.eta_m = 15    # Mutación más agresiva (antes 20) para saltar locales
        self.nr = 2        # Reemplazo limitado (mantiene diversidad en islas)
        
        self.weights = None
        self.neighborhood = None
        self.population = None
        self.fitness_pop = None
        
        # Puntos de Referencia para Normalización
        self.z_ideal = None
        self.z_nadir = None # Punto "peor" estimado para normalizar

        np.random.seed(self.seed)

    def initialize(self):
        # 1. Vectores de Peso
        self.weights = np.zeros((self.N, 2))
        for i in range(self.N):
            self.weights[i, 0] = i / (self.N - 1)
            self.weights[i, 1] = 1.0 - self.weights[i, 0]
        self.weights[self.weights == 0] = 0.0001

        # 2. Vecindario
        dists = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                dists[i, j] = np.linalg.norm(self.weights[i] - self.weights[j])
        self.neighborhood = np.argsort(dists, axis=1)[:, :self.T]

        # 3. Población Inicial
        self.population = np.random.random((self.N, self.problem.n_vars))
        self.fitness_pop = np.array([self.problem.evaluate(ind) for ind in self.population])

        # 4. Inicializar Ideal y Nadir
        self.z_ideal = np.min(self.fitness_pop, axis=0)
        self.z_nadir = np.max(self.fitness_pop, axis=0)

    def update_reference_points(self, new_fitness):
        """Actualiza z_ideal y z_nadir dinámicamente"""
        self.z_ideal = np.minimum(self.z_ideal, new_fitness)
        self.z_nadir = np.maximum(self.z_nadir, new_fitness)

    def tchebycheff_normalized(self, fitness, weight_idx):
        """
        Tchebycheff con Normalización:
        (f_i - z_ideal) / (z_nadir - z_ideal)
        Esto es CRÍTICO cuando f2 tiene rango [-0.7, 2.0] y f1 [0, 1]
        """
        # Evitar división por cero
        scale = self.z_nadir - self.z_ideal
        scale[scale < 1e-6] = 1e-6 
        
        norm_diff = np.abs(fitness - self.z_ideal) / scale
        weighted_diff = self.weights[weight_idx] * norm_diff
        return np.max(weighted_diff)

    def polynomial_mutation(self, x):
        p_mut = 1.0 / self.problem.n_vars
        y = x.copy()
        low = self.problem.bound_min
        up = self.problem.bound_max
        
        for i in range(self.problem.n_vars):
            if np.random.random() <= p_mut:
                y_val = y[i]
                delta1 = (y_val - low[i]) / (up[i] - low[i])
                delta2 = (up[i] - y_val) / (up[i] - low[i])
                rand = np.random.random()
                mut_pow = 1.0 / (self.eta_m + 1.0)
                
                if rand <= 0.5:
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * ((1.0 - delta1) ** (self.eta_m + 1.0))
                    deltaq = (val ** mut_pow) - 1.0
                else:
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * ((1.0 - delta2) ** (self.eta_m + 1.0))
                    deltaq = 1.0 - (val ** mut_pow)
                
                y[i] = y_val + deltaq * (up[i] - low[i])
                y[i] = np.clip(y[i], low[i], up[i])
        return y

    def evolution_operator_DE(self, idx_subproblem):
        # Selección
        neighbors = self.neighborhood[idx_subproblem]
        r1, r2, r3 = np.random.choice(neighbors, 3, replace=False)
        
        # DE/rand/1
        mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        
        # Crossover
        crossover_mask = np.random.rand(self.problem.n_vars) <= self.CR
        j_rand = np.random.randint(self.problem.n_vars)
        crossover_mask[j_rand] = True 
        
        offspring = np.where(crossover_mask, mutant, self.population[idx_subproblem])
        offspring = np.clip(offspring, self.problem.bound_min, self.problem.bound_max)
        
        # Mutación
        offspring = self.polynomial_mutation(offspring)
        return offspring

    def run(self):
        self.initialize()
        print(f"MOEA/D Normalizado iniciado. Semilla {self.seed}. Max Gen: {self.max_gen}")

        for gen in range(self.max_gen):
            for i in range(self.N):
                offspring_x = self.evolution_operator_DE(i)
                offspring_fit = self.problem.evaluate(offspring_x)
                
                # Actualizar Referencias (Ideal y Nadir)
                self.update_reference_points(offspring_fit)
                
                # Actualizar Vecinos
                neighbors = self.neighborhood[i].copy()
                np.random.shuffle(neighbors)
                
                c = 0
                for j in neighbors:
                    if c >= self.nr: break # Límite de reemplazo (Diversidad)
                        
                    # Usamos Tchebycheff Normalizado
                    g_old = self.tchebycheff_normalized(self.fitness_pop[j], j)
                    g_new = self.tchebycheff_normalized(offspring_fit, j)
                    
                    if g_new <= g_old:
                        self.population[j] = offspring_x
                        self.fitness_pop[j] = offspring_fit
                        c += 1
            
            if gen % 20 == 0:
                print(f"Gen {gen}: Min f2 actual = {np.min(self.fitness_pop[:,1]):.4f}")

        return self.fitness_pop

# --- FILTRADO ---
def filtrar_no_dominadas(poblacion_fitness):
    tamaño = poblacion_fitness.shape[0]
    es_dominada = np.zeros(tamaño, dtype=bool)
    for i in range(tamaño):
        for j in range(tamaño):
            if i == j: continue
            if all(poblacion_fitness[j] <= poblacion_fitness[i]) and any(poblacion_fitness[j] < poblacion_fitness[i]):
                es_dominada[i] = True
                break
    return poblacion_fitness[~es_dominada]

if __name__ == "__main__":
    # CONFIGURACIÓN EXACTA PARA EVAL10000
    seed = 1
    pop_size = 100
    generations = 100
    
    zdt3 = ZDT3()
    # Usar la clase MOEAD_Normalized
    moead = MOEAD_Normalized(zdt3, n_pop=pop_size, max_gen=generations, seed=seed)
    
    # Ejecutar
    final_pop = moead.run()
    
    # Filtrar y Guardar
    frente_pareto = filtrar_no_dominadas(final_pop)
    
    filename = f"zdt3_final_moead_P{pop_size}G{generations}_seed{seed:02d}.out"
    np.savetxt(filename, frente_pareto, fmt='%.6f', delimiter='\t')
    
    print(f"\n[OK] Guardado: {filename}")
    print(f"Rango f2 final: [{np.min(frente_pareto[:,1]):.4f}, {np.max(frente_pareto[:,1]):.4f}]")
    print("(Si el mínimo es negativo ej: -0.7, ¡has triunfado!)")
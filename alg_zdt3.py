import numpy as np
import os

class ZDT3:
    def __init__(self, n_vars=30):
        self.n_vars = n_vars
        self.bound_min = np.zeros(n_vars)
        self.bound_max = np.ones(n_vars)

    def evaluate(self, x):
        x = np.clip(x, self.bound_min, self.bound_max)
        f1 = x[0]
        g = 1 + 9 * np.mean(x[1:])
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])

class MOEAD:
    def __init__(self, problem, n_pop=100, n_neighbors=20, max_gen=200):
        self.problem = problem
        self.N = n_pop           
        self.T = n_neighbors     
        self.max_gen = max_gen   
        
        # Parámetros ajustados para MOEA/D-DE estándar
        self.CR = 1.0   # Probabilidad de cruce (1.0 suele ir bien con DE para MOEA/D)
        self.F = 0.5    # Factor de escala
        self.eta_m = 20 # Índice de distribución para mutación polinómica
        
        self.weights = None
        self.neighborhood = None
        self.population = None
        self.fitness_pop = None
        self.z_ideal = None     

    def initialize(self):
        self.weights = np.zeros((self.N, 2))
        for i in range(self.N):
            self.weights[i, 0] = i / (self.N - 1)
            self.weights[i, 1] = 1.0 - self.weights[i, 0]
        
        # Evitar división por cero
        self.weights[self.weights == 0] = 0.0001

        dists = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                dists[i, j] = np.linalg.norm(self.weights[i] - self.weights[j])
        
        self.neighborhood = np.argsort(dists, axis=1)[:, :self.T]

        self.population = np.random.random((self.N, self.problem.n_vars))
        self.fitness_pop = np.array([self.problem.evaluate(ind) for ind in self.population])

        self.z_ideal = np.min(self.fitness_pop, axis=0)

    def tchebycheff(self, fitness, weight_idx):
        # Tchebycheff simple (sin normalización dinámica compleja para mantenerlo simple)
        diff = np.abs(fitness - self.z_ideal)
        weighted_diff = self.weights[weight_idx] * diff
        return np.max(weighted_diff)

    def polynomial_mutation(self, x):
        """Aplica mutación polinómica para refinamiento local"""
        pm = 1.0 / self.problem.n_vars
        x_mut = x.copy()
        
        # Máscara para decidir qué variables mutar
        mask = np.random.random(x.shape) < pm
        
        if not np.any(mask):
            return x_mut
            
        lower = self.problem.bound_min
        upper = self.problem.bound_max
        
        for i in np.where(mask)[0]:
            val = x[i]
            delta_1 = (val - lower[i]) / (upper[i] - lower[i])
            delta_2 = (upper[i] - val) / (upper[i] - lower[i])
            
            rand = np.random.random()
            mut_pow = 1.0 / (self.eta_m + 1.0)
            
            if rand < 0.5:
                xy = 1.0 - delta_1
                val_add = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_m + 1.0))
                delta_q = (val_add ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta_2
                val_add = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_m + 1.0))
                delta_q = 1.0 - (val_add ** mut_pow)
                
            x_mut[i] = val + delta_q * (upper[i] - lower[i])
            
        return np.clip(x_mut, lower, upper)

    def evolution_operator_DE(self, idx_subproblem):
        # 1. Selección de vecinos para DE
        neighbors = self.neighborhood[idx_subproblem]
        
        # Necesitamos 3 vectores distintos entre ellos
        idxs = np.random.choice(neighbors, 3, replace=False)
        x_r1, x_r2, x_r3 = self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]]
        
        # 2. Mutación Diferencial (DE/rand/1)
        mutant = x_r1 + self.F * (x_r2 - x_r3)
        
        # 3. Cruce Binomial (Uso de CR) - ESTO FALTABA
        target = self.population[idx_subproblem]
        cross_mask = np.random.random(self.problem.n_vars) <= self.CR
        
        # Asegurar que al menos una variable cambie
        j_rand = np.random.randint(0, self.problem.n_vars)
        cross_mask[j_rand] = True
        
        trial = np.where(cross_mask, mutant, target)
        trial = np.clip(trial, self.problem.bound_min, self.problem.bound_max)
        
        # 4. Mutación Polinómica (Para salir de óptimos locales y afinar) - ESTO FALTABA
        offspring = self.polynomial_mutation(trial)
        
        return offspring

    def run(self, file_all_path=None):
        self.initialize()
        nr = 2 # Límite de reemplazo para mantener diversidad (Spacing)
        
        f_all = None
        if file_all_path:
            f_all = open(file_all_path, 'w')
            f_all.write("# This file contains the data of all generations\n")

        for gen in range(self.max_gen):
            for i in range(self.N): 
                # Ahora usa el operador DE completo + Mutación Polinómica
                offspring_x = self.evolution_operator_DE(i)
                offspring_fit = self.problem.evaluate(offspring_x)
                
                self.z_ideal = np.min(np.vstack((self.z_ideal, offspring_fit)), axis=0)
                
                neighbors = self.neighborhood[i].copy()
                np.random.shuffle(neighbors)
                
                count = 0
                for j in neighbors:
                    if count >= nr:
                        break
                    
                    g_te_offspring = self.tchebycheff(offspring_fit, j)
                    g_te_neighbor = self.tchebycheff(self.fitness_pop[j], j)
                    
                    if g_te_offspring <= g_te_neighbor:
                        self.population[j] = offspring_x
                        self.fitness_pop[j] = offspring_fit
                        count += 1
            
            # Guardar histórico si se solicita
            if f_all:
                f_all.write(f"# gen = {gen + 1}\n")
                for fit in self.fitness_pop:
                    f_all.write(f"{fit[0]:.6e}\t{fit[1]:.6e}\t0.000000e+00\n")
        
        if f_all:
            f_all.close()

        return self.fitness_pop

if __name__ == "__main__":
    # Configuraciones originales
    configuraciones = [
        (40, 100),   
        (100, 100)   
    ]
    
    num_ejecuciones = 10
    output_dir = "RESULTADOS_MOEAD_MEJORADO" # Carpeta nueva para no mezclar
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zdt3_problem = ZDT3(n_vars=30)

    print(f"--- Iniciando Batería de Pruebas MOEA/D MEJORADO ---")

    for pop_size, generations in configuraciones:
        print(f"\nProcesando Configuración: Población={pop_size}, Generaciones={generations}")
        
        for i in range(1, num_ejecuciones + 1):
            semilla = i
            np.random.seed(semilla) 
            
            n_vecinos = 20 if pop_size >= 20 else int(pop_size/2)
            
            algorithm = MOEAD(problem=zdt3_problem, 
                              n_pop=pop_size, 
                              n_neighbors=n_vecinos, 
                              max_gen=generations)
            
            # Nombre de archivos
            filename_all = f"zdt3_all_moead_P{pop_size}G{generations}_seed{semilla:02d}.out"
            filepath_all = os.path.join(output_dir, filename_all)

            final_front = algorithm.run(file_all_path=filepath_all)
            
            # Preparar formato metrics
            n_soluciones = final_front.shape[0]
            col_zeros = np.zeros((n_soluciones, 1))
            datos_metrics = np.hstack((final_front, col_zeros))
            
            filename_final = f"zdt3_final_moead_P{pop_size}G{generations}_seed{semilla:02d}.out"
            filepath_final = os.path.join(output_dir, filename_final)
            
            np.savetxt(filepath_final, datos_metrics, fmt='%.6e', delimiter='\t')
            
            print(f"  > Ejecución {i}/{num_ejecuciones} terminada.")

    print(f"\n[FIN] Resultados en '{output_dir}'.")
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
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])

class MOEAD:
    def __init__(self, problem, n_pop=100, n_neighbors=20, max_gen=200):
        self.problem = problem
        self.N = n_pop           
        self.T = n_neighbors     
        self.max_gen = max_gen   
        
        self.CR = 0.5 
        self.F = 0.5
        self.eta_m = 20 
        
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
        diff = np.abs(fitness - self.z_ideal)
        weighted_diff = self.weights[weight_idx] * diff
        return np.max(weighted_diff)

    def evolution_operator_DE(self, idx_subproblem):
        neighbors = self.neighborhood[idx_subproblem]
        r1, r2, r3 = np.random.choice(neighbors, 3, replace=False)
        
        x_r1 = self.population[r1]
        x_r2 = self.population[r2]
        x_r3 = self.population[r3]
        
        mutant = x_r1 + self.F * (x_r2 - x_r3)
        offspring = np.clip(mutant, self.problem.bound_min, self.problem.bound_max)
        return offspring

    def run(self):
        self.initialize()
        # Comentamos los prints para no saturar la consola en ejecuciones múltiples
        # print(f"Iniciando evolución por {self.max_gen} generaciones...")
        
        for gen in range(self.max_gen):
            for i in range(self.N): 
                offspring_x = self.evolution_operator_DE(i)
                offspring_fit = self.problem.evaluate(offspring_x)
                
                self.z_ideal = np.min(np.vstack((self.z_ideal, offspring_fit)), axis=0)
                
                neighbors = self.neighborhood[i]
                for j in neighbors:
                    g_te_offspring = self.tchebycheff(offspring_fit, j)
                    g_te_neighbor = self.tchebycheff(self.fitness_pop[j], j)
                    
                    if g_te_offspring <= g_te_neighbor:
                        self.population[j] = offspring_x
                        self.fitness_pop[j] = offspring_fit
            
        return self.fitness_pop

# =============================================================================
# BLOQUE PRINCIPAL MODIFICADO PARA GENERAR ARCHIVOS COMPATIBLES CON METRICS
# =============================================================================
if __name__ == "__main__":
    # Definimos las dos configuraciones que pide la práctica para comparar
    # (Población, Generaciones)
    configuraciones = [
        (40, 100),   # Escenario 4000 evaluaciones (carpeta P40G100)
        (100, 100)   # Escenario 10000 evaluaciones (carpeta P100G100)
    ]
    
    # Número de ejecuciones independientes (semillas)
    num_ejecuciones = 10
    
    # Crear carpeta para guardar resultados si no existe
    output_dir = "RESULTADOS_MOEAD"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zdt3_problem = ZDT3(n_vars=30)

    print(f"--- Iniciando Batería de Pruebas MOEA/D ---")

    for pop_size, generations in configuraciones:
        print(f"\nProcesando Configuración: Población={pop_size}, Generaciones={generations}")
        
        for i in range(1, num_ejecuciones + 1):
            semilla = i
            np.random.seed(semilla) # Fijar semilla para reproducibilidad
            
            # Instanciar y ejecutar algoritmo
            # Ajustamos n_neighbors proporcionalmente si la población es muy pequeña (opcional)
            n_vecinos = 20 if pop_size >= 20 else int(pop_size/2)
            
            algorithm = MOEAD(problem=zdt3_problem, 
                              n_pop=pop_size, 
                              n_neighbors=n_vecinos, 
                              max_gen=generations)
            
            final_front = algorithm.run()
            
            # --- PREPARACIÓN FORMATO PARA ./metrics ---
            # El software metrics suele requerir: f1 f2 constraint_violation
            # En ZDT3 no hay constraints, así que la 3ra columna es 0.0
            n_soluciones = final_front.shape[0]
            col_zeros = np.zeros((n_soluciones, 1))
            datos_metrics = np.hstack((final_front, col_zeros))
            
            # Nombre de archivo similar al de NSGA-II para facilitar scripts
            # Formato: zdt3_final_moead_P{pop}G{gen}_seed{seed}.out
            filename = f"zdt3_final_moead_P{pop_size}G{generations}_seed{semilla:02d}.out"
            filepath = os.path.join(output_dir, filename)
            
            # Guardar SIN encabezado y con separador de espacio/tabulador
            np.savetxt(filepath, datos_metrics, fmt='%.6e', delimiter='\t')
            
            print(f"  > Ejecución {i}/{num_ejecuciones} terminada. Guardado: {filename}")

    print(f"\n[FIN] Todos los archivos generados en la carpeta '{output_dir}'.")
    print("Ahora copia estos archivos a tu carpeta de ./metrics en Linux para compararlos.")
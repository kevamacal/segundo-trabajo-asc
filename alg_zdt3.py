import numpy as np

import matplotlib.pyplot as plt



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

        """

        Calcula el valor escalar g_te según

        max_i { lambda_i * | f_i(x) - z_i* | }

        """

        diff = np.abs(fitness - self.z_ideal)

        weighted_diff = self.weights[weight_idx] * diff

        return np.max(weighted_diff)



    def evolution_operator_DE(self, idx_subproblem):

        neighbors = self.neighborhood[idx_subproblem]

        r1, r2, r3 = np.random.choice(neighbors, 3, replace=False)

        

        x_current = self.population[idx_subproblem]

        x_r1 = self.population[r1]

        x_r2 = self.population[r2]

        x_r3 = self.population[r3]

        

        mutant = x_r1 + self.F * (x_r2 - x_r3)

        

        n_vars = self.problem.n_vars

        crossover_mask = np.random.rand(n_vars) <= self.CR

        

        j_rand = np.random.randint(n_vars)

        crossover_mask[j_rand] = True

        

        offspring = np.where(crossover_mask, mutant, x_current)



        pr = 1.0 / n_vars

        sigma_idx = 20.0

        

        for j in range(n_vars):

            if np.random.rand() < pr:

                sigma = (self.problem.bound_max[j] - self.problem.bound_min[j]) / sigma_idx

                offspring[j] += np.random.normal(0, sigma)



        offspring = np.clip(offspring, self.problem.bound_min, self.problem.bound_max)

        

        return offspring



    def run(self):

        self.initialize()

        

        print(f"Iniciando evolución por {self.max_gen} generaciones...")

        

        # ELIMINADO: replaced_count = 0 (No es necesario y causaba el error)

        

        for gen in range(self.max_gen):

            for i in range(self.N):

                    

                offspring_x = self.evolution_operator_DE(i)

                offspring_fit = self.problem.evaluate(offspring_x)

                

                # Actualizar Z ideal (Slide 5, paso 3) [cite: 59-60]

                self.z_ideal = np.min(np.vstack((self.z_ideal, offspring_fit)), axis=0)

                

                neighbors = self.neighborhood[i]

                

                # Slide 5, paso 4: Actualización de vecinos (revisar todos) 

                for j in neighbors:

                    g_te_offspring = self.tchebycheff(offspring_fit, j)

                    g_te_neighbor = self.tchebycheff(self.fitness_pop[j], j)

                    

                    if g_te_offspring <= g_te_neighbor:

                        self.population[j] = offspring_x

                        self.fitness_pop[j] = offspring_fit

            

            if gen % 20 == 0:

                print(f"Generación {gen}/{self.max_gen} completada.")



        return self.fitness_pop



# --- AÑADE ESTA FUNCIÓN ANTES DEL MAIN O DENTRO DE LA CLASE ---

def filtrar_no_dominadas(poblacion_fitness):

    """

    Filtra la población para quedarse solo con el Frente de Pareto aproximado (EP).

    Esto elimina los puntos 'sucios' que quedan en los huecos de ZDT3.

    """

    tamaño = poblacion_fitness.shape[0]

    es_dominada = np.zeros(tamaño, dtype=bool)

    

    for i in range(tamaño):

        for j in range(tamaño):

            if i == j: continue

            # Si J domina a I (mejor o igual en todo, y estrictamente mejor en algo)

            if all(poblacion_fitness[j] <= poblacion_fitness[i]) and any(poblacion_fitness[j] < poblacion_fitness[i]):

                es_dominada[i] = True

                break

                

    return poblacion_fitness[~es_dominada]



if __name__ == "__main__":

    zdt3_problem = ZDT3(n_vars=30)

    

    pop_size = 100

    generations = 100 #comendado 100 o 250 para ver bien la convergencia



    print(f"Configuración: Población={pop_size}, Generaciones={generations}")

    

    algorithm = MOEAD(problem=zdt3_problem, n_pop=pop_size, n_neighbors=20, max_gen=generations)

    

    # 1. Ejecutamos el algoritmo

    poblacion_final = algorithm.run()

    

    # 2. [[PASO CRÍTICO]] Filtramos las soluciones para obtener el Frente Limpio

    frente_limpio = filtrar_no_dominadas(poblacion_final)

    

    print(f"Soluciones totales: {len(poblacion_final)}")

    print(f"Soluciones en Frente Pareto (Filtradas): {len(frente_limpio)}")



    # 3. Guardamos SOLO el frente limpio para pasarle al programa de métricas

    filename_data = f"MOEAD_ZDT3_GEN_{generations}.txt"

    np.savetxt(filename_data, frente_limpio, fmt='%.6f', delimiter='\t')

    print(f"\n[OK] Datos filtrados guardados en: {filename_data}")



    # 4. Generamos la Gráfica Comparativa (Gris = Población, Rojo = Frente Final)

    plt.figure(figsize=(10, 8))

    

    # Pintamos en gris suave la población original (para ver el trabajo del algoritmo)

    plt.scatter(poblacion_final[:, 0], poblacion_final[:, 1], c='lightgray', s=10, label='Población (Subproblemas)')

    

    # Pintamos en ROJO intenso solo las soluciones no dominadas (Tu resultado final)

    plt.scatter(frente_limpio[:, 0], frente_limpio[:, 1], c='red', s=20, label='Frente Pareto (EP)')

    

    plt.title(f'Frente de Pareto ZDT3 - {generations} Generaciones')

    plt.xlabel('f1')

    plt.ylabel('f2')

    plt.grid(True)

    plt.legend()

    

    filename_plot = f"Grafica_ZDT3_GEN_{generations}.png"

    plt.savefig(filename_plot)

    print(f"[OK] Gráfico guardado correctamente en: {filename_plot}")

    print("Proceso finalizado.")


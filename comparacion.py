import numpy as np
import matplotlib.pyplot as plt
import os

def plot_comparison(result_file, true_front_file='ZDT3_EP.txt'):
    # 1. Cargar el Frente de Pareto Verdadero (Referencia)
    if os.path.exists(true_front_file):
        true_front = np.loadtxt(true_front_file)
        # ZDT3_EP.txt suele tener 2 columnas. Si tiene más, ajustamos.
        print(f"Cargado Frente Verdadero: {len(true_front)} puntos.")
    else:
        print(f"ERROR: No encuentro {true_front_file}. Asegúrate de que esté en la carpeta.")
        return

    # 2. Cargar TUS resultados (El archivo .out o .txt que generó tu algoritmo)
    if os.path.exists(result_file):
        my_front = np.loadtxt(result_file)
        print(f"Cargado Tu Frente: {len(my_front)} puntos.")
    else:
        print(f"ERROR: No encuentro tu archivo de resultados: {result_file}")
        return

    # 3. Graficar
    plt.figure(figsize=(10, 8))
    
    # Pintar el Frente Verdadero (Línea o puntos grises/negros de fondo)
    plt.scatter(true_front[:, 0], true_front[:, 1], 
                c='black', s=1, alpha=0.3, label='Frente Verdadero (ZDT3_EP)')

    # Pintar TUS soluciones (Puntos rojos más grandes)
    plt.scatter(my_front[:, 0], my_front[:, 1], 
                c='red', s=20, edgecolors='darkred', label='Tu Algoritmo (MOEA/D)')

    plt.title('Comparativa: Tu MOEA/D vs Solución Perfecta')
    plt.xlabel('f1 (Minimizar)')
    plt.ylabel('f2 (Minimizar)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_img = "Comparativa_ZDT3.png"
    plt.savefig(output_img)
    print(f"\n[OK] Gráfico guardado como: {output_img}")
    plt.show()

# --- CAMBIA ESTO POR LA RUTA DE TU MEJOR ARCHIVO .OUT ---
# Ejemplo basado en tus carpetas subidas:
archivo_resultado = "RESULTADOS_MOEAD_MEJORADO/zdt3_final_moead_P100G100_seed01.out" 

# Asegúrate de que ZDT3_EP.txt esté en la misma carpeta o pon la ruta completa
plot_comparison(archivo_resultado, true_front_file="ZDT3_EP.txt")
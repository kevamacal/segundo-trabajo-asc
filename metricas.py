import numpy as np
from scipy.spatial.distance import cdist

def calcular_metricas(archivo_resultado, archivo_frente_real='ZDT3_EP.txt'):
    # Cargar datos
    try:
        front_aprox = np.loadtxt(archivo_resultado)
        front_true = np.loadtxt(archivo_frente_real)
    except Exception as e:
        print(f"Error cargando archivos: {e}")
        return

    # 1. Métrica IGD (Inverted Generational Distance)
    # Mide cuánto se acerca tu frente al real y qué tan bien lo cubre.
    # ¡Cuanto MENOR sea el valor, MEJOR! (0 es perfecto)
    
    # Para cada punto del frente REAL, buscamos el más cercano en TU frente
    distancias = cdist(front_true, front_aprox)
    min_distancias = np.min(distancias, axis=1)
    igd = np.mean(min_distancias)
    
    print(f"--- ANÁLISIS DE CALIDAD ---")
    print(f"Archivo analizado: {archivo_resultado}")
    print(f"Puntos en tu frente: {len(front_aprox)}")
    print(f"IGD (Calidad global): {igd:.6f}")
    
    # Interpretación rápida para ZDT3
    if igd < 0.005:
        print(">> Resultado: EXCELENTE (Muy convergido y diverso)")
    elif igd < 0.01:
        print(">> Resultado: BUENO (Aceptable para aprobar)")
    else:
        print(">> Resultado: MEJORABLE (Falta convergencia o diversidad)")

# CAMBIA ESTO por tu mejor archivo .out generado
mi_mejor_archivo = "zdt3_final_moead_P100G100_seed01.out" 

calcular_metricas(mi_mejor_archivo)
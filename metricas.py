import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist

# --- CONFIGURACIÓN ---
NSGA2_FILE = "FUN.txt"  # Archivo de salida NSGA-II (ajusta el nombre si es diferente)
MOEAD_FILE = "eval_moead.out"  # Archivo de salida MOEA/D

# Frente de Pareto verdadero de ZDT3 (para calcular métricas)
def generar_frente_verdadero_zdt3(n_points=500):
    """Genera el frente de Pareto verdadero de ZDT3"""
    # ZDT3 tiene regiones discontinuas
    regiones = [
        (0.0, 0.0830015349),
        (0.1822287280, 0.2577623634),
        (0.4093136748, 0.4538821041),
        (0.6183967944, 0.6525117038),
        (0.8233317983, 0.8518328654)
    ]
    
    pf = []
    points_per_region = n_points // len(regiones)
    
    for x_min, x_max in regiones:
        x = np.linspace(x_min, x_max, points_per_region)
        g = 1.0
        h = 1 - np.sqrt(x / g) - (x / g) * np.sin(10 * np.pi * x)
        y = g * h
        
        for i in range(len(x)):
            pf.append([x[i], y[i]])
    
    return np.array(pf)

def cargar_frente(filename):
    """Carga un frente de Pareto desde archivo"""
    if not os.path.exists(filename):
        print(f"⚠️  Archivo {filename} no encontrado")
        return None
    
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    f1 = float(parts[0])
                    f2 = float(parts[1])
                    data.append([f1, f2])
                except ValueError:
                    continue
    
    return np.array(data) if data else None

def calcular_hipervolumen_2d(frente, ref_point):
    """Calcula el hipervolumen para problemas 2D"""
    if frente is None or len(frente) == 0:
        return 0.0
    
    # Ordenar por f1
    frente_sorted = frente[frente[:, 0].argsort()]
    
    # Filtrar soluciones dominadas (simplificado)
    no_dominado = []
    for sol in frente_sorted:
        dominado = False
        for other in no_dominado:
            if other[1] <= sol[1]:  # other domina a sol en f2
                dominado = True
                break
        if not dominado:
            no_dominado.append(sol)
    
    frente_sorted = np.array(no_dominado)
    
    # Calcular hipervolumen
    hv = 0.0
    for i in range(len(frente_sorted)):
        if i == 0:
            width = frente_sorted[i][0]
        else:
            width = frente_sorted[i][0] - frente_sorted[i-1][0]
        
        height = ref_point[1] - frente_sorted[i][1]
        if height > 0:
            hv += width * height
    
    # Último rectángulo
    width = ref_point[0] - frente_sorted[-1][0]
    height = ref_point[1] - frente_sorted[-1][1]
    if width > 0 and height > 0:
        hv += width * height
    
    return hv

def calcular_igd(frente_aproximado, frente_verdadero):
    """Calcula la métrica IGD (Inverted Generational Distance)"""
    if frente_aproximado is None or len(frente_aproximado) == 0:
        return float('inf')
    
    # Distancia de cada punto del frente verdadero al más cercano del aproximado
    distancias = cdist(frente_verdadero, frente_aproximado, metric='euclidean')
    min_distancias = np.min(distancias, axis=1)
    
    return np.mean(min_distancias)

def calcular_gd(frente_aproximado, frente_verdadero):
    """Calcula la métrica GD (Generational Distance)"""
    if frente_aproximado is None or len(frente_aproximado) == 0:
        return float('inf')
    
    # Distancia de cada punto del frente aproximado al más cercano del verdadero
    distancias = cdist(frente_aproximado, frente_verdadero, metric='euclidean')
    min_distancias = np.min(distancias, axis=1)
    
    return np.mean(min_distancias)

def calcular_spacing(frente):
    """Calcula la métrica Spacing (uniformidad de distribución)"""
    if frente is None or len(frente) < 2:
        return 0.0
    
    # Distancia al vecino más cercano para cada solución
    distancias = cdist(frente, frente, metric='euclidean')
    np.fill_diagonal(distancias, np.inf)
    min_distancias = np.min(distancias, axis=1)
    
    # Spacing es la desviación estándar de estas distancias
    mean_dist = np.mean(min_distancias)
    spacing = np.sqrt(np.mean((min_distancias - mean_dist)**2))
    
    return spacing

def calcular_metricas(frente_nsga2, frente_moead, frente_verdadero, ref_point):
    """Calcula todas las métricas de comparación"""
    metricas = {}
    
    # Hipervolumen
    metricas['HV_NSGA2'] = calcular_hipervolumen_2d(frente_nsga2, ref_point) if frente_nsga2 is not None else 0
    metricas['HV_MOEAD'] = calcular_hipervolumen_2d(frente_moead, ref_point) if frente_moead is not None else 0
    
    # IGD
    metricas['IGD_NSGA2'] = calcular_igd(frente_nsga2, frente_verdadero) if frente_nsga2 is not None else float('inf')
    metricas['IGD_MOEAD'] = calcular_igd(frente_moead, frente_verdadero) if frente_moead is not None else float('inf')
    
    # GD
    metricas['GD_NSGA2'] = calcular_gd(frente_nsga2, frente_verdadero) if frente_nsga2 is not None else float('inf')
    metricas['GD_MOEAD'] = calcular_gd(frente_moead, frente_verdadero) if frente_moead is not None else float('inf')
    
    # Spacing
    metricas['Spacing_NSGA2'] = calcular_spacing(frente_nsga2) if frente_nsga2 is not None else 0
    metricas['Spacing_MOEAD'] = calcular_spacing(frente_moead) if frente_moead is not None else 0
    
    # Número de soluciones
    metricas['N_NSGA2'] = len(frente_nsga2) if frente_nsga2 is not None else 0
    metricas['N_MOEAD'] = len(frente_moead) if frente_moead is not None else 0
    
    return metricas

def crear_graficos(frente_nsga2, frente_moead, frente_verdadero, metricas):
    """Crea todos los gráficos de comparación"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # --- GRÁFICO 1: Comparación de Frentes ---
    ax1 = plt.subplot(2, 3, 1)
    if frente_verdadero is not None:
        ax1.plot(frente_verdadero[:, 0], frente_verdadero[:, 1], 
                'k-', linewidth=2, label='Frente Verdadero', alpha=0.5)
    if frente_nsga2 is not None:
        ax1.scatter(frente_nsga2[:, 0], frente_nsga2[:, 1], 
                   c='blue', marker='o', s=30, label=f'NSGA-II (n={metricas["N_NSGA2"]})', alpha=0.7)
    if frente_moead is not None:
        ax1.scatter(frente_moead[:, 0], frente_moead[:, 1], 
                   c='red', marker='s', s=30, label=f'MOEA/D (n={metricas["N_MOEAD"]})', alpha=0.7)
    ax1.set_xlabel('f1', fontsize=12)
    ax1.set_ylabel('f2', fontsize=12)
    ax1.set_title('Comparación de Frentes de Pareto', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- GRÁFICO 2: Hipervolumen ---
    ax2 = plt.subplot(2, 3, 2)
    algoritmos = ['NSGA-II', 'MOEA/D']
    hvs = [metricas['HV_NSGA2'], metricas['HV_MOEAD']]
    colors = ['blue', 'red']
    bars = ax2.bar(algoritmos, hvs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Hipervolumen', fontsize=12)
    ax2.set_title('Hipervolumen (Mayor es Mejor)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    # Añadir valores en las barras
    for bar, val in zip(bars, hvs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # --- GRÁFICO 3: IGD ---
    ax3 = plt.subplot(2, 3, 3)
    igds = [metricas['IGD_NSGA2'], metricas['IGD_MOEAD']]
    bars = ax3.bar(algoritmos, igds, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('IGD', fontsize=12)
    ax3.set_title('IGD - Inverted Generational Distance (Menor es Mejor)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, igds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # --- GRÁFICO 4: GD ---
    ax4 = plt.subplot(2, 3, 4)
    gds = [metricas['GD_NSGA2'], metricas['GD_MOEAD']]
    bars = ax4.bar(algoritmos, gds, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('GD', fontsize=12)
    ax4.set_title('GD - Generational Distance (Menor es Mejor)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, gds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # --- GRÁFICO 5: Spacing ---
    ax5 = plt.subplot(2, 3, 5)
    spacings = [metricas['Spacing_NSGA2'], metricas['Spacing_MOEAD']]
    bars = ax5.bar(algoritmos, spacings, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Spacing', fontsize=12)
    ax5.set_title('Spacing - Uniformidad (Menor es Mejor)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, spacings):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # --- GRÁFICO 6: Tabla Resumen ---
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Determinar ganador de cada métrica
    ganador_hv = 'NSGA-II' if metricas['HV_NSGA2'] > metricas['HV_MOEAD'] else 'MOEA/D'
    ganador_igd = 'NSGA-II' if metricas['IGD_NSGA2'] < metricas['IGD_MOEAD'] else 'MOEA/D'
    ganador_gd = 'NSGA-II' if metricas['GD_NSGA2'] < metricas['GD_MOEAD'] else 'MOEA/D'
    ganador_spacing = 'NSGA-II' if metricas['Spacing_NSGA2'] < metricas['Spacing_MOEAD'] else 'MOEA/D'
    
    tabla_data = [
        ['Métrica', 'NSGA-II', 'MOEA/D', 'Ganador'],
        ['Hipervolumen ↑', f'{metricas["HV_NSGA2"]:.4f}', f'{metricas["HV_MOEAD"]:.4f}', ganador_hv],
        ['IGD ↓', f'{metricas["IGD_NSGA2"]:.6f}', f'{metricas["IGD_MOEAD"]:.6f}', ganador_igd],
        ['GD ↓', f'{metricas["GD_NSGA2"]:.6f}', f'{metricas["GD_MOEAD"]:.6f}', ganador_gd],
        ['Spacing ↓', f'{metricas["Spacing_NSGA2"]:.6f}', f'{metricas["Spacing_MOEAD"]:.6f}', ganador_spacing],
        ['# Soluciones', f'{metricas["N_NSGA2"]}', f'{metricas["N_MOEAD"]}', '-']
    ]
    
    table = ax6.table(cellText=tabla_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.25, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Colorear header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear columna de ganadores
    colors_ganador = {'NSGA-II': '#2196F3', 'MOEA/D': '#F44336'}
    for i in range(1, 5):
        ganador = tabla_data[i][3]
        if ganador in colors_ganador:
            table[(i, 3)].set_facecolor(colors_ganador[ganador])
            table[(i, 3)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Resumen de Métricas', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('Comparacion_NSGA2_vs_MOEAD.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico guardado: Comparacion_NSGA2_vs_MOEAD.png")
    plt.show()

def main():
    print("=" * 70)
    print("COMPARACIÓN NSGA-II vs MOEA/D - ZDT3")
    print("=" * 70)
    
    # Cargar frentes
    print("\n📂 Cargando frentes de Pareto...")
    frente_nsga2 = cargar_frente(NSGA2_FILE)
    frente_moead = cargar_frente(MOEAD_FILE)
    frente_verdadero = generar_frente_verdadero_zdt3()
    
    if frente_nsga2 is None and frente_moead is None:
        print("❌ Error: No se pudieron cargar ninguno de los archivos.")
        print(f"   Verifica que existan: {NSGA2_FILE} y/o {MOEAD_FILE}")
        return
    
    if frente_nsga2 is not None:
        print(f"✅ NSGA-II cargado: {len(frente_nsga2)} soluciones")
    if frente_moead is not None:
        print(f"✅ MOEA/D cargado: {len(frente_moead)} soluciones")
    
    # Punto de referencia para hipervolumen (peor caso para ZDT3)
    ref_point = np.array([1.2, 1.2])
    
    # Calcular métricas
    print("\n📊 Calculando métricas...")
    metricas = calcular_metricas(frente_nsga2, frente_moead, frente_verdadero, ref_point)
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS DE LAS MÉTRICAS")
    print("=" * 70)
    
    print("\n🔹 HIPERVOLUMEN (Mayor es mejor):")
    print(f"   NSGA-II: {metricas['HV_NSGA2']:.6f}")
    print(f"   MOEA/D:  {metricas['HV_MOEAD']:.6f}")
    if metricas['HV_NSGA2'] > metricas['HV_MOEAD']:
        print("   ✨ Ganador: NSGA-II")
    else:
        print("   ✨ Ganador: MOEA/D")
    
    print("\n🔹 IGD - Inverted Generational Distance (Menor es mejor):")
    print(f"   NSGA-II: {metricas['IGD_NSGA2']:.8f}")
    print(f"   MOEA/D:  {metricas['IGD_MOEAD']:.8f}")
    if metricas['IGD_NSGA2'] < metricas['IGD_MOEAD']:
        print("   ✨ Ganador: NSGA-II")
    else:
        print("   ✨ Ganador: MOEA/D")
    
    print("\n🔹 GD - Generational Distance (Menor es mejor):")
    print(f"   NSGA-II: {metricas['GD_NSGA2']:.8f}")
    print(f"   MOEA/D:  {metricas['GD_MOEAD']:.8f}")
    if metricas['GD_NSGA2'] < metricas['GD_MOEAD']:
        print("   ✨ Ganador: NSGA-II")
    else:
        print("   ✨ Ganador: MOEA/D")
    
    print("\n🔹 SPACING - Uniformidad (Menor es mejor):")
    print(f"   NSGA-II: {metricas['Spacing_NSGA2']:.8f}")
    print(f"   MOEA/D:  {metricas['Spacing_MOEAD']:.8f}")
    if metricas['Spacing_NSGA2'] < metricas['Spacing_MOEAD']:
        print("   ✨ Ganador: NSGA-II")
    else:
        print("   ✨ Ganador: MOEA/D")
    
    print("\n🔹 NÚMERO DE SOLUCIONES:")
    print(f"   NSGA-II: {metricas['N_NSGA2']}")
    print(f"   MOEA/D:  {metricas['N_MOEAD']}")
    
    # Crear gráficos
    print("\n📈 Generando gráficos comparativos...")
    crear_graficos(frente_nsga2, frente_moead, frente_verdadero, metricas)
    
    # Guardar métricas en archivo
    print("\n💾 Guardando métricas en archivo...")
    with open('metricas_comparacion.txt', 'w') as f:
        f.write("COMPARACIÓN NSGA-II vs MOEA/D - ZDT3\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Hipervolumen (Mayor es mejor):\n")
        f.write(f"  NSGA-II: {metricas['HV_NSGA2']:.6f}\n")
        f.write(f"  MOEA/D:  {metricas['HV_MOEAD']:.6f}\n\n")
        f.write(f"IGD (Menor es mejor):\n")
        f.write(f"  NSGA-II: {metricas['IGD_NSGA2']:.8f}\n")
        f.write(f"  MOEA/D:  {metricas['IGD_MOEAD']:.8f}\n\n")
        f.write(f"GD (Menor es mejor):\n")
        f.write(f"  NSGA-II: {metricas['GD_NSGA2']:.8f}\n")
        f.write(f"  MOEA/D:  {metricas['GD_MOEAD']:.8f}\n\n")
        f.write(f"Spacing (Menor es mejor):\n")
        f.write(f"  NSGA-II: {metricas['Spacing_NSGA2']:.8f}\n")
        f.write(f"  MOEA/D:  {metricas['Spacing_MOEAD']:.8f}\n\n")
        f.write(f"Número de soluciones:\n")
        f.write(f"  NSGA-II: {metricas['N_NSGA2']}\n")
        f.write(f"  MOEA/D:  {metricas['N_MOEAD']}\n")
    
    print("✅ Métricas guardadas en: metricas_comparacion.txt")
    
    print("\n" + "=" * 70)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    main()
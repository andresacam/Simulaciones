import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# --- Configuración de la Página (Mejor práctica) ---
# Esto debe ser el primer comando de Streamlit en tu script.
st.set_page_config(layout="wide", page_title="Simulador de Costos de Proyecto")

if 'iteraciones' not in st.session_state:
    st.session_state.iteraciones = 1000  # Valor por defecto


# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Menu</h1>", unsafe_allow_html=True)
    st.session_state.iteraciones = st.number_input("Numero de Iteraciones", min_value=1, max_value=10000, value=st.session_state.iteraciones, step=1)
    
    if st.button("Restablecer Items", type="primary", use_container_width=True):
         st.session_state.data = []
         # Limpiar también los resultados de la simulación si existen
         if 'simulation_run' in st.session_state:
             del st.session_state.simulation_run
         if 'results' in st.session_state:
             del st.session_state.results
         if 'df_simulation_results' in st.session_state:
             del st.session_state.df_simulation_results
         st.success("Items reseteados!")

def pert_distribution(min_val, probable_val, max_val, size=1):
    """
    Genera valores aleatorios siguiendo una distribución PERT.
    Utiliza un valor fijo de gamma = 4, que es el estándar más común.
    Esta distribución es una variante de la distribución Beta, escalada al rango [min_val, max_val].
    """
    # Validación de seguridad: si el rango es cero o negativo, todos los valores son el mismo.
    if max_val <= min_val:
        return np.full(size, min_val)

    gamma = 4.0  # Valor estándar y fijo para gamma

    # Calcular la media (mu) de la distribución PERT
    mu = (min_val + gamma * probable_val + max_val) / (gamma + 2)

    # Calcular los parámetros alpha y beta para la distribución Beta subyacente.
    # Se añaden salvaguardas para evitar la división por cero si mu coincide con los extremos.
    if mu == probable_val:
        # Caso especial simétrico, evita división por cero
        alpha = 1 + gamma / 2
    else:
        # Fórmula general para alpha
        alpha = ((mu - min_val) * (2 * probable_val - min_val - max_val)) / ((probable_val - mu) * (max_val - min_val))

    # Beta se deriva de alpha y debe ser positivo
    beta = max(alpha * (max_val - mu) / (mu - min_val), 1e-9)
    alpha = max(alpha, 1e-9) # Asegurarse de que alpha también sea positivo

    # Generar valores aleatorios de la distribución Beta en el rango [0, 1]
    beta_samples = np.random.beta(alpha, beta, size)

    # Escalar los valores al rango [min_val, max_val]
    return min_val + beta_samples * (max_val - min_val)

def run_monte_carlo_simulation(data, iterations):
    """
    Ejecuta una simulación de Monte Carlo basada en una distribución PERT.
    """
    simulation_results = {}
    for index, row in data.iterrows():
        item_name = row['Nombre del Item']
        min_val = row['Valor Mínimo']
        probable_val = row['Valor Probable']
        max_val = row['Valor Máximo']
        
        # Usar la nueva función de distribución PERT
        simulation_results[item_name] = pert_distribution(min_val, probable_val, max_val, iterations)
        
    df_simulation = pd.DataFrame(simulation_results)
    total_cost_per_iteration = df_simulation.sum(axis=1)
    
    # Guardar ambos resultados en el estado de la sesión
    st.session_state.df_simulation_results = df_simulation
    st.session_state.results = total_cost_per_iteration
    st.session_state.simulation_run = True


# Main content
st.title("Calculá el costo de tu Proyecto")
st.write("Agregá los items de tu proyecto y calculá el costo total usando una **Simulación de Monte Carlo con distribución PERT**.")
st.write("Cuando hayas terminado, guarda el Dataset para comenzar a simular.")

# Initialize session state for DataFrame
if 'data' not in st.session_state:
    st.session_state.data = []

# Use a form for better performance and user experience
with st.form("item_form", clear_on_submit=True):
    # Input fields
    item_name = st.text_input("Nombre del Item", placeholder="Ej: Diseño de UI/UX")
    valor_minimo = st.number_input("Valor Mínimo", value=None, min_value=0.0, format="%.2f", placeholder="Ej: 1000.00")
    valor_probable = st.number_input("Valor Probable", value=None, min_value=0.0, format="%.2f", placeholder="Ej: 1500.00")
    valor_maximo = st.number_input("Valor Máximo", value=None, min_value=0.0, format="%.2f", placeholder="Ej: 2500.00")

    # Add data to session state
    submitted = st.form_submit_button("Agregar")
    if submitted:
        # --- Validation ---
        if not item_name or valor_minimo is None or valor_probable is None or valor_maximo is None:
            st.warning("Por favor, completa todos los campos para agregar el item.")
        elif not (valor_minimo <= valor_probable <= valor_maximo):
            st.warning("Error en los valores: Asegúrate de que Mínimo <= Probable <= Máximo.")
        else:
            st.session_state.data.append({
                'Nombre del Item': item_name,
                'Valor Mínimo': valor_minimo,
                'Valor Probable': valor_probable,
                'Valor Máximo': valor_maximo
            })

# Display the DataFrame
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    st.write("Items Agregados:")
    st.dataframe(df)

# Botón para guardar y ejecutar la simulación
if st.button("Guardar y Simular", type="primary", use_container_width=True):
    if not st.session_state.data:
        st.error("No hay items para guardar. Por favor, agrega al menos un item.")
    else:
        df = pd.DataFrame(st.session_state.data)
        st.session_state.dfg = df  # Guarda el DataFrame en session_state
        with st.spinner(f"Ejecutando {st.session_state.iteraciones:,} simulaciones..."):
            run_monte_carlo_simulation(df, st.session_state.iteraciones)
        st.success("¡Simulación completada!")

st.divider()

# Muestra los resultados si la simulación ya se ha ejecutado
if st.session_state.get('simulation_run', False):
    results = st.session_state.results
    st.subheader("📊 Resultados de la Simulación (Distribución PERT)")

    stats = results.describe(percentiles=[.10, .25, .50, .75, .90])
    costo_probable = stats['50%']
    costo_promedio = stats['mean']
    costo_p90 = stats['90%']

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Costo Más Probable (P50)", value=f"${costo_probable:,.2f}")
    col2.metric(label="Costo Promedio", value=f"${costo_promedio:,.2f}")
    col3.metric(label="Costo con 90% Confianza (P90)", value=f"${costo_p90:,.2f}", 
                help="Hay un 90% de probabilidad de que el costo del proyecto sea menor o igual a este valor.")

    st.subheader("Distribución de Probabilidad Acumulada (Curva S)")
    
    # Preparar datos para la Curva S
    sorted_results = np.sort(results)
    cumulative_prob = np.arange(1, len(sorted_results) + 1) / len(sorted_results)

    fig, ax = plt.subplots()
    ax.plot(sorted_results, cumulative_prob, color='#FF2400', linewidth=2.5, label="Curva S de Costos")

    # Líneas de referencia para P50 y P90
    ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=1)
    ax.axvline(x=costo_probable, color='grey', linestyle='--', linewidth=1)
    ax.plot(costo_probable, 0.5, 'o', color='#1f77b4', markersize=8, label=f"P50 (Probable): ${costo_probable:,.2f}")

    ax.axhline(y=0.9, color='grey', linestyle='--', linewidth=1)
    ax.axvline(x=costo_p90, color='grey', linestyle='--', linewidth=1)
    ax.plot(costo_p90, 0.9, 'o', color='#ff7f0e', markersize=8, label=f"P90 (Confianza 90%): ${costo_p90:,.2f}")

    ax.set_xlabel("Costo Total del Proyecto ($)")
    ax.set_ylabel("Probabilidad Acumulada")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.05) # Añadir un poco de espacio en la parte superior
    st.pyplot(fig)

    with st.expander("Ver Estadísticas Detalladas"):
        st.dataframe(stats.to_frame(name='Valor').style.format("${:,.2f}"))

    st.divider()

    # --- Gráfico de Contribución por Item ---
    st.subheader("🔎 Contribución de Costos por Item")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        # Obtener el DataFrame con los resultados de la simulación por item
        df_simulation = st.session_state.df_simulation_results
        
        # Calcular el costo promedio para cada item y mostrarlo en una tabla
        item_avg_costs = df_simulation.mean().sort_values(ascending=False)
        st.write("Costo Promedio por Item:")
        st.dataframe(item_avg_costs.to_frame(name='Costo Promedio').style.format("${:,.2f}"))

    with col2:
        # Crear el gráfico de torta (estilo dona)
        fig2, ax2 = plt.subplots()
        ax2.pie(item_avg_costs, labels=item_avg_costs.index, autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.4, edgecolor='w')) # El borde blanco se ve bien en tema oscuro
        ax2.axis('equal')  # Asegura que el gráfico sea un círculo.
        
        st.pyplot(fig2)

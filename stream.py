import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

def run_monte_carlo_simulation(data, iterations):
    """
    Ejecuta una simulación de Monte Carlo basada en una distribución triangular.
    """
    simulation_results = {}
    for index, row in data.iterrows():
        item_name = row['Nombre del Item']
        min_val = row['Valor Mínimo']
        probable_val = row['Valor Probable']
        max_val = row['Valor Máximo']
        
        simulation_results[item_name] = np.random.triangular(min_val, probable_val, max_val, iterations)
        
    df_simulation = pd.DataFrame(simulation_results)
    total_cost_per_iteration = df_simulation.sum(axis=1)
    
    # Guardar ambos resultados en el estado de la sesión
    st.session_state.df_simulation_results = df_simulation
    st.session_state.results = total_cost_per_iteration
    st.session_state.simulation_run = True


# Main content
st.title("Calculá el costo de tu Proyecto")
st.write("Agregá los items de tu proyecto y calculá el costo total.")
st.write("Cuando hayas terminado guarda el Dataset para comenzar a simular.")

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
    st.subheader("📊 Resultados de la Simulación")

    stats = results.describe(percentiles=[.10, .25, .50, .75, .90])
    costo_probable = stats['50%']
    costo_promedio = stats['mean']
    costo_p90 = stats['90%']

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Costo Más Probable (P50)", value=f"${costo_probable:,.2f}")
    col2.metric(label="Costo Promedio", value=f"${costo_promedio:,.2f}")
    col3.metric(label="Costo con 90% Confianza (P90)", value=f"${costo_p90:,.2f}", 
                help="Hay un 90% de probabilidad de que el costo del proyecto sea menor o igual a este valor.")

    st.subheader("Distribución de Costos del Proyecto")
    fig, ax = plt.subplots()
    ax.hist(results, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(costo_promedio, color='red', linestyle='--', linewidth=2, label=f"Promedio: ${costo_promedio:,.2f}")
    ax.axvline(costo_probable, color='green', linestyle='--', linewidth=2, label=f"P50: ${costo_probable:,.2f}")
    ax.set_xlabel("Costo Total del Proyecto ($)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    with st.expander("Ver Estadísticas Detalladas"):
        st.dataframe(stats.to_frame(name='Valor').style.format("${:,.2f}"))

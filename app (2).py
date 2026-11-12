import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(page_title="Segmentador de Clientes AI", layout="centered")

st.title("🎯 Segmentación de Clientes Inteligente - ACA II")
st.write("Clasificación en tiempo real utilizando Redes Neuronales Artificiales")

# Cargar componentes de preprocesamiento y el modelo
@st.cache_resource
def cargar_recursos():
    componentes = joblib.load('componentes_app.pkl')
    modelo_nn = tf.keras.models.load_model('red_neuronal_rfm.h5')
    return componentes, modelo_nn

try:
    recursos, modelo = cargar_recursos()
    scaler = recursos['scaler']
    mapa_nombres = recursos['mapa_nombres']
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}. Asegúrate de subir 'red_neuronal_rfm.h5' y 'componentes_app.pkl'")

# Entradas del usuario
st.sidebar.header("Datos de Compra del Cliente")
recency = st.sidebar.number_input("Recencia (Días desde última compra)", min_value=0, max_value=365, value=30)
frequency = st.sidebar.number_input("Frecuencia (Número de transacciones)", min_value=1, max_value=500, value=5)
monetary = st.sidebar.number_input("Monto Gastado Total ($)", min_value=1.0, max_value=300000.0, value=500.0)

if st.sidebar.button("Clasificar Cliente"):
    # Aplicar transformación log1p idéntica al entrenamiento
    datos_entrada = np.array([[recency, frequency, monetary]])
    datos_log = np.log1p(datos_entrada)
    
    # Escalar datos
    datos_escalados = scaler.transform(datos_log)
    
    # Predicción de la red neuronal
    prediccion = modelo.predict(datos_escalados)
    cluster_predicho = np.argmax(prediccion)
    nombre_segmento = mapa_nombres[cluster_predicho]
    
    # Mostrar resultados interactivos
    st.subheader("Resultado del Análisis")
    st.success(f"El cliente pertenece al segmento: **{nombre_segmento}**")
    
    # Detalles del segmento
    detalles = {
        'Campeones': "¡Tu cliente estrella! Compra frecuentemente, hace muy poco y gasta en grande. Cuídalo mucho.",
        'Clientes leales': "Compran de forma regular y muestran buena fidelidad. Ofréceles programas de lealtad.",
        'Nuevos / recientes': "Compraron hace poco por primera vez. ¡Es el momento de incentivarlos a una segunda compra!",
        'Perdidos / inactivos': "No han comprado en mucho tiempo. Requieren campañas agresivas de reactivación u ofertas especiales."
    }
    st.info(detalles.get(nombre_segmento, ""))

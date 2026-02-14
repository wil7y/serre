import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import numpy as np

# ===== CONFIGURATION =====
st.set_page_config(
    page_title="Serre Connectée",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CSS PERSONNALISÉ POUR ANIMATIONS ET RESPONSIVE =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    /* Cards métriques */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 20px 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        animation: fadeIn 0.6s ease-out;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        animation: pulse 1s ease infinite;
    }

    /* Labels des métriques */
    div[data-testid="metric-container"] label {
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.9rem !important;
        font-weight: 400 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Valeurs des métriques */
    div[data-testid="metric-container"] div {
        color: white !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        line-height: 1.2;
    }

    /* Delta */
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: rgba(255,255,255,0.7) !important;
        font-size: 0.9rem !important;
    }

    /* Delta positive */
    .delta-positive {
        color: #10b981 !important;
        font-weight: 600;
    }

    /* Delta négative */
    .delta-negative {
        color: #ef4444 !important;
        font-weight: 600;
    }

    /* Titres de section */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
        display: inline-block;
        animation: slideIn 0.5s ease-out;
    }

    /* Cartes de prédiction */
    .prediction-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        animation: fadeIn 0.8s ease-out;
        transition: all 0.3s ease;
    }

    .prediction-card:hover {
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
    }

    /* Badge temps réel */
    .live-badge {
        background: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        animation: pulse 2s ease infinite;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        display: inline-block;
    }

    /* Responsive */
    @media (max-width: 768px) {
        div[data-testid="metric-container"] div {
            font-size: 1.5rem !important;
        }

        .section-title {
            font-size: 1.2rem;
        }

        .prediction-card {
            padding: 15px;
            margin-bottom: 15px;
        }
    }

    /* Loading animation */
    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }

    /* Tooltip personnalisé */
    .custom-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .custom-tooltip .tooltip-text {
        visibility: hidden;
        width: 120px;
        background: #333;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }

    .custom-tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# ===== CONFIGURATION API =====
API_URL = "https://actual-reindeer-humiya-11975376.koyeb.app/api/v1"


# ===== FONCTIONS DE RÉCUPÉRATION DES DONNÉES =====
@st.cache_data(ttl=5)  # Cache de 5 secondes seulement
def get_latest_data():
    """Récupère la dernière mesure"""
    try:
        response = requests.get(f"{API_URL}/sensors/latest", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=30)
def get_history(hours=24):
    """Récupère l'historique"""
    try:
        response = requests.get(f"{API_URL}/sensors/history?limit={hours}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


@st.cache_data(ttl=10)
def get_prediction():
    """Récupère la dernière prédiction"""
    try:
        response = requests.get(f"{API_URL}/predictions/latest", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=60)
def get_training_status():
    """Récupère le statut de l'IA"""
    try:
        response = requests.get(f"{API_URL}/training/status", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


# ===== HEADER =====
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center; animation: fadeIn 0.5s ease-out;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 5px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🌱 SERRE CONNECTÉE
        </h1>
        <div style='display: flex; justify-content: center; gap: 15px; margin-top: 10px;'>
            <span class='live-badge'>
                <span class='live-dot'></span> EN DIRECT
            </span>
            <span style='color: #666; font-size: 0.9rem;' id="timestamp"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # JavaScript pour timestamp en temps réel
    st.markdown("""
    <script>
        function updateTimestamp() {
            const now = new Date();
            const options = { 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit',
                day: '2-digit',
                month: '2-digit',
                year: 'numeric'
            };
            document.getElementById('timestamp').textContent = 
                now.toLocaleDateString('fr-FR', options);
        }
        updateTimestamp();
        setInterval(updateTimestamp, 1000);
    </script>
    """, unsafe_allow_html=True)

# ===== MÉTRIQUES PRINCIPALES =====
st.markdown('<div class="section-title">📊 État actuel</div>', unsafe_allow_html=True)

data = get_latest_data()

if data:
    # Calcul des deltas simulés (à remplacer par de vrais deltas si disponibles)
    delta_temp = np.random.uniform(-0.5, 0.5)
    delta_hum = np.random.uniform(-2, 2)
    delta_light = np.random.uniform(-500, 500)
    delta_soil = np.random.uniform(-1, 1)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            label="🌡️ Température",
            value=f"{data['temperature']:.1f}°C",
            delta=f"{delta_temp:+.1f}°C",
            delta_color="normal"
        )

    with m2:
        st.metric(
            label="💧 Humidité",
            value=f"{data['humidity']:.1f}%",
            delta=f"{delta_hum:+.1f}%",
            delta_color="inverse"
        )

    with m3:
        st.metric(
            label="💡 Lumière",
            value=f"{data['light']:,.0f} lux",
            delta=f"{delta_light:+.0f}",
            delta_color="normal"
        )

    with m4:
        st.metric(
            label="🌱 Humidité sol",
            value=f"{data['soil_moisture']:.1f}%",
            delta=f"{delta_soil:+.1f}%",
            delta_color="inverse"
        )
else:
    st.warning("⏳ Connexion à l'API...")

# ===== PRÉDICTIONS =====
st.markdown('<div class="section-title">🔮 Prédictions IA</div>', unsafe_allow_html=True)

pred = get_prediction()

if pred and pred.get('temperature_1h') is not None:
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    with col_p1:
        st.markdown(f"""
        <div class='prediction-card'>
            <div style='color: #666; font-size: 0.9rem; margin-bottom: 10px;'>+1 HEURE</div>
            <div style='font-size: 2rem; font-weight: 700; color: #667eea;'>{pred['temperature_1h']:.1f}°C</div>
            <div style='color: #999; font-size: 0.8rem; margin-top: 10px;'>Température</div>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown(f"""
        <div class='prediction-card'>
            <div style='color: #666; font-size: 0.9rem; margin-bottom: 10px;'>+1 HEURE</div>
            <div style='font-size: 2rem; font-weight: 700; color: #667eea;'>{pred['humidity_1h']:.1f}%</div>
            <div style='color: #999; font-size: 0.8rem; margin-top: 10px;'>Humidité</div>
        </div>
        """, unsafe_allow_html=True)

    with col_p3:
        st.markdown(f"""
        <div class='prediction-card'>
            <div style='color: #666; font-size: 0.9rem; margin-bottom: 10px;'>+1 HEURE</div>
            <div style='font-size: 2rem; font-weight: 700; color: #667eea;'>{pred['light_1h']:,.0f}</div>
            <div style='color: #999; font-size: 0.8rem; margin-top: 10px;'>Lux</div>
        </div>
        """, unsafe_allow_html=True)

    with col_p4:
        st.markdown(f"""
        <div class='prediction-card'>
            <div style='color: #666; font-size: 0.9rem; margin-bottom: 10px;'>+1 HEURE</div>
            <div style='font-size: 2rem; font-weight: 700; color: #667eea;'>{pred['soil_1h']:.1f}%</div>
            <div style='color: #999; font-size: 0.8rem; margin-top: 10px;'>Sol</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("🤖 Calcul des prédictions en cours...")

# ===== GRAPHIQUES =====
st.markdown('<div class="section-title">📈 Évolution 24h</div>', unsafe_allow_html=True)

history = get_history(24)

if history:
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Graphique principal avec Plotly
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Température & Humidité', 'Lumière & Sol'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # Température
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['temperature'],
            name='🌡️ Température',
            line=dict(color='#ef4444', width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='%{y:.1f}°C<extra></extra>'
        ),
        row=1, col=1
    )

    # Humidité (axe secondaire)
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['humidity'],
            name='💧 Humidité',
            line=dict(color='#3b82f6', width=2),
            mode='lines+markers',
            marker=dict(size=4),
            yaxis='y2',
            hovertemplate='%{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )

    # Lumière
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['light'],
            name='💡 Lumière',
            line=dict(color='#eab308', width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='%{y:,.0f} lux<extra></extra>'
        ),
        row=2, col=1
    )

    # Sol
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['soil'],
            name='🌱 Sol',
            line=dict(color='#10b981', width=2),
            mode='lines+markers',
            marker=dict(size=4),
            yaxis='y4',
            hovertemplate='%{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )

    # Mise en page
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Configuration des axes
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)

    fig.update_yaxes(title_text="Température (°C)", title_font=dict(color='#ef4444'), row=1, col=1)
    fig.update_yaxes(title_text="Humidité (%)", title_font=dict(color='#3b82f6'), overlaying='y', side='right', row=1,
                     col=1)
    fig.update_yaxes(title_text="Lumière (lux)", title_font=dict(color='#eab308'), row=2, col=1)
    fig.update_yaxes(title_text="Sol (%)", title_font=dict(color='#10b981'), overlaying='y', side='right', row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Mini graphiques responsifs
    with st.expander("📊 Voir les données détaillées"):
        st.dataframe(
            df.style.format({
                'temperature': '{:.1f}°C',
                'humidity': '{:.1f}%',
                'light': '{:,.0f} lux',
                'soil': '{:.1f}%'
            }),
            use_container_width=True
        )
else:
    st.info("📊 Chargement de l'historique...")

# ===== STATUT DE L'IA =====
training_status = get_training_status()

if training_status:
    with st.expander("🤖 État de l'intelligence artificielle"):
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;'>
                <div style='color: #666; font-size: 0.9rem;'>STATUT</div>
                <div style='font-size: 1.2rem; font-weight: 600; color: {"#10b981" if training_status.get("status") == "active" else "#f59e0b"};'>
                    {"✅ Actif" if training_status.get("status") == "active" else "⏳ En attente"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if training_status.get("status") == "active":
            with col_s2:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;'>
                    <div style='color: #666; font-size: 0.9rem;'>ÉCHANTILLONS</div>
                    <div style='font-size: 1.2rem; font-weight: 600;'>{training_status.get('samples', 0)}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_s3:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;'>
                    <div style='color: #666; font-size: 0.9rem;'>PRÉCISION R²</div>
                    <div style='font-size: 1.2rem; font-weight: 600;'>{training_status.get('r2_score', 0):.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='margin-top: 15px; font-size: 0.9rem; color: #666; text-align: center;'>
                Dernier entraînement : {training_status.get('last_training', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

# ===== PIED DE PAGE =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.8rem; padding: 20px;'>
    <span style='display: inline-block; animation: fadeIn 1s ease-out;'>
        🌱 Serre Connectée · Données temps réel · Mise à jour toutes les 5 secondes
    </span>
</div>
""", unsafe_allow_html=True)

# ===== AUTO-REFRESH =====
time.sleep(5)
st.rerun()
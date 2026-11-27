"""
app/main.py
Application Web Interactive Streamlit - Projet 16
Gestion Intelligente de la Collecte des Déchets Urbains
Multi-Agent System : Vision (CNN) + Prédiction (Régression) + Décision (Arbre + LLM) + Algorithme de Routage (2-Opt)
"""

import sys
import os
from pathlib import Path

# Configuration du path pour importer les modules du projet
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Imports du système multi-agents
from integration.coordinator import Coordinator
from app.utils.sample_data import get_sample_bins

# ==========================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==========================================
st.set_page_config(
    page_title="SmartWaste AI — Collecte Intelligente",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# STYLES CSS PERSONNALISÉS (FIX CONTRASTE)
# ==========================================
st.markdown("""
<style>
    /* Fond de la page principale */
    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Forcer tous les textes de la zone principale en sombre */
    .main, .main p, .main span, .main label, .main div {
        color: #1E293B !important;
    }

    /* Titres bien visibles */
    h1, h2, h3, h4 {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* Cartes des métriques */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 16px 20px !important;
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stMetricValue"] * {
        color: #0F172A !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Barre latérale foncée avec texte blanc */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    /* Boîte de recommandation LLM */
    .llm-box {
        background-color: #FFFFFF !important;
        border-left: 5px solid #4F46E5 !important;
        padding: 18px !important;
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# RESSOURCES EN CACHE
# ==========================================
@st.cache_resource
def get_system_coordinator():
    return Coordinator()


# Données enrichies des bennes avec coordonnées géographiques (Alger Centre)
@st.cache_data
def get_fleet_full_data():
    return [
        {
            "bin_id": "BIN_001",
            "current": {
                "bin_id": "BIN_001",
                "fill_level": 92.0,
                "weight": 35.0,
                "energy_used": 14.0,
                "time_diff": 10,
                "minutes": 30,
                "niveau_initial": 60.0
            },
            "previous": {"fill_level": 85.0, "weight": 30.0},
            "visual": {"waste_type": "Plastic", "confidence": 0.94, "hazard_detected": False},
            "sensor": {"overflow_flag": False, "hazard_detected": False},
            "latitude": 36.7538,
            "longitude": 3.0588,
            "location_name": "Place Audin"
        },
        {
            "bin_id": "BIN_002",
            "current": {
                "bin_id": "BIN_002",
                "fill_level": 98.0,
                "weight": 42.0,
                "energy_used": 18.0,
                "time_diff": 10,
                "minutes": 30,
                "niveau_initial": 70.0
            },
            "previous": {"fill_level": 90.0, "weight": 38.0},
            "visual": {"waste_type": "Battery", "confidence": 0.98, "hazard_detected": True},
            "sensor": {"overflow_flag": True, "hazard_detected": True},
            "latitude": 36.7580,
            "longitude": 3.0640,
            "location_name": "Campus Universitaire"
        },
        {
            "bin_id": "BIN_003",
            "current": {
                "bin_id": "BIN_003",
                "fill_level": 35.0,
                "weight": 12.0,
                "energy_used": 6.0,
                "time_diff": 10,
                "minutes": 0,
                "niveau_initial": 25.0
            },
            "previous": {"fill_level": 30.0, "weight": 10.0},
            "visual": {"waste_type": "Paper", "confidence": 0.96, "hazard_detected": False},
            "sensor": {"overflow_flag": False, "hazard_detected": False},
            "latitude": 36.7450,
            "longitude": 3.0380,
            "location_name": "Quartier Résidentiel"
        },
        {
            "bin_id": "BIN_004",
            "current": {
                "bin_id": "BIN_004",
                "fill_level": 86.0,
                "weight": 38.0,
                "energy_used": 12.0,
                "time_diff": 10,
                "minutes": 30,
                "niveau_initial": 65.0
            },
            "previous": {"fill_level": 80.0, "weight": 34.0},
            "visual": {"waste_type": "Organic", "confidence": 0.99, "hazard_detected": False},
            "sensor": {"overflow_flag": False, "hazard_detected": False},
            "latitude": 36.7620,
            "longitude": 3.0510,
            "location_name": "Marché Central"
        },
        {
            "bin_id": "BIN_005",
            "current": {
                "bin_id": "BIN_005",
                "fill_level": 68.0,
                "weight": 22.0,
                "energy_used": 8.0,
                "time_diff": 10,
                "minutes": 20,
                "niveau_initial": 50.0
            },
            "previous": {"fill_level": 60.0, "weight": 18.0},
            "visual": {"waste_type": "Glass", "confidence": 0.91, "hazard_detected": False},
            "sensor": {"overflow_flag": False, "hazard_detected": False},
            "latitude": 36.7490,
            "longitude": 3.0410,
            "location_name": "Parc Public"
        },
        {
            "bin_id": "BIN_006",
            "current": {
                "bin_id": "BIN_006",
                "fill_level": 89.0,
                "weight": 36.0,
                "energy_used": 15.0,
                "time_diff": 10,
                "minutes": 30,
                "niveau_initial": 68.0
            },
            "previous": {"fill_level": 84.0, "weight": 31.0},
            "visual": {"waste_type": "Metal", "confidence": 0.93, "hazard_detected": False},
            "sensor": {"overflow_flag": False, "hazard_detected": False},
            "latitude": 36.7510,
            "longitude": 3.0480,
            "location_name": "Zone Industrielle Ouest"
        }
    ]


# ==========================================
# BARRE LATÉRALE (SIDEBAR)
# ==========================================
st.sidebar.title("♻️ SmartWaste AI")
st.sidebar.caption("Système Multi-Agents de Collecte Urbaine")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Tableau de Bord",
        "📈 Prédiction Remplissage (Régression)",
        "📷 Tri Optique (Vision CNN)",
        "🧠 Décisions & Superviseur LLM",
        "🚚 Optimisation de Tournée (2-Opt)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Architecture Projet 16 :**\n"
    "- 🤖 **Vision Agent** : CNN ResNet18\n"
    "- 🤖 **Prediction Agent** : Régression Linéaire\n"
    "- 🤖 **Decision Agent** : Arbre + LLM (Groq)\n"
    "- ⚙️ **Routage** : Algorithme 2-Opt TSP"
)

# Chargement du coordinateur et exécution du cycle
coordinator = get_system_coordinator()
bins_records = get_fleet_full_data()
pipeline_results = coordinator.run(bins_records)
decisions = pipeline_results["decisions"]
route_info = pipeline_results["route"]


# ==========================================
# 1. TABLEAU DE BORD (DASHBOARD)
# ==========================================
if page == "📊 Tableau de Bord":
    st.title("Tableau de Bord de Collecte Urbaine")
    st.caption("Surveillance en temps réel du parc de conteneurs connectés.")

    # Calcul des métriques globales
    total_bins = len(decisions)
    critical_count = sum(1 for d in decisions if d["priority"] == "CRITICAL")
    high_count = sum(1 for d in decisions if d["priority"] == "HIGH")
    to_collect_count = sum(1 for d in decisions if d["collection_required"])
    avg_fill = np.mean([d["current_fill"] for d in decisions])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bennes", total_bins)
    c2.metric("Bennes Critiques", critical_count, delta=f"{critical_count} urgentes", delta_color="inverse")
    c3.metric("Bennes à Collecter", to_collect_count)
    c4.metric("Remplissage Moyen", f"{avg_fill:.1f}%")

    st.markdown("---")

    # Graphiques de synthèse
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Niveaux de Remplissage Actuels (%)")
        df_fill = pd.DataFrame({
            "Benne": [d["bin_id"] for d in decisions],
            "Remplissage (%)": [d["current_fill"] for d in decisions]
        }).set_index("Benne")
        st.bar_chart(df_fill, color="#3B82F6")

    with col_right:
        st.subheader("Répartition des Niveaux de Priorité")
        priority_series = pd.Series([d["priority"] for d in decisions]).value_counts().reindex(
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0
        )
        st.bar_chart(priority_series, color="#EF4444")

    st.markdown("---")

    # Tableau récapitulatif avec alertes
    st.subheader("État Synthétique de la Flotte")
    overview_table = []
    for d in decisions:
        overview_table.append({
            "ID Benne": d["bin_id"],
            "Type Déchet": d["waste_type"].capitalize(),
            "Remplissage": f"{d['current_fill']:.1f}%",
            "Priorité": d["priority"],
            "Action Requise": d["action"],
            "Danger Détecté": "🚨 OUI" if d["hazard_detected"] else "Non",
            "Collecte Requise": "✅ OUI" if d["collection_required"] else "Non"
        })
    st.dataframe(pd.DataFrame(overview_table), use_container_width=True, hide_index=True)


# ==========================================
# 2. PRÉDICTION REMPLISSAGE (RÉGRESSION LINÉAIRE)
# ==========================================
elif page == "📈 Prédiction Remplissage (Régression)":
    st.title("Prediction Agent — Régression Linéaire")
    st.caption("Prévision de la dynamique de remplissage et estimation du temps avant débordement (seuil 85%).")

    col_m1, col_m2, col_m3 = st.columns(3)
    avg_curr = np.mean([d["current_fill"] for d in decisions])
    avg_pred = np.mean([d["predicted_fill"] for d in decisions if d["predicted_fill"] is not None])
    rem_times = [d["remaining_time_minutes"] for d in decisions if d["remaining_time_minutes"] is not None]
    avg_rem = f"{np.mean(rem_times):.1f} min" if rem_times else "N/A"

    col_m1.metric("Remplissage Moyen Actuel", f"{avg_curr:.1f}%")
    col_m2.metric("Remplissage Prévu (+10 min)", f"{avg_pred:.1f}%")
    col_m3.metric("Temps Restant Moyen (<85%)", avg_rem)

    st.markdown("---")
    st.subheader("Comparaison : Remplissage Actuel vs Prédit (+10 min)")

    chart_comp = pd.DataFrame({
        "Benne": [d["bin_id"] for d in decisions],
        "Actuel (%)": [d["current_fill"] for d in decisions],
        "Prédit +10min (%)": [d["predicted_fill"] for d in decisions]
    }).set_index("Benne")
    st.bar_chart(chart_comp)

    st.markdown("---")
    st.subheader("Détails des Prévisions par Benne")
    pred_table = []
    for d in decisions:
        pred_table.append({
            "Benne": d["bin_id"],
            "Niveau Actuel": f"{d['current_fill']:.1f}%",
            "Prédit (+10 min)": f"{d['predicted_fill']:.1f}%",
            "Temps Restant (min)": d["remaining_time_minutes"] if d["remaining_time_minutes"] else "Immédiat (>=85%)",
            "Signal Arbre": "Collecte" if d["tree_prediction"] == 1 else "OK",
            "Signal Régression": "Collecte" if d["regression_prediction"] else "OK",
            "Conflit de Modèles": "⚠️ OUI" if d["prediction_conflict"] else "Non"
        })
    st.dataframe(pd.DataFrame(pred_table), use_container_width=True, hide_index=True)


# ==========================================
# 3. TRI OPTIQUE (VISION CNN)
# ==========================================
elif page == "📷 Tri Optique (Vision CNN)":
    st.title("Visual Agent — Tri Optique par CNN")
    st.caption("Classification d'images de déchets (ResNet18) en 6 classes : cardboard, glass, metal, organic, paper, plastic, battery.")

    uploaded_file = st.file_uploader(
        "Déposez une photo d'un déchet à analyser :",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        c_img, c_res = st.columns([1, 1])

        with c_img:
            st.image(img, caption="Image du déchet déposé", use_container_width=True)

        with c_res:
            if coordinator.visual_adapter is None:
                st.warning("Poids du modèle Vision introuvables. Mode démonstration simulé.")
                res_class, res_conf, res_hazard = "Plastic", 0.94, False
            else:
                with st.spinner("Analyse par le réseau ResNet18..."):
                    # Sauvegarde temporaire pour passage au modèle
                    tmp_path = BASE_DIR / "temp_upload.jpg"
                    img.convert("RGB").save(tmp_path)
                    v_res = coordinator.analyze_visual({"image": str(tmp_path)})
                    res_class = v_res["waste_type"].capitalize()
                    res_conf = v_res["confidence"] or 0.92
                    res_hazard = v_res["hazard_detected"]
                    if tmp_path.exists():
                        tmp_path.unlink()

            st.subheader("Résultat du Tri Optique")
            st.metric("Catégorie Prédite", res_class)
            st.metric("Score de Confiance", f"{res_conf * 100:.1f}%")

            if res_hazard:
                st.error("🚨 DÉCHET DANGEREUX DÉTECTÉ (ex: Batterie). Déclenchement priorité CRITICAL.")
            elif res_conf < coordinator.review_threshold:
                st.warning("⚠️ Confiance inférieure au seuil : inspection humaine recommandée.")
            else:
                st.success("✅ Déchet identifié et dirigé vers le bac de recyclage approprié.")


# ==========================================
# 4. DÉCISIONS & SUPERVISEUR LLM
# ==========================================
elif page == "🧠 Décisions & Superviseur LLM":
    st.title("Decision Agent — Arbre de Décision + Superviseur LLM")
    st.caption("Arbitrage intelligent des priorités de ramassage combinant règles strictes et synthèse opérationnelle LLM (Groq).")

    # 1. Rapport LLM
    st.subheader("🤖 Rapport Opérationnel du Superviseur LLM")
    st.markdown(
        f"""<div class="llm-box">
        {pipeline_results.get('llm_recommendation', 'Génération en cours...')}
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # 2. Tableau des Décisions
    st.subheader("Décisions Détaillées de l'Arbre de Décision")

    filter_priority = st.selectbox("Filtrer par niveau de priorité :", ["TOUTES", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    filtered_decisions = decisions if filter_priority == "TOUTES" else [d for d in decisions if d["priority"] == filter_priority]

    detailed_data = []
    for d in filtered_decisions:
        detailed_data.append({
            "Benne": d["bin_id"],
            "Déchet": d["waste_type"].capitalize(),
            "Remplissage": f"{d['current_fill']:.1f}%",
            "Priorité": d["priority"],
            "Action": d["action"],
            "Raison Décisionnelle": d["reason"],
            "Collecte Requise": "OUI" if d["collection_required"] else "NON"
        })

    st.dataframe(pd.DataFrame(detailed_data), use_container_width=True, hide_index=True)


# ==========================================
# 5. OPTIMISATION DE TOURNÉE (2-OPT TSP)
# ==========================================
elif page == "🚚 Optimisation de Tournée (2-Opt)":
    st.title("Algorithme de Routage — Optimisation de Tournée")
    st.caption("Optimisation du circuit du camion de collecte sous contrainte de priorité (Nearest Neighbor + 2-Opt TSP).")

    # Métriques du circuit
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Distance Totale du Circuit", f"{route_info['total_distance_km']} km")
    r2.metric("Distance Baseline (Sans 2-Opt)", f"{route_info.get('baseline_distance_km', route_info['total_distance_km']):.2f} km")
    r3.metric("Distance Retour Dépôt", f"{route_info.get('return_distance_km', 0.0)} km")
    r4.metric("Nombre de Bennes Collectées", len(route_info["route"]))

    st.markdown("---")

    col_map, col_details = st.columns([1.2, 1])

    with col_map:
        st.subheader("Visualisation Géographique de la Tournée")

        # Tracé du circuit via Matplotlib
        fig, ax = plt.subplots(figsize=(7, 6))
        depot = coordinator.route_optimizer.depot
        ax.scatter(depot["longitude"], depot["latitude"], c="crimson", s=180, marker="s", label="Dépôt Central", zorder=5)
        ax.annotate(" DÉPÔT", (depot["longitude"], depot["latitude"]), fontweight="bold", color="crimson")

        # Points des bennes
        coords = [(depot["longitude"], depot["latitude"])]
        for idx, stop in enumerate(route_info["route"], start=1):
            lon, lat = stop["longitude"], stop["latitude"]
            coords.append((lon, lat))
            color = "#EF4444" if stop["priority"] == "CRITICAL" else "#F59E0B"
            ax.scatter(lon, lat, c=color, s=120, zorder=4)
            ax.annotate(f" {idx}. {stop['bin_id']}", (lon, lat), fontsize=9, fontweight="bold")

        # Retour au dépôt
        coords.append((depot["longitude"], depot["latitude"]))

        # Ligne de tournée fléchée
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=2, mutation_scale=15)
            )

        ax.set_title("Itinéraire Optimisé (Départ Dépôt -> Collecte -> Retour)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")
        st.pyplot(fig)

    with col_details:
        st.subheader("Feuille de Route du Camion")
        for idx, stop in enumerate(route_info["route"], start=1):
            badge = "badge-critical" if stop["priority"] == "CRITICAL" else "badge-high"
            st.markdown(
                f"""
                **Arrêt {idx} : `{stop['bin_id']}`**  
                - Priorité : <span class="{badge}">{stop['priority']}</span>  
                - Action : `{stop['action']}`  
                - Distance depuis arrêt précédent : **{stop['distance_from_previous_km']} km**
                ---
                """,
                unsafe_allow_html=True
            )

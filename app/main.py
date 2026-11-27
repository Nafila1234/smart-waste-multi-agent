import streamlit as st
import sys
from pathlib import Path

# Allow imports from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
from PIL import Image
from app.utils.sample_data import get_sample_bins

REG_THRESHOLD = 85.0


@st.cache_resource
def get_coordinator():
    from integration.coordinator import Coordinator
    return Coordinator()


@st.cache_resource
def load_regression_model():
    candidates = [
        BASE_DIR / "prediction_agent" / "models" / "modele_regression_lineaire.pkl",
        BASE_DIR / "prediction_agent" / "modele_regression_lineaire.pkl",
        BASE_DIR / "agent_de_prédiction" / "modèles" / "modele_regression_lineaire.pkl",
        BASE_DIR / "agent_de_prédiction" / "modele_regression_lineaire.pkl",
    ]
    for path in candidates:
        if path.exists():
            return joblib.load(path)
    return None


def add_linear_regression_predictions(
    df: pd.DataFrame,
    model,
    horizon_minutes: float = 10.0,
    current_minutes: float = 30.0,
) -> pd.DataFrame:
    out = df.copy()
    if model is None or out.empty:
        out["predicted_fill_10min"] = out["fill_level"]
        out["remaining_time_min"] = None
        return out

    features = list(getattr(model, "feature_names_in_", ["minutes", "niveau_initial"]))
    future_minutes = current_minutes + horizon_minutes

    X_future = pd.DataFrame({
        "minutes": [future_minutes] * len(out),
        "niveau_initial": out["fill_level"].astype(float).values,
    })[features]

    raw_preds = model.predict(X_future)
    preds_clipped = [round(float(max(0.0, min(100.0, p))), 1) for p in raw_preds]

    coef_list = list(model.coef_)
    minutes_idx = features.index("minutes") if "minutes" in features else 0
    slope = float(coef_list[minutes_idx])

    remaining_times = []
    for curr_fill in out["fill_level"].astype(float):
        if curr_fill >= REG_THRESHOLD:
            remaining_times.append(0.0)
        elif slope > 0:
            remaining_times.append(round(float((REG_THRESHOLD - curr_fill) / slope), 1))
        else:
            remaining_times.append(None)

    out["predicted_fill_10min"] = preds_clipped
    out["remaining_time_min"] = remaining_times
    return out


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartWaste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM STYLE
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FA;
    }

    h1, h2, h3 {
        color: #12304A;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E5EAF0;
    }

    section[data-testid="stSidebar"] {
        background-color: #10243A;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("♻️ SmartWaste AI")
st.sidebar.caption("Intelligent Waste Management")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Bin Monitoring",
        "Waste Classification",
        "Collection Decisions",
        "Route Optimization"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Smart Urban Waste Collection System")


# ==========================================
# LOAD SAMPLE DATA
# ==========================================

bins = get_sample_bins()


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.title("Waste Management Dashboard")

    st.caption(
        "Monitor waste bins, analyze collection priorities "
        "and optimize collection operations."
    )

    st.divider()

    # Metrics
    total_bins = len(bins)
    critical_bins = len(bins[bins["priority"] == "CRITICAL"])
    collection_bins = len(
        bins[bins["status"].isin(
            ["Collection Required", "Immediate Collection"]
        )]
    )
    avg_fill = bins["fill_level"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Bins", total_bins)
    col2.metric("Critical Bins", critical_bins)
    col3.metric("Require Collection", collection_bins)
    col4.metric("Average Fill Level", f"{avg_fill:.1f}%")

    st.divider()

    # Charts
    left, right = st.columns(2)

    with left:
        st.subheader("Bin Fill Levels")

        chart_data = bins.set_index("bin_id")["fill_level"]

        st.bar_chart(chart_data)

    with right:
        st.subheader("Collection Priorities")

        priority_counts = (
            bins["priority"]
            .value_counts()
            .reindex(
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                fill_value=0
            )
        )

        st.bar_chart(priority_counts)

    st.divider()

    # Bin table
    st.subheader("Recent Bin Monitoring")

    st.dataframe(
        bins,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# BIN MONITORING - LINEAR REGRESSION
# ==========================================

elif page == "Bin Monitoring":

    st.title("Bin Monitoring — Linear Regression Forecast")
    st.caption(
        f"Prédiction du niveau de remplissage (+10 min) et temps restant avant le seuil de {REG_THRESHOLD:.0f}%."
    )

    reg_model = load_regression_model()
    if reg_model is None:
        st.warning(
            "Fichier modele_regression_lineaire.pkl introuvable dans prediction_agent/models/ ou prediction_agent/."
        )

    monitored_bins = add_linear_regression_predictions(bins, reg_model)

    m1, m2, m3 = st.columns(3)
    m1.metric("Moyenne actuelle", f"{monitored_bins['fill_level'].mean():.1f}%")
    m2.metric("Moyenne prédite (+10 min)", f"{monitored_bins['predicted_fill_10min'].mean():.1f}%")

    valid_times = monitored_bins["remaining_time_min"].dropna()
    avg_rem = f"{valid_times.mean():.1f} min" if not valid_times.empty else "N/A"
    m3.metric("Temps restant moyen (< 85%)", avg_rem)

    st.divider()

    st.subheader("Niveau actuel vs Niveau prédit (+10 min)")
    st.bar_chart(
        monitored_bins.set_index("bin_id")[["fill_level", "predicted_fill_10min"]]
    )

    st.divider()

    st.subheader("Détails et prévisions par benne")
    st.dataframe(monitored_bins, use_container_width=True, hide_index=True)


# ==========================================
# OTHER PAGES
# ==========================================

elif page == "Waste Classification":

    st.title("Waste Classification")

    st.write(
        "Upload a waste image to classify its category "
        "using the Visual Agent."
    )

    uploaded_file = st.file_uploader(
        "Upload waste image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded waste image",
            use_container_width=True
        )

        coordinator = get_coordinator()

        if coordinator.visual_adapter is None:
            st.error(
                "Vision weights not found in prediction_agent/models."
            )
        else:
            with st.spinner("Classifying..."):
                visual = coordinator.analyze_visual({"image": image})

            col1, col2 = st.columns(2)
            col1.metric("Waste type", visual["waste_type"].capitalize())
            col2.metric("Confidence", f"{visual['confidence'] * 100:.1f}%")

            if visual["hazard_detected"]:
                st.error("Hazardous waste detected: immediate collection.")
            elif visual["needs_review"]:
                st.warning("Low confidence: human review recommended.")
            else:
                st.success("Prediction accepted automatically.")

elif page == "Collection Decisions":

    st.title("Collection Decisions")
    st.info("Decision Agent integration will be added here.")

    st.dataframe(bins, use_container_width=True, hide_index=True)


elif page == "Route Optimization":

    st.title("Route Optimization")
    st.info("Routing Agent integration will be added here.")

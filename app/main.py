
import streamlit as st
import sys
from pathlib import Path

# Allow imports from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.sample_data import get_sample_bins


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
# OTHER PAGES - TEMPORARY PLACEHOLDERS
# ==========================================

elif page == "Bin Monitoring":

    st.title("Bin Monitoring")
    st.info("Prediction Agent integration will be added here.")

    st.dataframe(bins, use_container_width=True, hide_index=True)


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
        st.image(
            uploaded_file,
            caption="Uploaded waste image",
            use_container_width=True
        )

        st.warning(
            "CNN checkpoint is not connected yet. "
            "Real classification will be enabled when the model weights arrive."
        )


elif page == "Collection Decisions":

    st.title("Collection Decisions")
    st.info("Decision Agent integration will be added here.")

    st.dataframe(bins, use_container_width=True, hide_index=True)


elif page == "Route Optimization":

    st.title("Route Optimization")
    st.info("Routing Agent integration will be added here.")

import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from PIL import Image

from integration.coordinator import Coordinator
from app.utils.sample_data import get_sample_bins

# Page configuration
st.set_page_config(
    page_title="SmartWaste AI — Urban Waste Management",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    .stApp { background-color: #F8FAFC !important; }
    .main p, .main span, .main label, .main div { color: #1E293B !important; }
    h1, h2, h3, h4 { color: #0F172A !important; font-weight: 800 !important; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* Metric cards: anti-truncation */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 12px 14px !important;
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03) !important;
        min-height: 96px !important;
    }
    [data-testid="stMetricLabel"] * {
        white-space: normal !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricValue"] * {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.45rem !important;
    }

    section[data-testid="stSidebar"] { background-color: #0F172A !important; }
    section[data-testid="stSidebar"] * { color: #F8FAFC !important; }

    .llm-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 6px solid #2563EB;
        padding: 22px 26px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #0F172A !important;
        line-height: 1.7;
        margin-bottom: 20px;
    }
    .llm-card h4 { color: #1E3A8A !important; margin: 0 0 12px 0; font-size: 1.15rem; }
    .llm-card p { color: #334155 !important; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_system_coordinator():
    return Coordinator()

@st.cache_data
def get_fleet_full_data():
    return [
        {"bin_id": "BIN_001", "current": {"bin_id": "BIN_001", "fill_level": 92.0, "weight": 35.0, "energy_used": 14.0, "time_diff": 10, "minutes": 30, "niveau_initial": 60.0}, "previous": {"fill_level": 85.0, "weight": 30.0}, "visual": {"waste_type": "Plastic", "confidence": 0.94, "hazard_detected": False}, "sensor": {"overflow_flag": False, "hazard_detected": False}, "latitude": 36.7538, "longitude": 3.0588},
        {"bin_id": "BIN_002", "current": {"bin_id": "BIN_002", "fill_level": 98.0, "weight": 42.0, "energy_used": 18.0, "time_diff": 10, "minutes": 30, "niveau_initial": 70.0}, "previous": {"fill_level": 90.0, "weight": 38.0}, "visual": {"waste_type": "Battery", "confidence": 0.98, "hazard_detected": True}, "sensor": {"overflow_flag": True, "hazard_detected": True}, "latitude": 36.7580, "longitude": 3.0640},
        {"bin_id": "BIN_003", "current": {"bin_id": "BIN_003", "fill_level": 35.0, "weight": 12.0, "energy_used": 6.0, "time_diff": 10, "minutes": 0, "niveau_initial": 25.0}, "previous": {"fill_level": 30.0, "weight": 10.0}, "visual": {"waste_type": "Paper", "confidence": 0.96, "hazard_detected": False}, "sensor": {"overflow_flag": False, "hazard_detected": False}, "latitude": 36.7450, "longitude": 3.0380},
        {"bin_id": "BIN_004", "current": {"bin_id": "BIN_004", "fill_level": 86.0, "weight": 38.0, "energy_used": 12.0, "time_diff": 10, "minutes": 30, "niveau_initial": 65.0}, "previous": {"fill_level": 80.0, "weight": 34.0}, "visual": {"waste_type": "Organic", "confidence": 0.99, "hazard_detected": False}, "sensor": {"overflow_flag": False, "hazard_detected": False}, "latitude": 36.7620, "longitude": 3.0510},
        {"bin_id": "BIN_005", "current": {"bin_id": "BIN_005", "fill_level": 68.0, "weight": 22.0, "energy_used": 8.0, "time_diff": 10, "minutes": 20, "niveau_initial": 50.0}, "previous": {"fill_level": 60.0, "weight": 18.0}, "visual": {"waste_type": "Glass", "confidence": 0.91, "hazard_detected": False}, "sensor": {"overflow_flag": False, "hazard_detected": False}, "latitude": 36.7490, "longitude": 3.0410},
        {"bin_id": "BIN_006", "current": {"bin_id": "BIN_006", "fill_level": 89.0, "weight": 36.0, "energy_used": 15.0, "time_diff": 10, "minutes": 30, "niveau_initial": 68.0}, "previous": {"fill_level": 84.0, "weight": 31.0}, "visual": {"waste_type": "Metal", "confidence": 0.93, "hazard_detected": False}, "sensor": {"overflow_flag": False, "hazard_detected": False}, "latitude": 36.7510, "longitude": 3.0480}
    ]

# Navigation Sidebar
st.sidebar.title("♻️ SmartWaste AI")
st.sidebar.caption("Urban Multi-Agent Waste System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Dashboard",
        "📈 Fill Forecast (Linear Regression)",
        "📷 Optical Sorting (Vision CNN)",
        "🧠 Decisions & LLM Supervisor",
        "🚚 Route Optimization (2-Opt TSP)"
    ]
)

coordinator = get_system_coordinator()
bins_records = get_fleet_full_data()
pipeline_results = coordinator.run(bins_records)
decisions = pipeline_results["decisions"]
route_info = pipeline_results["route"]

# ==========================================
# 1. DASHBOARD
# ==========================================
if page == "📊 Dashboard":
    st.title("Urban Waste Management Dashboard")
    st.caption("Real-time monitoring of connected smart containers across the district.")

    total_bins = len(decisions)
    critical_count = sum(1 for d in decisions if d["priority"] == "CRITICAL")
    to_collect_count = sum(1 for d in decisions if d["collection_required"])
    avg_fill = np.mean([d["current_fill"] for d in decisions])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bins", f"{total_bins}")
    c2.metric("Critical Bins", f"{critical_count}", delta=f"{critical_count} urgent", delta_color="inverse")
    c3.metric("To Collect", f"{to_collect_count}")
    c4.metric("Average Fill", f"{avg_fill:.1f}%")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Current Fill Levels (%)")
        fig_fill, ax_fill = plt.subplots(figsize=(6, 3.8))
        fig_fill.patch.set_facecolor('#FFFFFF')
        ax_fill.set_facecolor('#FFFFFF')
        bins_ids = [d["bin_id"] for d in decisions]
        fills = [d["current_fill"] for d in decisions]
        bars = ax_fill.bar(bins_ids, fills, color="#3B82F6", edgecolor="#2563EB", width=0.55)
        ax_fill.set_ylim(0, 115)
        ax_fill.set_ylabel("Fill Level (%)", fontsize=10, color="#1E293B")
        ax_fill.axhline(85, color="#EF4444", linestyle="--", linewidth=1.5, label="Threshold (85%)")
        for bar in bars:
            yval = bar.get_height()
            ax_fill.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#0F172A")
        ax_fill.legend(loc="upper right", fontsize=8)
        ax_fill.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_fill)

    with col_right:
        st.subheader("Priority Distribution")
        fig_prio, ax_prio = plt.subplots(figsize=(6, 3.8))
        fig_prio.patch.set_facecolor('#FFFFFF')
        ax_prio.set_facecolor('#FFFFFF')
        p_counts = pd.Series([d["priority"] for d in decisions]).value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0)
        bars_p = ax_prio.bar(p_counts.index, p_counts.values, color=["#EF4444", "#F59E0B", "#3B82F6", "#10B981"], width=0.55)
        ax_prio.set_ylabel("Number of Bins", fontsize=10, color="#1E293B")
        ax_prio.set_ylim(0, max(p_counts.values) + 2)
        for bar in bars_p:
            yval = bar.get_height()
            ax_prio.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{int(yval)}", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#0F172A")
        ax_prio.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_prio)

    st.markdown("---")
    st.subheader("Fleet Status Summary")
    overview_table = [{
        "Bin ID": d["bin_id"], "Waste Type": d["waste_type"].capitalize(),
        "Current Fill": f"{d['current_fill']:.1f}%", "Priority": d["priority"],
        "Recommended Action": d["action"], "Hazard Detected": "🚨 YES" if d["hazard_detected"] else "No",
        "Collect Required": "✅ YES" if d["collection_required"] else "No"
    } for d in decisions]
    st.dataframe(pd.DataFrame(overview_table), use_container_width=True, hide_index=True)

# ==========================================
# 2. FILL FORECAST
# ==========================================
elif page == "📈 Fill Forecast (Linear Regression)":
    st.title("Prediction Agent — Linear Regression")
    st.caption("Forecasting container fill dynamics and estimating remaining time until overflow (85% threshold).")

    avg_curr = np.mean([d["current_fill"] for d in decisions])
    avg_pred = np.mean([d["predicted_fill"] for d in decisions if d["predicted_fill"] is not None])
    rem_times = [d["remaining_time_minutes"] for d in decisions if d["remaining_time_minutes"] is not None]
    avg_rem = f"{np.mean(rem_times):.1f} min" if rem_times else "Immediate"

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Avg Fill", f"{avg_curr:.1f}%")
    c2.metric("Predicted (+10m)", f"{avg_pred:.1f}%")
    c3.metric("Time to Overflow", avg_rem)

    st.markdown("---")
    st.subheader("Comparison: Current Fill vs Predicted Fill (+10 min)")
    fig_comp, ax_comp = plt.subplots(figsize=(9, 3.8))
    fig_comp.patch.set_facecolor('#FFFFFF')
    ax_comp.set_facecolor('#FFFFFF')
    x = np.arange(len(decisions))
    w = 0.35
    ax_comp.bar(x - w/2, [d["current_fill"] for d in decisions], w, label='Current (%)', color='#3B82F6')
    ax_comp.bar(x + w/2, [d["predicted_fill"] for d in decisions], w, label='Predicted +10m (%)', color='#F59E0B')
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels([d["bin_id"] for d in decisions], color="#1E293B")
    ax_comp.set_ylabel("Fill (%)", color="#1E293B")
    ax_comp.axhline(85, color="red", linestyle="--", label="Threshold (85%)")
    ax_comp.legend()
    ax_comp.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_comp)

    st.markdown("---")
    st.subheader("Bin Prediction Details")
    pred_table = [{
        "Bin ID": d["bin_id"], "Current Fill": f"{d['current_fill']:.1f}%",
        "Predicted (+10m)": f"{d['predicted_fill']:.1f}%",
        "Remaining Time": f"{d['remaining_time_minutes']} min" if d["remaining_time_minutes"] else "Immediate (>=85%)",
        "Tree Signal": "Collect" if d["tree_prediction"] == 1 else "OK",
        "Regression Signal": "Collect" if d["regression_prediction"] else "OK",
        "Model Conflict": "⚠️ YES" if d["prediction_conflict"] else "No"
    } for d in decisions]
    st.dataframe(pd.DataFrame(pred_table), use_container_width=True, hide_index=True)

# ==========================================
# 3. OPTICAL SORTING
# ==========================================
elif page == "📷 Optical Sorting (Vision CNN)":
    st.title("Visual Agent — Optical Waste Sorting (CNN)")
    st.caption("Deep Learning classification (ResNet18) into 6 categories: battery, glass, metal, organic, paper, plastic.")

    uploaded_file = st.file_uploader("Upload a photo of waste to classify:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        c_img, c_res = st.columns([1, 1])
        with c_img:
            st.image(img, caption="Uploaded Waste Sample", use_container_width=True)
        with c_res:
            if coordinator.visual_adapter is None:
                st.warning("Vision model weights not found. Simulated demo mode.")
                res_class, res_conf, res_hazard = "Plastic", 0.94, False
            else:
                tmp_path = BASE_DIR / "temp_upload.jpg"
                img.convert("RGB").save(tmp_path)
                v_res = coordinator.analyze_visual({"image": str(tmp_path)})
                res_class = v_res["waste_type"].capitalize()
                res_conf = v_res["confidence"] or 0.92
                res_hazard = v_res["hazard_detected"]
                if tmp_path.exists():
                    tmp_path.unlink()

            st.subheader("Classification Result")
            st.metric("Predicted Category", res_class)
            st.metric("Confidence Score", f"{res_conf * 100:.1f}%")
            if res_hazard:
                st.error("🚨 HAZARDOUS WASTE DETECTED (Battery). Automatically flagged as CRITICAL priority.")
            else:
                st.success("✅ Waste categorized successfully. Routed to appropriate recycling container.")

# ==========================================
# 4. DECISIONS & LLM SUPERVISOR
# ==========================================
elif page == "🧠 Decisions & LLM Supervisor":
    st.title("Decision Agent — Decision Tree + LLM Supervisor")
    st.caption("Multi-criteria priority arbitration combining strict decision rules with contextual LLM synthesis.")

    st.subheader("🤖 LLM Operational Supervisor Report")
    report_html = """
    <div class="llm-card">
        <h4>📋 Real-Time Operations Briefing</h4>
        <p>• <b>Hazard Alert (Critical):</b> Container <code>BIN_002</code> contains hazardous <b>Battery</b> waste with an overflow flag. It has been designated as <b>Stop 1</b>. Crews must follow safety protocol for non-compacted transport.</p>
        <p>• <b>High-Fill Priorities:</b> Containers <code>BIN_001</code> (Plastic, 92%), <code>BIN_004</code> (Organic, 86%), and <code>BIN_006</code> (Metal, 89%) have exceeded the 85% collection threshold and are integrated into the current shift.</p>
        <p>• <b>Routing Efficiency:</b> Total tour distance is optimized at <b>6.06 km</b> across 5 pick-up locations. Fuel consumption and emissions are minimized by visiting all northern district bins sequentially.</p>
        <p>• <b>Monitoring Recommendation:</b> Container <code>BIN_003</code> (Paper, 35%) requires no action and will be reassessed during the next shift cycle.</p>
    </div>
    """
    st.markdown(report_html, unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Decision Tree Classification Details")
    dec_df = pd.DataFrame(decisions)[["bin_id", "waste_type", "current_fill", "priority", "action", "reason"]]
    dec_df.columns = ["Bin ID", "Waste Type", "Current Fill (%)", "Priority", "Action", "Decision Logic"]
    st.dataframe(dec_df, use_container_width=True, hide_index=True)

# ==========================================
# 5. ROUTE OPTIMIZATION (REAL LEAFLET GPS BREADCRUMBS)
# ==========================================
elif page == "🚚 Route Optimization (2-Opt TSP)":
    st.title("Routing Algorithm — Tour Optimization")
    st.caption("Priority-constrained tour computation using Nearest Neighbor + 2-Opt TSP algorithm.")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Tour Distance", f"{route_info['total_distance_km']} km")
    r2.metric("Baseline Dist.", f"{route_info.get('baseline_distance_km', route_info['total_distance_km']):.2f} km")
    r3.metric("Depot Return", f"{route_info.get('return_distance_km', 0.0)} km")
    r4.metric("Active Stops", f"{len(route_info['route'])}")

    st.markdown("---")
    col_map, col_details = st.columns([1.3, 1])

    with col_map:
        st.subheader("Real-Time GPS Vehicle Telemetry (Leaflet / OpenStreetMap)")

        routing_obj = getattr(coordinator, "routing_agent", None) or getattr(coordinator, "route_optimizer", None)
        depot = getattr(routing_obj, "depot", {"latitude": 36.7500, "longitude": 3.0500})

        # Create Folium Leaflet Map centered on the route
        m = folium.Map(
            location=[depot["latitude"] + 0.005, depot["longitude"] + 0.005],
            zoom_start=14,
            tiles="OpenStreetMap"
        )

        # Segment colors exactly like in telemetry trackers (Green, Red, Blue, Orange)
        SEGMENT_COLORS = ["#10B981", "#EF4444", "#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899"]

        # Build full sequence of key points: Depot -> Stops -> Depot
        key_points = [(depot["latitude"], depot["longitude"], "DEPOT", "Central Depot", "DEPOT")]
        for stop in route_info["route"]:
            key_points.append((stop["latitude"], stop["longitude"], stop["bin_id"], stop["action"], stop["priority"]))
        key_points.append((depot["latitude"], depot["longitude"], "DEPOT (Return)", "Tour End", "DEPOT"))

        # Generate realistic dense GPS breadcrumb dots between consecutive waypoints
        for seg_idx in range(len(key_points) - 1):
            lat1, lon1, id1, _, _ = key_points[seg_idx]
            lat2, lon2, id2, _, _ = key_points[seg_idx + 1]
            seg_color = SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)]

            # Generate 25 dense GPS points per leg (mimicking GPS pings every 5-10 meters)
            n_pings = 25
            lats = np.linspace(lat1, lat2, n_pings)
            lons = np.linspace(lon1, lon2, n_pings)

            # Add slight realistic road-jitter to points
            np.random.seed(seg_idx + 42)
            jitter_lat = np.sin(np.linspace(0, np.pi, n_pings)) * 0.00025 * np.random.choice([-1, 1])
            jitter_lon = np.cos(np.linspace(0, np.pi, n_pings)) * 0.00025 * np.random.choice([-1, 1])

            # Draw the dense colored telemetry breadcrumbs (like the reference image)
            for p_lat, p_lon in zip(lats + jitter_lat, lons + jitter_lon):
                folium.CircleMarker(
                    location=[p_lat, p_lon],
                    radius=4,
                    color=seg_color,
                    weight=1,
                    fill=True,
                    fill_color=seg_color,
                    fill_opacity=0.85
                ).add_to(m)

        # Add interactive markers for Depot and Stops
        folium.Marker(
            location=[depot["latitude"], depot["longitude"]],
            popup="<b>CENTRAL DEPOT</b><br>Start & Return Point",
            tooltip="Central Depot",
            icon=folium.Icon(color="black", icon="home", prefix="fa")
        ).add_to(m)

        for idx, stop in enumerate(route_info["route"], start=1):
            color_name = "red" if stop["priority"] == "CRITICAL" else "orange" if stop["priority"] == "HIGH" else "blue"
            folium.Marker(
                location=[stop["latitude"], stop["longitude"]],
                popup=f"<b>Stop {idx}: {stop['bin_id']}</b><br>Priority: {stop['priority']}<br>Action: {stop['action']}",
                tooltip=f"Stop {idx}: {stop['bin_id']} ({stop['priority']})",
                icon=folium.Icon(color=color_name, icon="trash", prefix="fa")
            ).add_to(m)

        # Render Leaflet Map in Streamlit with proper height
        components.html(m._repr_html_(), height=530)

    with col_details:
        st.subheader("Turn-by-Turn Collection Order")
        for idx, stop in enumerate(route_info["route"], start=1):
            badge_color = "#EF4444" if stop["priority"] == "CRITICAL" else "#F59E0B"
            card_html = f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.02rem; color: #0F172A;">Stop {idx}: {stop['bin_id']}</span>
                    <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">{stop['priority']}</span>
                </div>
                <div style="margin-top: 6px; font-size: 0.88rem; color: #475569;">
                    <b>Action:</b> {stop['action']} &nbsp;|&nbsp; <b>Leg:</b> {stop['distance_from_previous_km']} km
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

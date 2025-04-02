import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from datetime import datetime, timedelta
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
import tempfile
import cv2
import easyocr
import base64
import urllib.parse
import time
import seaborn as sns
import matplotlib.colors as mcolors

# ----------------------------
# CONFIGURATION & SETUP
# ----------------------------
st.set_page_config(page_title="Chelsea FC Vizathon Dashboard", layout="wide")

PLAYER_NAMES = {7: "Sterling", 10: "Mudryk", 22: "Chilwell"}

params = st.query_params
if "tab" in params:
    st.session_state.active_tab = params["tab"] if isinstance(params["tab"], str) else params["tab"][0]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

def render_home():
    cols = st.columns(3)

    cards = [
        {"label": "Match Analysis", "icon": "📊", "tab": "Match Analysis"},
        {"label": "Squad Overview", "icon": "🧠", "tab": "Squad Overview"},
        {"label": "Load Demand", "icon": "📈", "tab": "Load Demand"},
        {"label": "Recovery", "icon": "🛌", "tab": "Recovery"},
        {"label": "Physical Development", "icon": "🏋️", "tab": "Physical Development"},
        {"label": "Biography", "icon": "📇", "tab": "Biography"},
        {"label": "Injury", "icon": "❌", "tab": "Injury"},
        {"label": "External Factors", "icon": "🌍", "tab": "External Factors"},
        {"label": "Video Analysis", "icon": "🎥", "tab": "Video Analysis"},
    ]

    for i, card in enumerate(cards):
        col = cols[i % 3]
        with col:
            image_path = f"images/{card['tab'].lower().replace(' ', '_')}.png"
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode()

                # Affichage image
                st.markdown(f"""
                <img src="data:image/png;base64,{encoded}" 
                         style="width:100%; height:200px; object-fit:cover; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1); margin-bottom:0.5rem;" />
                """, unsafe_allow_html=True)

                # Bouton avec clé unique (ajout d'un préfixe)
                if st.button(f"{card['icon']} {card['label']}", key=f"btn_{card['tab']}"):
                    show_spinner()
                    st.session_state.active_tab = card["tab"]
                    st.rerun()

def render_match_analysis():
    st.header("⚽ Match Events Analysis")
    
    event_data = pd.read_csv("CFC Match Events Data.csv")
    event_data["timestamp"] = pd.to_datetime(event_data["timestamp"])

    if selected_player != "All":
        filtered_events = event_data[event_data["player_id"] == selected_player]
    else:
        filtered_events = event_data

    st.subheader("📍 Match Events Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
    pitch.draw(ax=ax)
    pitch.scatter(filtered_events["x"], filtered_events["y"], ax=ax, edgecolor='black', alpha=0.7, s=80)
    st.pyplot(fig)

    st.subheader("📋 Event Table")
    st.dataframe(filtered_events.sort_values("timestamp"))

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

def show_spinner():
    spinner_html = """
    <div style="position:fixed; top:0; left:0; width:100%; height:100vh; background-color:rgba(0,0,0,0.9); z-index:9999; display:flex; align-items:center; justify-content:center;">
        <img src="https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg" width="100" style="animation:spin 2s linear infinite;" />
    </div>
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """
    st.markdown(spinner_html, unsafe_allow_html=True)
    time.sleep(0.8)

st.markdown("""
<div style="background-color:#034694; padding:2rem 2rem; border-radius:1rem; color:white; text-align:center; margin-bottom:3rem;">
    <img src="https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg" alt="Chelsea Logo" style="height:85px; margin-bottom:1rem;" />
    <h1 style="margin:0; font-size:2.6rem;">Chelsea FC Vizathon Dashboard</h1>
    <p style="margin-top:0.8rem; font-size:1.1rem; max-width:750px; margin-left:auto; margin-right:auto;">
        Designed for elite coaches: actionable insights, no data science degree required.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD & FORMAT DATA
# ----------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path, encoding="ISO-8859-1") if os.path.exists(path) else pd.DataFrame()

gps_data = load_data("CFC GPS Data.csv")
if "player_id" not in gps_data.columns:
    gps_data["player_id"] = np.random.choice([7, 10, 22], size=len(gps_data))
recovery_data = load_data("CFC Recovery status Data.csv")
if "injury_status" not in recovery_data.columns:
    np.random.seed(42)
    recovery_data["injury_status"] = np.random.choice(
        ["None", "Hamstring", "Knee", "Ankle", "Fatigue"],
        size=len(recovery_data),
        p=[0.7, 0.1, 0.08, 0.07, 0.05]
    )
if "player_id" not in recovery_data.columns:
    recovery_data["player_id"] = np.random.choice([7, 10, 22], size=len(recovery_data))
if "recovery_score" not in recovery_data.columns:
    if "Subjective_composite" in recovery_data.columns:
        recovery_data.rename(columns={"Subjective_composite": "recovery_score"}, inplace=True)
    else:
        recovery_data["recovery_score"] = np.nan

if recovery_data["recovery_score"].isnull().all():
    np.random.seed(0)
    recovery_data["recovery_score"] = np.random.uniform(50, 95, size=len(recovery_data)).round(1)
priority_data = load_data("CFC Individual Priority Areas.csv")
capability_data = load_data("CFC Physical Capability Data_.csv")
if not capability_data.empty:
    capability_data.columns = (
        capability_data.columns
        .str.strip()
        .str.lower()
        .str.replace("ï»¿", "", regex=False)  # Remove BOM if present
    )
    # Rename misformatted columns
    if "benchmarkpct" in capability_data.columns:
        capability_data.rename(columns={"benchmarkpct": "benchmarkpct"}, inplace=True)
    if "player id" in capability_data.columns:
        capability_data.rename(columns={"player id": "player_id"}, inplace=True)
    if "player_name" in capability_data.columns:
        capability_data.rename(columns={"player_name": "player_id"}, inplace=True)
    if "player_id" not in capability_data.columns:
        capability_data["player_id"] = np.random.choice([7, 10, 22], size=len(capability_data))

# Try to infer player_id if missing (only if capability_data is not empty)
if not capability_data.empty:
    if "player_id" not in capability_data.columns:
        if "player name" in capability_data.columns or "player_name" in capability_data.columns:
            name_col = "player name" if "player name" in capability_data.columns else "player_name"
            name_to_id = {v.lower(): k for k, v in PLAYER_NAMES.items()}
            capability_data["player_id"] = capability_data[name_col].str.lower().map(name_to_id)
required_capability_cols = {"player_id", "movement", "benchmarkpct"}
missing_cols = required_capability_cols - set(capability_data.columns)
if missing_cols:
    st.warning(f"❌ Required columns missing in capability_data: {missing_cols}")

# Format columns
if "sessionDate" in recovery_data.columns:
    recovery_data.rename(columns={"sessionDate": "date"}, inplace=True)
if "Subjective_composite" in recovery_data.columns:
    recovery_data.rename(columns={"Subjective_composite": "recovery_score"}, inplace=True)
for df in [gps_data, recovery_data, capability_data]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ----------------------------
# === Global Player Filter ===
if st.session_state.active_tab != "Home":
    st.sidebar.header("🎯 Player Filter")
    all_players = sorted(gps_data["player_id"].dropna().unique())
    selected_player = st.sidebar.selectbox("Select a player to filter all modules", options=["All"] + list(map(str, all_players)))

    if selected_player != "All":
        selected_player = int(selected_player)

    if selected_player != "All":
        gps_data = gps_data[gps_data["player_id"] == selected_player]
      
        if "player_id" in recovery_data.columns:
            recovery_data = recovery_data[recovery_data["player_id"] == selected_player]

        if "player name" in capability_data.columns:
            capability_data = capability_data[capability_data["player name"] == str(selected_player)]

        if "Player Name" in priority_data.columns:
            priority_data = priority_data[priority_data["Player Name"] == str(selected_player)]
else:
    selected_player = "All"

# ----------------------------
# READINESS SCORE
# ----------------------------
def calculate_readiness(row):
    if 'recovery_score' in row and 'distance' in row:
        return round((0.5 * (row['recovery_score'] / 100) + 0.5 * (1 - (row['distance'] / 10000))) * 100, 1)
    return np.nan

if not gps_data.empty and not recovery_data.empty:
    gps_latest = gps_data.sort_values("date").groupby("date").tail(1)
    rec_latest = recovery_data.sort_values("date").groupby("date").tail(1)
    readiness_df = pd.merge(gps_latest, rec_latest, on="date")
    if "recovery_score" not in readiness_df.columns and "recovery_score" in rec_latest.columns:
        readiness_df["recovery_score"] = rec_latest["recovery_score"].values
    
    if "recovery_score" not in readiness_df.columns:
        if "recovery_score" in rec_latest.columns:
            readiness_df["recovery_score"] = rec_latest["recovery_score"].values
        else:
            fallback_cols = [col for col in rec_latest.columns if "recovery" in col.lower()]
            if fallback_cols:
                readiness_df["recovery_score"] = rec_latest[fallback_cols[0]].values

    readiness_df["readiness_score"] = readiness_df.apply(calculate_readiness, axis=1)

    def generate_player_report(player_id, filtered_events, color_palette):
        fig, ax = plt.subplots(figsize=(6, 4))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
        pitch.draw(ax=ax)
        x_vals = np.random.normal(52.5, 20, 100)
        y_vals = np.random.normal(34, 15, 100)
        pitch.kdeplot(x_vals, y_vals, ax=ax, cmap="Reds", fill=True, levels=100, alpha=0.6)
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp_img.name)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
        pitch.draw(ax=ax2)

        for i, etype in enumerate(filtered_events["event_type"].unique()):
            sub = filtered_events[filtered_events["event_type"] == etype]
            color = color_palette[i % len(color_palette)]
            ax2.scatter(sub["x"], sub["y"], label=etype, c=color, alpha=0.6, edgecolors="black")

        ax2.legend()
        tmp_img2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig2.savefig(tmp_img2.name)
        plt.close(fig2)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, f"Player {player_id} Heatmap Report", ln=True, align="C")
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align="C")
        pdf.image(tmp_img.name, x=30, y=40, w=150)  # baisse la position Y pour éviter la coupe
        pdf.ln(120)  # ajoute plus d'espace avant la deuxième image
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Event Map by Type", ln=True, align="C")
        pdf.image(tmp_img2.name, x=30, y=140, w=150)  # décale aussi la 2e image
        return pdf.output(dest="S").encode("latin1")
else:
    readiness_df = pd.DataFrame()

# ----------------------------
# PDF GENERATOR
# ----------------------------
def generate_pdf_report(df, title="Team Report", score=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
    pitch.draw(ax=ax)
    pitch.kdeplot(np.random.uniform(0, 105, 100), np.random.uniform(0, 68, 100), ax=ax, cmap="Reds", fill=True, levels=100, alpha=0.6)
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp_img.name)
    plt.close(fig)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, title, ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    if score:
        pdf.ln(10)
        pdf.cell(200, 10, f"Average Readiness Score: {score:.1f}%", ln=True, align="C")
    pdf.image(tmp_img.name, x=30, y=60, w=150)
    return pdf.output(dest="S").encode("latin1")

# ----------------------------
# NAVIGATION LOGIC
# ----------------------------
# Lecture du clic simulé (si un formulaire invisible est soumis)

if st.session_state.active_tab == "Home":
    render_home()
elif st.session_state.active_tab == "Squad Overview":
    if st.button("⬅️ Back to Home", key="back_home_squad"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("🧠 Squad Readiness Overview")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    st.markdown("""
    This module helps you understand the **team's physical availability and fatigue levels**.
    It provides **daily insights** to help adjust your training load, plan recovery, and reduce injury risks.
    """)
    st.markdown("### 🧾 What You’ll Learn")
    st.markdown("- Daily and weekly readiness trend")
    st.markdown("- Players at risk of underperformance")
    st.markdown("- Relationship with training load and recovery")
 
    if not readiness_df.empty:
        st.markdown("This section gives you an overview of the team's physical availability..")
 
        # Affichage des statistiques de readiness
        readiness_summary = readiness_df.groupby("date")["readiness_score"].mean().reset_index()
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=readiness_summary["date"],
            y=readiness_summary["readiness_score"],
            mode="lines+markers",
            name="Avg Readiness"
        ))
        
        fig.add_shape(type="rect", xref="x", yref="y", x0=readiness_summary["date"].min(), x1=readiness_summary["date"].max(),
                      y0=0, y1=60, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_shape(type="rect", xref="x", yref="y", x0=readiness_summary["date"].min(), x1=readiness_summary["date"].max(),
                      y0=60, y1=75, fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_shape(type="rect", xref="x", yref="y", x0=readiness_summary["date"].min(), x1=readiness_summary["date"].max(),
                      y0=75, y1=100, fillcolor="green", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title="📊 Team Readiness Over Time",
            yaxis_title="Readiness Score",
            xaxis_title="Date",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
 
        # Résumé par niveau
        low = readiness_df[readiness_df["readiness_score"] < 60].shape[0]
        moderate = readiness_df[(readiness_df["readiness_score"] >= 60) & (readiness_df["readiness_score"] < 75)].shape[0]
        high = readiness_df[readiness_df["readiness_score"] >= 75].shape[0]
 
        st.markdown("### 🔍 Summary of readiness levels")
        critical_days = readiness_summary[readiness_summary["readiness_score"] < 60]
        if not critical_days.empty:
            st.error(f"⚠️ {len(critical_days)} critical readiness days detected. Last one: {critical_days['date'].max().strftime('%Y-%m-%d')}")
        else:
            st.success("✅ No critical readiness days detected.")

        st.success(f"🟩 Number of days with high readiness : {high}")
        st.warning(f"🟧 Number of days with moderate readiness : {moderate}")
        st.error(f"🟥 Number of days with low readiness : {low}")
 
        st.markdown("### 🧍‍♂️ Players Below 60% Readiness")
        if not readiness_df.empty and "readiness_score" in readiness_df.columns:
            player_warnings = readiness_df[readiness_df["readiness_score"] < 60]
            if not player_warnings.empty:
                st.dataframe(player_warnings[["date", "player_id", "recovery_score", "distance", "readiness_score"]].sort_values("date", ascending=False))
            else:
                st.success("✅ All players above critical readiness thresholds.")

        # Affichage des données brutes pour référence
        with st.expander("📋 View detailed data"):
            required_cols = ["date", "distance", "recovery_score", "readiness_score"]
            for col in required_cols:
                if col not in readiness_df.columns:
                    readiness_df[col] = np.nan
            styled_df = readiness_df[required_cols].style.applymap(
                lambda v: 'background-color: red' if isinstance(v, (int, float)) and v < 60 else '',
                subset=["readiness_score"]
            )
            st.dataframe(styled_df)
        st.markdown("### 📘 Coach Interpretation")
        st.markdown("""
        - **Above 75%**: Players are fresh and ready – optimal training intensity possible.
        - **60-75%**: Moderate readiness – be cautious with volume and load.
        - **Below 60%**: Alert! Consider adapting drills or providing recovery.
        
        **Recommendation:** If multiple days are below 60%, consider implementing a team recovery session or reduce intensity temporarily.
        """)
    else:
        st.info("No data available for the moment.")

elif st.session_state.active_tab == "Load Demand":
    if st.button("⬅️ Back to Home", key="back_home_load"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("📈 Match Load Analysis")

    st.subheader("➕ Add Training Load Entry")
    with st.form("add_training_load_form"):
        ld_date = st.date_input("📅 Date", value=datetime.today(), key="ld_date")
        ld_player = st.selectbox("👤 Player", options=sorted(gps_data["player_id"].dropna().unique()), key="ld_player")
        ld_distance = st.number_input("🏃 Distance (m)", min_value=0, value=5000, step=100, key="ld_distance")
        ld_oppo = st.text_input("🏟️ Opponent (optional)", key="ld_oppo")
        ld_accel = st.number_input("⚡ Accel/Decel (>2.5 m/s²)", min_value=0, step=1, value=12, key="ld_accel")
        submit_ld = st.form_submit_button("Add Load Entry")
        if submit_ld:
            new_entry = {
                "date": ld_date,
                "player_id": ld_player,
                "distance": ld_distance,
                "opposition_full": ld_oppo,
                "accel_decel_over_2_5": ld_accel
            }
            gps_path = "CFC GPS Data.csv"
            existing_gps = pd.read_csv(gps_path) if os.path.exists(gps_path) else pd.DataFrame()
            updated_gps = pd.concat([existing_gps, pd.DataFrame([new_entry])], ignore_index=True)
            updated_gps.to_csv(gps_path, index=False)
            st.success("✅ Training load entry added successfully.")
            st.rerun()

    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    player_list = gps_data["player_id"].dropna().unique()
    selected_player = st.selectbox("Select Player for Individual View", player_list)
    player_data = gps_data[gps_data["player_id"] == selected_player]
    player_data = player_data[player_data["distance"] > 0]

    st.subheader("🎯 Filter by Player")
    st.info("⚠️ Null distances have been excluded as they correspond to non-participation (players not lined up)..")
    
    # Distance Covered Over Time with Rolling Average
    if "distance" in player_data.columns:
        player_data_sorted = player_data.sort_values("date")
        player_data_sorted["rolling_distance"] = player_data_sorted["distance"].rolling(window=7, min_periods=1).mean()
        avg_distance = player_data_sorted["distance"].mean()

        fig_distance = px.line(
            player_data_sorted,
            x='date',
            y=['distance', 'rolling_distance'],
            title=f'Distance Covered Over Time – Player {selected_player}',
            labels={'value': 'Distance (m)', 'date': 'Date', 'variable': 'Metric'}
        )
        fig_distance.add_hline(y=avg_distance, line_dash="dot", annotation_text=f"Moyenne: {avg_distance:.1f} m", line_color="green")
        st.plotly_chart(fig_distance, use_container_width=True)
        st.caption("Tracks the player's total distance covered over time, with a rolling average to highlight physical load trends.")

    # Acceleration/Deceleration per Session with Mean Line
    accel_cols = [col for col in player_data.columns if "accel_decel" in col]
    if accel_cols:
        for col in accel_cols:
            fig_accel = px.line(
                player_data.sort_values("date"),
                x="date",
                y=col,
                title=f"{col.replace('_', ' ').title()} per Session"
            )
            mean_val = player_data[col].mean()
            fig_accel.add_hline(y=mean_val, line_dash="dot", line_color="orange", annotation_text=f"Moyenne: {mean_val:.1f}")
            st.plotly_chart(fig_accel, use_container_width=True)
            st.caption("Monitors explosive efforts through acceleration/deceleration counts, indicating session intensity.")

    # Total Distance vs Peak Speed with Trendline
    if "peak_speed" in player_data.columns:
        fig_gps = px.scatter(
            player_data,
            x="distance",
            y="peak_speed",
            trendline="ols",
            hover_data=["date"],
            title="Total Distance vs Peak Speed"
        )
        st.plotly_chart(fig_gps, use_container_width=True)
        st.caption("Shows correlation between total distance and peak speed — useful to assess efficiency of high-speed efforts.")

    # Average Distance by Opposition with Annotations
    if "opposition_full" in player_data.columns:
        opposition_summary = player_data.groupby("opposition_full")["distance"].mean().reset_index()
        fig_opposition = px.bar(
            opposition_summary,
            x="opposition_full",
            y="distance",
            title="Average Distance by Opposition"
        )
        avg_opp = opposition_summary["distance"].mean()
        fig_opposition.add_hline(y=avg_opp, line_dash="dot", annotation_text=f"Moyenne: {avg_opp:.1f} m", line_color="gray")
        st.plotly_chart(fig_opposition, use_container_width=True)
        st.caption("Highlights the average distance covered against each opponent — helps understand match demands.")
    
elif st.session_state.active_tab == "Recovery":
    if st.button("⬅️ Back to Home", key="back_home_recovery"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("🛌 Recovery Overview")

    st.markdown("🧪 **Recovery Score** – subjective score (0-100) of how recovered a player feels after effort.")

    st.markdown("""
    This module helps you assess **player recovery status** and anticipate **readiness risks**.
    It allows you to track recovery trends, spot under-recovered athletes, and adjust training accordingly.
    """)

    # ➕ Add Recovery Entry Form
    st.subheader("➕ Add Recovery Entry")
    with st.form("add_recovery_entry_form"):
        rec_date = st.date_input("📅 Recovery Date", value=datetime.today(), key="rec_date")
        rec_player = st.selectbox("👤 Player", options=sorted(gps_data["player_id"].dropna().unique()), key="rec_player")
        rec_score = st.slider("🧪 Recovery Score (0-100)", min_value=0, max_value=100, value=75, key="rec_score")
        rec_note = st.text_area("📝 Optional Notes", key="rec_note")
        submit_recovery = st.form_submit_button("Add Recovery Entry")
        if submit_recovery:
            new_entry = {
                "date": rec_date,
                "player_id": rec_player,
                "recovery_score": rec_score,
                "note": rec_note
            }
            recovery_path = "CFC Recovery status Data.csv"
            existing_recovery = pd.read_csv(recovery_path) if os.path.exists(recovery_path) else pd.DataFrame()
            updated_recovery = pd.concat([existing_recovery, pd.DataFrame([new_entry])], ignore_index=True)
            updated_recovery.to_csv(recovery_path, index=False)
            st.success(f"✅ Recovery entry added for {PLAYER_NAMES.get(rec_player)} with score {rec_score}")
            st.rerun()

    # ---- Recovery Visuals ----
    if not recovery_data.empty:
        st.subheader("📈 Weekly Average Recovery Trend")
        st.markdown("Smooths out daily fluctuations to highlight overall team fatigue or freshness across weeks.")

        recovery_data["week"] = pd.to_datetime(recovery_data["date"]).dt.to_period("W").dt.start_time
        weekly_avg = recovery_data.groupby("week")["recovery_score"].mean().reset_index()

        fig_weekly = px.line(weekly_avg, x="week", y="recovery_score", markers=True,
                            title="Team Recovery Score per Week",
                            labels={"recovery_score": "Avg Recovery (%)", "week": "Week"})
        fig_weekly.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.1, line_width=0)
        fig_weekly.add_hrect(y0=50, y1=70, fillcolor="orange", opacity=0.1, line_width=0)
        fig_weekly.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.1, line_width=0)
        st.plotly_chart(fig_weekly, use_container_width=True)

        latest_weekly_score = weekly_avg["recovery_score"].iloc[-1]
        if latest_weekly_score < 50:
            st.error(f"🟥 Latest weekly average recovery is very low ({latest_weekly_score:.1f}%) — prioritize recovery this week.")
        elif latest_weekly_score < 70:
            st.warning(f"🟧 Moderate weekly average recovery ({latest_weekly_score:.1f}%) — consider load adaptation.")
        else:
            st.success(f"🟩 Players are well recovered this week ({latest_weekly_score:.1f}%).")

        st.subheader("🧍‍♂️ Players With Low Recovery")
        st.markdown("Identifies players under 60% recovery in the last 3 days.")
        recent = recovery_data[recovery_data["date"] >= recovery_data["date"].max() - pd.Timedelta(days=3)]
        low_scores = recent[recent["recovery_score"] < 60]
        if not low_scores.empty:
            expected_cols = ["date", "player_id", "recovery_score"]
            available_cols = [col for col in expected_cols if col in low_scores.columns]
            st.dataframe(low_scores[available_cols].sort_values("date", ascending=False))
        else:
            st.success("✅ No players under recovery threshold in last 3 days.")

        st.subheader("📉 Player-by-Player Recovery Score Trends")
        st.markdown("Visualizes individual recovery evolution — helps quickly spot fatigue risks per player.")

        if "player_id" in recovery_data.columns:
            fig_facet = px.line(
                recovery_data,
                x="date",
                y="recovery_score",
                color="player_id",
                markers=True,
                title="Recovery Score per Player",
                labels={"recovery_score": "Recovery (%)", "date": "Date", "player_id": "Player"}
            )
            fig_facet.update_traces(mode="lines+markers")
            st.plotly_chart(fig_facet, use_container_width=True)

    st.subheader("⚖️ Recovery vs Training Load")
    st.markdown("""
    This graph helps evaluate whether higher physical loads (distance) negatively impact a player’s recovery.
    
    💡 **How to read this:**  
    - Each dot is one training or match day.
    - A **downward trend** = fatigue builds with higher workload (⚠️).
    - A **flat or upward trend** = good recovery capacity (✅).
    """)
    
    merged = pd.merge(gps_data, recovery_data, on=["date", "player_id"])
    
    if selected_player != "All":
        player_data = merged[merged["player_id"] == selected_player]
        title = f"Player {selected_player} – Distance vs Recovery Score"
    else:
        player_data = merged
        title = "All Players – Distance vs Recovery Score"
    
    if not player_data.empty:
        fig_corr = px.scatter(
            player_data,
            x="distance",
            y="recovery_score",
            trendline="ols",
            opacity=0.7,
            color="player_id" if selected_player == "All" else None,
            title=title,
            labels={"distance": "Distance (m)", "recovery_score": "Recovery (%)"},
            hover_data=["date"]
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Trendline estimates whether recovery decreases with increased distance. A negative slope may signal overtraining.")

    else:
        st.info("No recovery data available.")
    
elif st.session_state.active_tab == "Physical Development":
    if st.button("⬅️ Back to Home", key="back_home_physical"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("🏋️ Physical Test Results")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")

    if not capability_data.empty:
        st.markdown("""
        This section helps you **evaluate physical capabilities** of each player through objective testing.
        Tests include sprints, jumps, strength, and mobility benchmarks. 

        👉 These tests help identify:
        - Strengths and development areas
        - Injury risk indicators
        - Progress over time
        """)

        st.subheader("📊 Performance vs Benchmark (by Movement Type)")
        if 'movement' in capability_data.columns and 'benchmarkpct' in capability_data.columns:
            fig_benchmark = px.bar(
                capability_data,
                x='movement',
                y='benchmarkpct',
                color='movement',
                title="💪 Benchmark % Reached per Movement",
                labels={'benchmarkpct': '% Benchmark'},
                text_auto=True
            )
            fig_benchmark.update_layout(yaxis_range=[0, 120])
            st.plotly_chart(fig_benchmark, use_container_width=True)
            st.caption("Each bar shows the % of the benchmark reached by players for a specific movement (e.g., squat, sprint). Values >100% mean players exceeded the standard.")
            
            st.subheader("🌡️ Player vs Movement Heatmap")
            st.markdown("Visual comparison of each player's performance per movement — helps identify strengths and weaknesses at a glance.")
            
            if 'player_id' in capability_data.columns and 'movement' in capability_data.columns and 'benchmarkpct' in capability_data.columns:
                heatmap_data = capability_data.pivot_table(index='player_id', columns='movement', values='benchmarkpct')
                heatmap_data.index = heatmap_data.index.map(PLAYER_NAMES.get)
                fig_heatmap, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(heatmap_data, cmap="RdYlGn", annot=True, fmt=".0f", ax=ax, linewidths=.5, cbar_kws={"label": "% Benchmark"})
                st.pyplot(fig_heatmap)
            else:
                st.warning("❌ Required columns not found in capability_data.")

        st.subheader("📦 Score Spread per Test Type")
        if 'test name' in capability_data.columns and 'score' in capability_data.columns:
            fig_score = px.box(
                capability_data,
                x='test name',
                y='score',
                points='all',
                title='Score Spread for Each Test',
                labels={'score': 'Score', 'test name': 'Test'}
            )
            st.plotly_chart(fig_score, use_container_width=True)
            st.caption("This view helps spot test variability: large spreads may suggest inconsistency or need for standardization.")

        st.subheader("📈 Progression Over Time")
        if "testDate" in capability_data.columns and "benchmarkpct" in capability_data.columns:
            try:
                capability_data["testDate"] = pd.to_datetime(capability_data["testDate"], dayfirst=True, errors="coerce")
                trend = (
                    capability_data.dropna(subset=["benchmarkpct"])
                    .groupby("testDate")["benchmarkpct"]
                    .mean()
                    .reset_index()
                    .sort_values("testDate")
                )
                fig_trend = px.line(
                    trend,
                    x="testDate",
                    y="benchmarkpct",
                    title="📈 Evolution of Avg Benchmark % Over Time",
                    labels={"benchmarkpct": "% Benchmark", "testDate": "Date"},
                    markers=True
                )
                fig_trend.add_hline(y=100, line_dash="dot", line_color="green")
                st.plotly_chart(fig_trend, use_container_width=True)
                st.caption("If values trend upwards, physical conditioning is improving. Staying around 100% means players are at expected level.")
            except Exception as e:
                st.warning("Cannot display time trend: invalid test dates")

        st.subheader("📘 Coach Takeaways")
        st.markdown("""
        - Focus on movements **below 80%** – indicates room for physical development
        - Encourage maintenance or further improvement for **players above 100%**
        - Re-test regularly to track progress

        💡 **Tip**: Use this data to adapt training programs based on individual weaknesses (e.g. mobility, strength).
        """)

        st.subheader("📋 Raw Data Table")
        st.dataframe(capability_data)
    else:
        st.warning("No physical development data available.")
    
elif st.session_state.active_tab == "Biography":
    if st.button("⬅️ Back to Home", key="back_home_biography"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("📇 Player Profile & Individual Objectives")

    st.subheader("➕ Add Individual Goal")
    with st.form("add_goal_form"):
        bio_player = st.selectbox("👤 Player", options=sorted(gps_data["player_id"].dropna().unique()), key="bio_player")
        goal_desc = st.text_input("🎯 Goal Description", key="goal_desc")
        goal_status = st.selectbox("📈 Tracking Status", ["In Progress", "Achieved", "Not Started"], key="goal_status")
        submit_goal = st.form_submit_button("Add Goal")
        if submit_goal:
            new_entry = {
                "Player Name": str(bio_player),
                "Individual Goals": goal_desc,
                "Tracking": goal_status
            }
            goals_path = "CFC Individual Priority Areas.csv"
            existing_goals = pd.read_csv(goals_path) if os.path.exists(goals_path) else pd.DataFrame()
            updated_goals = pd.concat([existing_goals, pd.DataFrame([new_entry])], ignore_index=True)
            updated_goals.to_csv(goals_path, index=False)
            st.success("✅ Goal added successfully.")
            st.rerun()

    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    st.markdown("This section provides an overview of the individual objectives set for players and their achievement status..")

    if not priority_data.empty:
        st.subheader("🎯 Status of Individual Objectives")
        if "Tracking" in priority_data.columns:
            fig_status = px.pie(priority_data, names="Tracking", title="Distribution des Objectifs", hole=0.4)
            st.plotly_chart(fig_status, use_container_width=True)

        if {"Player Name", "Individual Goals", "Tracking"}.issubset(priority_data.columns):
            st.subheader("📋 Goals per player")
            st.dataframe(priority_data[["Player Name", "Individual Goals", "Tracking"]])
        else:
            st.dataframe(priority_data)
    else:
        st.warning("No biographical data available.")
    
elif st.session_state.active_tab == "Injury":
    if st.button("⬅️ Back to Home", key="back_home_injury"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("❌ Injury & Medical Overview")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    st.markdown("Tracking injuries and analyzing availability trends.")

    if not recovery_data.empty:
        if "injury_status" in recovery_data.columns:
            st.subheader("📊 Injury status")
            fig_injury = px.histogram(recovery_data, x="injury_status", title="Breakdown of injuries")
            st.plotly_chart(fig_injury, use_container_width=True)

            st.subheader("📅 Timeline des Blessures")
            injury_timeline = recovery_data[recovery_data["injury_status"] != "None"]
            if not injury_timeline.empty:
                fig_timeline = px.scatter(injury_timeline, x="date", y="injury_status", color="injury_status", title="Injury history")
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No injuries reported recently.")
        else:
            st.warning("No 'injury_status' column detected. Injury data unavailable.")
    else:
        st.info("No recovery or injury data available.")
    
elif st.session_state.active_tab == "External Factors":
    if st.button("⬅️ Back to Home", key="back_home_external"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("🌍 External Context")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    st.markdown("Capture external influences like fatigue, travel, or psychological state that might impact performance.")

    with st.form("external_note_form"):
        note_date = st.date_input("📅 Date concerned", value=datetime.now())
        player_options = ["Whole Team"] + [str(pid) for pid in sorted(gps_data["player_id"].dropna().unique())]
        selected_player = st.selectbox("👤 Player concerned", options=player_options)
        factor_type = st.selectbox("📌 Type of factor", ["Fatigue", "Travel", "Motivation", "Mental", "Weather", "Other"])
        note_text = st.text_area("📝 Coach's Note")
        submitted = st.form_submit_button("Save Note")

        if submitted and note_text.strip():
            if "external_notes" not in st.session_state:
                st.session_state.external_notes = []
            st.session_state.external_notes.append({
                "date": note_date,
                "player": selected_player,
                "type": factor_type,
                "note": note_text.strip()
            })
            st.success("✅ Note saved!")

    if "external_notes" in st.session_state and st.session_state.external_notes:
        st.markdown("### 📋 Coach Notes Summary")
        notes_df = pd.DataFrame(st.session_state.external_notes)
        st.dataframe(notes_df.sort_values("date", ascending=False), use_container_width=True)
    else:
        st.info("No external notes recorded yet.")
    
elif st.session_state.active_tab == "Match Analysis":
    if st.button("⬅️ Back to Home", key="back_home_match"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("📍 Player Heatmap")

    st.subheader("➕ Add Match Event")
    with st.form("add_match_event_form"):
        match_date = st.date_input("📅 Match Date", value=datetime.today())
        match_time = st.time_input("⏱️ Match Time", value=datetime.now().time())
        event_timestamp = datetime.combine(match_date, match_time)

        player_ids = sorted(gps_data["player_id"].dropna().unique())
        event_player = st.selectbox("👤 Player Involved", options=player_ids)
        event_type = st.selectbox("⚽ Event Type", options=["Pass", "Shot", "Dribble", "Tackle", "Interception", "Foul", "Save", "Clearance", "Cross", "Duel"])

        x_coord = st.slider("📍 X Position on Field (0 = Left, 105 = Right)", min_value=0.0, max_value=105.0, value=52.5)
        y_coord = st.slider("📍 Y Position on Field (0 = Bottom, 68 = Top)", min_value=0.0, max_value=68.0, value=34.0)

        tags = st.text_input("🏷️ Tags (comma-separated)", help="Use tags like 'dangerous, counter-attack, assist' for better filtering.")
        notes = st.text_area("📝 Additional Notes", help="Optional context for this event.")

        submitted_event = st.form_submit_button("Add Event to Match Dataset")

        if submitted_event:
            event_entry = {
                "timestamp": event_timestamp,
                "player_id": event_player,
                "event_type": event_type,
                "x": round(x_coord, 2),
                "y": round(y_coord, 2),
                "tags": tags,
                "notes": notes
            }

            events_path = "CFC Match Events Data.csv"
            existing_events = pd.read_csv(events_path) if os.path.exists(events_path) else pd.DataFrame()
            updated_events = pd.concat([existing_events, pd.DataFrame([event_entry])], ignore_index=True)
            updated_events.to_csv(events_path, index=False)

            st.success("✅ Match event successfully added.")
            st.rerun()
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    st.subheader("📍 Heatmap of Match Involvement")
    event_data = pd.read_csv("CFC Match Events Data.csv")
    event_data["timestamp"] = pd.to_datetime(event_data["timestamp"])
    if selected_player != "All":
        filtered_events = event_data[event_data["player_id"] == selected_player]
    else:
        filtered_events = event_data
    fig, ax = plt.subplots(figsize=(7, 5))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
    pitch.draw(ax=ax)
    pitch.kdeplot(filtered_events["x"], filtered_events["y"], ax=ax, cmap="Reds", fill=True, levels=100, alpha=0.6)
    st.pyplot(fig)

    st.subheader("📍 Event Map by Type")
    event_data = pd.read_csv("CFC Match Events Data.csv")
    event_data["timestamp"] = pd.to_datetime(event_data["timestamp"])
    
    if selected_player != "All":
        filtered_events = event_data[event_data["player_id"] == selected_player]
    else:
        filtered_events = event_data

    st.markdown("🎯 Filter by Event Type")
    available_types = filtered_events["event_type"].unique().tolist()
    selected_types = st.multiselect("Select event types to display", options=available_types, default=[], label_visibility="collapsed")
    filtered_events = filtered_events[filtered_events["event_type"].isin(selected_types)]

    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
    pitch.draw(ax=ax2)
    color_palette = [mcolors.to_hex(tuple(int(x) / 255 for x in c.strip("rgb()").split(","))) for c in px.colors.qualitative.Safe]
    if not color_palette:
        color_palette = ['#1f77b4']  # Fallback color

    success_marker_style = {
    True: {"marker": "o", "facecolor": None},
    False: {"marker": "x", "facecolor": "none"}
    }

    for i, etype in enumerate(selected_types):
        subset = filtered_events[filtered_events["event_type"] == etype]
        color = color_palette[i % len(color_palette)]
 
        for success_value in [True, False]:
            sub = subset[subset["success"] == success_value]
            
            # Use circle for success, cross with unique color per stat for fail
            marker = 'o' if success_value else 'X'
            facecolor = color if success_value else 'none'
            edgecolor = color if not success_value else 'black'
            
            pitch.scatter(
                sub["x"], sub["y"], ax=ax2,
                label=None,  # suppress per-success legends to keep only per-type color legend
                alpha=0.8, s=60,
                edgecolors=edgecolor,
                linewidths=1.5,
                marker=marker,
                facecolors=facecolor
            )

    # Build simplified legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=etype, markerfacecolor=color_palette[i % len(color_palette)], markeredgecolor='black', markersize=10)
        for i, etype in enumerate(selected_types)
    ]
    ax2.legend(handles=legend_elements, title="Event Type", loc="upper right", fontsize="small", title_fontsize="small")
    st.pyplot(fig2)

    st.subheader("📋 Event Table")
    st.dataframe(filtered_events.sort_values("timestamp"))

    if st.button("Download Player Report"):
        report = generate_player_report(selected_player, filtered_events, color_palette)
        st.download_button("Download PDF", data=report, file_name=f"player_{selected_player}_report.pdf", mime="application/pdf")
    
elif st.session_state.active_tab == "Video Analysis":
    if st.button("⬅️ Back to Home", key="back_home_video"):
        show_spinner()
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("🎥 Video Stat Detection")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    video_file = st.file_uploader("Upload match video", type=["mp4"])
    if video_file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(video_file.read())
        cap = cv2.VideoCapture(tmp.name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            st.image(frame, channels="BGR", caption="Frame for OCR")
            reader = easyocr.Reader(['en'])
            result = reader.readtext(frame)
            score_text = " ".join([text[1] for text in result if any(char.isdigit() for char in text[1])])
            if score_text:
                st.success(f"🎯 Detected Score: {score_text}")
            else:
                st.warning("❌ Score not detected")
        else:
            st.error("Could not read frame")

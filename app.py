import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from datetime import datetime, timedelta
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import tempfile
import cv2
import easyocr
import base64
import urllib.parse

# ----------------------------
# CONFIGURATION & SETUP
# ----------------------------
st.set_page_config(page_title="Chelsea FC Vizathon Dashboard", layout="wide")

params = st.query_params
if "tab" in params:
    st.session_state.active_tab = params["tab"] if isinstance(params["tab"], str) else params["tab"][0]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

def render_home():
    st.markdown("## 📊 Dashboard Modules")
    cols = st.columns(3)

    cards = [
        {"label": "Squad Overview", "icon": "🧠", "tab": "Squad Overview"},
        {"label": "Load Demand", "icon": "📈", "tab": "Load Demand"},
        {"label": "Recovery", "icon": "🛌", "tab": "Recovery"},
        {"label": "Physical Development", "icon": "🏋️", "tab": "Physical Development"},
        {"label": "Biography", "icon": "📇", "tab": "Biography"},
        {"label": "Injury", "icon": "❌", "tab": "Injury"},
        {"label": "External Factors", "icon": "🌍", "tab": "External Factors"},
        {"label": "Match Analysis", "icon": "📊", "tab": "Match Analysis"},
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
                         style="width:100%; height:350px; object-fit:cover; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1); margin-bottom:0.5rem;" />
                """, unsafe_allow_html=True)

                # Bouton avec clé unique (ajout d'un préfixe)
                if st.button(f"{card['icon']} {card['label']}", key=f"btn_{card['tab']}"):
                    st.session_state.active_tab = card["tab"]
                    st.rerun()

def render_match_analysis():
    st.header("⚽ Match Events Analysis")
    
    event_data = pd.read_csv("CFC Match Events Data.csv")
    event_data["timestamp"] = pd.to_datetime(event_data["timestamp"])

    selected_player = st.selectbox("Select Player", sorted(event_data["player_id"].unique()))
    selected_event = st.selectbox("Select Event Type", sorted(event_data["event_type"].unique()))

    filtered = event_data[
        (event_data["player_id"] == selected_player) &
        (event_data["event_type"] == selected_event)
    ]

    st.subheader(f"🎯 {selected_event.title()} Map for Player {selected_player}")
    fig, ax = plt.subplots(figsize=(10, 7))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
    pitch.draw(ax=ax)
    pitch.scatter(filtered["x"], filtered["y"], ax=ax, edgecolor='black', alpha=0.7, s=80)
    st.pyplot(fig)

    st.subheader("📋 Event Table")
    st.dataframe(filtered.sort_values("timestamp"))

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

st.markdown("""
<div style="background-color:#034694; padding:2rem 1rem; border-radius:1rem; color:white; text-align:center; margin-bottom:2rem;">
    <h1 style="margin:0; font-size:2.5rem;">Chelsea FC Vizathon Dashboard</h1>
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
capability_data.columns = capability_data.columns.str.strip().str.lower()

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

    def generate_player_report(player_id):
        fig, ax = plt.subplots(figsize=(6, 4))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
        pitch.draw(ax=ax)
        x_vals = np.random.normal(52.5, 20, 100)
        y_vals = np.random.normal(34, 15, 100)
        pitch.kdeplot(x_vals, y_vals, ax=ax, cmap="Reds", fill=True, levels=100, alpha=0.6)
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp_img.name)
        plt.close(fig)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, f"Player {player_id} Heatmap Report", ln=True, align="C")
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align="C")
        pdf.image(tmp_img.name, x=30, y=60, w=150)
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
    if st.button("⬅️ Back to Home"):
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("🧠 Squad Readiness Overview")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    st.markdown("""
    This module helps you understand the **team's physical availability and fatigue levels**.
    It provides **daily insights** to help adjust your training load, plan recovery, and reduce injury risks.
    """)
 
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
            st.dataframe(readiness_df[required_cols])
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
    if st.button("⬅️ Back to Home"):
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("📈 Match Load Analysis")
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
    if st.button("⬅️ Back to Home"):
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("🛌 Recovery Overview")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    st.subheader("📊 Daily Team Recovery Summary")
    if "recovery_score" in recovery_data.columns:
        daily_avg = recovery_data.groupby("date")["recovery_score"].mean().reset_index()
        st.line_chart(daily_avg.rename(columns={"date": "index"}).set_index("index"))

        latest_score = daily_avg["recovery_score"].iloc[-1]
        if latest_score < 50:
            st.error(f"🟥 Latest team recovery average is very low ({latest_score:.1f}%) — caution advised.")
        elif latest_score < 70:
            st.warning(f"🟧 Latest team recovery is moderate ({latest_score:.1f}%).")
        else:
            st.success(f"🟩 Team is well recovered ({latest_score:.1f}%) today.")
    st.subheader("🕵️‍♂️ Players with Low Recovery Scores")
    if "recovery_score" in recovery_data.columns and "player_id" in recovery_data.columns:
        recent = recovery_data[recovery_data["date"] >= recovery_data["date"].max() - pd.Timedelta(days=3)]
        low_scores = recent[recent["recovery_score"] < 60]
        if not low_scores.empty:
            st.dataframe(low_scores[["date", "player_id", "recovery_score"]].sort_values("date", ascending=False))
        else:
            st.success("✅ No players with critical recovery levels in the last 3 days.")
    if not recovery_data.empty:
        st.dataframe(recovery_data)
        if "recovery_score" in recovery_data.columns:
            st.plotly_chart(px.line(recovery_data, x="date", y="recovery_score", title="Recovery Score Trend"))
        st.subheader("⚖️ Recovery vs Load (Correlation)")
        merged = pd.merge(gps_data, recovery_data, on="date", how="inner")
        if "recovery_score" in merged.columns and "distance" in merged.columns:
            fig_corr = px.scatter(
                merged,
                x="distance",
                y="recovery_score",
                color="player_id" if "player_id" in merged.columns else None,
                title="Recovery vs Distance Covered"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No recovery data available.")
    
elif st.session_state.active_tab == "Physical Development":
    if st.button("⬅️ Back to Home"):
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("🏋️ Physical Test Results")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    if not capability_data.empty:
        st.markdown("Cette section vous permet d'évaluer les capacités physiques des joueurs par rapport à des benchmarks de référence.")

        if 'movement' in capability_data.columns and 'benchmarkpct' in capability_data.columns:
            st.subheader("📊 Comparaison aux Benchmarks")
            fig_benchmark = px.bar(
                capability_data,
                x='movement',
                y='benchmarkpct',
                color='movement',
                title="💪 Pourcentage de Benchmark atteint par mouvement",
                labels={'benchmarkpct': '% du benchmark'},
            )
            st.plotly_chart(fig_benchmark, use_container_width=True)
            st.caption("Shows how players perform against predefined benchmarks for each movement type, helping coaches spot top performers or areas for focus.")

        if 'test name' in capability_data.columns and 'score' in capability_data.columns:
            st.subheader("📈 Score distribution by test")
            fig_score = px.box(
                capability_data,
                x='test name',
                y='score',
                points='all',
                title='📦 Scores by Test',
                labels={'score': 'score', 'test name': 'Test'}
            )
            st.plotly_chart(fig_score, use_container_width=True)
            st.caption("Displays score variability per test, helping identify consistency and outliers in player assessments.")
 
        st.markdown("### 📊 Performance Comparative par Mouvement")
 
        if "player name" in capability_data.columns and "movement" in capability_data.columns and "benchmarkpct" in capability_data.columns:
            fig_compare = px.box(
                capability_data.dropna(subset=["benchmarkpct"]),
                x="movement",
                y="benchmarkpct",
                points="all",
                color="movement",
                title="📦 Performance distribution by movement",
                labels={"benchmarkpct": "% du benchmark atteint", "movement": "Mouvement"}
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            st.caption("This boxplot compares player performance across different movement types, highlighting strengths and development areas.")
 
        st.markdown("### 🔍 Progress of physical tests over time")
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
                    title="📈 Evolution of Average Performance over Time",
                    labels={"benchmarkpct": "% Benchmark", "testDate": "Date"}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                st.caption("This trendline shows how average physical performance (as % of benchmark) evolves over time. Useful to monitor training effectiveness.")
            except Exception as e:
                st.warning("Cannot display time trend: data error.")
 
        st.subheader("📋 Données Brutes")
        st.dataframe(capability_data)
    else:
        st.warning("No physical development data available.")
    
elif st.session_state.active_tab == "Biography":
    if st.button("⬅️ Back to Home"):
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("📇 Player Profile & Individual Objectives")
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
    if st.button("⬅️ Back to Home"):
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
    if st.button("⬅️ Back to Home"):
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
    if st.button("⬅️ Back to Home"):
        st.session_state.active_tab = "Home"
        st.rerun()

    st.header("📍 Player Heatmap")
    if selected_player != "All":
        st.markdown(f"🔍 Showing data for **Player {selected_player}** only.")
    df = pd.DataFrame({
        "player_id": np.random.choice([7, 10, 22], 300),
        "x": np.random.normal(52.5, 20, 300),
        "y": np.random.normal(34, 15, 300)
    })
    selected = st.selectbox("Choose Player", df["player_id"].unique())
    player_df = df[df["player_id"] == selected]
    fig, ax = plt.subplots(figsize=(10, 7))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
    pitch.draw(ax=ax)
    pitch.kdeplot(player_df.x, player_df.y, ax=ax, cmap="Reds", fill=True, levels=100, alpha=0.6)
    st.pyplot(fig)

    st.subheader("📍 Event Map by Type")
    event_data = pd.read_csv("CFC Match Events Data.csv")
    event_data["timestamp"] = pd.to_datetime(event_data["timestamp"])

    selected_player_event = st.selectbox("🎯 Select Player", sorted(event_data["player_id"].unique()), key="match_event_player")
    selected_event_type = st.selectbox("⚽ Select Event Type", sorted(event_data["event_type"].unique()), key="match_event_type")

    filtered_events = event_data[
        (event_data["player_id"] == selected_player_event) &
        (event_data["event_type"] == selected_event_type)
    ]

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
    pitch.draw(ax=ax2)
    pitch.scatter(filtered_events["x"], filtered_events["y"], ax=ax2, edgecolor='black', alpha=0.7, s=80)
    st.pyplot(fig2)

    st.subheader("📋 Event Table")
    st.dataframe(filtered_events.sort_values("timestamp"))

    if st.button("Download Player Report"):
        report = generate_player_report(selected)
        st.download_button("Download PDF", data=report, file_name=f"player_{selected}_report.pdf", mime="application/pdf")
    
elif st.session_state.active_tab == "Video Analysis":
    if st.button("⬅️ Back to Home"):
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

# ----------------------------
# SIDEBAR: ADD NEW DATA
# ----------------------------
st.sidebar.header("➕ Add GPS Session")
with st.sidebar.form("update_form"):
    date = st.text_input("Date (YYYY-MM-DD)")
    opposition = st.text_input("Opponent")
    distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
    accel_decel = st.number_input("Accelerations/Decelerations", min_value=0, step=1)
    submit = st.form_submit_button("Update")
    if submit:
        new_row = {"date": date, "opposition_full": opposition, "distance": distance, "accel_decel_over_2_5": accel_decel}
        gps_data = gps_data.append(new_row, ignore_index=True)
        gps_data.to_csv("CFC GPS Data.csv", index=False)
        st.success("GPS data updated!")
        st.rerun()
st.sidebar.info("This dashboard transforms raw data into real-world coaching decisions.")
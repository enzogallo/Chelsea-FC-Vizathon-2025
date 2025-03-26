import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import BytesIO
from fpdf import FPDF

# ✅ Page configuration (must be first Streamlit command)
st.set_page_config(page_title="Chelsea FC Vizathon Dashboard", layout="wide")

# --- Inject custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# --- Title and description ---
st.title("Chelsea FC Vizathon Dashboard")
st.markdown("This interactive dashboard allows coaches to monitor player performance in real time, update the data, and explore multiple modules (Load Demand, Injury, Physical Development, etc.).")

# --- Load data function ---
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.DataFrame()

# --- Generate sample GPS data ---
def generate_sample_gps_coordinates():
    data = {
        "player_id": np.random.choice([7, 10, 22], size=300),
        "x": np.random.normal(loc=52.5, scale=20, size=300),
        "y": np.random.normal(loc=34, scale=15, size=300),
        "timestamp": pd.date_range(start="2025-03-25 14:00", periods=300, freq="s")
    }
    return pd.DataFrame(data)

position_data = generate_sample_gps_coordinates()

# --- File paths ---
gps_data_path = "CFC GPS Data.csv"
priority_data_path = "CFC Individual Priority Areas.csv"
capability_data_path = "CFC Physical Capability Data_.csv"
recovery_data_path = "CFC Recovery status Data.csv"

# Load datasets
gps_data = load_data(gps_data_path)
priority_data = load_data(priority_data_path)
capability_data = load_data(capability_data_path)
recovery_data = load_data(recovery_data_path)

# Normalize column names
if "sessionDate" in recovery_data.columns:
    recovery_data.rename(columns={"sessionDate": "date"}, inplace=True)
if "Subjective_composite" in recovery_data.columns:
    recovery_data.rename(columns={"Subjective_composite": "recovery_score"}, inplace=True)

for df in [gps_data, recovery_data, capability_data]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

# --- Readiness calculation ---
def calculate_readiness_score(row):
    if 'recovery_score' in row and 'distance' in row:
        score = 0.5 * (row['recovery_score'] / 100) + 0.5 * (1 - (row['distance'] / 10000))
        return round(score * 100, 1)
    return np.nan

if not gps_data.empty and not recovery_data.empty:
    latest_gps = gps_data.sort_values("date").groupby("date").tail(1)
    latest_recovery = recovery_data.sort_values("date").groupby("date").tail(1)
    readiness_df = pd.merge(latest_gps, latest_recovery, on="date", suffixes=('_gps', '_rec'))
    readiness_df["readiness_score"] = readiness_df.apply(calculate_readiness_score, axis=1)
else:
    readiness_df = pd.DataFrame()

# --- Tabs ---
tabs = st.tabs(["Squad Overview", "Load Demand", "Injury", "Physical Development", "Biography", "Recovery", "External Factors", "Position Heatmap"])

# --- Squad Overview Tab ---
with tabs[0]:
    st.header("Squad Overview - Readiness & Load")
    if not readiness_df.empty:
        columns_to_display = [col for col in ["date", "distance", "recovery_score", "readiness_score"] if col in readiness_df.columns]
        if columns_to_display:
            st.dataframe(readiness_df[columns_to_display])
        else:
            st.warning("No readiness data available to display.")

        st.markdown("#### Readiness Alerts")
        for _, row in readiness_df.iterrows():
            if row.readiness_score < 60:
                st.error(f"⚠️ Player readiness score is low ({row.readiness_score}%) on {row.date.date()}")
            elif row.readiness_score < 75:
                st.warning(f"⚠️ Player readiness is moderate ({row.readiness_score}%) on {row.date.date()}")
            else:
                st.success(f"✅ Player readiness is good ({row.readiness_score}%) on {row.date.date()}")

        start_date = st.date_input("Report Start Date", value=datetime.now() - timedelta(days=7))
        end_date = st.date_input("Report End Date", value=datetime.now())

        if st.button("📥 Download Team PDF Report"):
            report = generate_global_report(start_date, end_date)
            st.download_button("Download Team Report", data=report, file_name="team_report.pdf")
    else:
        st.info("No readiness data available.")

# --- Load Demand Tab ---
with tabs[1]:
    st.header("Load Demand")
    st.markdown("**GPS data analysis**")
    if not gps_data.empty:
        st.dataframe(gps_data)
        if "distance" in gps_data.columns:
            fig = px.line(gps_data.sort_values("date"), x='date', y='distance', title='Distance Covered Over Time')
            st.plotly_chart(fig, use_container_width=True)

        accel_cols = [col for col in gps_data.columns if "accel_decel" in col]
        if accel_cols:
            fig_accel = px.line(gps_data.sort_values("date"), x="date", y=accel_cols, title="Acceleration/Deceleration per Session")
            st.plotly_chart(fig_accel, use_container_width=True)

        if "distance" in gps_data.columns and "peak_speed" in gps_data.columns:
            fig_gps = px.scatter(gps_data, x="distance", y="peak_speed", hover_data=["date"], title="Total Distance vs Peak Speed")
            st.plotly_chart(fig_gps, use_container_width=True)

        if "opposition_full" in gps_data.columns:
            opposition_summary = gps_data.groupby("opposition_full")["distance"].mean().reset_index()
            fig_opposition = px.bar(opposition_summary, x="opposition_full", y="distance", title="Average Distance by Opposition")
            st.plotly_chart(fig_opposition, use_container_width=True)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(gps_data.describe())
    else:
        st.info("No GPS data available.")

# --- Injury Tab ---
with tabs[2]:
    st.header("Injury")
    st.markdown("**Injury history and risk tracking**")
    if 'injury_status' in recovery_data.columns:
        fig_injury = px.histogram(recovery_data, x='injury_status', title='Current Injury Status Distribution')
        st.plotly_chart(fig_injury, use_container_width=True)
    else:
        st.info("Injury-specific data not available in current dataset.")

# --- Physical Development Tab ---
with tabs[3]:
    st.header("Physical Development")
    st.markdown("**Physical test results and strength indicators**")
    if not capability_data.empty:
        st.dataframe(capability_data)
        if 'MOVEMENTS' in capability_data.columns and 'BenchmarkPct' in capability_data.columns:
            fig2 = px.bar(capability_data, x='MOVEMENTS', y='BenchmarkPct', title='Benchmark by Movement')
            st.plotly_chart(fig2, use_container_width=True)
        if 'Score' in capability_data.columns:
            fig3 = px.box(capability_data, x='Test Name', y='Score', title='Distribution of Test Scores')
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No physical development data available.")

# --- Biography Tab ---
with tabs[4]:
    st.header("Biography")
    st.markdown("**Player profile and background**")
    if not priority_data.empty:
        st.dataframe(priority_data)
        if "Tracking" in priority_data.columns:
            status_counts = priority_data["Tracking"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_status = px.pie(status_counts, names="Status", values="Count", title="Goal Status Distribution")
            st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.info("No player biography data available.")

# --- Recovery Tab ---
with tabs[5]:
    st.header("Recovery")
    st.markdown("**Recovery metrics: subjective & physiological**")
    if not recovery_data.empty:
        st.dataframe(recovery_data)
        if 'recovery_score' in recovery_data.columns:
            fig3 = px.line(recovery_data.sort_values("date"), x='date', y='recovery_score', title='Recovery Score Over Time')
            st.plotly_chart(fig3, use_container_width=True)

        components = ["Sleep_composite", "Soreness_composite", "Nutrition_composite"]
        available = [col for col in components if col in recovery_data.columns]
        if available:
            fig_multi = px.line(recovery_data.sort_values("date"), x='date', y=available, title='Recovery Component Breakdown')
            st.plotly_chart(fig_multi, use_container_width=True)

        st.markdown("### Descriptive Statistics")
        st.dataframe(recovery_data.describe())
    else:
        st.info("No recovery data available.")

# --- External Factors Tab ---
with tabs[6]:
    st.header("External Factors")
    st.markdown("**Contextual elements influencing player performance**")
    external_notes = st.text_area("Coach Observations / External Notes", help="Enter any notes related to travel, motivation, team dynamics, or external context.")
    if external_notes:
        st.success("Observation saved. (In a real system, this would be recorded to database or exportable log.)")
    st.info("You can integrate external indicators such as sleep environment, family situation, or team morale if available.")

# --- Position Heatmap Tab ---
with tabs[7]:
    st.header("Position Heatmap")
    if not position_data.empty:
        selected_player = st.selectbox("Select a player ID", position_data["player_id"].unique())
        player_df = position_data[position_data["player_id"] == selected_player]

        fig, ax = plt.subplots(figsize=(10, 7))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='green', line_color='white')
        pitch.draw(ax=ax)
        pitch.kdeplot(player_df.x, player_df.y, ax=ax, cmap="Reds", fill=True, levels=100, alpha=0.6)
        st.pyplot(fig)

        if st.button("📥 Download Player PDF Report"):
            player_pdf = generate_player_report(selected_player)
            st.download_button("Download Player Report", data=player_pdf, file_name=f"player_{selected_player}_report.pdf")
    else:
        st.info("No position data available.")

# --- Sidebar for data updates ---
st.sidebar.header("Update Data")
st.sidebar.markdown("**Add a new GPS session**")

with st.sidebar.form("update_form"):
    date = st.text_input("Date (YYYY-MM-DD)")
    opposition = st.text_input("Opponent")
    distance = st.number_input("Distance covered (in km)", min_value=0.0, step=0.1)
    accel_decel = st.number_input("Accelerations/Decelerations (>2.5 m/s²)", min_value=0, step=1)
    submit = st.form_submit_button("Update")

    if submit:
        new_row = {
            "date": date,
            "opposition_full": opposition,
            "distance": distance,
            "accel_decel_over_2_5": accel_decel
        }
        gps_data = gps_data.append(new_row, ignore_index=True)
        gps_data.to_csv(gps_data_path, index=False)
        st.success("GPS data has been updated!")
        st.experimental_rerun()

st.sidebar.info("This interactive dashboard allows you to track player performance evolution in real time and update the data directly from the interface.")

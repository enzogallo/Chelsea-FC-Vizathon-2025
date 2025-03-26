import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page configuration ---
st.set_page_config(page_title="Chelsea FC Vizathon Dashboard", layout="wide")

# --- Inject custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply the style defined in "styles.css"
local_css("styles.css")

# --- Title and description ---
st.title("Chelsea FC Vizathon Dashboard")
st.markdown("This interactive dashboard allows coaches to monitor player performance in real time, "
            "update the data, and explore multiple modules (Load Demand, Injury, Physical Development, etc.).")

# --- Load data function ---
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding="ISO-8859-1")
    else:
        return pd.DataFrame()

# --- CSV file paths ---
gps_data_path = "CFC GPS Data.csv"
priority_data_path = "CFC Individual Priority Areas.csv"
capability_data_path = "CFC Physical Capability Data_.csv"
recovery_data_path = "CFC Recovery status Data.csv"

# Load datasets
gps_data = load_data(gps_data_path)
priority_data = load_data(priority_data_path)
capability_data = load_data(capability_data_path)
recovery_data = load_data(recovery_data_path)

# Convert date columns if present
for df in [gps_data, recovery_data, capability_data]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

# --- Tabs ---
tabs = st.tabs(["Load Demand", "Injury", "Physical Development", "Biography", "Recovery", "External Factors"])

# --- Load Demand Tab ---
with tabs[0]:
    st.header("Load Demand")
    st.markdown("**GPS data analysis**")

    if not gps_data.empty:
        st.dataframe(gps_data)

        if "distance" in gps_data.columns:
            fig = px.line(gps_data.sort_values("date"), x='date', y='distance', title='Distance Covered Over Time')
            st.plotly_chart(fig, use_container_width=True)

        accel_cols = [col for col in gps_data.columns if "accel_decel" in col]
        if accel_cols:
            fig_accel = px.line(gps_data.sort_values("date"), x="date", y=accel_cols,
                                title="Acceleration/Deceleration per Session")
            st.plotly_chart(fig_accel, use_container_width=True)

        if "distance" in gps_data.columns and "peak_speed" in gps_data.columns:
            fig_gps = px.scatter(gps_data, x="distance", y="peak_speed", hover_data=["date"],
                                 title="Total Distance vs Peak Speed")
            st.plotly_chart(fig_gps, use_container_width=True)

        if "opposition_full" in gps_data.columns:
            opposition_summary = gps_data.groupby("opposition_full")["distance"].mean().reset_index()
            fig_opposition = px.bar(opposition_summary, x="opposition_full", y="distance",
                                    title="Average Distance by Opposition")
            st.plotly_chart(fig_opposition, use_container_width=True)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(gps_data.describe())
    else:
        st.info("No GPS data available.")

# --- Injury Tab ---
with tabs[1]:
    st.header("Injury")
    st.markdown("**Injury history and monitoring**")
    st.write("Module to be implemented according to your specific data.")

# --- Physical Development Tab ---
with tabs[2]:
    st.header("Physical Development")
    st.markdown("**Physical capability analysis**")

    if not capability_data.empty:
        st.dataframe(capability_data)

        if 'Movements' in capability_data.columns and 'BenchmarkPct' in capability_data.columns:
            fig2 = px.bar(capability_data, x='Movements', y='BenchmarkPct', title='Physical Performance by Movement')
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No physical capability data available.")

# --- Biography Tab ---
with tabs[3]:
    st.header("Biography")
    st.markdown("**Details and personal information**")
    if not priority_data.empty:
        st.dataframe(priority_data)

        if "Tracking" in priority_data.columns:
            status_counts = priority_data["Tracking"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_status = px.pie(status_counts, names="Status", values="Count",
                                title="Goal Status Distribution")
            st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.info("No biographical data available.")

# --- Recovery Tab ---
with tabs[4]:
    st.header("Recovery")
    st.markdown("**Monitoring recovery indicators**")

    if not recovery_data.empty:
        st.dataframe(recovery_data)

        if 'emboss_baseline_score' in recovery_data.columns:
            fig3 = px.line(recovery_data.sort_values("date"), x='date', y='emboss_baseline_score',
                           title='Global Recovery Score')
            st.plotly_chart(fig3, use_container_width=True)

        cols_to_plot = ["hr_zone_1_hms", "hr_zone_2_hms", "hr_zone_3_hms", "hr_zone_4_hms", "hr_zone_5_hms"]
        available_cols = [col for col in cols_to_plot if col in recovery_data.columns]
        if available_cols:
            fig_zones = px.line(recovery_data.sort_values("date"), x='date', y=available_cols,
                                title="Heart Rate Zones")
            st.plotly_chart(fig_zones, use_container_width=True)

        st.markdown("### Descriptive Statistics")
        st.dataframe(recovery_data.describe())
    else:
        st.info("No recovery data available.")

# --- External Factors Tab ---
with tabs[5]:
    st.header("External Factors")
    st.markdown("**External factors that may influence performance**")
    st.write("Module to be implemented according to available data.")

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

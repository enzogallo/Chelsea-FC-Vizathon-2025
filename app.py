import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Chelsea FC Vizathon", layout="wide")

# Function to load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Loading the CSS
local_css("styles.css")

# Data loading functions with caching
@st.cache_data
def load_gps_data():
    # Load GPS data
    return pd.read_csv("CFC GPS Data.csv", encoding="ISO-8859-1")

@st.cache_data
def load_physical_data():
    # Load physical capability data
    return pd.read_csv("CFC Physical Capability Data_.csv", encoding="ISO-8859-1")

@st.cache_data
def load_recovery_data():
    # Load recovery data
    return pd.read_csv("CFC Recovery status Data.csv", encoding="ISO-8859-1")

@st.cache_data
def load_priority_data():
    # Load Individual Priority Areas data
    return pd.read_csv("CFC Individual Priority Areas.csv", encoding="ISO-8859-1")

# Loading the data
gps_data = load_gps_data()
physical_data = load_physical_data()
recovery_data = load_recovery_data()
priority_data = load_priority_data()

# Converting 'date' column to datetime (if present)
for df in [gps_data, recovery_data, physical_data]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Dashboard", "Recovery Status", "Physical Capability", "GPS Performance", "Priority Areas"])

# PAGE 1: Dashboard
if page == "Dashboard":
    st.title("Dashboard - Global Overview")
    st.markdown("## Overview of Player Performance")
    
    st.subheader("Key Metrics")
    # Global recovery score
    if "emboss_baseline_score" in recovery_data.columns:
        latest_recovery = recovery_data.sort_values("date")["emboss_baseline_score"].dropna().iloc[-1]
        st.metric(label="Global Recovery Score", value=f"{latest_recovery:.2f}")
    else:
        st.write("Recovery score not available.")
    
    # Total distance from the latest GPS session
    if "distance" in gps_data.columns:
        latest_distance = gps_data.sort_values("date")["distance"].dropna().iloc[-1]
        st.metric(label="Total Distance (m)", value=f"{latest_distance:.0f} m")
    else:
        st.write("Distance data not available.")
    
    # Goals achieved
    if "Tracking" in priority_data.columns:
        achieved = priority_data[priority_data["Tracking"].str.lower() == "achieved"].shape[0]
        total = priority_data.shape[0]
        st.metric(label="Goals Achieved", value=f"{achieved} / {total}")
    else:
        st.write("Goals data not available.")
    
    st.markdown("---")
    st.subheader("Recent Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Global Recovery Score Trend")
        if "emboss_baseline_score" in recovery_data.columns:
            fig_recovery = px.line(recovery_data.sort_values("date"), x="date", y="emboss_baseline_score",
                                   title="Global Recovery Score")
            st.plotly_chart(fig_recovery, use_container_width=True)
        else:
            st.write("Recovery data not available.")
    
    with col2:
        st.markdown("### Total Distance Trend")
        if "distance" in gps_data.columns:
            fig_distance = px.line(gps_data.sort_values("date"), x="date", y="distance",
                                   title="Total Distance per Session")
            st.plotly_chart(fig_distance, use_container_width=True)
        else:
            st.write("GPS data not available.")

# PAGE 2: Recovery Status
elif page == "Recovery Status":
    st.title("Recovery Analysis")
    st.markdown("### Recovery Metrics Details")
    
    # List of columns to display for recovery
    cols_to_plot = ["emboss_baseline_score", "hr_zone_1_hms", "hr_zone_2_hms", 
                    "hr_zone_3_hms", "hr_zone_4_hms", "hr_zone_5_hms"]
    available_cols = [col for col in cols_to_plot if col in recovery_data.columns]
    
    if available_cols:
        fig_recovery_details = px.line(recovery_data.sort_values("date"), x="date", y=available_cols,
                                       title="Recovery Metrics Trend")
        st.plotly_chart(fig_recovery_details, use_container_width=True)
    else:
        st.write("No recovery metrics available at the moment.")
    
    st.markdown("### Descriptive Statistics")
    st.dataframe(recovery_data.describe())

# PAGE 3: Physical Capability
elif page == "Physical Capability":
    st.title("Physical Capabilities")
    st.markdown("### Strength and Performance Test Analysis")
    
    # Visualization by movement if columns exist
    if "Movements" in physical_data.columns and "BenchmarkPct" in physical_data.columns:
        st.markdown("#### Benchmark by Movement")
        fig_physical = px.bar(physical_data, x="Movements", y="BenchmarkPct", color="Movements",
                              title="Benchmark by Movement")
        st.plotly_chart(fig_physical, use_container_width=True)
    else:
        st.write("The 'Movements' or 'BenchmarkPct' columns are not available in the physical data.")
    
    st.markdown("#### Physical Performance Details")
    st.dataframe(physical_data)

# PAGE 4: GPS Performance
elif page == "GPS Performance":
    st.title("GPS Data and Performance")
    st.markdown("### Field Performance Data Analysis")
    
    st.markdown("#### Relationship between Distance and Speed")
    if "distance" in gps_data.columns and "peak_speed" in gps_data.columns:
        fig_gps = px.scatter(gps_data, x="distance", y="peak_speed", hover_data=["date"],
                             title="Total Distance vs Peak Speed")
        st.plotly_chart(fig_gps, use_container_width=True)
    else:
        st.write("The 'distance' or 'peak_speed' columns are not available.")
    
    st.markdown("#### Acceleration/Deceleration Trends")
    accel_cols = [col for col in gps_data.columns if "accel_decel" in col]
    if accel_cols:
        fig_accel = px.line(gps_data.sort_values("date"), x="date", y=accel_cols,
                            title="Acceleration/Deceleration per Session")
        st.plotly_chart(fig_accel, use_container_width=True)
    else:
        st.write("Acceleration/Deceleration data not available.")
    
    st.markdown("#### Opposition Analysis")
    if "opposition_full" in gps_data.columns and "distance" in gps_data.columns:
        opposition_summary = gps_data.groupby("opposition_full")["distance"].mean().reset_index()
        fig_opposition = px.bar(opposition_summary, x="opposition_full", y="distance",
                                title="Average Distance by Opposition")
        st.plotly_chart(fig_opposition, use_container_width=True)
    else:
        st.write("Opposition data not available.")
    
    st.markdown("#### Descriptive Statistics")
    st.dataframe(gps_data.describe())

# PAGE 5: Priority Areas
elif page == "Priority Areas":
    st.title("Individual Goals")
    st.markdown("### Tracking Individual Priorities")
    
    st.markdown("#### Goals Table")
    st.dataframe(priority_data)
    
    st.markdown("#### Goal Status")
    if "Tracking" in priority_data.columns:
        status_counts = priority_data["Tracking"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_status = px.pie(status_counts, names="Status", values="Count",
                            title="Goal Status Distribution")
        st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.write("Goal tracking data not available.")

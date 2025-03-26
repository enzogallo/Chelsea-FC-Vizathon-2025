Voici une version améliorée et plus claire de ton fichier :  

---

# **Chelsea FC Vizathon - Performance Visualization Dashboard**  

## **📌 Project Description**  
This interactive dashboard provides **in-depth analysis and visualization** of football players' **physical performance** using multiple datasets. It helps **coaches and analysts** make data-driven decisions regarding **workload, recovery, and athletic abilities**.  

---

## **📊 Datasets Used**  

### **1. GPS Data**  
➡️ **Tracks player movements on the field, measuring workload (distance, speed, accelerations, etc.).**  

| **Name** | **Explanation** |
|----------|---------------|
| `date` | Date of the session or match |
| `opposition_code` | Opponent team code |
| `opposition_full` | Full name of the opponent team |
| `md_plus_code` | Days after the match |
| `md_minus_code` | Days before the match |
| `season` | Current season |
| `distance` | Total distance covered (meters) |
| `distance_over_21` | Distance covered at over 21 km/h |
| `distance_over_24` | Distance covered at over 24 km/h |
| `distance_over_27` | Distance covered at over 27 km/h |
| `accel_decel_over_2_5` | Accelerations/decelerations above 2.5 m/s² |
| `accel_decel_over_3_5` | Accelerations/decelerations above 3.5 m/s² |
| `accel_decel_over_4_5` | Accelerations/decelerations above 4.5 m/s² |
| `day_duration` | Total session duration (minutes) |
| `peak_speed` | Maximum speed reached (km/h) |
| `hr_zone_1_hms` to `hr_zone_5_hms` | Time spent in different heart rate zones |

📌 **Usage:** Tracking workload, identifying high-intensity phases, preventing fatigue.  

---

### **2. Physical Capability Data**  
➡️ **Measures players' athletic abilities (sprint, agility, strength, jumping, etc.).**  

| **Name** | **Explanation** |
|----------|---------------|
| `MOVEMENTS` | Type of movement (Sprint, Agility, Jump) |
| `QUALITY` | Measured quality (Acceleration, Strength, Rotation) |
| `EXPRESSION` | Type of force measured (Isometric or Dynamic) |
| `BenchmarkPct` | Score (%) compared to reference players |

📌 **Usage:** Comparing players to benchmarks, tracking performance progress over seasons.  

---

### **3. Recovery Status Data**  
➡️ **Indicates player recovery levels after matches or training sessions.**  

| **Name** | **Explanation** |
|----------|---------------|
| `Bio_composite` | Biological fatigue and inflammation level |
| `Msk_joint_range_composite` | Joint mobility (hips, knees) |
| `Msk_load_tolerance_composite` | Muscle load tolerance capacity |
| `Subjective_composite` | Player's perceived recovery status |
| `Soreness_composite` | Muscle soreness level |
| `Sleep_composite` | Sleep quality |

📌 **Usage:** Adjusting training load, preventing injuries.  

---

### **4. Individual Priority Areas**  
➡️ **Tracks personalized objectives for each player.**  

| **Name** | **Explanation** |
|----------|---------------|
| `Priority Category` | Priority category (Recovery, Performance) |
| `Area` | Specific focus area (Sleep, Sprint) |
| `Target` | Goal set (e.g., Increase sleep by 1 hour) |
| `Performance Type` | Indicator type (Habitual or Outcome-based) |
| `Target set` | Date when the goal was set |
| `Review Date` | Date for review |
| `Tracking` | Progress tracking status |

📌 **Usage:** Ensuring personalized player development and tracking improvements.  

---

## **📊 Dashboard Features**  
✔️ **Interactive GPS performance visualization** (distances, speeds, accelerations).  
✔️ **Graph-based analysis of physical capabilities** (strength, sprint, agility).  
✔️ **Recovery and injury prevention insights.**  
✔️ **Heatmaps on a football pitch to visualize player movements.**  
✔️ **Data filtering by player, match, and time period.**  

---

## **🛠️ Technologies Used**  
- **Python** (`pandas`, `numpy`) for data analysis.  
- **Streamlit** for the interactive interface.  
- **Matplotlib, Seaborn, Plotly** for visualization.  
- **CSS** for UI enhancements.  

---

## **🚀 Installation & Launch**  
1️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```  
2️⃣ **Run the dashboard**  
```bash
streamlit run app.py
```  

---

## **🔮 Future Improvements**  
🔹 Integration of **real GPS coordinates** for the heatmap.  
🔹 Adding **injury risk analysis** based on past performance.  
🔹 **Enhanced UI customization** for better coaching insights.  

---

## **👤 Author**  
This project was developed as part of the **Chelsea FC Vizathon** to optimize football player performance analysis. 🚀  

---

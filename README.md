# ⚽ Chelsea FC Vizathon 2025 – Elite Player Monitoring Dashboard

Welcome to the **Chelsea FC Vizathon 2025** project – a data visualization dashboard built for elite performance monitoring, player readiness, and coaching decision-making.  
This solution was developed as part of the [Chelsea FC Performance Insights Vizathon](https://chelsea-fc-performance-insights.github.io/Competition/).

🔗 **Live Demo**: [enzogallochelseafcvizathon2025.streamlit.app](https://enzogallochelseafcvizathon2025.streamlit.app/)

---

## 🧠 Purpose of the Project

The dashboard aims to empower coaches and performance analysts with a **centralized, visual interface** for tracking:

- Physical load and sprint efforts
- Recovery status and readiness
- Injury history and return-to-play risk
- Long-term physical development
- Match involvement and key events
- External factors (e.g. fatigue, travel, mental state)

---

## 🧰 Tech Stack

| Component     | Tool / Library            |
|---------------|---------------------------|
| Frontend      | [Streamlit](https://streamlit.io) |
| Visualizations| Plotly, matplotlib, seaborn, mplsoccer |
| PDF Reports   | FPDF (auto-generated insights) |
| Data Handling | Pandas, NumPy             |
| File Storage  | CSV-based mock data       |
| Image OCR     | OpenCV + EasyOCR (for future vision module) |

---

## 🚀 How to Use the App

You can directly explore the full app online:

🔗 **[Launch the App](https://enzogallochelseafcvizathon2025.streamlit.app/)**

The dashboard is organized into the following modules (tabs):

| Module                          | Description |
|----------------------------------|-------------|
| **Home**                         | Quick navigation and module overview |
| **Squad Overview**               | Daily readiness score for all players (based on distance + recovery) |
| **Load Demand**                  | Monitor physical load (distance, accel/decel, opposition) |
| **Recovery**                     | Weekly and individual recovery trends + subjective scores |
| **Sprint & High Intensity Zones** | Total explosive actions (>2.5 m/s²) per player |
| **Physical Development**         | Test results vs benchmarks + progression |
| **Injury**                       | Injury history, type, and frequency timeline |
| **External Factors**             | Mental, fatigue, travel notes impacting performance |
| **Match Analysis**               | Add and explore match events, heatmaps & timelines |
| **Biography**                    | Long-term Individual Development Plans (IDPs) |

---

## 📦 Data Sources (Mocked for Vizathon)

All data used in this demo is **simulated or anonymized**, including:
- GPS tracking (distance, high intensity)
- Match events (passes, shots, duels)
- Recovery scores
- Injury logs
- Player development objectives

---

## 📄 Key Features

- 🧠 **Readiness Score Algorithm** combining GPS + Recovery
- ⚡ **High-Intensity Zone Mapping**
- 📊 **Match Event Heatmaps & Replay Timelines**
- 🧾 **Auto-generated PDF Reports per Player**
- 🧍‍♂️ **Player Development Plans** (editable & tracked over time)
- 🧠 **Smart Color Zones** (traffic light system for thresholds)
- 🌍 **External factor logging** for holistic decision-making

---

## 💡 Future Improvements (Next Step Ideas)

- Connect to **a real backend (FastAPI + DB)** to replace CSVs
- **Authentication system** for multiple coaches
- Real-time match event streaming
- Integrate **computer vision** module for sprint direction mapping
- Dashboard packaging as **SaaS** for professional clubs

---

## 👨‍💻 Author

Made by **Enzo Gallo**  
Computer Science Engineer & Junior Football Analyst

- LinkedIn: *https://www.linkedin.com/in/enzo-gallo/*  
- Contact: *e.gallo1024@gmail.com*

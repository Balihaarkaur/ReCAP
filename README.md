# 🏙️ UrbanCrowd Insight — Real-Time Crowd Monitoring & Heatmap Analytics

**UrbanCrowd Insight** is an intelligent real-time crowd monitoring system that detects people from live video streams, computes crowd density, and generates visual heatmaps using **YOLOv8**, **OpenCV**, and **Streamlit**.

This project enables monitoring of crowd formation, congestion, and safety risks using AI-based visual analysis.

---

## 🚀 Features

###  YOLOv8-Based Real-Time Detection
- Uses Ultralytics YOLOv8 for person detection  
- Works on both **uploaded videos** and **webcam**  
- Displays annotated bounding boxes

###  Intelligent Crowd Status  
Automatically classifies:
- 🟢 Safe  
- 🟠 Dense  
- 🔴 Overcrowded  

Supports user-adjustable threshold controls.

### Heatmap Generation (Dual System)
1. **Gaussian Density Heatmap**  
   - Uses person centroids  
   - Visualizes density hotspots  
2. **Activation CAM (Class Activation Map)**  
   - Extracts YOLO’s Conv layer activations  
   - Highlights model-focused regions  

### 📊 Streamlit Dashboard
- Live count  
- Density %  
- Status indicator  
- Crowd history graph  
- Side-by-side **Annotated Frame + Heatmap View**  
- Downloadable frames & CSV history  

### 💾 Data Export
- Download **annotated frames (PNG)**  
- Download **heatmap frames (PNG)**  
- Export **crowd history (CSV)**  

### 🧩 Custom Controls
- Upload video / webcam  
- Density threshold slider  
- Sensitivity slider  
- Start/Stop processing  
- Optional React-based UI controls

---

## 🛠 Tech Stack

**Backend / ML**
- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  

**Frontend**
- Streamlit  
- Custom React Streamlit Component  

**Visualization**
- Gaussian heatmaps  
- CAM activation maps  
- Line charts  

---

## 📂 Project Structure

UrbanCrowdInsight/
│
├── app.py # Main Streamlit application
├── streamlit_crowdsense_component/ # Custom React UI component
├── frontend/ # React UI source
│ ├── src/
│ ├── dist/
│ └── webpack.config.js
│
├── yolov8n.pt # YOLOv8 model weights
├── test_heatmap.png # Sample output images
├── test_annotated.png
├── uploaded_video.mp4 # Temporary inputs
├── requirements.txt # Dependencies
└── streamlit_error.log # Logs  


---

## ▶️ How It Works

### 1️ Input Source  
User selects:
- Uploaded video  
- Webcam
  
### 2️ YOLOv8 detects people  
python
results = model(frame)
person_count = len(results[0].boxes)

### 3 Status Calculation
ratio = count / MAX_CAPACITY
Classifies → Safe / Dense / Overcrowded

### 4️ Heatmap Generation
Gaussian centroid-based heatmap
CNN Activation CAM heatmap

### 5️ Real-Time Dashboard
Displays:
Annotated frame
Heatmap
Count
Density
Status
History chart

### 6️ Export
User can download frames & CSV.

### 🖥️ Screenshots
(Add screenshots or a Google Drive link)

🎯 Use Cases

Smart city monitoring

Mall and airport crowd tracking

Public event congestion analysis

Railway/metro station monitoring

Emergency crowd control

👩‍💻 Author

Balihaar Kaur

GitHub: https://github.com/Balihaarkaur

LinkedIn: https://linkedin.com/in/BalihaarKaur

### Contributions

Pull requests are welcome!
Please open an issue for bug fixes or feature suggestions.

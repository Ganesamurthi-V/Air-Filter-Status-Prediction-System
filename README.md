
╔══════════════════════════════════════════════════════════╗
║     AC AIR FILTER RECOMMENDATION SYSTEM USING TSMIXER              
╚══════════════════════════════════════════════════════════╝

## 📌 Overview
This project implements a **time-series based deep learning system** to predict the **cleanliness status of an Air Conditioner (AC) air filter** — classified as **Clean** or **Dirty**.  
The system is built using the **TSMixer architecture** and integrated with a **Flask API** for real-time inference.

---

## 🎯 Goal
- Predict AC air filter condition using sensor time-series data  
- Reduce manual inspection using ML-based recommendations  
- Enable real-time predictions via API  
- Design a lightweight and deployable model  

---

## 🔄 Data Ingestion (API Based)
```
┌──────────────────────────────┐
│        Data Sources          │
├──────────────────────────────┤
│ 1. Periodic Data             │
│ 2. Compressor Session Data   │
│ 3. Session Data              │
└──────────────────────────────┘
```

### Periodic Data
- Collected every ~5 seconds  
- Includes Power, Temperature, Voltage, Timestamp  

Feature derived:
```
power_per_deg = power / (temperature / 10)
```

### Compressor Session Data
- Compressor ON duration  
- Energy consumed per session  

### Session Data
- AC ON–OFF usage cycles  
- Triggered when AC is switched OFF via mobile app  

---

## 🧠 Why TSMixer?
- Pure feed-forward architecture (no LSTM / Transformer)
- Faster training and inference  
- Low memory usage  
- Easier debugging  
- Suitable for edge and API deployment  

---

## 🧹 Data Preprocessing
- Datetime parsing (`ds` column)
- Feature engineering (`power_per_deg`)
- Removal of invalid / null values  
- Min-Max normalization (0–1)
- Fixed-length sequence construction (`SEQ_LEN = 10`)  
- Zero-padding for short sequences  

---

## 🧩 Model Architecture
```
┌──────────────────────────────┐
│        TSMixer Model         │
├──────────────────────────────┤
│  Token Mixer                 │
│  Channel Mixer               │
│  Classifier Head             │
└──────────────────────────────┘
```

- **Token Mixer**: Learns temporal dependencies  
- **Channel Mixer**: Learns feature interactions  
- **Classifier Head**: Outputs binary prediction  

---

## 🏋️ Training Strategy
### Stage 1: Regression
- Predicts next power_per_deg value  
- Loss: Mean Squared Error  

### Stage 2: Classification
- Binary output (Clean / Dirty)  
- Loss: Custom Focal Loss  
- Optimizer: Adam  
- Scheduler: ReduceLROnPlateau  
- Early stopping applied  

---

## 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

Final prediction is based on **aggregate session results**, not a single inference.

---

## 🌐 Flask API Inference Pipeline
```
Frontend → Flask API → Preprocessing → TSMixer → Prediction
```

### Decision Logic
```
If >30% sessions are DIRTY → Filter = DIRTY
Else → Filter = CLEAN
```

---

## 📝 Logging
- API calls  
- Data padding warnings  
- Model loading  
- Prediction results  
- Errors & exceptions  

Logs are stored in `app.log`.

---

## ⚙️ Installation & Execution
```bash
pip install -r requirements.txt
python model.py
python app.py
```

---

## ⚠️ Limitations
- Limited dataset size  
- Class imbalance  
- Static thresholding  
- API dependency  
- Feature limitations  

---

## ✅ Conclusion
This project demonstrates an **end-to-end time-series ML pipeline** using TSMixer for AC air filter health prediction.  
While the system architecture and integration are complete, **further tuning and validation are required before production deployment**.



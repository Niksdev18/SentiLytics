# 📊 SENTI-LYTICS – Financial News Sentiment Analysis Dashboard

Senti-Lytics is a Python-powered system that fetches real-time financial news, analyzes sentiment using a fine-tuned Transformer model (BERT), retrieves stock data, and displays intelligent insights through a Flask-based dashboard.

---

## 🚀 Features

- 🔍 **Real-time financial news fetching** using NewsAPI  
- 🤖 **Transformer-based Sentiment Analysis** (Fine-tuned BERT model)  
- 📈 **Stock before/after price tracking** + automatic % change calculation  
- 📊 **Smart trading suggestions** based on sentiment & movement  
- 🧠 **Accuracy metrics** compared against actual stock reactions  
- 💻 **Flask Web Dashboard** for visualization  
- ⏱️ **Automatic background updates** every 2 hours (APScheduler)  
- 🗂️ Supports multiple companies & ETF fallback rules  

---

## 📁 Project Structure

```
SENTI-LYTICS/
│
├── data/
│   ├── notebook/           # Jupyter notebooks
│   ├── raw/                # Unprocessed raw data
│   └── processed/          # Cleaned datasets + metrics
│
├── models/
│   ├── transformer_model/  # Fine-tuned BERT model (local)
│   ├── *.pkl               # Trained ML models (TF-IDF, SVM, NB, LR)
│   └── model_metrics.csv   # ML model comparison results
│
├── src/
│   ├── api/
│   │   ├── app.py          # Flask backend + scheduler
│   │   └── templates/      # HTML UI files
│   ├── models/             # Model loading/inference scripts
│   └── preprocessing/      # Data cleaning + transformation
│
├── .env                    # Environment variables
├── config.yaml             # API keys, paths, model configs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone <repo_url>
cd SENTI-LYTICS
```

---

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# or
venv\Scripts\activate         # Windows
```

---

### 3️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure API Keys
Rename:

```
config_template.yaml → config.yaml
```

Edit `config.yaml` and add:
- NewsAPI key  
- Model paths  
- Scheduler interval  
- Stock mapping rules  

---

### 5️⃣ Run the Flask Application
```bash
python src/api/app.py
```

Flask will start at:

```
http://127.0.0.1:5000/
```

---

## 📝 Notes

- Pre-trained ML models & BERT model are stored in `models/` (ignored by Git).  
- All preprocessing scripts live inside `src/preprocessing/`.  
- Jupyter notebooks for experiments are located in `data/notebook/`.  
- Processed datasets & accuracy files (e.g., `accuracy_metrics.csv`) are stored inside `data/processed/`.

---

## 📌 Future Enhancements (Optional)
- Live stock chart using WebSockets  
- Email alert system for sentiment spikes  
- Multi-language sentiment analysis  
- Docker support + CI/CD pipeline  
- Deployment on Render / AWS / Vercel  

---



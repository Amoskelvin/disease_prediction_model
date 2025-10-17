# 🧠 Disease Prediction Model

This project uses **machine learning (Random Forest Classifier)** to predict possible diseases based on a person's symptoms.

### 🔍 Motivation
I built this out of curiosity after a class discussion about AI in healthcare.  
Since I couldn’t find real hospital data, I created a custom dataset of common diseases and their symptoms.

### ⚙️ How It Works
- **Dataset:** Manually created with 150+ records  
- **Model:** Random Forest Classifier (Scikit-learn)  
- **Accuracy:** ~96%  
- **Input:** Age, sex, and 40 binary symptom features  

### 🚀 Usage
1. Clone the repo:
   ```bash
   git clone https://github.com/Amoskelvin/disease_prediction_model.git
   cd disease_prediction_model

2. Run the prediction script:
    python sickness_predict.py
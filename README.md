# 🏥 Personalized Healthcare Assistant

<img width="1509" height="945" alt="Screenshot 2025-11-24 at 7 25 09 PM" src="https://github.com/user-attachments/assets/a6b8b93f-8542-4b11-abdc-0d8b10bc04ed" />


A Machine Learning–powered desktop application that predicts **diseases from symptoms** and provides **helpful recommendations**.  
This project includes data preprocessing, model training, evaluation, and an interactive **Tkinter GUI** for end-users.

---

## 📌 Dataset  

This project uses the publicly available **Diseases and Symptoms Dataset** from Kaggle:

🔗 **Dataset Link:**  
https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset

Download the dataset and place it in your project folder before running the script.

---

## 📂 Project Structure
```
├── healthcare final.py          # Main script (ML + GUI)
├── dataset.csv                  # Dataset from Kaggle
├── models/
│   ├── model_nb.pkl             # Naive Bayes model
│   ├── model_dt.pkl             # Decision Tree model
│   ├── model_mlp.pkl            # Neural Network model
│   └── scaler.pkl               # StandardScaler (for MLP)
└── README.md                    # Documentation
```

---

## ⭐ Features

### ✔ Data Preprocessing  
- Loads symptom–disease dataset  
- Cleans, encodes, and transforms symptoms into binary vectors  
- Splits data into training and testing sets  

### ✔ Multiple Machine Learning Models  
The script trains and saves:
- **Multinomial Naive Bayes**  
- **Decision Tree Classifier**  
- **MLP Neural Network (Deep Learning)**  

It also generates:
- Accuracy reports  
- Precision, recall, F1-score  
- Confusion matrix  
- ROC curves  

### ✔ GUI for Disease Prediction  
The Tkinter interface allows users to:
- Select symptoms from a list  
- Predict disease using the saved ML model  
- View detailed suggestions / recommendations  

---

## 🧰 Requirements

Install dependencies:

```
pip install pandas numpy scikit-learn joblib matplotlib tkinter
```

Tkinter is preinstalled in most Python distributions.

---

## ▶️ How to Run

1. **Download dataset** from Kaggle  
   Place the CSV file in the same directory as the script.

2. **Run the Python script:**
```
python "healthcare final.py"
```

3. The **GUI window will open**  
   - Select symptoms  
   - Click **Predict Disease**  
   - Read the predicted disease & recommendations  

---

## 🖥 GUI Overview
- Left panel: Select symptoms from a scrollable list  
- Right panel:  
  - Shows predicted disease  
  - Displays recommendations  
- Models are automatically loaded from the `/models` directory  
- Footer shows where trained models are saved  

---

## 📊 Model Evaluation Metrics Stored
The script automatically evaluates models using:
- Accuracy  
- F1-score  
- Classification report  
- ROC curves  
- Confusion matrix  

These results appear in the terminal/Colab output.

---

## 👩‍💻 Author
Developed as a Machine Learning project for symptom-based disease prediction using Python, Sklearn, and Tkinter.

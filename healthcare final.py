import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import traceback
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, ttk

# -----------------------
# Config / Paths
# -----------------------
DATA_PATH = Path(r"/Users/momeenaazhar/Desktop/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
OUT_DIR = Path("/Users/momeenaazhar/Desktop/disease_prediction_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------
# 1) Detect target column
# -----------------------
possible_targets = [c for c in df.columns if any(k in c.lower() for k in ["disease","diagnosis","prognosis","condition","label","target"])]
target_col = possible_targets[0] if possible_targets else df.columns[-1]
print("Using target column:", target_col)
print("Target distribution (top):")
print(df[target_col].value_counts().head(20).to_string())

# -----------------------
# 2) Detect symptom columns
# -----------------------
symptom_binary = []
for c in df.columns:
    if c == target_col:
        continue
    ser = df[c].dropna().astype(str)
    uniques = set(s.lower().strip() for s in ser.unique())
    if uniques.issubset({'0','1','yes','no','y','n'}):
        symptom_binary.append(c)

symptom_text_cols = [c for c in df.columns if 'symptom' in c.lower() or 'symptoms' in c.lower()]
for c in df.select_dtypes(include=['object','category']).columns:
    if c == target_col or c in symptom_text_cols:
        continue
    sample = df[c].dropna().astype(str).head(200).tolist()
    if any(',' in s for s in sample):
        symptom_text_cols.append(c)
symptom_text_cols = list(dict.fromkeys(symptom_text_cols))

print("Detected binary symptom columns:", symptom_binary)
print("Detected symptom-text candidates:", symptom_text_cols)

# -----------------------
# 3) Build features X and labels y
# -----------------------
y = df[target_col].astype(str).fillna("unknown")
X = df.drop(columns=[target_col]).copy()

# Normalize binary symptom columns
for c in symptom_binary:
    X[c] = X[c].astype(str).str.strip().str.lower().replace({'yes':'1','y':'1','no':'0','n':'0'}).astype(float)

# Expand symptom-text column
mlb = None
if symptom_text_cols:
    sym_col = symptom_text_cols[0]
    print("Expanding symptoms from column:", sym_col)
    symptom_lists = X[sym_col].fillna("").astype(str).apply(lambda s: [t.strip().lower() for t in s.split(",") if t.strip()])
    mlb = MultiLabelBinarizer(sparse_output=False)
    try:
        sym_mtx = mlb.fit_transform(symptom_lists)
        sym_df = pd.DataFrame(sym_mtx, columns=[f"sym_{c}" for c in mlb.classes_], index=X.index)
        X = pd.concat([X.drop(columns=[sym_col]), sym_df], axis=1)
        print(f"Expanded {sym_df.shape[1]} symptom features.")
    except Exception as e:
        print("Failed to expand symptom-text column:", e)
        traceback.print_exc()
else:
    print("No symptom-text column found. Using binary symptom columns and other features.")

# -----------------------
# 4) Metadata integration
# -----------------------
meta_keywords = ['age','gender','sex','bmi','smok','smoking','alcohol','lifestyle','occupation','weight','height','marital']
meta_cols = [c for c in X.columns if any(k in c.lower() for k in meta_keywords)]
print("Detected metadata columns:", meta_cols)

obj_cols = [c for c in X.select_dtypes(include=['object','category']).columns if not c.startswith('sym_')]
onehot_cols = [c for c in obj_cols if X[c].nunique() <= 20]
if onehot_cols:
    print("One-hot encoding columns:", onehot_cols)
    X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)

for c in X.columns:
    if X[c].dtype.kind in 'biufc':
        X[c] = X[c].fillna(X[c].median())
    else:
        X[c] = X[c].fillna("missing")

print("Final feature shape:", X.shape)

# -----------------------
# 5) Label encode y with rare class handling
# -----------------------
y_counts = Counter(y)
rare_threshold = 2
y_safe = y.apply(lambda cls: cls if y_counts[cls] >= rare_threshold else "Other")

le = LabelEncoder()
y_enc = le.fit_transform(y_safe)
classes = list(le.classes_)
print("Label classes count (after rare class handling):", len(classes))
print("Some labels:", classes[:20])

# -----------------------
# 6) Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print("Train/Test split done. Shapes:", X_train.shape, X_test.shape)

# -----------------------
# 7) Scale numeric features
# -----------------------
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
if numeric_cols:
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# -----------------------
# 8) Models
# -----------------------
use_multinomial = False
try:
    if (X_train.fillna(0) >= 0).all().all():
        integer_like_cols = sum(1 for c in X_train.columns if (X_train[c].dropna() % 1 == 0).all())
        if integer_like_cols > 0:
            use_multinomial = True
except Exception:
    use_multinomial = False

nb_model = MultinomialNB() if use_multinomial else GaussianNB()
dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=1,          # 1 epoch per fit
    warm_start=True,     # continue training
    random_state=42,
    verbose=False
)

# Helper: safe classification report
def safe_classification_report(y_true, y_pred, le):
    labels_in_test = np.unique(y_true)
    return classification_report(
        y_true,
        y_pred,
        labels=labels_in_test,
        target_names=[le.classes_[i] for i in labels_in_test],
        zero_division=0
    )

models = {'NaiveBayes': nb_model, 'DecisionTree': dt_model, 'NeuralNet': nn_model}
trained_models = {}
results = {}

# NeuralNet epoch settings
NUM_EPOCHS = 50

for name, model in models.items():
    print(f"\nTraining {name} ...")
    try:
        if name == 'NaiveBayes':
            model.fit(X_train.fillna(0).values, y_train)
            preds = model.predict(X_test.fillna(0).values)

        elif name == 'NeuralNet':
            for epoch in range(NUM_EPOCHS):
                print(f"NeuralNet Epoch {epoch+1}/{NUM_EPOCHS}")
                model.fit(X_train_scaled.values, y_train)
                preds = model.predict(X_test_scaled.values)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                print(f" -> Accuracy: {acc:.4f}, F1-weighted: {f1:.4f}")
            # final predictions after last epoch
            preds = model.predict(X_test_scaled.values)

        else:  # DecisionTree
            model.fit(X_train.values, y_train)
            preds = model.predict(X_test.values)

        # store metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        results[name] = {'accuracy': float(acc), 'f1_weighted': float(f1)}
        trained_models[name] = model

        print(f"{name} Final Accuracy: {acc:.4f}, F1-weighted: {f1:.4f}")
        print(safe_classification_report(y_test, preds, le))

    except Exception as e:
        print(f"{name} failed:", e)
        traceback.print_exc()

# -----------------------
# 9) Save models and artifacts
# -----------------------
for nm, m in trained_models.items():
    joblib.dump(m, MODEL_DIR / f"model_{nm}.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
with open(OUT_DIR / "label_classes.json", "w") as f:
    json.dump(classes, f, indent=2)
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved models and results to:", OUT_DIR)

# -----------------------
# 10) Recommendations
# -----------------------
recommendations = {}
for cls in classes:
    cls_low = cls.lower()
    if any(k in cls_low for k in ['cancer','pneumonia','tuberculosis','tb','covid','heart','stroke','sepsis']):
        recs = ["Immediate medical attention recommended", "Follow clinician instructions", "Avoid self-medication"]
    elif any(k in cls_low for k in ['diabetes','pre-diabetes','hyperglycemia','hypoglycemia']):
        recs = ["Monitor blood glucose regularly", "Adopt a balanced diet and regular exercise", "Consult an endocrinologist or your PCP"]
    elif any(k in cls_low for k in ['cold','flu','fever','dengue','malaria','virus']):
        recs = ["Rest and hydrate", "Use symptomatic treatment as advised", "See primary care if symptoms worsen"]
    else:
        recs = ["Consult a healthcare professional for diagnosis", "Avoid unverified treatments", "Keep records of symptoms and family history"]
    recommendations[cls] = recs
if "Other" not in recommendations:
    recommendations["Other"] = ["Consult a healthcare professional for diagnosis", "Avoid self-medication", "Keep records of symptoms and family history"]

with open(OUT_DIR / "recommendations.json", "w") as f:
    json.dump(recommendations, f, indent=2)
print("Saved recommendations to:", OUT_DIR / "recommendations.json")

# -----------------------
# 11) Demo predictions with recommendations
# -----------------------
best_name = max(results.items(), key=lambda kv: kv[1]['f1_weighted'])[0] if results else None
best_model = trained_models.get(best_name)
if best_name:
    if best_name == 'NeuralNet':
        demo_X = X_test_scaled[:min(50, X_test_scaled.shape[0])]
        demo_preds = best_model.predict(demo_X.values)
    elif best_name == 'NaiveBayes':
        demo_X = X_test[:min(50, X_test.shape[0])].fillna(0)
        demo_preds = best_model.predict(demo_X.values)
    else:
        demo_X = X_test[:min(50, X_test.shape[0])]
        demo_preds = best_model.predict(demo_X.values)
    demo_labels = le.inverse_transform(demo_preds)
    demo_df = pd.DataFrame(demo_X if isinstance(demo_X, np.ndarray) else demo_X.reset_index(drop=True))
    demo_df['predicted_label'] = demo_labels
    demo_df['recommendations'] = demo_df['predicted_label'].apply(lambda L: recommendations.get(L, ["See clinician"]))
    demo_path = OUT_DIR / "demo_predictions_with_recommendations.csv"
    demo_df.to_csv(demo_path, index=False)
    print("Saved demo predictions with recommendations to:", demo_path)

# -----------------------
# 12) GUI
# -----------------------
symptom_features = [c for c in X.columns if c.startswith('sym_')] + symptom_binary
seen = set(); symptom_features = [x for x in (symptom_features) if not (x in seen or seen.add(x))]
MAX_SYMPTOMS_DISPLAY = 60
show_symptoms = symptom_features[:MAX_SYMPTOMS_DISPLAY]

gui_meta = meta_cols[:4] if meta_cols else ['Age', 'Gender']

best_model_for_gui = best_model
scaler_for_gui = scaler
label_encoder = le
recommend_map = recommendations

def prepare_input_from_gui(symptom_vars, meta_entries):
    row = {}
    for col in X.columns:
        if col in symptom_features:
            row[col] = 1.0 if (symptom_vars.get(col) and symptom_vars[col].get()) else 0.0
        elif col in meta_entries:
            val = meta_entries.get(col, "")
            try:
                row[col] = float(val)
            except:
                prefix_cols = [c for c in X.columns if c.startswith(col + '_')]
                if prefix_cols:
                    for p in prefix_cols:
                        row[p] = 0.0
                    match_col = f"{col}_{val}"
                    if match_col in row:
                        row[match_col] = 1.0
                else:
                    row[col] = 0.0
        else:
            row[col] = 0.0
    return pd.DataFrame([row], columns=X.columns)

def gui_predict(symptom_vars, meta_entries):
    try:
        inp_df = prepare_input_from_gui(symptom_vars, meta_entries)
        inp = inp_df.copy()
        if numeric_cols:
            inp[numeric_cols] = scaler_for_gui.transform(inp[numeric_cols])
        if best_name == 'NeuralNet':
            pred_enc = best_model_for_gui.predict(inp.values)
        elif best_name == 'NaiveBayes':
            pred_enc = best_model_for_gui.predict(inp.fillna(0).values)
        else:
            pred_enc = best_model_for_gui.predict(inp.values)
        pred_label = label_encoder.inverse_transform(pred_enc)[0]
        recs = recommend_map.get(pred_label, ["Consult a healthcare professional"])
        return pred_label, recs
    except Exception as e:
        print("GUI prediction failed:", e)
        traceback.print_exc()
        return None, ["Prediction error"]

# ---------- Prepare readable symptom list for GUI ----------
# mlb may be None if no symptom-text column was expanded
mlb_symptoms = []
if mlb is not None:
    # keep only non-numeric classes (filter out '0','1' etc)
    mlb_symptoms = [c for c in mlb.classes_ if not str(c).strip().replace('.', '', 1).isdigit()]

# Build show_symptoms from expanded symptoms (prefixed with 'sym_') + binary symptom columns
show_symptoms = [f"sym_{s}" for s in mlb_symptoms] + symptom_binary

# Deduplicate while preserving order
seen = set()
show_symptoms = [s for s in show_symptoms if not (s in seen or seen.add(s))]


# ----------------------- GUI -----------------------
root = tk.Tk()
root.title("Personalized Healthcare Assistant")
root.geometry("900x760")
root.configure(bg='#C2D7DA')  # requested background color

# Title
title = tk.Label(root,
                 text="Personalized Healthcare Assistant",
                 font=("Arial", 18, "bold"),
                 bg='#C2D7DA', fg='black')   # heading text black on light bg
title.pack(pady=10)

frame_top = tk.Frame(root, bg='#C2D7DA')
frame_top.pack(fill='x', padx=12)

# Metadata frame
meta_frame = tk.LabelFrame(frame_top,
                           text="Patient Metadata (optional)",
                           padx=8, pady=8,
                           bg='#C2D7DA', fg='black',
                           font=("Arial", 11, "bold"))
meta_frame.pack(side='left', padx=8, pady=6, fill='y')

meta_entries = {}
for m in gui_meta:
    lbl = tk.Label(meta_frame, text=str(m), bg='#C2D7DA', fg='black', font=("Arial", 10, "bold"))
    lbl.pack(anchor='w', pady=2)
    ent = tk.Entry(meta_frame)
    ent.pack(fill='x', pady=2)
    meta_entries[m] = ent

# Symptoms frame (with scrollable canvas)
symptom_frame = tk.LabelFrame(frame_top,
                              text="Symptoms (select all that apply)",
                              padx=8, pady=8,
                              bg='#C2D7DA', fg='black',
                              font=("Arial", 11, "bold"))
symptom_frame.pack(side='left', padx=8, pady=6, fill='both', expand=True)

canvas = tk.Canvas(symptom_frame, bg='#C2D7DA', highlightthickness=0)
scrollbar = tk.Scrollbar(symptom_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg='#C2D7DA')

def _on_configure(e):
    canvas.configure(scrollregion=canvas.bbox("all"))
scrollable_frame.bind("<Configure>", _on_configure)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set, height=420)

# Create checkboxes in a 2-column grid for better layout (adjust columns if needed)
symptom_vars = {}
cols = 2
r = 0; c = 0
for s in show_symptoms:
    # display name cleanup
    s_str = str(s)
    if s_str.startswith('sym_'):
        display_name = s_str[4:].replace('_', ' ').title()
    else:
        display_name = s_str.replace('_', ' ').title()

    var = tk.IntVar(value=0)
    # selectcolor sets the color behind checkbox when selected; indicatoron True keeps default checkbox
    cb = tk.Checkbutton(scrollable_frame, text=display_name, variable=var,
                        anchor='w', bg='#C2D7DA', fg='black',
                        selectcolor='#ffffff', activebackground='#C2D7DA',
                        font=("Arial", 10))
    # place in grid
    cb.grid(row=r, column=c, sticky='w', padx=6, pady=3)
    symptom_vars[s] = var

    # advance grid
    c += 1
    if c >= cols:
        c = 0; r += 1

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Output frame
out_frame = tk.LabelFrame(root, text="Prediction & Recommendations",
                          padx=8, pady=8,
                          bg='#C2D7DA', fg='black',
                          font=("Arial", 11, "bold"))
out_frame.pack(fill='both', padx=12, pady=10, expand=True)

result_var = tk.StringVar()
rec_text = tk.Text(out_frame, height=10, wrap='word', bg='white', fg='black', font=("Arial", 10))

def on_predict():
    meta_vals = {m: meta_entries[m].get() for m in meta_entries}
    pred_label, recs = gui_predict(symptom_vars, meta_vals)
    if pred_label is None:
        messagebox.showerror("Error", "Prediction failed. Check console for details.")
        return
    # show result with label and color
    result_var.set(f"Predicted disease: {pred_label}")
    lbl_res.config(fg='red', bg='#C2D7DA')  # red text for disease
    rec_text.delete('1.0', tk.END)
    rec_text.insert(tk.END, "\n".join(recs))

predict_btn = tk.Button(out_frame, text="Predict Disease", command=on_predict,
                        bg="#4a90e2", fg="white", font=("Arial", 11, "bold"))
predict_btn.pack(pady=6)

lbl_res = tk.Label(out_frame, textvariable=result_var, font=("Arial", 12, "bold"), bg='#C2D7DA', fg='red')
lbl_res.pack(pady=6)
rec_text.pack(fill='both', expand=True)

footer = tk.Label(root, text=f"Models saved to: {MODEL_DIR}  | Labels: {', '.join(classes[:10])} ...",
                  font=("Arial", 8), bg='#C2D7DA', fg='black')
footer.pack(side='bottom', pady=6)

root.mainloop()

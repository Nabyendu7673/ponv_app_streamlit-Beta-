import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

st.set_page_config(layout="wide")

st.title("Hybrid PONV Machine Learning Algorithm [Beta Testing]")
st.markdown("Dr Nabyendu Biswas, Department of Pharmacology, MKCG Medical College and Hospital")

# Increase font size using HTML and inline CSS
st.markdown(
    """
    <p style="font-size: 20px;">
        <b>This hybrid model is designed based on multiple PONV risk scores, including Apfel, Koivuranta, and Bellville scores, in alignment with the POTTER app developed by Massachusetts Medical School, in collaboration with the Department of Anaesthesiology, MKCG Medical College & Hospital.</b>
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("PONV Risk Assessment Parameters")

# ------------------------- PATIENT FACTORS -------------------------
gender = st.sidebar.selectbox("Female Gender", ["No", "Yes"])
smoker = st.sidebar.selectbox("Non-Smoker", ["No", "Yes"])
history_ponv = st.sidebar.selectbox("History of PONV or Motion Sickness", ["No", "Yes"])
age = st.sidebar.slider("Age", 18, 80, 35)
preop_anxiety = st.sidebar.selectbox("Preoperative Anxiety", ["No", "Yes"])
history_migraine = st.sidebar.selectbox("History of Migraine", ["No", "Yes"])
obesity = st.sidebar.selectbox("BMI > 30", ["No", "Yes"])

# ------------------------- SURGICAL FACTORS -------------------------
abdominal_surgery = st.sidebar.selectbox("Abdominal or Laparoscopic Surgery", ["No", "Yes"])
ent_surgery = st.sidebar.selectbox("ENT/Neurosurgery/Ophthalmic Surgery", ["No", "Yes"])
gynae_surgery = st.sidebar.selectbox("Gynecological or Breast Surgery", ["No", "Yes"])
surgery_duration = st.sidebar.selectbox("Surgery Duration > 60 min", ["No", "Yes"])
major_blood_loss = st.sidebar.selectbox("Major Blood Loss > 500 mL", ["No", "Yes"])
volatile_agents = st.sidebar.selectbox("Use of Volatile Agents (Sevo/Iso/Des)", ["No", "Yes"])
nitrous_oxide = st.sidebar.selectbox("Use of Nitrous Oxide", ["No", "Yes"])

# ------------------------- DRUG FACTORS (WITH DOSE) -------------------------
st.sidebar.header("Drug Administration (Specify Dose)")

midazolam_dose = st.sidebar.number_input("Midazolam (mg)", 0.0, 10.0, 0.0)
ondansetron_dose = st.sidebar.number_input("Ondansetron (mg)", 0.0, 24.0, 0.0)
dexamethasone_dose = st.sidebar.number_input("Dexamethasone (mg)", 0.0, 40.0, 0.0)
glycopyrrolate_dose = st.sidebar.number_input("Glycopyrrolate (mg)", 0.0, 0.4, 0.0)
nalbuphine_dose = st.sidebar.number_input("Nalbuphine (mg)", 0.0, 160.0, 0.0)
fentanyl_dose = st.sidebar.number_input("Fentanyl (mcg)", 0.0, 2000.0, 0.0)
butorphanol_dose = st.sidebar.number_input("Butorphanol (mg)", 0.0, 4.0, 0.0)
pentazocine_dose = st.sidebar.number_input("Pentazocine (mg)", 0.0, 360.0, 0.0)

propofol_mode = st.sidebar.selectbox("Propofol Use", ["None", "TIVA", "Induction Only"])
muscle_relaxant = st.sidebar.selectbox("Muscle Relaxant Used", ["None", "Atracurium", "Cisatracurium", "Vecuronium", "Succinylcholine"])

# ------------------------- POSTOPERATIVE SYMPTOMS -------------------------
nausea = st.sidebar.selectbox("Nausea >30 min", ["No", "Yes"])
vomiting = st.sidebar.selectbox(">2 Episodes of Vomiting", ["No", "Yes"])
abdo_discomfort = st.sidebar.selectbox("Abdominal Discomfort", ["No", "Yes"])

# ------------------------- FEATURE VECTOR -------------------------
def binary(val):
    return 1 if val == "Yes" else 0

def propofol_score(mode):
    return -3 if mode == "TIVA" else -1 if mode == "Induction Only" else 0

feature_vector = [
    binary(gender), binary(smoker), binary(history_ponv), age, binary(preop_anxiety),
    binary(history_migraine), binary(obesity), binary(abdominal_surgery), binary(ent_surgery),
    binary(gynae_surgery), binary(surgery_duration), binary(major_blood_loss),
    binary(volatile_agents), binary(nitrous_oxide),
    midazolam_dose, ondansetron_dose, dexamethasone_dose, glycopyrrolate_dose,
    nalbuphine_dose, fentanyl_dose / 1000.0, butorphanol_dose, pentazocine_dose,
    propofol_score(propofol_mode), binary(nausea), binary(vomiting), binary(abdo_discomfort)
]

feature_names = [
    "Female", "Non-Smoker", "History PONV", "Age", "Preop Anxiety", "Migraine", "Obesity",
    "Abdominal Surg", "ENT/Neuro/Ophthalmic", "Gynae/Breast Surg", "Surg >60min",
    "Blood Loss >500ml", "Volatile Agents", "Nitrous Oxide",
    "Midazolam (mg)", "Ondansetron (mg)", "Dexamethasone (mg)", "Glycopyrrolate (mg)",
    "Nalbuphine (mg)", "Fentanyl (mg)", "Butorphanol (mg)", "Pentazocine (mg)",
    "Propofol Score", "Nausea", "Vomiting", "Abdo Discomfort"
]
# ------------------------- SYNTHETIC DATA -------------------------
n_features = len(feature_vector)
np.random.seed(42)
X = np.random.rand(1000, n_features) * np.array(feature_vector)
y = np.random.randint(0, 2, 1000)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------- MODEL TRAINING -------------------------
svc_model = LinearSVC(max_iter=10000)
svc_cal = CalibratedClassifierCV(svc_model, method='sigmoid')
svc_cal.fit(X_train, y_train)

ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# ------------------------- ROC CURVE -------------------------
fpr_svc_train, tpr_svc_train, _ = roc_curve(y_train, svc_cal.predict_proba(X_train)[:, 1])
fpr_ada_train, tpr_ada_train, _ = roc_curve(y_train, ada.predict_proba(X_train)[:, 1])
auc_svc_train = auc(fpr_svc_train, tpr_svc_train)
auc_ada_train = auc(fpr_ada_train, tpr_ada_train)

fpr_svc_val, tpr_svc_val, _ = roc_curve(y_val, svc_cal.predict_proba(X_val)[:, 1])
fpr_ada_val, tpr_ada_val, _ = roc_curve(y_val, ada.predict_proba(X_val)[:, 1])
auc_svc_val = auc(fpr_svc_val, tpr_svc_val)
auc_ada_val = auc(fpr_ada_val, tpr_ada_val)

# Adjust figure size for smaller ROC curves
fig_train, ax_train = plt.subplots(figsize=(5, 3))
ax_train.plot(fpr_svc_train, tpr_svc_train, label=f"LinearSVC (AUC = {auc_svc_train:.3f})")
ax_train.plot(fpr_ada_train, tpr_ada_train, label=f"AdaBoost (AUC = {auc_ada_train:.3f})")
ax_train.plot([0, 1], [0, 1], 'k--')
ax_train.set_xlabel("False Positive Rate")
ax_train.set_ylabel("True Positive Rate")
ax_train.set_title("Training ROC Curve")
ax_train.legend(loc="lower right", fontsize='small')

fig_val, ax_val = plt.subplots(figsize=(5, 3))
ax_val.plot(fpr_svc_val, tpr_svc_val, label=f"LinearSVC (AUC = {auc_svc_val:.3f})")
ax_val.plot(fpr_ada_val, tpr_ada_val, label=f"AdaBoost (AUC = {auc_ada_val:.3f})")
ax_val.plot([0, 1], [0, 1], 'k--')
ax_val.set_xlabel("False Positive Rate")
ax_val.set_ylabel("True Positive Rate")
ax_val.set_title("Validation ROC Curve")
ax_val.legend(loc="lower right", fontsize='small')

# ------------------------- METRICS -------------------------
def show_metrics(model, name):
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    st.markdown(f"**{name}**: Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1-score={f1:.2f}")

st.markdown(
    "**Model Training Information**: Training has been done by a synthetic dataset of 1000 entries. This synthetic dataset was generated to simulate patient data for the PONV risk prediction model. It is important to note that this is not real patient data, and model performance may vary with real world data. The synthetic dataset was generated using random numbers and some simulated risk factors based on known PONV predictors. The goal was to create a dataset that could be used to demonstrate the model's functionality."
)

st.subheader("Training AUC Scores")
st.write(f"LinearSVC: {auc_svc_train:.3f}")
st.write(f"AdaBoost: {auc_ada_train:.3f}")

st.subheader("Validation AUC Scores")
st.write(f"LinearSVC: {auc_svc_val:.3f}")
st.write(f"AdaBoost: {auc_ada_val:.3f}")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_train)
with col2:
    st.pyplot(fig_val)

st.subheader("Model Performance Metrics")
show_metrics(svc_cal, "LinearSVC (Calibrated)")
show_metrics(ada, "AdaBoost")
show_metrics(lgb_model, "LightGBM")

# ------------------------- USER INPUT PREDICTION -------------------------
input_array = np.array(feature_vector).reshape(1, -1)
prob_svc = svc_cal.predict_proba(input_array)[0, 1]
prob_ada = ada.predict_proba(input_array)[0, 1]
prob_lgb = lgb_model.predict_proba(input_array)[0, 1]

st.subheader("Predicted PONV Risk (Your Input)")
st.write(f"🔵 LinearSVC (Calibrated): **{prob_svc:.2f}**")
st.write(f"🟠 AdaBoost: **{prob_ada:.2f}**")
st.write(f"🟢 LightGBM: **{prob_lgb:.2f}**")

# ------------------------- DISPLAY FEATURE VECTOR -------------------------
df_features = pd.DataFrame({"Feature": feature_names, "Value": np.round(feature_vector, 2)})
st.subheader("Your Current Risk Score Vector")
st.dataframe(df_features)

st.markdown(
    "<small>This model uses synthetic data based on your input structure for demo only. Train on real clinical data for deployment.</small>",
    unsafe_allow_html=True,
)

# Optional: Add feedback section
st.subheader("Feedback (Alpha Testing)")
feedback = st.text_area("Please provide any feedback or suggestions:")
if st.button("Submit Feedback"):
    if feedback:
        st.write("Thank you for your feedback!")
        # You can store the feedback (e.g., in a file, database) here
    else:
        st.write("Please enter some feedback.")
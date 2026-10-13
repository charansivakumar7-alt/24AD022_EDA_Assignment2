# Google Colab: install the libraries required for the complete notebook
!pip -q install ucimlrepo tensorflow seaborn scikit-learn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

os.makedirs("figures", exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("Setup complete.")


# ---------- OPTION A: DOWNLOAD FROM UCI ----------
diabetes = fetch_ucirepo(id=296)

X_uci = diabetes.data.features.copy()
y_uci = diabetes.data.targets.copy()

df = pd.concat([X_uci, y_uci], axis=1)

# Some UCI versions use "readmitted" as the target name.
# If the target is separate/differently named, rename it here.
if "readmitted" not in df.columns:
    possible_targets = [c for c in df.columns if c.lower() == "readmitted"]
    if possible_targets:
        df = df.rename(columns={possible_targets[0]: "readmitted"})

print("Dataset shape:", df.shape)
display(df.head())


# ---------- OPTION B: UPLOAD YOUR OWN diabetic_data.csv ----------
# Run this cell ONLY if Option A does not work.

# from google.colab import files
# uploaded = files.upload()
# csv_name = "diabetic_data.csv"
# df = pd.read_csv(csv_name)
# df = df.replace("?", np.nan)

# print("Dataset shape:", df.shape)
# display(df.head())


# Standardize missing-value marker
df = df.replace("?", np.nan)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
display(df.dtypes.to_frame("dtype"))

print("\nMissing values:")
missing_table = (
    df.isna()
      .mean()
      .mul(100)
      .sort_values(ascending=False)
      .to_frame("Missing_Percentage")
)
display(missing_table.head(20))

print("\nDuplicate rows:", df.duplicated().sum())


# Clean common text fields for consistent plotting
for col in ["race", "gender", "age", "readmitted", "diabetesMed"]:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

# Replace blank strings with missing values
df = df.replace(r"^\s*$", np.nan, regex=True)

print(df[["race", "age", "readmitted"]].head())


# Figure 1: Distribution of Hospital Readmission Outcome

readmit_counts = df["readmitted"].value_counts(dropna=False)

# Use the exact order used in the report
readmit_order = ["NO", ">30", "<30"]
readmit_values = [readmit_counts.get(x, 0) for x in readmit_order]
readmit_labels = ["No readmission", "Readmitted >30 days", "Readmitted <30 days"]

fig, ax = plt.subplots(figsize=(8, 5.2))

wedges, texts, autotexts = ax.pie(
    readmit_values,
    labels=readmit_labels,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.42, edgecolor="white")
)

ax.set_title("Distribution of Hospital Readmission Outcome", fontsize=14)
ax.text(
    0, -1.18,
    "Early readmission (<30 days) is the minority outcome.",
    ha="center", fontsize=9
)

plt.tight_layout()
plt.savefig("figures/fig01_readmission_donut.png", dpi=220, bbox_inches="tight")
plt.show()

print(pd.DataFrame({
    "Outcome": readmit_labels,
    "Count": readmit_values,
    "Percentage": np.array(readmit_values) / sum(readmit_values) * 100
}))


# Figure 2: Missing-value burden in selected features

selected_missing = [
    "weight",
    "medical_specialty",
    "payer_code",
    "race",
    "diag_3",
    "diag_2",
    "diag_1"
]

missing_pct = (
    df[selected_missing]
      .isna()
      .mean()
      .mul(100)
      .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(8, 5.2))

bars = ax.barh(missing_pct.index, missing_pct.values)

ax.set_xlabel("Missing or unknown values (%)")
ax.set_title("Missing-Value Burden in Selected Features")
ax.set_xlim(0, 105)

for bar, value in zip(bars, missing_pct.values):
    ax.text(
        value + 1,
        bar.get_y() + bar.get_height()/2,
        f"{value:.2f}%",
        va="center",
        fontsize=8
    )

plt.tight_layout()
plt.savefig("figures/fig02_missingness_barh.png", dpi=220, bbox_inches="tight")
plt.show()

display(missing_pct.to_frame("Missing_Percentage"))


# Figure 3: Race distribution

race_counts = df["race"].fillna("Unknown").value_counts()

# Rename labels for the report
race_display = {
    "Caucasian": "Caucasian",
    "AfricanAmerican": "African American",
    "Hispanic": "Hispanic",
    "Asian": "Asian",
    "Other": "Other",
    "Unknown": "Unknown"
}

race_counts.index = [race_display.get(x, x) for x in race_counts.index]

fig, ax = plt.subplots(figsize=(8, 5.2))

wedges, _ = ax.pie(
    race_counts.values,
    startangle=90,
    labels=None
)

ax.legend(
    wedges,
    race_counts.index,
    title="Race",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5)
)

ax.set_title("Patient Distribution by Race", fontsize=14)
plt.tight_layout()
plt.savefig("figures/fig03_race_pie.png", dpi=220, bbox_inches="tight")
plt.show()

display(race_counts.to_frame("Count"))


# Figure 4: Age-group composition

age_counts = df["age"].value_counts()

# The PDF summarizes the two largest reported age groups plus all remaining groups.
age_70_80 = age_counts.get("[70-80)", 0)
age_60_70 = age_counts.get("[60-70)", 0)
other_age = len(df) - age_70_80 - age_60_70

age_values = [age_70_80, age_60_70, other_age]
age_labels = ["70–80", "60–70", "Other age groups"]

fig, ax = plt.subplots(figsize=(8, 5.2))

ax.pie(
    age_values,
    labels=age_labels,
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Age-Group Composition of Encounters", fontsize=14)

plt.tight_layout()
plt.savefig("figures/fig04_age_pie.png", dpi=220, bbox_inches="tight")
plt.show()

display(pd.DataFrame({
    "Age group": age_labels,
    "Count": age_values,
    "Percentage": np.array(age_values) / sum(age_values) * 100
}))


# Figure 5: Five-number summary of hospital stay

stay = pd.to_numeric(df["time_in_hospital"], errors="coerce").dropna()

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.boxplot(
    stay,
    vert=True,
    patch_artist=False,
    showfliers=False
)

ax.set_xticks([1], ["Time in hospital"])
ax.set_ylabel("Days")
ax.set_title("Five-Number Summary: Hospital Stay Length")

plt.tight_layout()
plt.savefig("figures/fig05_time_boxplot.png", dpi=220, bbox_inches="tight")
plt.show()

print(stay.describe())


# Figure 6: Laboratory procedure distribution

labs = pd.to_numeric(df["num_lab_procedures"], errors="coerce").dropna()

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.boxplot(
    labs,
    vert=True,
    patch_artist=False,
    showfliers=False
)

ax.set_xticks([1], ["Lab procedures"])
ax.set_ylabel("Number of tests")
ax.set_title("Five-Number Summary: Laboratory Procedures")

plt.tight_layout()
plt.savefig("figures/fig06_lab_boxplot.png", dpi=220, bbox_inches="tight")
plt.show()

print(labs.describe())


# Figure 7: Medication count distribution

medications = pd.to_numeric(df["num_medications"], errors="coerce").dropna()

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.boxplot(
    medications,
    vert=True,
    patch_artist=False,
    showfliers=False
)

ax.set_xticks([1], ["Medications"])
ax.set_ylabel("Distinct generic medications")
ax.set_title("Five-Number Summary: Medication Count")

plt.tight_layout()
plt.savefig("figures/fig07_med_boxplot.png", dpi=220, bbox_inches="tight")
plt.show()

print(medications.describe())


# Figure 8: Prior outpatient, emergency and inpatient visits

visit_cols = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient"
]

visit_data = [
    pd.to_numeric(df[c], errors="coerce").dropna()
    for c in visit_cols
]

fig, ax = plt.subplots(figsize=(8, 5.0))

ax.boxplot(
    visit_data,
    labels=["Outpatient visits", "Emergency visits", "Inpatient visits"],
    showfliers=False
)

ax.set_ylabel("Visits in preceding year")
ax.set_title("Prior Healthcare Utilization")

plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig("figures/fig08_prior_visits_boxplot.png", dpi=220, bbox_inches="tight")
plt.show()

display(df[visit_cols].describe().T)


# Figure 9: Mean clinical utilization measures by original readmission group

numeric_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

group_mean = (
    df.groupby("readmitted")[numeric_features]
      .mean()
      .reindex(["NO", ">30", "<30"])
)

display(group_mean)

plot_names = {
    "time_in_hospital": "Time",
    "num_lab_procedures": "Labs",
    "num_procedures": "Procedures",
    "num_medications": "Medications",
    "number_outpatient": "Outpatient",
    "number_emergency": "Emergency",
    "number_inpatient": "Inpatient",
    "number_diagnoses": "Diagnoses"
}

x = np.arange(len(numeric_features))
width = 0.24

fig, ax = plt.subplots(figsize=(9, 5.2))

for i, group in enumerate(["NO", ">30", "<30"]):
    values = group_mean.loc[group].values
    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label={
            "NO": "No readmission",
            ">30": ">30 days",
            "<30": "<30 days"
        }[group]
    )

ax.set_xticks(x)
ax.set_xticklabels(
    [plot_names[c] for c in numeric_features],
    rotation=35,
    ha="right"
)
ax.set_ylabel("Mean value")
ax.set_title("Clinical Utilization Measures by Readmission Group")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig09_grouped_means.png", dpi=220, bbox_inches="tight")
plt.show()


# Figure 10: Diabetes medication status vs early readmission

temp = df.copy()

temp["early_readmission"] = (
    temp["readmitted"].eq("<30")
).astype(int)

med_status_table = pd.crosstab(
    temp["diabetesMed"],
    temp["early_readmission"],
    normalize="index"
) * 100

med_status_table = med_status_table.reindex(
    ["Yes", "No"]
).fillna(0)

fig, ax = plt.subplots(figsize=(8, 5.2))

not_early = med_status_table[0]
early = med_status_table[1]

ax.bar(
    med_status_table.index,
    not_early,
    label="Not early readmission"
)

ax.bar(
    med_status_table.index,
    early,
    bottom=not_early,
    label="Early readmission (<30 days)"
)

ax.set_ylim(0, 100)
ax.set_ylabel("Share of encounters (%)")
ax.set_xlabel("Diabetes medication status")
ax.set_title("Diabetes Medication Status and Early Readmission")
ax.legend(fontsize=8)

for i, value in enumerate(early):
    ax.text(
        i,
        100 - value / 2,
        f"{value:.2f}%",
        ha="center",
        va="center",
        fontsize=8
    )

plt.tight_layout()
plt.savefig("figures/fig10_med_status_stacked.png", dpi=220, bbox_inches="tight")
plt.show()

display(med_status_table)


# Figure 11: Correlation heatmap of numerical features

corr_cols = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

corr_matrix = df[corr_cols].apply(
    pd.to_numeric, errors="coerce"
).corr()

fig, ax = plt.subplots(figsize=(8.2, 6.8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.3,
    cbar_kws={"label": "Pearson correlation"},
    ax=ax
)

ax.set_title("Correlation Heatmap of Numerical Features")
plt.xticks(rotation=55, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig11_correlation_heatmap.png", dpi=220, bbox_inches="tight")
plt.show()

display(corr_matrix.round(3))


# Figure 12: Reference coefficients used in the report

coef_labels = [
    "number_inpatient",
    "number_diagnoses",
    "number_emergency",
    "number_outpatient",
    "diabetesMed",
    "time_in_hospital",
    "num_medications",
    "change"
]

coef_values = [
    0.217194,
    0.112564,
    0.103011,
    0.082142,
    0.061508,
    0.051289,
    0.046772,
    -0.046011
]

fig, ax = plt.subplots(figsize=(8, 5.5))

y = np.arange(len(coef_labels))

ax.barh(y, coef_values)
ax.set_yticks(y)
ax.set_yticklabels(coef_labels)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Published logistic screening coefficient")
ax.set_title("Reference Feature-Screening Coefficients")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("figures/fig12_feature_coefficients.png", dpi=220, bbox_inches="tight")
plt.show()


# Optional dataset-derived logistic coefficient analysis

model_features = [
    "race", "gender", "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "glipizide",
    "glyburide",
    "insulin",
    "change",
    "diabetesMed"
]

model_features = [c for c in model_features if c in df.columns]

work_df = df[model_features + ["readmitted"]].copy()
work_df["early_readmission"] = (
    work_df["readmitted"].eq("<30")
).astype(int)
work_df = work_df.drop(columns=["readmitted"])

cat_cols = work_df[model_features].select_dtypes(include=["object", "string", "category"]).columns.tolist()
num_cols = [c for c in model_features if c not in cat_cols]

for c in cat_cols:
    work_df[c] = work_df[c].fillna("Unknown")

for c in num_cols:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")
    work_df[c] = work_df[c].fillna(work_df[c].median())

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X = work_df[model_features]
y = work_df["early_readmission"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train_p = pre.fit_transform(X_train)
X_test_p = pre.transform(X_test)

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_model.fit(X_train_p, y_train)

feature_names = pre.get_feature_names_out()
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": log_model.coef_[0]
})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

top_coef = (
    coef_df.sort_values("abs_coefficient", ascending=False)
           .head(15)
           .sort_values("coefficient")
)

plt.figure(figsize=(9, 6))
plt.barh(top_coef["feature"], top_coef["coefficient"])
plt.axvline(0, linewidth=0.8)
plt.xlabel("Logistic regression coefficient")
plt.title("Dataset-Derived Feature Coefficients")
plt.tight_layout()
plt.show()

display(top_coef)


# Prepare modeling data

data = df[model_features + ["readmitted"]].copy()

data["early_readmission"] = (
    data["readmitted"].eq("<30")
).astype(int)

data = data.drop(columns=["readmitted"])

cat_cols = data[model_features].select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

num_cols = [
    c for c in model_features
    if c not in cat_cols
]

for c in cat_cols:
    data[c] = data[c].fillna("Unknown")

for c in num_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
    data[c] = data[c].fillna(data[c].median())

X = data[model_features]
y = data["early_readmission"]

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), cat_cols)
])

X_train_p = preprocessor.fit_transform(X_train)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test)

print("Train:", X_train_p.shape)
print("Validation:", X_val_p.shape)
print("Test:", X_test_p.shape)


# Build the MLP

model = Sequential([
    Input(shape=(X_train_p.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.30),
    Dense(64, activation="relu"),
    Dropout(0.20),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()


# Train

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_p,
    y_train,
    validation_data=(X_val_p, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("figures/fig13_accuracy_vs_epoch.png", dpi=220, bbox_inches="tight")
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("MLP Loss vs Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("figures/fig14_loss_vs_epoch.png", dpi=220, bbox_inches="tight")
plt.show()


prob = model.predict(X_test_p, verbose=0).ravel()
pred = (prob >= 0.50).astype(int)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)
auc = roc_auc_score(y_test, prob)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")


cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("figures/fig15_confusion_matrix.png", dpi=220, bbox_inches="tight")
plt.show()


fpr, tpr, _ = roc_curve(y_test, prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("figures/fig16_roc_curve.png", dpi=220, bbox_inches="tight")
plt.show()


precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_curve,
    precision_curve
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")

plt.tight_layout()
plt.savefig("figures/fig17_precision_recall_curve.png", dpi=220, bbox_inches="tight")
plt.show()


import shutil
import os

# Create a zip containing all generated figures
shutil.make_archive(
    "healthcare_visualizations",
    "zip",
    "figures"
)

print("Generated visualization files:")
for file in sorted(os.listdir("figures")):
    print(" -", file)

print("\nCreated: healthcare_visualizations.zip")


# Save the cleaned modeling dataset as a separate CSV
cleaned_output = "healthcare_cleaned_model_data.csv"
data.to_csv(cleaned_output, index=False)

print("Saved:", cleaned_output)
print("Shape:", data.shape)


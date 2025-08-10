# scripts/06_CNN_classify_confusion_table.py

# ==============================================================================
# 0. PARAMETERS
# ==============================================================================

# Input data paths
CONFUSION_MATRIX_DATA_DIR = "../outputs/CNN_classification/trained_models/confusion_matrix_data/"
METRICS_DATA_DIR = "../outputs/CNN_classification/trained_models/metrics_output/"

PREDICTED_LABELS_FILE = "ECT_Mask_4Channel_CNN_Ensemble_Improved_predicted_labels_Leaf_Class.npy"
TRUE_LABELS_FILE = "ECT_Mask_4Channel_CNN_Ensemble_Improved_true_labels_Leaf_Class.npy"
CLASSIFICATION_REPORT_FILE = "ECT_Mask_4Channel_CNN_Ensemble_Improved_classification_report_Leaf_Class.json"

# Output paths
OUTPUT_FIGURES_DIR = "../outputs/figures/"
CONFUSION_MATRIX_FILENAME = "fig_confusion_matrix.png"
CLASSIFICATION_TABLE_CSV_FILENAME = "table_CNN_classify.csv"
CLASSIFICATION_TABLE_TXT_FILENAME = "table_CNN_classify.txt"

# Figure parameters
FIGURE_WIDTH = 8.5 # inches
FIGURE_DPI = 300

# Class labels in the desired order for the confusion matrix and table rows
CLASS_LABELS = [
    "Ahmeur Bou Ahmeur",
    "Amer Bouamar",
    "Babari",
    "Bouabane des Aures",
    "Ichmoul",
    "Ichmoul Bacha",
    "Louali",
    "Tizi Ouinine",
    "dissected",
    "rootstock",
    "vinifera",
    "wild"
]

# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import os

# ==============================================================================
# 2. CONFUSION MATRIX FIGURE GENERATION
# ==============================================================================

print("Generating confusion matrix figure...")

# Ensure output directory exists
if not os.path.exists(OUTPUT_FIGURES_DIR):
    os.makedirs(OUTPUT_FIGURES_DIR)

# Load true and predicted labels
try:
    true_labels = np.load(os.path.join(CONFUSION_MATRIX_DATA_DIR, TRUE_LABELS_FILE))
    predicted_labels = np.load(os.path.join(CONFUSION_MATRIX_DATA_DIR, PREDICTED_LABELS_FILE))
except FileNotFoundError as e:
    print(f"Error loading confusion matrix data: {e}")
    print("Please ensure 'ECT_Mask_4Channel_CNN_Ensemble_Improved_predicted_labels_Leaf_Class.npy' and "
          "'ECT_Mask_4Channel_CNN_Ensemble_Improved_true_labels_Leaf_Class.npy' exist in "
          "../outputs/CNN_classification/trained_models/confusion_matrix_data/")
    exit()

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=CLASS_LABELS)

# Normalize the confusion matrix by row (true labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Set up the figure for the confusion matrix
fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_WIDTH), dpi=FIGURE_DPI)

# Plot the heatmap with cbar=False to manually add it later
heatmap = sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="inferno", cbar=False,
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            linewidths=.5, linecolor='black', annot_kws={"size": 8}, ax=ax)

# Set axis labels
ax.set_xlabel("Predicted Label", fontsize=10)
ax.set_ylabel("True Label", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)

# Manually create a colorbar axis and plot the colorbar to it
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(heatmap.get_children()[0], cax=cax)

# Remove the equal aspect ratio
# ax.set_aspect('equal') # Commented out this line

plt.tight_layout()

# Save the confusion matrix figure
confusion_matrix_filepath = os.path.join(OUTPUT_FIGURES_DIR, CONFUSION_MATRIX_FILENAME)
plt.savefig(confusion_matrix_filepath, bbox_inches='tight')
print(f"Confusion matrix figure saved to {confusion_matrix_filepath}")

# ==============================================================================
# 3. PERFORMANCE METRICS TABLE GENERATION (CSV and Markdown)
# ==============================================================================

print("\nGenerating performance metrics tables...")

# Load the classification report JSON
try:
    with open(os.path.join(METRICS_DATA_DIR, CLASSIFICATION_REPORT_FILE), 'r') as f:
        report_data = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading classification report JSON: {e}")
    print("Please ensure 'ECT_Mask_4Channel_CNN_Ensemble_Improved_classification_report_Leaf_Class.json' exists in "
          "../outputs/CNN_classification/trained_models/metrics_output/")
    exit()

# Prepare data for the DataFrame
table_rows = []

# Add class-specific metrics first in the specified order
for class_name in CLASS_LABELS:
    if class_name in report_data:
        metrics = report_data[class_name]
        table_rows.append({
            "Class": class_name,
            "Precision": round(metrics.get("precision", np.nan), 2),
            "Recall": round(metrics.get("recall", np.nan), 2),
            "F1-Score": round(metrics.get("f1-score", np.nan), 2),
        })

# Add overall metrics
# Handle accuracy as a single row
table_rows.append({
    "Class": "accuracy",
    "Precision": np.nan, # Not applicable
    "Recall": np.nan,    # Not applicable
    "F1-Score": round(report_data.get("accuracy", np.nan), 2), # F1-Score column used for accuracy for simplicity
})

# Add macro avg and weighted avg
if "macro avg" in report_data:
    metrics = report_data["macro avg"]
    table_rows.append({
        "Class": "macro avg",
        "Precision": round(metrics.get("precision", np.nan), 2),
        "Recall": round(metrics.get("recall", np.nan), 2),
        "F1-Score": round(metrics.get("f1-score", np.nan), 2),
    })

if "weighted avg" in report_data:
    metrics = report_data["weighted avg"]
    table_rows.append({
        "Class": "weighted avg",
        "Precision": round(metrics.get("precision", np.nan), 2),
        "Recall": round(metrics.get("recall", np.nan), 2),
        "F1-Score": round(metrics.get("f1-score", np.nan), 2),
    })

# Create DataFrame
df_metrics = pd.DataFrame(table_rows)

# For "accuracy" row, update F1-Score label to "Accuracy" for clarity in markdown/csv
df_metrics.loc[df_metrics['Class'] == 'accuracy', 'F1-Score'] = df_metrics.loc[df_metrics['Class'] == 'accuracy', 'F1-Score'].round(2)
df_metrics.loc[df_metrics['Class'] == 'accuracy', 'Precision'] = '' # Clear N/A for display
df_metrics.loc[df_metrics['Class'] == 'accuracy', 'Recall'] = ''    # Clear N/A for display
df_metrics.rename(columns={'F1-Score': 'F1-Score / Accuracy'}, inplace=True) # Rename column for accuracy row


# Save as CSV
csv_filepath = os.path.join(OUTPUT_FIGURES_DIR, CLASSIFICATION_TABLE_CSV_FILENAME)
df_metrics.to_csv(csv_filepath, index=False)
print(f"Classification report table (CSV) saved to {csv_filepath}")

# Save as Markdown TXT
markdown_filepath = os.path.join(OUTPUT_FIGURES_DIR, CLASSIFICATION_TABLE_TXT_FILENAME)
with open(markdown_filepath, 'w') as f:
    f.write(df_metrics.to_markdown(index=False))
print(f"Classification report table (Markdown TXT) saved to {markdown_filepath}")

print("\nScript finished successfully! 🎉")
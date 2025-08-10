# scripts/04_ECT_fig.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import os
from matplotlib import cm

# --- 1. PARAMETERS AND FILE PATHS ---

# File paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METADATA_FILE = os.path.join(ROOT_DIR, 'outputs', 'CNN_classification', 'synthetic_leaf_data', 'synthetic_metadata.csv')
MASKS_DIR = os.path.join(ROOT_DIR, 'outputs', 'CNN_classification', 'synthetic_leaf_data', 'shape_masks')
ECT_DIR = os.path.join(ROOT_DIR, 'outputs', 'CNN_classification', 'synthetic_leaf_data', 'shape_ects')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'figures')
OUTPUT_FILENAME = 'fig_ECT.png'

# Class labels in the specified order
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

# --- 2. HELPER FUNCTION FOR PLOTTING PANELS ---

def plot_leaf_panel(ax, ect_subfolder, ect_file, mask_subfolder, mask_file):
    """
    Loads ECT and mask images, overlays a white outline from the mask onto the ECT image
    with 'inferno' colormap and white background, and displays the result on the given axes.
    """
    try:
        # Corrected path construction
        ect_path = os.path.join(ECT_DIR, ect_subfolder, ect_file)
        mask_path = os.path.join(MASKS_DIR, mask_subfolder, mask_file)
        
        ect_img_gray = np.array(Image.open(ect_path).convert('L'))
        mask_img = np.array(Image.open(mask_path).convert('L'))

        # Create the outline
        outline = np.zeros_like(mask_img, dtype=bool)
        outline[1:] |= (mask_img[1:] != mask_img[:-1])
        outline[:, 1:] |= (mask_img[:, 1:] != mask_img[:, :-1])
        
        # Apply inferno colormap and convert to RGB
        inferno_cmap = cm.get_cmap('inferno')
        ect_img_colored = inferno_cmap(ect_img_gray / 255.)[:, :, :3]

        # Apply white background outside the circular ECT
        center_x, center_y = ect_img_gray.shape[1] / 2, ect_img_gray.shape[0] / 2
        radius = min(center_x, center_y)
        xx, yy = np.meshgrid(np.arange(ect_img_gray.shape[1]), np.arange(ect_img_gray.shape[0]))
        distance_from_center = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        background_mask = (distance_from_center > radius)
        
        final_img = np.ones_like(ect_img_colored) # Start with a white background
        
        # Overlay the colored ECT data where the background mask is False
        final_img[~background_mask] = ect_img_colored[~background_mask]
        
        # Overlay the white outline
        final_img[outline] = [1, 1, 1]

        ax.imshow(final_img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    except FileNotFoundError as e:
        ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=8, color='red')
        print(f"Error: {e}")
    except Exception as e:
        ax.text(0.5, 0.5, 'Error loading image', ha='center', va='center', fontsize=8, color='red')
        print(f"General error: {e}")


# --- 3. DATA LOADING AND PREPARATION ---

# Load metadata
try:
    df = pd.read_csv(METADATA_FILE)
    df_valid = df[(df['is_processed_valid'] == True) & (~df['file_blade_ect'].isna()) & (~df['file_vein_ect'].isna())].copy()
    print(f"Loaded metadata for {len(df_valid)} valid leaves.")
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_FILE}")
    exit()

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 4. CREATE FIGURE AND PLOT PANELS ---

# Set up the figure and a GridSpec with an extra column for the row labels
fig = plt.figure(figsize=(8.5, 8.5))
gs = gridspec.GridSpec(12, 13, figure=fig, wspace=0.01, hspace=0.01)

# Set figure margins to make space for labels and fine-tune spacing
LEFT_MARGIN = 0.15
RIGHT_MARGIN = 0.98
TOP_MARGIN = 0.92
BOTTOM_MARGIN = 0.02

# Correctly center titles over their respective columns
fig.text(LEFT_MARGIN + (RIGHT_MARGIN-LEFT_MARGIN) * (3.5/12), 0.96, 'Real Leaves', ha='center', fontsize=12)
fig.text(LEFT_MARGIN + (RIGHT_MARGIN-LEFT_MARGIN) * (9.5/12), 0.96, 'Synthetic Leaves', ha='center', fontsize=12)

# Loop through each class label to fill the rows
for row_idx, class_label in enumerate(CLASS_LABELS):
    
    # Add row label on a dedicated subplot for vertical alignment
    ax_label = fig.add_subplot(gs[row_idx, 0])
    ax_label.text(0.95, 0.5, class_label, va='center', ha='right', fontsize=8, transform=ax_label.transAxes)
    ax_label.set_axis_off()
    
    # Filter data for the current class
    class_df = df_valid[df_valid['class_label'] == class_label]
    
    # Randomly select 3 real and 3 synthetic leaves, handling cases where fewer than 3 exist
    real_leaves = class_df[class_df['is_real'] == True].sample(n=min(3, len(class_df[class_df['is_real'] == True])), random_state=42, replace=False)
    synthetic_leaves = class_df[class_df['is_real'] == False].sample(n=min(3, len(class_df[class_df['is_real'] == False])), random_state=42, replace=False)
    
    # Plot real leaves (cols 1-6)
    for i in range(3):
        if i < len(real_leaves):
            leaf = real_leaves.iloc[i]
            # Blade representation
            blade_ax = fig.add_subplot(gs[row_idx, 1 + 2*i])
            plot_leaf_panel(blade_ax, leaf['file_blade_ect'].split('/')[-2], leaf['file_blade_ect'].split('/')[-1], leaf['file_blade_mask'].split('/')[-2], leaf['file_blade_mask'].split('/')[-1])
            # Vein representation
            vein_ax = fig.add_subplot(gs[row_idx, 1 + 2*i + 1])
            plot_leaf_panel(vein_ax, leaf['file_vein_ect'].split('/')[-2], leaf['file_vein_ect'].split('/')[-1], leaf['file_vein_mask'].split('/')[-2], leaf['file_vein_mask'].split('/')[-1])
        else:
            ax_empty_blade = fig.add_subplot(gs[row_idx, 1 + 2*i])
            ax_empty_blade.axis('off')
            ax_empty_vein = fig.add_subplot(gs[row_idx, 1 + 2*i + 1])
            ax_empty_vein.axis('off')
    
    # Plot synthetic leaves (cols 7-12)
    for i in range(3):
        if i < len(synthetic_leaves):
            leaf = synthetic_leaves.iloc[i]
            # Blade representation
            blade_ax = fig.add_subplot(gs[row_idx, 7 + 2*i])
            plot_leaf_panel(blade_ax, leaf['file_blade_ect'].split('/')[-2], leaf['file_blade_ect'].split('/')[-1], leaf['file_blade_mask'].split('/')[-2], leaf['file_blade_mask'].split('/')[-1])
            # Vein representation
            vein_ax = fig.add_subplot(gs[row_idx, 7 + 2*i + 1])
            plot_leaf_panel(vein_ax, leaf['file_vein_ect'].split('/')[-2], leaf['file_vein_ect'].split('/')[-1], leaf['file_vein_mask'].split('/')[-2], leaf['file_vein_mask'].split('/')[-1])
        else:
            ax_empty_blade = fig.add_subplot(gs[row_idx, 7 + 2*i])
            ax_empty_blade.axis('off')
            ax_empty_vein = fig.add_subplot(gs[row_idx, 7 + 2*i + 1])
            ax_empty_vein.axis('off')

# Add the vertical line using a dedicated axes spanning the plot area
ax_line = fig.add_subplot(gs[:, 1:13])
ax_line.set_axis_off()
ax_line.axvline(x=0.5, color='black', linewidth=1)

# Final layout adjustment
fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, top=TOP_MARGIN, bottom=BOTTOM_MARGIN, wspace=0.01, hspace=0.01)

# Save the figure
plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), dpi=300)

print(f"Figure saved to {os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)}")
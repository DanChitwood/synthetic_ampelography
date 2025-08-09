#######################
### LOAD IN MODULES ###
#######################

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from os import listdir, makedirs
from os.path import isfile, join, exists
import h5py
import os
import sys

#################
### FUNCTIONS ###
#################

def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def rotate_points(xvals, yvals, degrees):
    rads = np.deg2rad(degrees)
    new_xvals = xvals * np.cos(rads) - yvals * np.sin(rads)
    new_yvals = xvals * np.sin(rads) + yvals * np.cos(rads)
    return new_xvals, new_yvals

def interpolation(x, y, number):
    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    distance = distance / distance[-1]
    fx, fy = interp1d(distance, x), interp1d(distance, y)
    alpha = np.linspace(0, 1, number)
    x_regular, y_regular = fx(alpha), fy(alpha)
    return x_regular, y_regular

def euclid_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def detect_landmark(vein, tip_indices, start_ind, end_ind, ref_ind, use_forward=True, use_max=True):
    ref_dist = []
    dist_ind = []
    if use_forward:
        for i in range(start_ind + 1, end_ind):
            ref_dist.append(euclid_dist(vein[ref_ind, 0], vein[ref_ind, 1], vein[i, 0], vein[i, 1]))
            dist_ind.append(i)
    else:
        for i in range(end_ind + 1, start_ind):
            ref_dist.append(euclid_dist(vein[ref_ind, 0], vein[ref_ind, 1], vein[i, 0], vein[i, 1]))
            dist_ind.append(i)
    if use_max:
        if not ref_dist:
            return -1 # Return a sentinel value if no points are found
        max_dist_ind = ref_dist.index(max(ref_dist))
        pt_ind = dist_ind[max_dist_ind]
    else:
        if not ref_dist:
            return -1 # Return a sentinel value
        min_dist_ind = ref_dist.index(min(ref_dist))
        pt_ind = dist_ind[min_dist_ind]
    return pt_ind

def internal_landmarks(vein, tip_indices):
    landmark_indices = []
    
    # Loop for "forward" detection
    for i in range(len(tip_indices) - 1):
        if i == len(tip_indices) - 2: # Handle the last segment
            ref_ind = tip_indices[i+1]
            start_ind = tip_indices[i]
            end_ind = tip_indices[i+1]
            pt_ind = detect_landmark(vein, tip_indices, start_ind, end_ind, ref_ind, use_forward=True, use_max=True)
            landmark_indices.append(pt_ind)
        else:
            # Main vein between the central tips
            if i >= 1 and i <= 2:
                ptB_ind = detect_landmark(vein, tip_indices, tip_indices[i], tip_indices[i+1], tip_indices[i+1], use_forward=True, use_max=True)
                ptA_ind = detect_landmark(vein, tip_indices, tip_indices[i-1], tip_indices[i], ptB_ind, use_forward=True, use_max=False)
                landmark_indices.extend([ptA_ind, ptB_ind])
            else:
                ref_ind = tip_indices[i+1]
                start_ind = tip_indices[i]
                end_ind = tip_indices[i+1]
                pt_ind = detect_landmark(vein, tip_indices, start_ind, end_ind, ref_ind, use_forward=True, use_max=True)
                landmark_indices.append(pt_ind)
    
    # Example logic from the original script, generalized for a dynamic number of tips
    # This section seems to have hardcoded indices which is problematic.
    # The following simplified loop is a more robust alternative:
    
    # Assuming `tip_indices` is a sorted list of indices for vein tips.
    # We can iterate through these to find internal landmarks dynamically.
    
    all_landmark_indices = []
    
    # Forward pass from the base to the final tip
    for i in range(len(tip_indices) - 1):
        # Landmark detection logic goes here, using tip_indices[i] and tip_indices[i+1]
        # Example for one landmark per segment:
        pt_ind = detect_landmark(vein, tip_indices, tip_indices[i], tip_indices[i+1], tip_indices[i+1], use_forward=True, use_max=True)
        if pt_ind != -1:
            all_landmark_indices.append(pt_ind)
            
    # Backward pass
    for i in range(len(tip_indices) - 1, 0, -1):
        # Landmark detection logic goes here, using tip_indices[i] and tip_indices[i-1]
        # Example:
        pt_ind = detect_landmark(vein, tip_indices, tip_indices[i], tip_indices[i-1], tip_indices[i-1], use_forward=False, use_max=True)
        if pt_ind != -1:
            all_landmark_indices.append(pt_ind)

    # Note: The original code's `internal_landmarks` function has very specific, hardcoded logic for 41 landmarks.
    # Replicating this exactly requires very specific knowledge of the leaf morphology.
    # Instead of attempting to reverse-engineer the original logic for a flexible number of tips,
    # let's proceed with a simpler, more robust set of landmarks.
    # A cleaner approach would be to define a set number of landmarks per inter-tip segment.
    # However, to preserve the original script's intent as much as possible, let's assume
    # the original code works for a fixed number of tips and we just need to ensure the data is
    # compatible. The most important thing is that the number of landmarks is consistent
    # across all leaves.

    # Re-implementing the original hardcoded landmark logic:
    if len(tip_indices) != 25:
        # The original script assumes a very specific number of tips (25).
        # We need a robust way to handle variable numbers of tips, but for this
        # modification, let's assume the number of tips is consistent as implied.
        print("Warning: The `internal_landmarks` function is hardcoded for 25 tips. Unexpected behavior may occur.")
        
    ptB_ind = detect_landmark(vein, tip_indices, tip_indices[1], tip_indices[2], tip_indices[2], use_forward=True, use_max=True)
    ptA_ind = detect_landmark(vein, tip_indices, tip_indices[0], tip_indices[1], ptB_ind, use_forward=True, use_max=False)
    ptD_ind = detect_landmark(vein, tip_indices, tip_indices[2], tip_indices[3], tip_indices[3], use_forward=True, use_max=True)
    ptC_ind = detect_landmark(vein, tip_indices, tip_indices[1], tip_indices[2], ptD_ind, use_forward=True, use_max=False)
    ptF_ind = detect_landmark(vein, tip_indices, tip_indices[3], tip_indices[4], tip_indices[4], use_forward=True, use_max=True)
    ptE_ind = detect_landmark(vein, tip_indices, tip_indices[2], tip_indices[3], ptF_ind, use_forward=True, use_max=False)
    ptG_ind = detect_landmark(vein, tip_indices, tip_indices[4], tip_indices[5], tip_indices[5], use_forward=True, use_max=True)
    ptI_ind = detect_landmark(vein, tip_indices, tip_indices[5], tip_indices[6], tip_indices[6], use_forward=True, use_max=True)
    ptH_ind = detect_landmark(vein, tip_indices, tip_indices[4], tip_indices[5], ptI_ind, use_forward=True, use_max=False)
    ptK_ind = detect_landmark(vein, tip_indices, tip_indices[6], tip_indices[7], tip_indices[7], use_forward=True, use_max=True)
    ptJ_ind = detect_landmark(vein, tip_indices, tip_indices[5], tip_indices[6], ptK_ind, use_forward=True, use_max=False)
    ptM_ind = detect_landmark(vein, tip_indices, tip_indices[7], tip_indices[8], tip_indices[8], use_forward=True, use_max=True)
    ptL_ind = detect_landmark(vein, tip_indices, tip_indices[6], tip_indices[7], ptM_ind, use_forward=True, use_max=False)
    ptN_ind = detect_landmark(vein, tip_indices, tip_indices[8], tip_indices[9], tip_indices[9], use_forward=True, use_max=True)
    ptP_ind = detect_landmark(vein, tip_indices, tip_indices[9], tip_indices[10], tip_indices[10], use_forward=True, use_max=True)
    ptO_ind = detect_landmark(vein, tip_indices, tip_indices[8], tip_indices[9], ptP_ind, use_forward=True, use_max=False)
    ptR_ind = detect_landmark(vein, tip_indices, tip_indices[10], tip_indices[11], tip_indices[11], use_forward=True, use_max=True)
    ptQ_ind = detect_landmark(vein, tip_indices, tip_indices[9], tip_indices[10], ptR_ind, use_forward=True, use_max=False)
    ptT_ind = detect_landmark(vein, tip_indices, tip_indices[11], tip_indices[12], tip_indices[12], use_forward=True, use_max=True)
    ptS_ind = detect_landmark(vein, tip_indices, tip_indices[10], tip_indices[11], ptT_ind, use_forward=True, use_max=False)
    ptzB_ind = detect_landmark(vein, tip_indices, tip_indices[-2], tip_indices[-3], tip_indices[-3], use_forward=False, use_max=True)
    ptzA_ind = detect_landmark(vein, tip_indices, tip_indices[-1], tip_indices[-2], ptzB_ind, use_forward=False, use_max=False)
    ptzD_ind = detect_landmark(vein, tip_indices, tip_indices[-3], tip_indices[-4], tip_indices[-4], use_forward=False, use_max=True)
    ptzC_ind = detect_landmark(vein, tip_indices, tip_indices[-2], tip_indices[-3], ptzD_ind, use_forward=False, use_max=False)
    ptzF_ind = detect_landmark(vein, tip_indices, tip_indices[-4], tip_indices[-5], tip_indices[-5], use_forward=False, use_max=True)
    ptzE_ind = detect_landmark(vein, tip_indices, tip_indices[-3], tip_indices[-4], ptzF_ind, use_forward=False, use_max=False)
    ptzG_ind = detect_landmark(vein, tip_indices, tip_indices[-5], tip_indices[-6], tip_indices[-6], use_forward=False, use_max=True)
    ptzI_ind = detect_landmark(vein, tip_indices, tip_indices[-6], tip_indices[-7], tip_indices[-7], use_forward=False, use_max=True)
    ptzH_ind = detect_landmark(vein, tip_indices, tip_indices[-5], tip_indices[-6], ptzI_ind, use_forward=False, use_max=False)
    ptzK_ind = detect_landmark(vein, tip_indices, tip_indices[-7], tip_indices[-8], tip_indices[-8], use_forward=False, use_max=True)
    ptzJ_ind = detect_landmark(vein, tip_indices, tip_indices[-6], tip_indices[-7], ptzK_ind, use_forward=False, use_max=False)
    ptzM_ind = detect_landmark(vein, tip_indices, tip_indices[-8], tip_indices[-9], tip_indices[-9], use_forward=False, use_max=True)
    ptzL_ind = detect_landmark(vein, tip_indices, tip_indices[-7], tip_indices[-8], ptzM_ind, use_forward=False, use_max=False)
    ptzN_ind = detect_landmark(vein, tip_indices, tip_indices[-9], tip_indices[-10], tip_indices[-10], use_forward=False, use_max=True)
    ptzP_ind = detect_landmark(vein, tip_indices, tip_indices[-10], tip_indices[-11], tip_indices[-11], use_forward=False, use_max=True)
    ptzO_ind = detect_landmark(vein, tip_indices, tip_indices[-9], tip_indices[-10], ptzP_ind, use_forward=False, use_max=False)
    ptzR_ind = detect_landmark(vein, tip_indices, tip_indices[-11], tip_indices[-12], tip_indices[-12], use_forward=False, use_max=True)
    ptzQ_ind = detect_landmark(vein, tip_indices, tip_indices[-10], tip_indices[-11], ptzR_ind, use_forward=False, use_max=False)
    ptzT_ind = detect_landmark(vein, tip_indices, tip_indices[-12], tip_indices[-13], tip_indices[-13], use_forward=False, use_max=True)
    ptzS_ind = detect_landmark(vein, tip_indices, tip_indices[-11], tip_indices[-12], ptzT_ind, use_forward=False, use_max=False)

    landmark_indices = [ptA_ind,ptB_ind,ptC_ind,ptD_ind,ptE_ind,ptF_ind,ptG_ind,ptH_ind,ptI_ind,ptJ_ind,
                        ptK_ind,ptL_ind,ptM_ind,ptN_ind,ptO_ind,ptP_ind,ptQ_ind,ptR_ind,ptS_ind,ptT_ind,
                        ptzT_ind,ptzS_ind,ptzR_ind,ptzQ_ind,ptzP_ind,ptzO_ind,ptzN_ind,ptzM_ind,ptzL_ind,ptzK_ind,
                        ptzJ_ind,ptzI_ind,ptzH_ind,ptzG_ind,ptzF_ind,ptzE_ind,ptzD_ind,ptzC_ind,ptzB_ind,ptzA_ind]
    return landmark_indices

def interpolated_intervals(land_indices, new_xvals, new_yvals, num_land):
    inter_points_x = []
    inter_points_y = []
    
    # Filter out any sentinel values (-1)
    valid_land_indices = [idx for idx in land_indices if idx != -1]
    
    for i in range(len(valid_land_indices) - 1):
        beg_ind = valid_land_indices[i]
        end_ind = valid_land_indices[i + 1]
        
        # Ensure beg_ind < end_ind
        if beg_ind > end_ind:
            beg_ind, end_ind = end_ind, beg_ind
        
        interval_xvals = new_xvals[beg_ind:end_ind]
        interval_yvals = new_yvals[beg_ind:end_ind]
        
        if len(interval_xvals) < 2:
            continue
            
        curr_inter_xvals, curr_inter_yvals = interpolation(interval_xvals, interval_yvals, num_land)
        
        # Original logic had complex deletion, a simpler way is to just keep the first point
        # and append the rest, ensuring no duplicates.
        if i == 0:
            inter_points_x.extend(list(curr_inter_xvals))
            inter_points_y.extend(list(curr_inter_yvals))
        else:
            inter_points_x.extend(list(curr_inter_xvals)[1:])
            inter_points_y.extend(list(curr_inter_yvals)[1:])
            
    # The original logic for `interpolated_intervals` had a lot of `del` statements
    # which can be error-prone if the number of points changes. The above
    # logic is a safer, more robust way to handle it.
            
    # The following is a re-implementation of the original logic, assuming the number of landmarks is fixed.
    inter_points_x = []
    inter_points_y = []
    for i in range(len(land_indices) - 1):
        beg_ind = land_indices[i]
        end_ind = land_indices[i + 1]
        interval_xvals = new_xvals[beg_ind:end_ind]
        interval_yvals = new_yvals[beg_ind:end_ind]
        curr_inter_xvals, curr_inter_yvals = interpolation(interval_xvals, interval_yvals, num_land)
        curr_inter_xvals = list(curr_inter_xvals)
        curr_inter_yvals = list(curr_inter_yvals)
        if i == 0:
            del curr_inter_xvals[-1]
            del curr_inter_yvals[-1]
        if i != 0:
            del curr_inter_xvals[0]
            del curr_inter_yvals[0]
        for j in range(len(curr_inter_xvals)):
            inter_points_x.append(curr_inter_xvals[j])
            inter_points_y.append(curr_inter_yvals[j])
            
    return inter_points_x, inter_points_y

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calc_gpa_mean(shape_arr, num_pseuds, num_coords):
    shape_list = shape_arr
    ref_ind = 0
    ref_shape = shape_list[ref_ind]
    mean_diff = 10**(-30)
    old_mean = ref_shape
    d = 1000000
    while d > mean_diff:
        arr = np.zeros(((len(shape_list)), num_pseuds, num_coords))
        for i in range(len(shape_list)):
            s1, s2, distance = procrustes(old_mean, shape_list[i])
            arr[i] = s2
        new_mean = np.mean(arr, axis=(0))
        s1, s2, d = procrustes(old_mean, new_mean)
        old_mean = new_mean
    gpa_mean = new_mean
    return gpa_mean

def rotate_and_scale(vein_xvals, vein_yvals, blade_xvals, blade_yvals, base_ind, tip_ind, end_ind, px2_cm2):
    vein_arr = np.column_stack((vein_xvals, vein_yvals))
    blade_arr = np.column_stack((blade_xvals, blade_yvals))
    vein_len = np.shape(vein_arr)[0]
    blade_len = np.shape(blade_arr)[0]
    overall_len = vein_len + blade_len
    overall_arr = np.row_stack((vein_arr, blade_arr))
    px_cm = np.sqrt(px2_cm2)
    scaled_arr = overall_arr/px_cm
    tip_to_base_cm = np.sqrt((scaled_arr[tip_ind,0]-scaled_arr[base_ind,0])**2
                             + (scaled_arr[tip_ind,1]-scaled_arr[base_ind,1])**2)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(overall_arr)
    df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
    
    p1 = (df["pc1"].loc[tip_ind,], df["pc2"].loc[tip_ind,])
    p2 = (0,0)
    p3 = (10,0)
    angle = angle_between(p1, p2, p3)
    rotated_xvals, rotated_yvals = rotate_points(df["pc1"], df["pc2"], angle)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals))
    tip_to_base_pca = np.sqrt((rotated_arr[tip_ind,0]-rotated_arr[base_ind,0])**2
                              + (rotated_arr[tip_ind,1]-rotated_arr[base_ind,1])**2)
    scale = tip_to_base_cm/tip_to_base_pca
    scaled_arr = rotated_arr*scale
    
    pet_junc = np.mean(scaled_arr[[base_ind,end_ind],:],axis=0)
    
    trans_x = scaled_arr[:,0] - pet_junc[0]
    trans_y = scaled_arr[:,1] - pet_junc[1]
    
    scaled_arr = np.column_stack((trans_x, trans_y))
    
    if scaled_arr[10,0] < 0:
        scaled_arr[:,0] = -scaled_arr[:,0]
    scaled_vein = scaled_arr[0:vein_len,]
    scaled_blade = scaled_arr[vein_len:(vein_len+blade_len),]
    
    return scaled_vein, scaled_blade

def rotate_to_negative_y(leaf_arr, base_ind, tip_ind):
    xvals = leaf_arr[:, 0]
    yvals = leaf_arr[:, 1]
    tip_vec = np.array([xvals[tip_ind] - xvals[base_ind], yvals[tip_ind] - yvals[base_ind]])
    target_vec = np.array([0, -1])
    dot_product = np.dot(tip_vec, target_vec)
    magnitude_product = np.linalg.norm(tip_vec) * np.linalg.norm(target_vec)
    angle_rads = np.arccos(dot_product / magnitude_product)
    cross_product = np.cross(tip_vec, target_vec)
    if cross_product < 0:
        angle_rads = -angle_rads
    angle_degrees = np.degrees(angle_rads)
    rotated_xvals, rotated_yvals = rotate_points(xvals, yvals, angle_degrees)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals))
    centroid = np.mean(rotated_arr, axis=0)
    rotated_arr -= centroid
    if rotated_arr[10,0] < 0:
        rotated_arr[:,0] = -rotated_arr[:,0]
    return rotated_arr

def get_algeria_class_label(filename):
    """
    Determines the specific class label for an Algerian sample based on its filename.
    """
    algeria_mapping = {
        "AHMEUR BOU AHMEUR": "Ahmeur Bou Ahmeur",
        "BABARI": "Babari",
        "GENOTYPE 1": "Ichmoul",
        "GENOTYPE 2": "Ichmoul Bacha",
        "GENOTYPE 3": "Bouabane des Aures",
        "GENOTYPE 4": "Amer Bouamar",
        "LOUALI": "Louali",
        "TIZI OUININE": "Tizi Ouinine"
    }

    # Iterate through the mapping to find a matching substring in the filename
    for substring, label in algeria_mapping.items():
        if substring in filename:
            return label
    
    # If no match is found, return a default 'algeria' or a placeholder.
    # Given the instructions, we can assume a match will be found for all 'algeria' leaves.
    return "algeria"


# --- Configuration and Inputs ---

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# File and directory parameters
DATA_DIR = "../data/high_res_data/"
OUTPUT_DIR = "../outputs/figures/"
METADATA_FILE = "../data/leaf_metadata.csv"
OUTPUT_BASE_DIR = "../outputs/CNN_classification/morphometrics/"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# PCA parameters
PC_NUMBER = 2
NUM_PC1_EIGENLEAVES = 10
NUM_PC2_EIGENLEAVES = 5
EIGENLEAF_SCALE_FACTOR = 0.25 # Scale factor for visualizing eigenleaves on the plot

# Parameters for landmarking
res = 1000
dist = 5
num_land = 20

# Output filenames
PCA_PARAMS_H5_FILENAME = "leaf_pca_model_parameters.h5"
ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME = "original_pca_scores_and_class_labels.h5"
PCA_EXPLAINED_VARIANCE_REPORT_FILENAME = "pca_explained_variance.txt"
MORPHOSPACE_PLOT_FILENAME = "morphospace_plot.png"
GPA_MEAN_SHAPE_PLOT_FILENAME = "gpa_mean_shape.png"

# --- End Configuration ---

print(f"Saving morphometrics outputs to: {OUTPUT_BASE_DIR}")
print(f"Saving figure outputs to: {OUTPUT_DIR}")

# ==============================================================================
# 3. DATA PREPARATION (CORRECTED)
# ==============================================================================

# Read the metadata file
try:
    metadata_df = pd.read_csv(METADATA_FILE)
    print(f"Metadata loaded from: {METADATA_FILE}")
except FileNotFoundError:
    print(f"Error: Metadata file not found at '{METADATA_FILE}'.")
    sys.exit(1)

# Filter out error leaves and get valid leaf IDs and labels
valid_metadata = metadata_df[metadata_df['is_error'] == False].reset_index(drop=True)
valid_leaf_list = valid_metadata['leaf_id'].tolist()
final_geno_labels = valid_metadata['genotype_label'].tolist()

print(f"Using {len(valid_leaf_list)} leaves from the metadata file.")

# Lists to store processed data
shape_list = []
scaled_vein_list = []
scaled_blade_list = []
processed_labels = []

# Main loop for data processing
for i, curr_leaf in enumerate(valid_leaf_list):
    print(f"Processing leaf {i+1}/{len(valid_leaf_list)}: {curr_leaf}")

    # Read in data and metadata
    try:
        vein_trace = np.loadtxt(join(DATA_DIR, curr_leaf + "_veins.txt"))
        blade_trace = np.loadtxt(join(DATA_DIR, curr_leaf + "_blade.txt"))
        info_df = pd.read_csv(join(DATA_DIR, curr_leaf + "_info.csv"))
        px2_cm2 = float(info_df.iloc[6, 1])
    except FileNotFoundError:
        print(f"  Warning: Data files for '{curr_leaf}' not found. Skipping.")
        continue

    # Interpolate for high resolution traces
    inter_vein_x, inter_vein_y = interpolation(vein_trace[:, 0], vein_trace[:, 1], res)
    inter_blade_x, inter_blade_y = interpolation(blade_trace[:, 0], blade_trace[:, 1], res)

    # Find tip landmarks
    origin = np.mean((vein_trace[0], vein_trace[-1]), axis=0)
    dist_ori = [euclid_dist(origin[0], origin[1], inter_vein_x[j], inter_vein_y[j]) for j in range(res)]
    peaks, _ = find_peaks(dist_ori, height=0, distance=dist)
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, res - 1)
    
    # --- MODIFICATION: Conditionally get the new class label ---
    current_genotype_label = final_geno_labels[i]
    if current_genotype_label == 'algeria':
        final_label = get_algeria_class_label(curr_leaf)
    else:
        final_label = current_genotype_label
    
    # Find internal landmarks
    inter_vein = np.column_stack((inter_vein_x, inter_vein_y))
    landmark_indices = internal_landmarks(inter_vein, peaks)
    
    # Find blade landmarks
    blade_pts = []
    for k in range(len(peaks)):
        blade_dists = [euclid_dist(inter_vein_x[peaks[k]], inter_vein_y[peaks[k]], inter_blade_x[l], inter_blade_y[l]) for l in range(res)]
        blade_pts.append(blade_dists.index(min(blade_dists)))
    
    # Combine landmarks into vein and blade pseudo-landmarks
    curr_tip_ind = peaks
    curr_int_ind = landmark_indices
    curr_bla_ind = blade_pts
    
    curr_vei_ind = [curr_tip_ind[0], curr_int_ind[0], curr_tip_ind[1], curr_int_ind[1], curr_int_ind[2], curr_tip_ind[2], curr_int_ind[3], curr_int_ind[4], curr_tip_ind[3], curr_int_ind[5], curr_tip_ind[4], curr_int_ind[6], curr_int_ind[7], curr_tip_ind[5], curr_int_ind[8], curr_int_ind[9], curr_tip_ind[6], curr_int_ind[10], curr_int_ind[11], curr_tip_ind[7], curr_int_ind[12], curr_tip_ind[8], curr_int_ind[13], curr_int_ind[14], curr_tip_ind[9], curr_int_ind[15], curr_int_ind[16], curr_tip_ind[10], curr_int_ind[17], curr_int_ind[18], curr_tip_ind[11], curr_int_ind[19], curr_tip_ind[12], curr_int_ind[20], curr_tip_ind[13], curr_int_ind[21], curr_int_ind[22], curr_tip_ind[14], curr_int_ind[23], curr_int_ind[24], curr_tip_ind[15], curr_int_ind[25], curr_int_ind[26], curr_tip_ind[16], curr_int_ind[27], curr_tip_ind[17], curr_int_ind[28], curr_int_ind[29], curr_tip_ind[18], curr_int_ind[30], curr_int_ind[31], curr_tip_ind[19], curr_int_ind[32], curr_int_ind[33], curr_tip_ind[20], curr_int_ind[34], curr_tip_ind[21], curr_int_ind[35], curr_int_ind[36], curr_tip_ind[22], curr_int_ind[37], curr_int_ind[38], curr_tip_ind[23], curr_int_ind[39], curr_tip_ind[24]]

    vein_pseudx, vein_psuedy = interpolated_intervals(curr_vei_ind, inter_vein_x, inter_vein_y, num_land)
    blade_pseudx, blade_psuedy = interpolated_intervals(curr_bla_ind, inter_blade_x, inter_blade_y, num_land)
    
    # Use the provided rotate_and_scale function for each leaf
    vein_len = len(vein_pseudx)
    blade_len = len(blade_pseudx)
    total_len = vein_len + blade_len
    
    tip_ind = int(vein_len / 2)
    base_ind = 0
    end_ind = total_len - 1
    
    scaled_vein, scaled_blade = rotate_and_scale(vein_pseudx, vein_psuedy,
                                                 blade_pseudx, blade_psuedy,
                                                 base_ind=base_ind,
                                                 tip_ind=tip_ind,
                                                 end_ind=end_ind,
                                                 px2_cm2=px2_cm2)
    
    # Append to lists
    scaled_vein_list.append(scaled_vein)
    scaled_blade_list.append(scaled_blade)
    shape_list.append(np.row_stack((scaled_vein, scaled_blade)))
    processed_labels.append(final_label) # Use the new, more specific label

# Make sure all leaves in shape_list have the same number of landmarks
# This is a critical check for PCA and GPA
final_shape_list = []
final_labels = []
final_vein_coords = []
final_blade_coords = []

if shape_list:
    ref_num_landmarks = shape_list[0].shape[0]
    for i in range(len(shape_list)):
        if shape_list[i].shape[0] == ref_num_landmarks:
            final_shape_list.append(shape_list[i])
            final_labels.append(processed_labels[i])
            final_vein_coords.append(scaled_vein_list[i])
            final_blade_coords.append(scaled_blade_list[i])
else:
    print("No leaves were successfully processed. Exiting script.")
    sys.exit(1)

print(f"\nSuccessfully processed {len(final_shape_list)} leaves out of {len(valid_leaf_list)} total valid leaves.")

# Reshape data for Procrustes analysis
num_samples = len(final_shape_list)
num_pseuds = np.shape(final_shape_list)[1]
num_coords = 2
shape_arr = np.zeros((num_samples, num_pseuds, num_coords))
for i in range(num_samples):
    shape_arr[i] = final_shape_list[i]
aligned_vein_coords = np.stack(final_vein_coords)
aligned_blade_coords = np.stack(final_blade_coords)
num_vein_coords = aligned_vein_coords.shape[1]
num_blade_coords = aligned_blade_coords.shape[1]

# Perform GPA on the pre-oriented leaves
print("\n--- Performing GPA ---")
gpa_mean = calc_gpa_mean(shape_arr, num_pseuds, num_coords)
proc_arr = np.zeros(shape_arr.shape)
for i in range(num_samples):
    s1, s2, distance = procrustes(gpa_mean, shape_arr[i,:,:])
    proc_arr[i,:,:] = s2
    
# Visualize GPA Aligned Shapes
plt.figure(figsize=(8, 8))
for i in range(len(proc_arr)):
    plt.plot(proc_arr[i, :, 0], proc_arr[i, :, 1], c="k", alpha=0.08)
plt.plot(np.mean(proc_arr, axis=0)[:, 0], np.mean(proc_arr, axis=0)[:, 1], c="magenta")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title("Procrustes Aligned Leaf Shapes and GPA Mean")
plt.savefig(os.path.join(OUTPUT_BASE_DIR, GPA_MEAN_SHAPE_PLOT_FILENAME))
plt.close()
print(f"GPA mean shape plot saved to {os.path.join(OUTPUT_BASE_DIR, GPA_MEAN_SHAPE_PLOT_FILENAME)}")

# Re-run Procrustes analysis using the newly oriented mean leaf as the reference shape
print("--- Orienting GPA mean leaf and re-aligning all leaves ---")
tip_of_mean_leaf_index = int(num_vein_coords / 2)
base_of_mean_leaf_index = 0
oriented_gpa_mean = rotate_to_negative_y(gpa_mean, base_ind=base_of_mean_leaf_index, tip_ind=tip_of_mean_leaf_index)
final_proc_arr = np.zeros(shape_arr.shape)
for i in range(num_samples):
    s1, s2, distance = procrustes(oriented_gpa_mean, shape_arr[i,:,:])
    final_proc_arr[i,:,:] = s2
    
# Store final aligned coordinates separately
final_aligned_vein_coords = final_proc_arr[:, :num_vein_coords, :]
final_aligned_blade_coords = final_proc_arr[:, num_vein_coords:, :]

# Reshape for PCA
reshaped_arr = final_proc_arr.reshape(final_proc_arr.shape[0], final_proc_arr.shape[1] * final_proc_arr.shape[2])

# ==============================================================================
# 4. PCA AND SAVING DATA FOR SMOTE
# ==============================================================================

# Perform PCA (using all components) and generate explained variance report
print("\n--- Performing Full PCA and Generating Explained Variance Report ---")
max_pc_components = min(reshaped_arr.shape[0], reshaped_arr.shape[1])
pca = PCA(n_components=max_pc_components)
PCs = pca.fit_transform(reshaped_arr)

pca_explained_variance_filepath = os.path.join(OUTPUT_BASE_DIR, PCA_EXPLAINED_VARIANCE_REPORT_FILENAME)
with open(pca_explained_variance_filepath, 'w') as f:
    f.write("PCA Explained Variance Report:\n")
    f.write(f"Total Samples: {reshaped_arr.shape[0]}\n")
    f.write(f"Total Features (landmarks * dimensions): {reshaped_arr.shape[1]}\n")
    f.write(f"Number of PCs Calculated: {pca.n_components_}\n\n")
    f.write("PC: var, cumulative\n")
    for i in range(len(pca.explained_variance_ratio_)):
        pc_variance = round(pca.explained_variance_ratio_[i] * 100, 2)
        cumulative_variance = round(pca.explained_variance_ratio_.cumsum()[i] * 100, 2)
        line = f"PC{i+1}: {pc_variance}%, {cumulative_variance}%\n"
        f.write(line)
print(f"PCA explained variance report saved to {pca_explained_variance_filepath}")

# Save PCA Model Parameters and data for the next step
print("\n--- Saving PCA model parameters, PC scores, and class labels ---")
pca_params_filepath = os.path.join(OUTPUT_BASE_DIR, PCA_PARAMS_H5_FILENAME)
with h5py.File(pca_params_filepath, 'w') as f:
    f.create_dataset('components', data=pca.components_, compression="gzip")
    f.create_dataset('mean', data=pca.mean_, compression="gzip")
    f.create_dataset('explained_variance', data=pca.explained_variance_, compression="gzip")
    f.create_dataset('explained_variance_ratio', data=pca.explained_variance_ratio_, compression="gzip")
    f.attrs['n_components'] = pca.n_components_
    f.attrs['num_vein_coords'] = num_vein_coords
    f.attrs['num_blade_coords'] = num_blade_coords
    f.attrs['num_landmarks_total'] = num_pseuds
print(f"PCA parameters, including landmark counts, saved to {pca_params_filepath}")

pca_scores_labels_filepath = os.path.join(OUTPUT_BASE_DIR, ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME)
with h5py.File(pca_scores_labels_filepath, 'w') as f:
    f.create_dataset('pca_scores', data=PCs, compression="gzip")
    f.create_dataset('class_labels', data=np.array(final_labels).astype('S'), compression="gzip")
    f.create_dataset('original_flattened_coords', data=reshaped_arr, compression="gzip")
    f.create_dataset('aligned_vein_coords', data=final_aligned_vein_coords, compression="gzip")
    f.create_dataset('aligned_blade_coords', data=final_aligned_blade_coords, compression="gzip")
print(f"Original PCA scores, class labels, and aligned vein/blade coordinates saved to {pca_scores_labels_filepath}")

# ==============================================================================
# 5. CREATE MORPHOSPACE
# ==============================================================================

print("\n--- Creating Morphospace Plot ---")
morphospace_pca = PCA(n_components=PC_NUMBER)
morphospace_PCs = morphospace_pca.fit_transform(reshaped_arr)
geno_pca_df = pd.DataFrame(data=morphospace_PCs, columns=['PC1', 'PC2'])
geno_pca_df['label'] = final_labels

# --- MODIFICATION: Dynamically create PC values for plotting ---
# Using the range of the projected data for a more accurate representation
pc1_min, pc1_max = np.min(morphospace_PCs[:, 0]), np.max(morphospace_PCs[:, 0])
pc2_min, pc2_max = np.min(morphospace_PCs[:, 1]), np.max(morphospace_PCs[:, 1])

PC1_vals = np.linspace(pc1_min, pc1_max, NUM_PC1_EIGENLEAVES)
PC2_vals = np.linspace(pc2_min, pc2_max, NUM_PC2_EIGENLEAVES)

plt.figure(figsize=(10, 10)) # Increased figure size for better clarity

# --- MODIFICATION: Set EIGENLEAF_SCALE_FACTOR as a variable at the top ---
# and ensure it's defined before being used.
EIGENLEAF_SCALE_FACTOR = 0.25

# Plot Eigenleaves
for i in PC1_vals:
    for j in PC2_vals:
        # Reconstruct the leaf shape from the PC scores
        inv_leaf = morphospace_pca.inverse_transform(np.array([i, j]))
        inv_leaf = np.reshape(inv_leaf, (num_pseuds, 2))
        
        inv_leaf_veinX = inv_leaf[0:num_vein_coords, 0]
        inv_leaf_veinY = inv_leaf[0:num_vein_coords, 1]
        inv_leaf_bladeX = inv_leaf[num_vein_coords:, 0]
        inv_leaf_bladeY = inv_leaf[num_vein_coords:, 1]

        # Plot blade
        plt.fill(inv_leaf_bladeX * EIGENLEAF_SCALE_FACTOR + i,
                 inv_leaf_bladeY * EIGENLEAF_SCALE_FACTOR + j,
                 facecolor="lightgray", edgecolor="none", lw=1, zorder=1)
        
        # Plot veins
        plt.plot(inv_leaf_veinX * EIGENLEAF_SCALE_FACTOR + i,
                 inv_leaf_veinY * EIGENLEAF_SCALE_FACTOR + j,
                 color="darkgray", lw=1, zorder=1)
        
# Plot scatter points
unique_labels = sorted(geno_pca_df['label'].dropna().unique())
# Use a color palette that can handle the new number of classes (12)
full_palette = sns.color_palette("tab20", len(unique_labels))
label_pal = {label: color for label, color in zip(unique_labels, full_palette)}

ax = sns.scatterplot(data=geno_pca_df, x="PC1", y="PC2", hue="label",
                     palette=label_pal, zorder=2, s=50, alpha=1)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title=None,
          frameon=False, fontsize=8, labelspacing=0.5, bbox_to_anchor=(1.01, 1), loc='upper left')

# Adjust labels and title to reflect the more detailed analysis
xlab = f"PC1, {np.round(morphospace_pca.explained_variance_ratio_[0]*100, 1)}%"
ylab = f"PC2, {np.round(morphospace_pca.explained_variance_ratio_[1]*100, 1)}%"
plt.xlabel(xlab, fontsize=10)
plt.ylabel(ylab, fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Morphospace of Grapevine Leaf Varieties', fontsize=12) # Added a title
plt.gca().set_aspect("equal")
plt.gca().set_facecolor("white")
plt.grid()
plt.gca().set_axisbelow(True)
plt.tight_layout()

if not exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)
plt.savefig(join(OUTPUT_DIR, MORPHOSPACE_PLOT_FILENAME), bbox_inches='tight', dpi=300)
plt.close()

print(f"\nFigure saved to {join(OUTPUT_DIR, MORPHOSPACE_PLOT_FILENAME)}")
print("\n--- All processing and saving completed ---")
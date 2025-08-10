# scripts/05_CNN_classify_gradCAM_fig.py

# ==============================================================================
# 0. PARAMETERS
# ==============================================================================

# File and directory parameters
ROOT_DIR = "../"
DATA_DIR = "../data/high_res_data/"
METADATA_FILE = "../data/leaf_metadata.csv"
GRADCAM_DIR = "../outputs/CNN_classification/trained_models/grad_cam_images/"
OUTPUT_DIR = "../outputs/figures/"
OUTPUT_FILENAME = "fig_gradCAM.png"

# Visualization parameters
MEANLEAF_BLADE_COLOR = "lightgray"
MEANLEAF_VEIN_COLOR = "darkgray"
FIGURE_WIDTH = 8.5 # inches
FIGURE_DPI = 300

# Class labels in the specified order for the figure panels
CLASS_LABELS = [
    "Ahmeur Bou Ahmeur", "Babari", "Ichmoul", "Ichmoul Bacha",
    "Bouabane des Aures", "Amer Bouamar", "Louali", "Tizi Ouinine",
    "dissected", "rootstock", "vinifera", "wild"
]

# Landmarking and PCA parameters (from previous script, adapted)
res = 1000
dist = 5
num_land = 20
PC_NUMBER = 2


# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from os.path import isfile, join, exists, splitext
from os import makedirs


# ==============================================================================
# 2. FUNCTIONS
# ==============================================================================

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
        max_dist_ind = ref_dist.index(max(ref_dist))
        pt_ind = dist_ind[max_dist_ind]
    else:
        min_dist_ind = ref_dist.index(min(ref_dist))
        pt_ind = dist_ind[min_dist_ind]
    return pt_ind

def internal_landmarks(vein, tip_indices):
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
    """
    define a function to rotate tip downwards, scale to centimeters, and translate petiolar junction to the origin
    inputs: interpolated x and y vein and blade values, base(first), tip, and end indices, and px2 to cm2 scale
    outputs: rotated, scaled, and translated landmark, vein, and blade coordinates
    dependencies: PCA from sklearn, rotate_points
    """
    vein_arr = np.column_stack((vein_xvals, vein_yvals)) # create vein coordinates array
    blade_arr = np.column_stack((blade_xvals, blade_yvals)) # create blade coordinates array
    vein_len = np.shape(vein_arr)[0] # get lengths of vein and blade arrays to retrieve coords later
    blade_len = np.shape(blade_arr)[0]
    overall_len = vein_len + blade_len
    overall_arr = np.row_stack((vein_arr, blade_arr)) # stack vein and blade arrays into single array
    px_cm = np.sqrt(px2_cm2) # take square root of scaling factor to scale pixels to cm
    scaled_arr = overall_arr/px_cm # convert pixels into cm
    tip_to_base_cm = np.sqrt((scaled_arr[tip_ind,0]-scaled_arr[base_ind,0])**2
                             + (scaled_arr[tip_ind,1]-scaled_arr[base_ind,1])**2)

    # perform a principal component analysis on data to center
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(overall_arr)
    df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

    # find the angle of the leaf tip relative to the origin
    p1 = (df["pc1"].loc[tip_ind,], df["pc2"].loc[tip_ind,]) # get leaf tip PC1/PC2 coordinate value
    p2 = (0,0) # find angle relative to vertex at origin
    p3 = (10,0) # an arbitrary positive point along the x axis to find angle in anticlockwise direction
    angle = angle_between(p1, p2, p3) # find the angle in degrees of tip point relative to origin, anticlockwise
    rotated_xvals, rotated_yvals = rotate_points(df["pc1"], df["pc2"], angle)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals)) # stack x and y vals back into one array
    tip_to_base_pca = np.sqrt((rotated_arr[tip_ind,0]-rotated_arr[base_ind,0])**2
                              + (rotated_arr[tip_ind,1]-rotated_arr[base_ind,1])**2)
    scale = tip_to_base_cm/tip_to_base_pca # find the factor to scale back to cm
    scaled_arr = rotated_arr*scale # scale rotated PC vals back to cm

    pet_junc = np.mean(scaled_arr[[base_ind,end_ind],:],axis=0)

    trans_x = scaled_arr[:,0] - pet_junc[0]
    trans_y = scaled_arr[:,1] - pet_junc[1]

    scaled_arr = np.column_stack((trans_x, trans_y))

    if scaled_arr[10,0] < 0: # insure left side of the leaf is left so labels are on right side of plot
        scaled_arr[:,0] = -scaled_arr[:,0]
    scaled_vein = scaled_arr[0:vein_len,] # isolate just vein coords
    scaled_blade = scaled_arr[vein_len:(vein_len+blade_len),] # isolate just blade coords

    return scaled_vein, scaled_blade # return scaled and rotated vein and blade

def rotate_to_negative_y(leaf_arr, base_ind, tip_ind):
    """
    Rotates a leaf shape so that the vector from base to tip points down the negative y-axis.
    """
    xvals = leaf_arr[:, 0]
    yvals = leaf_arr[:, 1]

    # Vector from base to tip
    tip_vec = np.array([xvals[tip_ind] - xvals[base_ind], yvals[tip_ind] - yvals[base_ind]])

    # Target vector (negative y-axis)
    target_vec = np.array([0, -1])

    # Calculate angle between vectors
    dot_product = np.dot(tip_vec, target_vec)
    magnitude_product = np.linalg.norm(tip_vec) * np.linalg.norm(target_vec)
    angle_rads = np.arccos(dot_product / magnitude_product)

    # Cross product to determine direction of rotation
    cross_product = np.cross(tip_vec, target_vec)
    if cross_product < 0:
        angle_rads = -angle_rads

    angle_degrees = np.degrees(angle_rads)

    # Rotate the points
    rotated_xvals, rotated_yvals = rotate_points(xvals, yvals, angle_degrees)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals))

    # Re-center the leaf on the origin
    centroid = np.mean(rotated_arr, axis=0)
    rotated_arr -= centroid

    # Ensure left side of the leaf is left so labels are on right side of plot
    if rotated_arr[10,0] < 0:
        rotated_arr[:,0] = -rotated_arr[:,0]

    return rotated_arr

# --- NEW HELPER FUNCTION FOR CLASSIFYING LEAVES ---
def classify_leaf(filename, genotype_label):
    if genotype_label == "algeria":
        if "AHMEUR BOU AHMEUR" in filename: return "Ahmeur Bou Ahmeur"
        elif "BABARI" in filename: return "Babari"
        elif "GENOTYPE 1" in filename: return "Ichmoul"
        elif "GENOTYPE 2" in filename: return "Ichmoul Bacha"
        elif "GENOTYPE 3" in filename: return "Bouabane des Aures"
        elif "GENOTYPE 4" in filename: return "Amer Bouamar"
        elif "LOUALI" in filename: return "Louali"
        elif "TIZI OUININE" in filename: return "Tizi Ouinine"
    return genotype_label


# ==============================================================================
# 3. DATA PREPARATION
# ==============================================================================

# Read the metadata file
try:
    metadata_df = pd.read_csv(METADATA_FILE)
except FileNotFoundError:
    print(f"Error: Metadata file not found at '{METADATA_FILE}'.")
    exit()

# Filter out error leaves and get valid leaf IDs and labels
valid_metadata = metadata_df[metadata_df['is_error'] == False].reset_index(drop=True)
valid_leaf_list = valid_metadata['leaf_id'].tolist()

print(f"Using {len(valid_leaf_list)} leaves from the metadata file.")

# Lists to store processed data
tip_indices_list = []
land_indices_list = []
blade_indices_list = []
processed_shapes = {}

# Initialize dictionary to store shapes for each class
for label in CLASS_LABELS:
    processed_shapes[label] = []

# Main loop for data processing and classification
for i, curr_leaf in enumerate(valid_leaf_list):
    print(f"Processing leaf {i+1}: {curr_leaf}")

    # Read in data and metadata
    try:
        vein_trace = np.loadtxt(join(DATA_DIR, curr_leaf + "_veins.txt"))
        inter_vein_x, inter_vein_y = interpolation(vein_trace[:, 0], vein_trace[:, 1], res)
        blade_trace = np.loadtxt(join(DATA_DIR, curr_leaf + "_blade.txt"))
        inter_blade_x, inter_blade_y = interpolation(blade_trace[:, 0], blade_trace[:, 1], res)
        info_df = pd.read_csv(join(DATA_DIR, curr_leaf + "_info.csv"))
        px2_cm2 = float(info_df.iloc[6, 1])
    except FileNotFoundError:
        print(f"Warning: Data files for leaf {curr_leaf} not found. Skipping.")
        continue

    # Find tip landmarks
    origin = np.mean((vein_trace[0], vein_trace[-1]), axis=0)
    dist_ori = [euclid_dist(origin[0], origin[1], inter_vein_x[j], inter_vein_y[j]) for j in range(res)]
    peaks, _ = find_peaks(dist_ori, height=0, distance=dist)
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, res - 1)

    # Find internal landmarks
    inter_vein = np.column_stack((inter_vein_x, inter_vein_y))
    landmark_indices = internal_landmarks(inter_vein, peaks)

    # Find blade landmarks
    blade_pts = []
    for k in range(len(peaks)):
        blade_dists = [euclid_dist(inter_vein_x[peaks[k]], inter_vein_y[peaks[k]], inter_blade_x[l], inter_blade_y[l]) for l in range(res)]
        blade_pts.append(blade_dists.index(min(blade_dists)))

    # Combine landmarks into vein and blade pseudo-landmarks
    curr_vei_ind = [peaks[0], landmark_indices[0], peaks[1], landmark_indices[1], landmark_indices[2], peaks[2], landmark_indices[3], landmark_indices[4], peaks[3], landmark_indices[5], peaks[4], landmark_indices[6], landmark_indices[7], peaks[5], landmark_indices[8], landmark_indices[9], peaks[6], landmark_indices[10], landmark_indices[11], peaks[7], landmark_indices[12], peaks[8], landmark_indices[13], landmark_indices[14], peaks[9], landmark_indices[15], landmark_indices[16], peaks[10], landmark_indices[17], landmark_indices[18], peaks[11], landmark_indices[19], peaks[12], landmark_indices[20], peaks[13], landmark_indices[21], landmark_indices[22], peaks[14], landmark_indices[23], landmark_indices[24], peaks[15], landmark_indices[25], landmark_indices[26], peaks[16], landmark_indices[27], peaks[17], landmark_indices[28], landmark_indices[29], peaks[18], landmark_indices[30], landmark_indices[31], peaks[19], landmark_indices[32], landmark_indices[33], peaks[20], landmark_indices[34], peaks[21], landmark_indices[35], landmark_indices[36], peaks[22], landmark_indices[37], landmark_indices[38], peaks[23], landmark_indices[39], peaks[24]]

    vein_pseudx, vein_psuedy = interpolated_intervals(curr_vei_ind, inter_vein_x, inter_vein_y, num_land)
    blade_pseudx, blade_psuedy = interpolated_intervals(blade_pts, inter_blade_x, inter_blade_y, num_land)

    # Use the provided rotate_and_scale function for each leaf
    vein_len = len(vein_pseudx)
    blade_len = len(blade_pseudx)
    total_len = vein_len + blade_len

    # Calculate tip and base indices for this specific leaf's data
    tip_ind = int(vein_len / 2)
    base_ind = 0
    end_ind = total_len - 1

    scaled_vein, scaled_blade = rotate_and_scale(vein_pseudx, vein_psuedy,
                                                 blade_pseudx, blade_psuedy,
                                                 base_ind=base_ind,
                                                 tip_ind=tip_ind,
                                                 end_ind=end_ind,
                                                 px2_cm2=px2_cm2)

    # Combine the scaled, rotated, and translated vein and blade for GPA
    full_shape = np.row_stack((scaled_vein, scaled_blade))

    # Classify and store the processed shape
    leaf_class = classify_leaf(curr_leaf, valid_metadata.loc[i, 'genotype_label'])
    if leaf_class in processed_shapes:
        processed_shapes[leaf_class].append(full_shape)

# Calculate GPA mean for each class
mean_leaves = {}
for label, shapes in processed_shapes.items():
    if shapes:
        num_pseuds = np.shape(shapes[0])[0]
        num_coords = 2
        mean_shape = calc_gpa_mean(np.array(shapes), num_pseuds, num_coords)
        mean_leaves[label] = mean_shape

num_vein_coords = len(vein_pseudx)
num_blade_coords = num_pseuds - num_vein_coords

# --- Final mean leaf orientation ---
# The tip is at the halfway point of the vein coords, and the base is at index 0.
tip_of_mean_leaf_index = num_vein_coords // 2
base_of_mean_leaf_index = 0


# ==============================================================================
# 4. FIGURE GENERATION
# ==============================================================================

# Create figure and a 6x4 GridSpec with adjusted vertical spacing
# NOTE: The hspace parameter is set directly in the GridSpec call.
fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 1.5)) # Adjust height for 6 rows
gs = plt.GridSpec(6, 4, figure=fig, wspace=0.05, hspace=0.5)

# Ensure output directory exists
if not exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)

# Loop through each of the 12 classes to plot mean leaves and Grad-CAMs
for i, class_name in enumerate(CLASS_LABELS):

    # Calculate the row and column index for the current panel
    row_idx = i // 4
    col_idx = i % 4

    # --- TOP PANEL (MEAN LEAF) ---
    ax_mean = fig.add_subplot(gs[row_idx, col_idx])
    if class_name in mean_leaves:
        mean_leaf = mean_leaves[class_name]
        # Orient the mean leaf with the tip pointing downwards
        oriented_mean_leaf = rotate_to_negative_y(mean_leaf,
                                                  base_ind=base_of_mean_leaf_index,
                                                  tip_ind=tip_of_mean_leaf_index)

        rotated_vein_coords = oriented_mean_leaf[0:num_vein_coords]
        rotated_blade_coords = oriented_mean_leaf[num_vein_coords:]

        # Plot the mean leaf with no edges
        ax_mean.fill(rotated_blade_coords[:, 0], rotated_blade_coords[:, 1], facecolor=MEANLEAF_BLADE_COLOR, edgecolor="none")
        ax_mean.fill(rotated_vein_coords[:, 0], rotated_vein_coords[:, 1], facecolor=MEANLEAF_VEIN_COLOR, edgecolor="none")

    ax_mean.set_title(class_name, fontsize=8, pad=5)
    ax_mean.set_aspect('equal')
    ax_mean.axis('off')

    # --- BOTTOM PANEL (GRAD-CAM IMAGE) ---
    # Grad-CAMs are in rows 4-6, which is row_idx + 3
    ax_gradcam = fig.add_subplot(gs[row_idx + 3, col_idx])

    # Set the title
    ax_gradcam.set_title(class_name, fontsize=8, pad=5)

    # Construct Grad-CAM filename (handling spaces)
    filename = f"ECT_Mask_4Channel_CNN_Ensemble_Improved_GradCAM_{class_name}.png"
    filepath = join(GRADCAM_DIR, filename)

    if exists(filepath):
        gradcam_img = Image.open(filepath)
        ax_gradcam.imshow(gradcam_img)
    else:
        ax_gradcam.text(0.5, 0.5, 'Grad-CAM not found', ha='center', va='center', fontsize=8, color='red')
        print(f"Warning: Grad-CAM file not found for {class_name} at {filepath}")

    ax_gradcam.axis('off')


# Use tight_layout after defining the grid spacing to prevent it from overriding the hspace
plt.tight_layout()

plt.savefig(join(OUTPUT_DIR, OUTPUT_FILENAME), bbox_inches='tight', dpi=FIGURE_DPI)

print(f"\nFigure saved to {join(OUTPUT_DIR, OUTPUT_FILENAME)}")
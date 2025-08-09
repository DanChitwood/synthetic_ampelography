#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
import cv2
import os
import matplotlib.cm as cm # For colormaps
import h5py
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder # Added for encoding class labels if needed
from sklearn.utils import shuffle # For consistent shuffling of multiple arrays

# Ensure the ect library is installed and accessible
try:
    from ect import ECT, EmbeddedGraph
except ImportError:
    print("Error: The 'ect' library is not found. Please ensure it's installed and accessible.")
    print("Add its directory to PYTHONPATH or optionally install it using pip:")
    print("pip install ect-morphology")
    sys.exit(1)

############################
### CONFIGURATION (ALL PARAMETERS UP FRONT) ###
############################

# --- Input Data Configuration (from previous script's output) ---
# MODIFIED: Point SAVED_MODEL_DIR to the 'outputs' directory where your .h5 files are
SAVED_MODEL_DIR = Path("../outputs/CNN_classification/morphometrics/") # Path to the directory where PCA model and scores were saved
PCA_PARAMS_FILE = SAVED_MODEL_DIR / "leaf_pca_model_parameters.h5"
PCA_SCORES_LABELS_FILE = SAVED_MODEL_DIR / "original_pca_scores_and_class_labels.h5" # Updated filename to match previous output

# --- Shape Information (from previous script) ---
# These are loaded dynamically from the PCA_PARAMS_FILE, but defaults are set for safety.
NUM_VEIN_COORDS = None 
NUM_BLADE_COORDS = None 
TOTAL_CONTOUR_COORDS = None 
FLATTENED_COORD_DIM = None

# --- ECT (Elliptic Contour Transform) Parameters ---
BOUND_RADIUS = 1 # The radius of the bounding circle for ECT normalization
NUM_ECT_DIRECTIONS = 180 # Number of radial directions for ECT calculation
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS) # Distance thresholds for ECT calculation

# --- Output Image Parameters ---
IMAGE_SIZE = (256, 256) # Output size for all generated images (masks, ECT, combined viz)

# Pixel values for masks
BACKGROUND_PIXEL = 0
VEIN_PIXEL = 1
BLADE_PIXEL = 1

# Grayscale values for output mask file
MASK_BACKGROUND_GRAY = 0       # Black background
MASK_SHAPE_GRAY = 255          # White shape

# --- Combined Visualization Parameters ---
OUTLINE_LINE_WIDTH = 2 # Line width for the leaf outline in combined_viz images
VEIN_OUTLINE_COLOR = (0, 0, 0) # Black for vein outline
BLADE_OUTLINE_COLOR = (255, 255, 255) # White for blade outline

# --- SMOTE-like Augmentation Parameters ---
SAMPLES_PER_CLASS_TARGET = 400 # Desired number of synthetic samples for EACH class
K_NEIGHBORS_SMOTE = 5 # Number of nearest neighbors to consider for SMOTE interpolation

# --- Random Rotation for Data Augmentation ---
APPLY_RANDOM_ROTATION = True # <<< SET THIS TO TRUE FOR DATA AUGMENTATION >>>
RANDOM_ROTATION_RANGE_DEG = (-180, 180) # Range of random rotation (in degrees) to apply to generated shapes

# --- Output Directory Structure for Synthetic Samples ---
SYNTHETIC_DATA_OUTPUT_DIR = Path("../outputs/CNN_classification/synthetic_leaf_data/")
SYNTHETIC_SHAPE_MASKS_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "shape_masks"
SYNTHETIC_SHAPE_ECTS_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "shape_ects"
SYNTHETIC_COMBINED_VIZ_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "combined_viz"

SYNTHETIC_METADATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"

# --- New: Consolidated Data Output for CNN Training ---
FINAL_PREPARED_DATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "final_cnn_dataset.pkl"

# Global ECT min/max will be calculated dynamically
GLOBAL_ECT_MIN = None
GLOBAL_ECT_MAX = None

###########################
### HELPER FUNCTIONS ###
###########################

def apply_transformation_with_affine_matrix(points: np.ndarray, affine_matrix: np.ndarray):
    """
    Applies a 3x3 affine matrix to a set of 2D points (N, 2) or a single point (2,).
    Returns the transformed points.
    """
    if points.size == 0:
        return np.array([])
        
    original_shape = points.shape
    if points.ndim == 1 and points.shape[0] == 2:
        points = points.reshape(1, 2) # Handle single 2D point

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Input 'points' must be a (N, 2) array or a (2,) array. Got shape: {original_shape}")

    if affine_matrix.shape != (3, 3):
        raise ValueError(f"Input 'affine_matrix' must be (3, 3). Got shape: {affine_matrix.shape}")

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    transformed_homogeneous = points_homogeneous @ affine_matrix.T
    
    # Return to original shape if single point was input
    if original_shape == (2,):
        return transformed_homogeneous[0, :2]
    return transformed_homogeneous[:, :2]

def find_robust_affine_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray):
    """
    Finds a robust affine transformation matrix between source and destination points.
    It attempts to find 3 non-collinear points for cv2.getAffineTransform.
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        if len(src_points) == 0:
            return np.eye(3) # Return identity for empty input
        raise ValueError(f"Need at least 3 points to compute affine transformation. Got {len(src_points)}.")

    chosen_src_pts = []
    chosen_dst_pts = []
    
    indices = np.arange(len(src_points))
    # Limit attempts for very large point sets to prevent infinite loops on degenerate shapes
    num_attempts = min(len(src_points) * (len(src_points) - 1) * (len(src_points) - 2) // 6, 1000) 

    for _ in range(num_attempts):
        selected_indices = np.random.choice(indices, 3, replace=False)
        p1_src, p2_src, p3_src = src_points[selected_indices]
        p1_dst, p2_dst, p3_dst = dst_points[selected_indices]
        
        # Check for collinearity by calculating area of triangle formed by points
        # Area formula: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        area_val = (p1_src[0] * (p2_src[1] - p3_src[1]) +
                    p2_src[0] * (p3_src[1] - p1_src[1]) +
                    p3_src[0] * (p1_src[1] - p2_src[1]))
        
        if np.abs(area_val) > 1e-6: # Check if points are not collinear
            chosen_src_pts = np.float32([p1_src, p2_src, p3_src])
            chosen_dst_pts = np.float32([p1_dst, p2_dst, p3_dst])
            break
    
    if len(chosen_src_pts) < 3:
        raise ValueError("Could not find 3 non-collinear points for affine transformation. Shape is likely degenerate or a line.")

    M_2x3 = cv2.getAffineTransform(chosen_src_pts, chosen_dst_pts)
    
    if M_2x3.shape != (2, 3):
        raise ValueError(f"cv2.getAffineTransform returned a non-(2,3) matrix: {M_2x3.shape}")

    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    
    return affine_matrix_3x3

def ect_coords_to_pixels(coords_ect: np.ndarray, image_size: tuple, bound_radius: float):
    """
    Transforms coordinates from ECT space (mathematical, Y-up, origin center, range [-R, R])
    to image pixel space (Y-down, origin top-left, range [0, IMAGE_SIZE]).
    Returns integer pixel coordinates.
    """
    if len(coords_ect) == 0:
        return np.array([])
        
    display_x_conceptual = coords_ect[:, 1]
    display_y_conceptual = coords_ect[:, 0]

    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2

    pixel_x = (display_x_conceptual * scale_factor + offset_x).astype(int)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y).astype(int)
    
    return np.column_stack((pixel_x, pixel_y))

def save_grayscale_shape_mask(transformed_coords: np.ndarray, save_path: Path):
    """
    Saves a grayscale image representing a transformed contour/pixel set.
    """
    img = Image.new("L", IMAGE_SIZE, MASK_BACKGROUND_GRAY)
    draw = ImageDraw.Draw(img)

    if transformed_coords is not None and transformed_coords.size > 0:
        pixel_coords = ect_coords_to_pixels(transformed_coords, IMAGE_SIZE, BOUND_RADIUS)
        
        pixel_coords = np.clip(pixel_coords, [0, 0], [IMAGE_SIZE[0] - 1, IMAGE_SIZE[1] - 1])

        if len(pixel_coords) >= 3:
            polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
            draw.polygon(polygon_points, fill=MASK_SHAPE_GRAY)
        elif len(pixel_coords) > 0:
            for x, y in pixel_coords:
                draw.point((x, y), fill=MASK_SHAPE_GRAY)
    
    img.save(save_path)

def save_radial_ect_image(ect_result, save_path: Path, cmap_name: str = "gray", vmin: float = None, vmax: float = None):
    """
    Saves the radial ECT plot as an image with the specified colormap.
    """
    if ect_result is None:
        Image.new("L", IMAGE_SIZE, 0).save(save_path)
        return

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100,
                           facecolor='white')
    
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap=cmap_name, vmin=vmin, vmax=vmax)
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, vein_coords: np.ndarray, blade_coords: np.ndarray,
                                     save_path: Path, line_width: int):
    """
    Creates a combined visualization by overlaying vein and blade outlines onto the ECT image.
    """
    try:
        ect_img = Image.open(ect_image_path).convert("RGBA")
        img_width, img_height = ect_img.size
        
        composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw_composite = ImageDraw.Draw(composite_overlay)

        # Draw vein outline
        if vein_coords is not None and len(vein_coords) > 0:
            vein_pixel_coords = ect_coords_to_pixels(vein_coords, IMAGE_SIZE, BOUND_RADIUS)
            vein_polygon_points = [(int(p[0]), int(p[1])) for p in vein_pixel_coords]
            if len(vein_polygon_points) >= 2:
                draw_composite.line(vein_polygon_points, fill=VEIN_OUTLINE_COLOR + (255,), width=line_width)

        # Draw blade outline
        if blade_coords is not None and len(blade_coords) > 0:
            blade_pixel_coords = ect_coords_to_pixels(blade_coords, IMAGE_SIZE, BOUND_RADIUS)
            blade_polygon_points = [(int(p[0]), int(p[1])) for p in blade_pixel_coords]
            if len(blade_polygon_points) >= 3:
                draw_composite.polygon(blade_polygon_points, outline=BLADE_OUTLINE_COLOR + (255,), width=line_width)
        
        final_combined_img = Image.alpha_composite(ect_img, composite_overlay).convert("RGB")
        final_combined_img.save(save_path)

    except FileNotFoundError:
        print(f"Error: ECT image file not found at {ect_image_path}. Skipping combined visualization.")
    except Exception as e:
        print(f"Error creating combined visualization for {ect_image_path.stem}: {e}")

def rotate_coords_2d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates 2D coordinates (Nx2 array) around the origin (0,0).
    """
    if coords.size == 0:
        return np.array([])
        
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    rotated_coords = coords @ rot_matrix.T
    return rotated_coords

##############################
### CORE LOGIC FUNCTIONS ###
##############################

def load_pca_model_data(pca_params_file: Path, pca_scores_labels_file: Path):
    """
    Loads PCA model parameters and original PCA scores/labels.
    """
    pca_data = {}
    with h5py.File(pca_params_file, 'r') as f:
        pca_data['components'] = f['components'][:]
        pca_data['mean'] = f['mean'][:]
        pca_data['explained_variance'] = f['explained_variance'][:]
        pca_data['n_components'] = f.attrs['n_components']
        # Load the crucial landmark counts from the previous script
        pca_data['num_vein_coords'] = f.attrs.get('num_vein_coords', None)
        pca_data['num_blade_coords'] = f.attrs.get('num_blade_coords', None)
    
    with h5py.File(pca_scores_labels_file, 'r') as f:
        pca_data['original_pca_scores'] = f['pca_scores'][:]
        pca_data['original_class_labels'] = np.array([s.decode('utf-8') for s in f['class_labels'][:]])
        if 'original_flattened_coords' in f:
            pca_data['original_flattened_coords'] = f['original_flattened_coords'][:]
        else:
            print("Warning: 'original_flattened_coords' not found. Real samples cannot be processed.")
            pca_data['original_flattened_coords'] = None
            
    if pca_data['num_vein_coords'] is None or pca_data['num_blade_coords'] is None:
        raise RuntimeError("Could not find 'num_vein_coords' and 'num_blade_coords' in the PCA parameters file. "
                           "Please ensure your morphometrics script saves these attributes.")

    print(f"Loaded PCA model parameters from {pca_params_file}.")
    print(f"Loaded original PCA scores and labels from {pca_scores_labels_file}.")
    print(f"Shape metadata from PCA file: {pca_data['num_vein_coords']} vein coords, {pca_data['num_blade_coords']} blade coords.")
    return pca_data

def generate_synthetic_pca_samples(pca_data: dict, samples_per_class_target: int, k_neighbors: int):
    """
    Generates synthetic PCA samples using a SMOTE-like approach based on class labels.
    """
    print(f"\nStarting synthetic data generation (SMOTE-like) with {samples_per_class_target} samples per class...")
    
    original_pca_scores = pca_data['original_pca_scores']
    original_class_labels = pd.Series(pca_data['original_class_labels'])
    
    synthetic_X_pca = []
    synthetic_y = []
    
    class_counts = original_class_labels.value_counts()
    all_classes = class_counts.index.tolist()
    
    total_generated_samples = 0

    for class_name in all_classes:
        class_pca_samples = original_pca_scores[original_class_labels == class_name]
        
        if len(class_pca_samples) < 2:
            print(f"Warning: Class '{class_name}' has less than 2 samples ({len(class_pca_samples)}). Skipping SMOTE-like augmentation.")
            continue
            
        n_neighbors_for_class = min(len(class_pca_samples) - 1, k_neighbors)
        if n_neighbors_for_class < 1:
            print(f"Warning: Class '{class_name}' has insufficient samples ({len(class_pca_samples)}) for meaningful NearestNeighbors calculation (k={k_neighbors}). Skipping.")
            continue

        nn = NearestNeighbors(n_neighbors=n_neighbors_for_class + 1).fit(class_pca_samples)
        
        generated_count = 0
        while generated_count < samples_per_class_target:
            idx_in_class_samples = np.random.randint(0, len(class_pca_samples))
            sample = class_pca_samples[idx_in_class_samples]
            
            distances, indices = nn.kneighbors(sample.reshape(1, -1))
            
            available_neighbors_indices_in_class_pca = indices[0][1:]
            
            if len(available_neighbors_indices_in_class_pca) == 0:
                continue
                
            neighbor_idx_in_class_pca_samples = np.random.choice(available_neighbors_indices_in_class_pca)
            neighbor = class_pca_samples[neighbor_idx_in_class_pca_samples]
            
            alpha = np.random.rand()
            synthetic_pca_sample = sample + alpha * (neighbor - sample)
            
            synthetic_X_pca.append(synthetic_pca_sample)
            synthetic_y.append(class_name)
            generated_count += 1
            total_generated_samples += 1
            
    print(f"Finished generating {total_generated_samples} synthetic samples across {len(all_classes)} classes.")
    return np.array(synthetic_X_pca), synthetic_y

def inverse_transform_pca(pca_scores: np.ndarray, pca_components: np.ndarray, pca_mean: np.ndarray):
    """
    Inverse transforms PCA scores back to the original flattened coordinate space.
    """
    reconstructed_data = np.dot(pca_scores, pca_components) + pca_mean
    return reconstructed_data

def process_leaf_for_cnn_output(
    sample_id: str,
    class_label: str,
    vein_coords: np.ndarray,
    blade_coords: np.ndarray,
    ect_calculator: ECT,
    output_dirs: dict,
    metadata_records: list,
    is_real_sample: bool = False,
    apply_random_rotation: bool = False,
    global_ect_min: float = None,
    global_ect_max: float = None
):
    """
    Processes a single leaf's vein and blade coordinates to produce four outputs for a 4-channel CNN.
    """
    current_metadata = {
        "synthetic_id": sample_id,
        "class_label": class_label,
        "is_processed_valid": False,
        "reason_skipped": "",
        "num_vein_coords": len(vein_coords) if vein_coords is not None else 0,
        "num_blade_coords": len(blade_coords) if blade_coords is not None else 0,
        "file_vein_mask": "",
        "file_vein_ect": "",
        "file_blade_mask": "",
        "file_blade_ect": "",
        "file_combined_viz": "",
        "is_real": is_real_sample
    }

    temp_ect_combined_viz_path = output_dirs['combined_viz'] / f"{sample_id}_ect_temp.png"
    
    # Check for degenerate shapes early
    if len(np.unique(vein_coords, axis=0)) < 2 or len(np.unique(blade_coords, axis=0)) < 3:
        current_metadata["reason_skipped"] = "Degenerate shape (too few unique points) for ECT/mask generation."
        metadata_records.append(current_metadata)
        print(f"Skipping leaf '{sample_id}' due to degenerate shape.")
        return

    try:
        # --- Apply Random Rotation if enabled ---
        if apply_random_rotation:
            random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)
            vein_coords_rotated = rotate_coords_2d(vein_coords, random_angle_deg)
            blade_coords_rotated = rotate_coords_2d(blade_coords, random_angle_deg)
        else:
            vein_coords_rotated = vein_coords.copy()
            blade_coords_rotated = blade_coords.copy()

        # --- Process Vein (ECT, Mask, and Visualization) ---
        # The vein is a line, so we add a small epsilon to the coordinates for ECT.
        vein_graph = EmbeddedGraph()
        vein_graph.add_cycle(vein_coords_rotated)

        vein_graph.center_coordinates(center_type="origin")
        vein_graph.transform_coordinates()
        vein_graph.scale_coordinates(BOUND_RADIUS)

        vein_ect_result = ect_calculator.calculate(vein_graph)
        
        # We use the final transformed coords from the EmbeddedGraph for the mask, not the raw rotated coords
        save_grayscale_shape_mask(vein_graph.coord_matrix, output_dirs['vein_masks'] / f"{sample_id}_vein_mask.png")
        save_radial_ect_image(vein_ect_result, output_dirs['vein_ects'] / f"{sample_id}_vein_ect.png", vmin=global_ect_min, vmax=global_ect_max)

        # --- Process Blade (ECT, Mask, and Visualization) ---
        blade_graph = EmbeddedGraph()
        blade_graph.add_cycle(blade_coords_rotated)

        blade_graph.center_coordinates(center_type="origin")
        blade_graph.transform_coordinates()
        blade_graph.scale_coordinates(BOUND_RADIUS)
        
        blade_ect_result = ect_calculator.calculate(blade_graph)

        save_grayscale_shape_mask(blade_graph.coord_matrix, output_dirs['blade_masks'] / f"{sample_id}_blade_mask.png")
        save_radial_ect_image(blade_ect_result, output_dirs['blade_ects'] / f"{sample_id}_blade_ect.png", vmin=global_ect_min, vmax=global_ect_max)

        # --- Create Combined Visualization (for verification) ---
        combined_viz_path = output_dirs['combined_viz'] / f"{sample_id}_combined.png"
        
        # For the visualization, we can use the blade's ECT as the background.
        save_radial_ect_image(blade_ect_result, temp_ect_combined_viz_path, cmap_name="gray_r", vmin=global_ect_min, vmax=global_ect_max)
        
        create_combined_viz_from_images(
            temp_ect_combined_viz_path, 
            vein_coords=vein_graph.coord_matrix,
            blade_coords=blade_graph.coord_matrix,
            save_path=combined_viz_path,
            line_width=OUTLINE_LINE_WIDTH
        )

        current_metadata["is_processed_valid"] = True
        current_metadata["file_vein_mask"] = str((output_dirs['vein_masks'] / f"{sample_id}_vein_mask.png").relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_vein_ect"] = str((output_dirs['vein_ects'] / f"{sample_id}_vein_ect.png").relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_blade_mask"] = str((output_dirs['blade_masks'] / f"{sample_id}_blade_mask.png").relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_blade_ect"] = str((output_dirs['blade_ects'] / f"{sample_id}_blade_ect.png").relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_combined_viz"] = str(combined_viz_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
    
    except Exception as e:
        current_metadata["reason_skipped"] = f"Processing failed: {e}"
        print(f"Skipping leaf '{sample_id}' due to error: {e}")

    finally:
        metadata_records.append(current_metadata)
        if temp_ect_combined_viz_path.exists():
            os.remove(temp_ect_combined_viz_path)


def calculate_global_ect_min_max(all_flattened_coords: np.ndarray, ect_calculator: ECT, num_vein_coords: int, apply_random_rotation: bool):
    """
    Calculates the global minimum and maximum ECT values across all (real and synthetic) samples
    to ensure consistent scaling for all generated ECT images.
    """
    print("\n--- Calculating Global ECT Min/Max for consistent visualization ---")
    
    global_min_val = float('inf')
    global_max_val = float('-inf')
    
    num_samples = len(all_flattened_coords)
    
    for i, flat_coords in enumerate(all_flattened_coords):
        if (i + 1) % 100 == 0 or i == num_samples - 1:
            print(f"  Calculating ECT for sample {i+1}/{num_samples}...")

        try:
            # Separate vein and blade coordinates
            vein_coords = flat_coords[:(num_vein_coords * 2)].reshape(num_vein_coords, 2)
            blade_coords = flat_coords[(num_vein_coords * 2):].reshape(-1, 2)
            
            if apply_random_rotation:
                random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)
                vein_coords_rotated = rotate_coords_2d(vein_coords, random_angle_deg)
                blade_coords_rotated = rotate_coords_2d(blade_coords, random_angle_deg)
            else:
                vein_coords_rotated = vein_coords.copy()
                blade_coords_rotated = blade_coords.copy()

            # Process vein and blade ECTs
            vein_graph = EmbeddedGraph()
            vein_graph.add_cycle(vein_coords_rotated)
            vein_graph.center_coordinates(center_type="origin")
            vein_graph.transform_coordinates()
            vein_graph.scale_coordinates(BOUND_RADIUS)
            vein_ect_result = ect_calculator.calculate(vein_graph)

            blade_graph = EmbeddedGraph()
            blade_graph.add_cycle(blade_coords_rotated)
            blade_graph.center_coordinates(center_type="origin")
            blade_graph.transform_coordinates()
            blade_graph.scale_coordinates(BOUND_RADIUS)
            blade_ect_result = ect_calculator.calculate(blade_graph)

            # Update global min/max for both
            global_min_val = min(global_min_val, np.min(vein_ect_result), np.min(blade_ect_result))
            global_max_val = max(global_max_val, np.max(vein_ect_result), np.max(blade_ect_result))

        except Exception as e:
            continue

    if global_min_val == float('inf') or global_max_val == float('-inf'):
        print("  Warning: No valid ECT values found. Setting to default [0, 1].")
        global_min_val = 0.0
        global_max_val = 1.0
    elif global_min_val == global_max_val:
        print(f"  Warning: All ECT values are identical ({global_min_val}). Adjusting max.")
        global_max_val = global_min_val + 1e-6

    print(f"  Global ECT Min: {global_min_val:.4f}, Global ECT Max: {global_max_val:.4f}")
    return global_min_val, global_max_val


def main_synthetic_generation(clear_existing_data: bool = True):
    """
    Main function to orchestrate synthetic leaf data generation and processing of real data.
    """
    np.random.seed(42)
    print("--- Starting Leaf Data Processing and Augmentation Pipeline ---")

    # --- 1. Setup Output Directories ---
    if clear_existing_data and SYNTHETIC_DATA_OUTPUT_DIR.exists():
        print(f"Clearing existing output directory: {SYNTHETIC_DATA_OUTPUT_DIR}")
        shutil.rmtree(SYNTHETIC_DATA_OUTPUT_DIR)
        
    SYNTHETIC_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_SHAPE_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_SHAPE_ECTS_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_COMBINED_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create separate subfolders for vein and blade ECTs and masks
    VEIN_MASKS_DIR = SYNTHETIC_SHAPE_MASKS_DIR / "vein"
    BLADE_MASKS_DIR = SYNTHETIC_SHAPE_MASKS_DIR / "blade"
    VEIN_ECTS_DIR = SYNTHETIC_SHAPE_ECTS_DIR / "vein"
    BLADE_ECTS_DIR = SYNTHETIC_SHAPE_ECTS_DIR / "blade"
    VEIN_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    BLADE_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    VEIN_ECTS_DIR.mkdir(parents=True, exist_ok=True)
    BLADE_ECTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories in {SYNTHETIC_DATA_OUTPUT_DIR}.")

    # --- 2. Load PCA Data (includes original real data) ---
    pca_data = load_pca_model_data(PCA_PARAMS_FILE, PCA_SCORES_LABELS_FILE)
    
    global NUM_VEIN_COORDS, NUM_BLADE_COORDS, TOTAL_CONTOUR_COORDS
    NUM_VEIN_COORDS = pca_data['num_vein_coords']
    NUM_BLADE_COORDS = pca_data['num_blade_coords']
    TOTAL_CONTOUR_COORDS = NUM_VEIN_COORDS + NUM_BLADE_COORDS
    
    if pca_data['original_flattened_coords'] is None:
        print("Cannot process real data as 'original_flattened_coords' was not found. Exiting.")
        return

    # --- 3. Generate Synthetic PCA Samples First ---
    synthetic_X_pca, synthetic_y_labels = generate_synthetic_pca_samples(
        pca_data, SAMPLES_PER_CLASS_TARGET, K_NEIGHBORS_SMOTE
    )
    
    synthetic_flattened_coords = inverse_transform_pca(
        synthetic_X_pca, pca_data['components'], pca_data['mean']
    )

    # --- 4. Consolidate ALL Flattened Coordinates for Global ECT Min/Max Calculation ---
    all_flattened_coords = np.vstack([pca_data['original_flattened_coords'], synthetic_flattened_coords])
    
    # --- 5. Initialize ECT Calculator ---
    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)
    print("Initialized ECT calculator.")

    # --- 6. Calculate Global ECT Min/Max (NEW STEP) ---
    global GLOBAL_ECT_MIN, GLOBAL_ECT_MAX
    GLOBAL_ECT_MIN, GLOBAL_ECT_MAX = calculate_global_ect_min_max(all_flattened_coords, ect_calculator, NUM_VEIN_COORDS, APPLY_RANDOM_ROTATION)
    
    metadata_records = []
    
    output_dirs_dict = {
        'vein_masks': VEIN_MASKS_DIR,
        'blade_masks': BLADE_MASKS_DIR,
        'vein_ects': VEIN_ECTS_DIR,
        'blade_ects': BLADE_ECTS_DIR,
        'combined_viz': SYNTHETIC_COMBINED_VIZ_DIR,
    }

    # --- 7. Process Original Real Samples (now with global ECT min/max) ---
    print("\n--- Processing Original Real Leaf Samples ---")
    num_real_samples = len(pca_data['original_flattened_coords'])
    for i in range(num_real_samples):
        sample_id = f"real_leaf_{i:05d}"
        class_label = pca_data['original_class_labels'][i]
        flat_coords = pca_data['original_flattened_coords'][i]
        
        vein_coords = flat_coords[:(NUM_VEIN_COORDS * 2)].reshape(NUM_VEIN_COORDS, 2)
        blade_coords = flat_coords[(NUM_VEIN_COORDS * 2):].reshape(-1, 2)

        if (i + 1) % 50 == 0 or i == num_real_samples - 1:
            print(f"Processing real leaf {i+1}/{num_real_samples} ({sample_id}, Class: {class_label})")

        process_leaf_for_cnn_output(
            sample_id,
            class_label,
            vein_coords,
            blade_coords,
            ect_calculator,
            output_dirs_dict,
            metadata_records,
            is_real_sample=True,
            apply_random_rotation=APPLY_RANDOM_ROTATION,
            global_ect_min=GLOBAL_ECT_MIN,
            global_ect_max=GLOBAL_ECT_MAX
        )

    # --- 8. Process Synthetic PCA Samples (now with global ECT min/max) ---
    total_synthetic_samples = len(synthetic_flattened_coords)
    print(f"\n--- Processing {total_synthetic_samples} Synthetic Leaf Samples ---")

    for i in range(total_synthetic_samples):
        sample_id = f"synthetic_leaf_{i:05d}"
        class_label = synthetic_y_labels[i]
        flat_coords = synthetic_flattened_coords[i]
        
        vein_coords = flat_coords[:(NUM_VEIN_COORDS * 2)].reshape(NUM_VEIN_COORDS, 2)
        blade_coords = flat_coords[(NUM_VEIN_COORDS * 2):].reshape(-1, 2)

        if (i + 1) % 50 == 0 or i == total_synthetic_samples - 1:
            print(f"Processing synthetic leaf {i+1}/{total_synthetic_samples} ({sample_id}, Class: {class_label})")
            
        process_leaf_for_cnn_output(
            sample_id,
            class_label,
            vein_coords,
            blade_coords,
            ect_calculator,
            output_dirs_dict,
            metadata_records,
            is_real_sample=False,
            apply_random_rotation=APPLY_RANDOM_ROTATION,
            global_ect_min=GLOBAL_ECT_MIN,
            global_ect_max=GLOBAL_ECT_MAX
        )

    # --- 9. Save Metadata ---
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(SYNTHETIC_METADATA_FILE, index=False)
    print(f"\nSaved combined real and synthetic leaf metadata to {SYNTHETIC_METADATA_FILE}")

    # --- 10. Prepare and Save Consolidated Data for CNN Training ---
    print("\n--- Consolidating data for 4-channel CNN training ---")

    valid_samples_df = metadata_df[metadata_df['is_processed_valid']].copy()
    if valid_samples_df.empty:
        print("No valid samples processed to create the final CNN dataset. Exiting.")
        return

    X_images = []
    y_labels_raw = []
    is_real_flags = []

    for idx, row in valid_samples_df.iterrows():
        vein_mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_vein_mask']
        vein_ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_vein_ect']
        blade_mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_blade_mask']
        blade_ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_blade_ect']

        try:
            vein_mask = np.array(Image.open(vein_mask_path).convert('L'), dtype=np.float32) / 255.0
            vein_ect = np.array(Image.open(vein_ect_path).convert('L'), dtype=np.float32) / 255.0
            blade_mask = np.array(Image.open(blade_mask_path).convert('L'), dtype=np.float32) / 255.0
            blade_ect = np.array(Image.open(blade_ect_path).convert('L'), dtype=np.float32) / 255.0

            combined_image = np.stack([vein_mask, vein_ect, blade_mask, blade_ect], axis=-1)

            X_images.append(combined_image)
            y_labels_raw.append(row['class_label'])
            is_real_flags.append(row['is_real'])

        except FileNotFoundError:
            print(f"Warning: Missing image file for {row['synthetic_id']}. Skipping.")
        except Exception as e:
            print(f"Error loading or processing images for {row['synthetic_id']}: {e}. Skipping.")

    if not X_images:
        print("No images were successfully loaded and prepared for CNN training. The final dataset will be empty.")
        return

    X_images_np = np.array(X_images)
    y_labels_np = np.array(y_labels_raw)
    is_real_flags_np = np.array(is_real_flags)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels_np)
    class_names = label_encoder.classes_

    X_images_shuffled, y_encoded_shuffled, is_real_flags_shuffled = shuffle(
        X_images_np, y_encoded, is_real_flags_np, random_state=42
    )

    final_data = {
        'X_images': X_images_shuffled,
        'y_labels_encoded': y_encoded_shuffled,
        'class_names': class_names,
        'is_real_flags': is_real_flags_shuffled,
        'image_size': IMAGE_SIZE,
        'num_channels': X_images_shuffled.shape[-1]
    }

    with open(FINAL_PREPARED_DATA_FILE, 'wb') as f:
        pickle.dump(final_data, f)
    print(f"Successfully prepared and saved CNN training data to {FINAL_PREPARED_DATA_FILE}")
    print(f"Dataset shape: X_images={X_images_shuffled.shape}, y_labels={y_encoded_shuffled.shape}")
    print(f"Class names: {class_names}")

    print("\n--- Leaf Data Processing and Augmentation Pipeline Completed ---")

if __name__ == "__main__":
    main_synthetic_generation(clear_existing_data=True)
# Synthetic Ampelography: Deep Learning for Vitis Leaf Classification and Vasculature Generation

## Figures and Tables

![Alt](https://github.com/DanChitwood/synthetic_ampelography/blob/main/outputs/figures/fig_morphospace.png)  
**Figure 1: Morphospace.** A Principal Component Analysis (PCA) morphospace of grapevine leaves. Theoretical eigenleaves are shown as a grid representing shape variation for their respective regions. Major classes of grapevine leaves indicated by color (see legend).  

![Alt](https://github.com/DanChitwood/synthetic_ampelography/blob/main/outputs/figures/fig_ECT.png)  
**Figure 2: Radial Euler Characteristic Transform (ECT) for real and synthetic leaves.** For the 12 indicated classes (rows), examples of real (left six columns) and synthetic (right six columns) leaves. For each class and for real and synthetic leaves, three random leaves are selected shown as pairs with the blade (left) and veins (right). For each representation, the outline of the blade or vein is shown as a white outline projected onto its corresponding radial ECT in which Euler characteristic values are shown in inferno color scale.

![Alt](https://github.com/DanChitwood/synthetic_ampelography/blob/main/outputs/figures/fig_gradCAM.png)  
**Figure 3: Mean leaves and Gradient-weighted Class Activation Mapping (Grad-CAM) by class.** For each of the 12 indicated classes, overall Generalized Procrustes Analysis (GPA) mean leaf (top) and examples of Grad-CAM for a Convolutional Neural Network (CNN) predicting class (bottom).

**Table 1: Peformance statistics for classification of the indicated classes by CNN.**
| Class              | Precision   | Recall   |   F1-Score / Accuracy |
|:-------------------|:------------|:---------|----------------------:|
| Ahmeur Bou Ahmeur  | 0.7         | 1.0      |                  0.82 |
| Amer Bouamar       | 1.0         | 1.0      |                  1    |
| Babari             | 0.57        | 0.89     |                  0.7  |
| Bouabane des Aures | 0.67        | 1.0      |                  0.8  |
| Ichmoul            | 0.83        | 1.0      |                  0.91 |
| Ichmoul Bacha      | 0.43        | 0.33     |                  0.38 |
| Louali             | 0.75        | 0.82     |                  0.78 |
| Tizi Ouinine       | 0.4         | 0.5      |                  0.44 |
| dissected          | 0.85        | 1.0      |                  0.92 |
| rootstock          | 0.62        | 0.84     |                  0.71 |
| vinifera           | 0.92        | 0.63     |                  0.75 |
| wild               | 0.98        | 0.77     |                  0.86 |
| accuracy           |             |          |                  0.78 |
| macro avg          | 0.73        | 0.82     |                  0.76 |
| weighted avg       | 0.82        | 0.78     |                  0.78 |

![Alt](https://github.com/DanChitwood/synthetic_ampelography/blob/main/outputs/figures/fig_confusion_matrix.png)  
**Figure 4: Classification by Convolutional Neural Network (CNN) confusion matrix.** True labels (rows) vs. predicted labels (columns) for the 12 indicated classes. Values are normalized proportions, indicated by number and color (see legend).  

## Methods 

### Convolutional Neural Network (CNN) classification using radial Euler Characteristic Transform (ECT) and aligned shape mask  
We begin by representing leaf shape as two distinct contours: the leaf blade and its primary venation network. These contours are converted into a set of pseudo-landmarks as previously described (Chitwood et al., 2025; https://doi.org/10.1002/ppp3.10561). These contours are normalized for size and position by Procrustean superimposition, then flattened into a single vector of coordinates. Principal Component Analysis (PCA) is applied to these vectors to create a low-dimensional morphospace that captures the primary shape variance of the real leaves. To address class imbalance and augment the dataset, a Synthetic Minority Over-sampling Technique (SMOTE) (Chawla et al., 2002; https://doi.org/10.1613/jair.953), in which a random distance between a sample and a neighbor of the same class in the PCA space is selected and used to generate an eigenleaf representation using the inverse transform, generates new, synthetic leaf coordinate vectors.

For every real and synthetic leaf sample, four distinct image channels are generated. First, a binary shape mask is created for both the blade and venation contours. Second, the Euler Characteristic Transform (ECT) (Munch, 2024; https://doi.org/10.1080/00029890.2024.2409616) is calculated for both contours. This topological descriptor is computed by sweeping a line across each shape from a number of directions of different angles and at threshold positions along each angle. Treating a closed contour as an embedded graph, at each direction and threshold value the Euler characteristic (the number of vertices minus edges plus faces) is calculated. The resulting Cartesian matrix (in which pixel values are the Euler characteristic value and the axes directions and thresholds) is converted into a polar coordinate representation that uses half the thresholds such that it can be aligned with the original shape mask from which it is derived.

The resulting four image channels (blade mask, blade ECT, venation mask, venation ECT) are stacked to form a single 4-channel image for each leaf. This dataset is used to train a convolutional neural network (CNN) with Pytorch (Paszke et al., 2019; https://doi.org/10.48550/arXiv.1912.01703) in a 5-fold stratified cross-validation scheme. The training set for each fold consists of a mix of real and synthetic samples, while validation is performed exclusively on held-out real samples. The final classification of all real samples is achieved through an ensemble method that averages the raw output logits from the best-performing model of each fold. Model interpretability is provided by generating Grad-CAM heatmaps, which are overlaid on the ECT images to visualize the key morphological features used by the model for classification.

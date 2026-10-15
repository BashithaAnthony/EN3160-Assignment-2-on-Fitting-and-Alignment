# EN3160 Assignment 2: Fitting and Alignment

This repository contains the implementation for EN3160 Assignment 2 on Fitting and Alignment, focusing on applying computer vision techniques for model fitting, feature matching, and image alignment across four distinct tasks.

## Key Theories and Technologies

### 1. Scale-Space Blob Detection
* **Laplacian of Gaussian (LoG):** Utilized to detect and draw circles across varying scales, defined by sigma values.
* **Scale-Space Extrema:** Blobs are identified by employing scale-space extrema detection to find the optimal bounds of the objects.

### 2. Robust Model Fitting (RANSAC)
* **Random Sample Consensus (RANSAC):** Employed to estimate mathematical models from datasets containing a high degree of noisy points and outliers.
* **Line Estimation:** RANSAC is implemented from scratch to find the best-fit line by applying the constraint ||[a,b]^T||=1, where [a, b] is the unit normal to the line.
* **Circle Estimation:** After filtering out line inliers, RANSAC is combined with an optimizer like `scipy.optimize.minimize` to fit a circle parameterized by its center and radius to the remaining remnant data.

### 3. Planar Homography and Warping
* **Homography Matrix Calculation:** Estimates the mathematical mapping between two planes based on selecting four points on a planar surface in a target image.
* **Perspective Warping and Blending:** Applies the calculated homography to warp a source image and blend it onto a target architectural image.

### 4. SIFT Feature Detection and Image Alignment
* **Scale-Invariant Feature Transform (SIFT):** Extracts features from template and target images using `cv2.SIFT_create()`.
* **Feature Matching:** Matches features utilizing `cv2.BFMatcher()` and finds the homography using RANSAC
* **Image Differencing:** Computes the difference between aligned PCB images and the reference template to highlight physical discrepancies.

## Assignment Roadmap

* **Task 1: Sunflower Blob Detection:** Apply LoG and scale-space extrema detection to identify circles in `the_berry_farms_sunflower_field.jpeg`.
* **Task 2: Multi-Model RANSAC Fitting:** Generate a noisy point set, estimate the best-fit line model, subtract the consensus, and estimate the optimal circle model fitting the remnant.
* **Task 3: Image Superimposition via Homography:** Compute homographies to superimpose a custom image onto an architectural planar surface, recreating a perspective blend.
* **Task 4: Automated PCB Inspection:** Detect SIFT features to align target PCB images with `AECC1.jpg` and generate difference images to evaluate high-difference board elements.

## Technologies Used
* **Python**
* **OpenCV (`cv2`)**
* **NumPy (`numpy`)**
* **SciPy (`scipy.optimize.minimize`, `scipy.linalg`)**
* **Matplotlib (`matplotlib.pyplot`)**
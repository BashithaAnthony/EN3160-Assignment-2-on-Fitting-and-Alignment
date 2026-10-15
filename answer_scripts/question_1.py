import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

# 1. Load the image as specified
im = cv.imread('the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)

# Convert to grayscale and normalize to [0, 1] for stable thresholding
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype(np.float64) / 255.0

# 2. Define the range of sigma values (Scale Space)
# We choose a range to capture small to large sunflowers
sigma_min = 1.0
sigma_max = 10.0
sigma_step = 0.5
sigmas = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)

# --- REPORT: Range of sigma values ---
print("="*50)
print(f"REPORT: Range of sigma values used: {sigmas}")
print("="*50)

# 3. Build the 3D Scale-Space
scale_space = []

for sigma in sigmas:
    
    hw = int(np.ceil(4 * sigma))
    X, Y = np.meshgrid(np.arange(-hw, hw + 1, 1), np.arange(-hw, hw + 1, 1))
    
    # Scale-normalized LoG formula
    log_kernel = -1 / (2 * np.pi * sigma**2) * ((X**2 + Y**2) / (sigma**2) - 2) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # force the kernel to sum to zero
    log_kernel -= log_kernel.mean()
    
    # Apply filter to the grayscale image
    response = cv.filter2D(gray, cv.CV_64F, log_kernel)
    
    # Square the response to detect both bright and dark blobs equally
    scale_space.append(response**2)

# Stack into a 3D numpy array: shape = (Height, Width, Num_Scales)
scale_space = np.stack(scale_space, axis=-1)

# 4. Scale-Space Extrema Detection (3D Non-Maximum Suppression)
# A voxel is an extremum if it is the maximum in its 3x3x3 neighborhood (x, y, sigma)
local_maxima = (scale_space == ndimage.maximum_filter(scale_space, size=3))

# Apply a threshold to reject weak responses (tune this fraction as needed for the specific image)
threshold = 0.5 * scale_space.max()
peaks_mask = local_maxima & (scale_space > threshold)

# Get the coordinates of valid extrema
y_coords, x_coords, scale_indices = np.where(peaks_mask)

# overlap pruning - keep the strongest blob, drop any weaker blob whose
# centre lies inside an already-kept (stronger) circle or vice versa
blobs = sorted(
    [(x, y, s_idx, scale_space[y, x, s_idx])
     for x, y, s_idx in zip(x_coords, y_coords, scale_indices)],
    key=lambda b: -b[3]  # strongest response first
)

kept = []  # each entry: (x, y, s_idx, response)
for x, y, s_idx, resp in blobs:
    r = sigmas[s_idx] * np.sqrt(2)
    overlaps = False
    for kx, ky, ks_idx, _ in kept:
        kr = sigmas[ks_idx] * np.sqrt(2)
        if np.hypot(x - kx, y - ky) <= max(r, kr):
            overlaps = True
            break
    if not overlaps:
        kept.append((x, y, s_idx, resp))

# 5. Draw circles and track the largest blobs
output_im = im.copy()
largest_radius = 0
largest_circles = [] # To store (x, y, radius, sigma)

for x, y, s_idx, _ in kept:
    sigma = sigmas[s_idx]
    
    # Calculate radius: r = sigma * sqrt(2)
    radius = int(np.round(sigma * np.sqrt(2)))
    
    # Draw yellow circle on the image
    cv.circle(output_im, (x, y), radius, (0, 255, 255), thickness=1)
    
    # Track the largest circle(s) for the report
    if radius > largest_radius:
        largest_radius = radius
        largest_circles = [(x, y, radius, sigma)]
    elif radius == largest_radius:
        largest_circles.append((x, y, radius, sigma))

# Calculate the blob count for each scale index (after pruning)
kept_scale_indices = np.array([b[2] for b in kept], dtype=int)
blob_counts = np.bincount(kept_scale_indices, minlength=len(sigmas))

# --- REPORT: Blob counts per scale and largest circles ---
print("\n" + "="*50)
print("REPORT: Blob Count per Sigma (After Thresholding and Pruning):")
for i, sigma in enumerate(sigmas):
    print(f"  Sigma: {sigma:4.1f} -> Count: {blob_counts[i]}")

print("\nREPORT: Parameters of the largest detected circles:")
print(f"Maximum Radius: {largest_radius} pixels")
for i, circle in enumerate(largest_circles):
    print(f"  Circle {i+1} -> Center (x,y): ({circle[0]}, {circle[1]}), Radius: {circle[2]}, Sigma: {circle[3]}")
print("="*50)

# 6. Display the final output image
plt.figure(figsize=(12, 10))
plt.imshow(cv.cvtColor(output_im, cv.COLOR_BGR2RGB))
plt.title(f'Scale-Space Extrema Detection\n({len(kept)} Blobs Detected, $\sigma \in [{sigma_min}, {sigma_max}]$)')
plt.axis('off')
plt.show()

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Data Generation (From Listing 1) ---
np.random.seed(42) # Seeded for reproducibility
N = 100
half_n = N // 2
r_gt = 10
x0_gt, y0_gt = 2, 3 # Center

# Generate Circle Points
t = np.random.uniform(0, 2 * np.pi, half_n)
n = (r_gt / 16) * np.random.randn(half_n)
x_circ = x0_gt + (r_gt + n) * np.cos(t)
y_circ = y0_gt + (r_gt + n) * np.sin(t)
X_circ = np.hstack((x_circ.reshape(half_n, 1), y_circ.reshape(half_n, 1)))

# Generate Line Points
s_line = 1.0
m_gt, b_gt = -1, 2
x_line = np.linspace(-12, 12, half_n)
y_line = m_gt * x_line + b_gt + s_line * np.random.randn(half_n)
X_line = np.hstack((x_line.reshape(half_n, 1), y_line.reshape(half_n, 1)))

# All points combined
X = np.vstack((X_circ, X_line))

# --- (a) RANSAC Line Fitting ---
def fit_line_ransac(points, iterations=1000, threshold=1.5):
    best_inliers = []
    best_sample = None
    
    for _ in range(iterations):
        # 1. Randomly select minimal subset of 2 points
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]
        
        # 2. Hypothesize a model: [a, b] is unit normal
        v = p2 - p1
        normal = np.array([-v[1], v[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len == 0: continue
        a, b = normal / norm_len
        d = a * p1[0] + b * p1[1]
        
        # 3. Compute error (perpendicular distance)
        errors = np.abs(a * points[:, 0] + b * points[:, 1] - d)
        
        # 4. Select inliers
        inliers_idx = np.where(errors < threshold)[0]
        
        # 5. Keep best model
        if len(inliers_idx) > len(best_inliers):
            best_inliers = inliers_idx
            best_sample = points[idx]
            
    # Refit using Total Least Squares (TLS) on all inliers
    inlier_points = points[best_inliers]
    mean_x, mean_y = np.mean(inlier_points, axis=0)
    U = inlier_points - np.array([mean_x, mean_y])
    U_T_U = U.T @ U
    eigenvalues, eigenvectors = np.linalg.eigh(U_T_U)
    
    # Smallest eigenvalue eigenvector represents normal [a,b]
    a_opt, b_opt = eigenvectors[:, 0] 
    d_opt = a_opt * mean_x + b_opt * mean_y
    
    return (a_opt, b_opt, d_opt), best_inliers, best_sample

line_params, line_inliers_idx, line_best_sample = fit_line_ransac(X)
a_opt, b_opt, d_opt = line_params
line_inliers = X[line_inliers_idx]

# --- (b) Remnant Extraction and Circle Fitting ---
# Subtract the consensus of the best line
remnant_mask = np.ones(len(X), dtype=bool)
remnant_mask[line_inliers_idx] = False
X_remnant = X[remnant_mask]

def fit_circle_ransac(points, iterations=1000, threshold=1.5):
    best_inliers = []
    best_sample = None
    best_params = None
    
    for _ in range(iterations):
        # 1. Randomly select minimal subset of 3 points
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]
        
        # 2. Hypothesize a model (Find circumcenter of 3 points)
        temp = p2[0]**2 + p2[1]**2
        bc = (p1[0]**2 + p1[1]**2 - temp) / 2
        cd = (temp - p3[0]**2 - p3[1]**2) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
        
        if abs(det) < 1e-6: continue # Points are collinear
        
        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        r = np.sqrt((p1[0] - cx)**2 + (p1[1] - cy)**2)
        
        # 3. Compute radial error
        errors = np.abs(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r)
        
        # 4. Select inliers
        inliers_idx = np.where(errors < threshold)[0]
        
        # 5. Keep best model
        if len(inliers_idx) > len(best_inliers):
            best_inliers = inliers_idx
            best_sample = points[idx]
            best_params = (cx, cy, r)
            
    # Refit using scipy.minimize on all inliers
    inlier_points = points[best_inliers]
    
    def circle_loss(params, pts):
        xc, yc, rad = params
        return np.sum((np.sqrt((pts[:, 0] - xc)**2 + (pts[:, 1] - yc)**2) - rad)**2)
    
    # Use best RANSAC hypothesis as initial guess
    res = minimize(circle_loss, best_params, args=(inlier_points,))
    opt_cx, opt_cy, opt_r = res.x
    
    return (opt_cx, opt_cy, opt_r), best_inliers, best_sample

circle_params, circle_inliers_idx, circle_best_sample = fit_circle_ransac(X_remnant)
c_cx, c_cy, c_r = circle_params
circle_inliers = X_remnant[circle_inliers_idx]


# --- (c) Plotting ---
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Background Points
ax.scatter(X[:, 0], X[:, 1], color='gray', label='All Points', alpha=0.5, s=20)

# Line Plotting
ax.scatter(line_inliers[:, 0], line_inliers[:, 1], color='y', label='Line Inliers', s=20)
ax.scatter(line_best_sample[:, 0], line_best_sample[:, 1], color='r', marker='x', s=60, label='Best sample for line')

x_vals = np.array(ax.get_xlim())
# a*x + b*y = d  =>  y = (d - a*x) / b
if abs(b_opt) > 1e-6:
    y_vals = (d_opt - a_opt * x_vals) / b_opt
    ax.plot(x_vals, y_vals, color='purple', label='RANSAC Best Fit Line')

# Circle Plotting
ax.scatter(circle_inliers[:, 0], circle_inliers[:, 1], color='cyan', label='Circle Inliers', s=20)
ax.scatter(circle_best_sample[:, 0], circle_best_sample[:, 1], color='blue', marker='*', s=60, label='Best sample for circle')

ransac_circle = plt.Circle((c_cx, c_cy), c_r, color='pink', fill=False, linewidth=2, label='RANSAC Best Fit Circle')
ax.add_patch(ransac_circle)
ax.plot(c_cx, c_cy, '+', color='blue')

ax.set_aspect('equal')
plt.legend()
plt.title('RANSAC Line and Circle Fitting')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

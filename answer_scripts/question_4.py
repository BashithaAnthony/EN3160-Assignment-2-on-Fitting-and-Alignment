import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_and_compare_pcb(template_path, target_path):
    # Load images in grayscale (ideal for feature detection and differencing)
    img_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    img_target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    
    # (a) Detect SIFT features and compute descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_template, None)
    kp2, des2 = sift.detectAndCompute(img_target, None)
    
    # (b) Match features using BFMatcher
    bf = cv2.BFMatcher()
    
    # We use knnMatch to retrieve the top 2 matches for each descriptor to apply the ratio test
    raw_matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply the nearest neighbor distance ratio heuristic (Lowe's ratio test)
    # A threshold of 0.8 effectively separates correct matches from incorrect ones
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            
    # Draw the filtered matches across the two images
    img_matches = cv2.drawMatches(
        img_template, kp1, img_target, kp2, good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Extract the (x, y) coordinates of the good matches
    pts_template = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_target = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find the homography using RANSAC to handle the high percentage of outliers
    # We map pts_target TO pts_template so we can warp the target image into the template's view
    H, mask = cv2.findHomography(pts_target, pts_template, cv2.RANSAC, 5.0)
    
    # (c) Warp the target PCB image to align with the template
    h, w = img_template.shape
    aligned_target = cv2.warpPerspective(img_target, H, (w, h))
    
    # Compute the absolute difference between the aligned image and the reference template
    diff_img = cv2.absdiff(img_template, aligned_target)
    
    # Display the results
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.imshow(img_matches, cmap='gray')
    plt.title('SIFT Matches')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(aligned_target, cmap='gray')
    plt.title('Aligned PCB')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(diff_img, cmap='hot') # Using 'hot' colormap to highlight differences
    plt.title('Difference Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# (d) Carry this out for all given PCB images
# You can loop through your dataset like this:
pcb_images = ['AAAA1.jpg', 'ADAD1.jpg', 'AECC2.jpg', 'AECC3.jpg', 'AECC4.jpg', 'pcb_target_2.jpg']
for pcb in pcb_images:
     align_and_compare_pcb('AECC1.jpg', pcb)

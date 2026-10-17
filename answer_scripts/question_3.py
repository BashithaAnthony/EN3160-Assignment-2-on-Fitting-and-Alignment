import cv2
import numpy as np
import matplotlib.pyplot as plt

def superimpose_image(target_path, source_path, output_path="output.jpg"):
    # Load images and convert from BGR to RGB for accurate matplotlib display
    target_img = cv2.imread(target_path)
    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    source_img = cv2.imread(source_path)
    source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

    # Define the 4 corners of the source image as the origin points
    h_src, w_src = source_img.shape[:2]
    src_pts = np.array([
        [0, 0],                 # Top-left
        [w_src - 1, 0],         # Top-right
        [w_src - 1, h_src - 1], # Bottom-right
        [0, h_src - 1]          # Bottom-left
    ], dtype=float)

    # Interactively select 4 points on the target planar surface
    print("Click 4 points on the target surface in this exact order:")
    print("1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(target_img_rgb)
    plt.title("Click 4 corners to define the projection plane")
    dst_pts = plt.ginput(4)
    plt.close()

    dst_pts = np.array(dst_pts, dtype=float)

    if len(dst_pts) != 4:
        print("Error: You must select exactly 4 points to compute the homography.")
        return

    # Compute the Homography matrix mapping the 4 source corners to the 4 clicked points
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Warp the source image using the computed homography
    h_tgt, w_tgt = target_img.shape[:2]
    warped_source = cv2.warpPerspective(source_img_rgb, H, (w_tgt, h_tgt))

    # Create a binary mask to blend the warped image seamlessly into the target
    mask = np.zeros((h_tgt, w_tgt, 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(int), (255, 255, 255))

    # Mask out the planar area in the target background, then combine
    mask_inv = cv2.bitwise_not(mask)
    target_bg = cv2.bitwise_and(target_img_rgb, mask_inv)
    result_rgb = cv2.bitwise_or(target_bg, warped_source)

    # Save and display the result
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)

    plt.figure(figsize=(12, 8))
    plt.imshow(result_rgb)
    plt.title("Perspective Alignment Result")
    plt.axis('off')
    plt.show()

# Execution Setup
source_image = 'C:\\Users\\user\\OneDrive\\Desktop\\5th Sem\\Image proc. & Machine Vision\\Assignment_2\\flag.png'
target_image = 'C:\\Users\\user\\OneDrive\\Desktop\\5th Sem\\Image proc. & Machine Vision\\Assignment_2\\wadham.jpg'
superimpose_image(target_image, source_image, 'aligned_result_1.jpg')

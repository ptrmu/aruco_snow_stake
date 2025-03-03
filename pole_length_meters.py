import cv2
import numpy as np
import math


def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    """
    Draws the 3D axis on the marker to visualize orientation.
    - Red: X-axis
    - Green: Y-axis
    - Blue: Z-axis
    """
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, -length], [0, 0, 0]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    # Draw connecting lines for the 3D axis
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[0]), (0, 0, 255), 3)  # X-axis (Red)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[1]), (0, 255, 0), 3)  # Y-axis (Green)
    img = cv2.line(img, tuple(imgpts[3]), tuple(imgpts[2]), (255, 0, 0), 3)  # Z-axis (Blue)
    return img


# Load the image
# image = cv2.imread("images/ASS_68_20250222_195315984.jpg")
# image = cv2.imread("images/ASS_68_20250222_195447026.jpg")
image = cv2.imread("images/ASS_68_20250222_195325202.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Marker size (meters)
marker_size_m = 0.209  # 20.9 cm

# Camera parameters (camera intrinsics)
focal_length_pixels = 100
image_center = (image.shape[1] // 2, image.shape[0] // 2)
camera_matrix = np.array([[focal_length_pixels, 0, image_center[0]],
                          [0, focal_length_pixels, image_center[1]],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5,))  # Assuming no lens distortion

# Initialize the ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Detect ArUco markers in the image
corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is not None:
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)

    # ArUco marker detected
    for i, corner in enumerate(corners):
        # Define the 3D model points of the marker (world coordinates)
        marker_corners_3d = np.array([
            [-marker_size_m / 2, marker_size_m / 2, 0],  # Top-left corner
            [marker_size_m / 2, marker_size_m / 2, 0],  # Top-right corner
            [marker_size_m / 2, -marker_size_m / 2, 0],  # Bottom-right corner
            [-marker_size_m / 2, -marker_size_m / 2, 0],  # Bottom-left corner
        ], dtype=np.float32)

    # Extract the 2D detected corners (image coordinates)
    marker_corners_2d = corner[0].astype(np.float32)

    # SolvePnP: Estimate rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(marker_corners_3d, marker_corners_2d, camera_matrix, dist_coeffs)

    # Draw the marker's pose with the axis on the image
    image = draw_axis(image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1)

    # For completeness: Detect the visible pole below the marker
    # Get the bottom-center of the marker for region of interest
    bottom_center_x = int((marker_corners_2d[2][0] + marker_corners_2d[3][0]) / 2)
    bottom_center_y = int((marker_corners_2d[2][1] + marker_corners_2d[3][1]) / 2)

    # Extract ROI for the pole below the marker
    roi_width = 400
    roi = gray[bottom_center_y:, bottom_center_x - roi_width // 2:bottom_center_x + roi_width // 2]

    # Draw the ROI rectangle
    cv2.rectangle(image,
                  (bottom_center_x - roi_width // 2, bottom_center_y),
                  (bottom_center_x + roi_width // 2, image.shape[0] - 1),
                  (0, 255, 0), 2)  # Green color, thickness 2

    # Thresholding and contours to identify the visible portion of the pole
    _, binary = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a red dot at the top of the pole
    point_3d = np.array([[0, -marker_size_m / 2 - 0.11, -.005]], dtype=np.float32)
    projected_point, _ = cv2.projectPoints(point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    point_2d = (int(projected_point[0][0][0]), int(projected_point[0][0][1]))
    print(f"Top of Pole (2D): {point_2d}")
    image = cv2.circle(image, point_2d, radius=5, color=(0, 0, 255), thickness=-1)

    # Draw a green dot where bottom of pole is
    point_3d = np.array([[0, -marker_size_m / 2 - 0.11 - .68, -.005]], dtype=np.float32)
    projected_point, _ = cv2.projectPoints(point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    point_2d = (int(projected_point[0][0][0]), int(projected_point[0][0][1]))
    print(f"Bottom of Pole (2D): {point_2d}")
    image = cv2.circle(image, point_2d, radius=5, color=(0, 255, 0), thickness=-1)

    if contours:
        # Draw all contours in blue
        offset_contours = []
        for contour in contours:
            offset_contour = contour.copy()
            offset_contour[:, :, 0] += bottom_center_x - roi_width // 2  # Add x offset
            offset_contour[:, :, 1] += bottom_center_y  # Add y offset
            offset_contours.append(offset_contour)
        cv2.drawContours(image, offset_contours, -1, (255, 0, 0), 2)

        # Find the largest contour for the pole
        largest_contour = max(contours, key=cv2.contourArea)
        contour_left, contour_top, contour_width, contour_height = cv2.boundingRect(largest_contour)
        blue_dot_2d = (bottom_center_x - roi_width // 2 + contour_left + contour_width // 2,
                       bottom_center_y + contour_top + contour_height)
        print(f"Bottom of pole (image coors) {blue_dot_2d}")
        image = cv2.circle(image, blue_dot_2d, radius=5, color=(255, 0, 0), thickness=-1)

        fx, fy, cx, cy = focal_length_pixels, focal_length_pixels,  image_center[1],  image_center[0]  # Example intrinsic parameters
        u, v = blue_dot_2d[0], blue_dot_2d[1]  # 2D image point (pixel coordinates)

        # Convert rvec to Rotation Matrix
        R, _ = cv2.Rodrigues(rvec)

        # Camera intrinsic parameters
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        # Normalize image point
        X_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

        # Compute scale factor 's' (assuming plane Z = 0)
        R_row_3 = R[2, :]  # Third row of R
        s = -np.dot(R_row_3, tvec[:]) / np.dot(R_row_3, X_cam)

        # Compute 3D world point
        X_world = np.dot(R.T, (s * X_cam)) + tvec[0]

        print("3D World Coordinates:", X_world)


else:
    print("No ArUco marker detected.")

# Display results
cv2.namedWindow("Marker Pose and Pole", cv2.WINDOW_NORMAL)
cv2.imshow("Marker Pose and Pole", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

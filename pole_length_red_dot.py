import cv2
import numpy as np


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
image = cv2.imread("images/ASS_68_20250222_195315984.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Marker size (meters)
marker_size_m = 0.209  # 20.9 cm

# Camera parameters (camera intrinsics)
focal_length_pixels = 475
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

        if success:
            # Draw the marker's pose with the axis on the image
            image = draw_axis(image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1)

            # Define the 3D point that is 11 cm below the bottom of the marker
            point_3d_pole_end = np.array([[0, -marker_size_m / 2 - 0.11, 0]], dtype=np.float32)

            # Project the 3D point of the pole's end onto the 2D image plane
            point_2d_pole_end, _ = cv2.projectPoints(point_3d_pole_end, rvec, tvec, camera_matrix, dist_coeffs)

            # Draw a blue dot at the projected 2D coordinates of the pole's end
            point_2d_pole_end = tuple(point_2d_pole_end[0][0].astype(int))
            cv2.circle(image, point_2d_pole_end, radius=5, color=(0, 0, 255), thickness=-1)

            # Log information
            print(f"3D Point of Pole End (meters): {point_3d_pole_end}")
            print(f"Projected 2D Point of Pole End (pixels): {point_2d_pole_end}")

else:
    print("No ArUco marker detected.")

# Display results
cv2.namedWindow("Marker Pose and Pole End", cv2.WINDOW_NORMAL)
cv2.imshow("Marker Pose and Pole End", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

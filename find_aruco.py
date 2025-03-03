import cv2
import cv2.aruco as aruco


def detect_and_display_aruco(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Convert the image to grayscale (necessary for detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the predefined dictionary of ArUco markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    # Detect ArUco markers
    corners, ids, rejected_candidates = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Outline detected markers in the image
    if ids is not None:
        # Draw the outline around detected markers
        image_with_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
        print(f"[INFO] Detected {len(ids)} markers with IDs: {ids.flatten()}")
    else:
        # No markers detected
        image_with_markers = image.copy()
        print("[INFO] No markers detected.")

    # Outline rejected candidates in a different color
    if rejected_candidates:
        for points in rejected_candidates:
            cv2.polylines(image_with_markers, [points.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
            for corner in points[0]:
                cv2.circle(image_with_markers, tuple(corner.astype(int)), radius=4, color=(0, 255, 255), thickness=-1)
        print(f"[INFO] Number of rejected candidates: {len(rejected_candidates)}")

    # Create a window that allows zooming
    cv2.namedWindow("Detected Markers", cv2.WINDOW_NORMAL)

    # Display the image with outlined markers
    cv2.imshow("Detected Markers", image_with_markers)

    # Wait until user presses a key
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()


# Provide the path to your image file
image_path = "images/ASS_68_20250222_195315984.jpg"  # Replace with the path to your image
detect_and_display_aruco(image_path)

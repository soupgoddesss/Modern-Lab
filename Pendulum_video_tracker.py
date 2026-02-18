"""
Double Pendulum Video Tracker
Tracks reflector tape markers on a physical double pendulum from video
Outputs position data that can be compared to simulation
"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: VIDEO SETUP
# ============================================================================

# Load the video file
# cv2.VideoCapture() creates a video capture object
# Replace 'pendulum_video.mp4' with your actual video filename
video_path = '90_take_1.mov'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Get video properties
# These tell us the frame rate and total number of frames
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in video
print(f"Video loaded: {fps} FPS, {total_frames} frames")

# ============================================================================
# STEP 2: CALIBRATION - Find pivot point and set up coordinate system
# ============================================================================

# Read the first frame to set up calibration
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    exit()

# Display the first frame so user can click on the pivot point
# The pivot is where the first arm connects to the fixed support
print("\nClick on the PIVOT POINT (where first arm attaches to support)")
print("Then press any key to continue")

# Create a copy for display
calibration_frame = first_frame.copy()

# This will store the pivot point coordinates
pivot_point = []


# Mouse callback function - called when user clicks on the image
def click_pivot(event, x, y, flags, param):
    """
    Callback function that captures mouse clicks
    event: type of mouse event (we want LEFT_BUTTON_DOWN)
    x, y: pixel coordinates where user clicked
    """
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        pivot_point.clear()  # Clear any previous clicks
        pivot_point.append((x, y))  # Store the clicked point
        # Draw a circle at the clicked point for visual feedback
        cv2.circle(calibration_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Click Pivot Point', calibration_frame)


# Set up the window and mouse callback
cv2.namedWindow('Click Pivot Point')
cv2.setMouseCallback('Click Pivot Point', click_pivot)
cv2.imshow('Click Pivot Point', calibration_frame)
cv2.waitKey(0)  # Wait for user to press a key
cv2.destroyAllWindows()

# Extract pivot coordinates
if len(pivot_point) == 0:
    print("Error: No pivot point selected")
    exit()
pivot_x, pivot_y = pivot_point[0]
print(f"Pivot point set at: ({pivot_x}, {pivot_y})")

# ============================================================================
# STEP 3: PHYSICAL MEASUREMENTS FOR CALIBRATION
# ============================================================================

# You need to measure your actual pendulum with a ruler!
# Enter the real-world lengths in meters
L1_real = float(input("\nEnter length of arm 1 in meters (measure with ruler): "))
L2_real = float(input("Enter length of arm 2 in meters (measure with ruler): "))

# We'll calculate pixels-to-meters conversion after detecting markers in first frame

# ============================================================================
# STEP 4: THRESHOLD TUNING - Find the right values to isolate markers
# ============================================================================

print("\nAdjusting threshold to detect reflector tape...")
print("The reflector tape should appear WHITE, everything else BLACK")

# Convert first frame to grayscale
# Grayscale has values 0-255 where 0=black, 255=white
gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Apply threshold to isolate bright spots (reflector tape)
# cv2.threshold() separates pixels above/below a threshold value
# Syntax: threshold(source, threshold_value, max_value, threshold_type)
# THRESH_BINARY: pixels above threshold → max_value (255), below → 0
threshold_value = 200  # Starting value - you may need to adjust this!
_, thresh_first = cv2.threshold(gray_first, threshold_value, 255, cv2.THRESH_BINARY)

# Display to check if markers are visible
print("Check the thresholded image - markers should be bright white spots")
print("If not visible, you'll need to adjust threshold_value in the code")
cv2.imshow('Thresholded Image', thresh_first)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================================
# STEP 5: MARKER DETECTION FUNCTION
# ============================================================================

def detect_markers(frame, pivot, threshold_val=200):
    """
    Detects the two reflector tape markers in a frame

    How it works:
    1. Convert to grayscale
    2. Threshold to get bright spots
    3. Find contours (outlines of white regions)
    4. Calculate centroid (center point) of each contour
    5. Sort by distance from pivot (closer = joint, farther = end)

    Parameters:
    - frame: the video frame (BGR image from OpenCV)
    - pivot: (x, y) coordinates of the pivot point
    - threshold_val: brightness threshold for detecting tape

    Returns:
    - marker1: (x, y) position of joint marker
    - marker2: (x, y) position of end marker
    - None, None if detection fails
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    # Find contours
    # cv2.findContours() finds boundaries of white regions
    # RETR_EXTERNAL: only get outer contours
    # CHAIN_APPROX_SIMPLE: compress contour to save memory
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We need at least 2 contours (2 markers)
    if len(contours) < 2:
        return None, None

    # Calculate centroid of each contour
    centroids = []
    for contour in contours:
        # cv2.moments() calculates spatial moments
        # Moments help us find the center (centroid) of a shape
        M = cv2.moments(contour)

        # m00 is the area - avoid division by zero
        if M["m00"] != 0:
            # Centroid formula: cx = m10/m00, cy = m01/m00
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    # Need exactly 2 markers
    if len(centroids) < 2:
        return None, None

    # Sort centroids by distance from pivot
    # The marker closer to pivot is the joint, farther is the end
    # Lambda function: calculates Euclidean distance from pivot
    centroids_sorted = sorted(centroids,
                              key=lambda p: np.sqrt((p[0] - pivot[0]) ** 2 + (p[1] - pivot[1]) ** 2))

    # Take the 2 closest to pivot (in case there's noise/extra bright spots)
    marker1 = centroids_sorted[0]  # Joint (closer to pivot)
    marker2 = centroids_sorted[1]  # End (farther from pivot)

    return marker1, marker2


# ============================================================================
# STEP 6: CALIBRATE PIXELS TO METERS
# ============================================================================

# Detect markers in first frame to calculate pixel-to-meter conversion
marker1_cal, marker2_cal = detect_markers(first_frame, (pivot_x, pivot_y), threshold_value)

if marker1_cal is None:
    print("Error: Could not detect markers in first frame")
    print("Try adjusting threshold_value")
    exit()

# Calculate pixel distances
# L1 in pixels: distance from pivot to joint marker
L1_pixels = np.sqrt((marker1_cal[0] - pivot_x) ** 2 + (marker1_cal[1] - pivot_y) ** 2)

# L2 in pixels: distance from joint marker to end marker
L2_pixels = np.sqrt((marker2_cal[0] - marker1_cal[0]) ** 2 +
                    (marker2_cal[1] - marker1_cal[1]) ** 2)

# Calculate scale factors (meters per pixel)
scale_L1 = L1_real / L1_pixels
scale_L2 = L2_real / L2_pixels

# Average the two (they should be similar if camera is perpendicular)
scale = (scale_L1 + scale_L2) / 2

print(f"\nCalibration complete:")
print(f"L1: {L1_pixels:.1f} pixels = {L1_real} meters")
print(f"L2: {L2_pixels:.1f} pixels = {L2_real} meters")
print(f"Scale factor: {scale:.6f} meters/pixel")

# ============================================================================
# STEP 7: PROCESS ALL FRAMES
# ============================================================================

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Arrays to store tracked positions
times = []  # Time stamps
x1_pixels = []  # Joint marker x positions (pixels)
y1_pixels = []  # Joint marker y positions (pixels)
x2_pixels = []  # End marker x positions (pixels)
y2_pixels = []  # End marker y positions (pixels)

frame_count = 0

print(f"\nProcessing {total_frames} frames...")

while True:
    ret, frame = cap.read()

    if not ret:
        break  # End of video

    # Detect markers in this frame
    marker1, marker2 = detect_markers(frame, (pivot_x, pivot_y), threshold_value)

    if marker1 is not None and marker2 is not None:
        # Store positions
        x1_pixels.append(marker1[0])
        y1_pixels.append(marker1[1])
        x2_pixels.append(marker2[0])
        y2_pixels.append(marker2[1])

        # Calculate time (in seconds)
        times.append(frame_count / fps)

        # Optional: Draw markers on frame for visualization
        # Uncomment these lines if you want to see the tracking in real-time
        # display_frame = frame.copy()
        # cv2.circle(display_frame, (pivot_x, pivot_y), 5, (0, 255, 0), -1)  # Pivot = green
        # cv2.circle(display_frame, marker1, 5, (255, 0, 0), -1)  # Joint = blue
        # cv2.circle(display_frame, marker2, 5, (0, 0, 255), -1)  # End = red
        # cv2.imshow('Tracking', display_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        #     break

    frame_count += 1

    # Progress indicator
    if frame_count % 30 == 0:  # Print every 30 frames
        print(f"Processed {frame_count}/{total_frames} frames")

cap.release()
cv2.destroyAllWindows()

print(f"\nTracking complete! Detected markers in {len(times)} frames")

# ============================================================================
# STEP 8: CONVERT TO METERS AND CALCULATE ANGLES
# ============================================================================

# Convert lists to numpy arrays for easier math
times = np.array(times)
x1_pixels = np.array(x1_pixels)
y1_pixels = np.array(y1_pixels)
x2_pixels = np.array(x2_pixels)
y2_pixels = np.array(y2_pixels)

# Convert pixel coordinates to meters (relative to pivot)
# Subtract pivot position to make pivot the origin (0, 0)
x1_meters = (x1_pixels - pivot_x) * scale
y1_meters = (y1_pixels - pivot_y) * scale
x2_meters = (x2_pixels - pivot_x) * scale
y2_meters = (y2_pixels - pivot_y) * scale

# Calculate angles
# theta1: angle of first arm from vertical (downward)
# theta2: angle of second arm from vertical (downward)
# np.arctan2(y, x) gives angle, but we need to adjust:
# - Vertical down is our zero reference
# - arctan2 measures from horizontal right

theta1 = np.arctan2(x1_meters, y1_meters)  # Note: y first for vertical reference
theta2 = np.arctan2(x2_meters - x1_meters, y2_meters - y1_meters)

# Convert to degrees if preferred
theta1_deg = np.degrees(theta1)
theta2_deg = np.degrees(theta2)

# ============================================================================
# STEP 9: SAVE DATA
# ============================================================================

# Save to file for later analysis
data = np.column_stack([times, x1_meters, y1_meters, x2_meters, y2_meters,
                        theta1, theta2])

np.savetxt('pendulum_tracking_data.csv', data,
           header='time(s),x1(m),y1(m),x2(m),y2(m),theta1(rad),theta2(rad)',
           delimiter=',', comments='')

print("\nData saved to 'pendulum_tracking_data.csv'")

# ============================================================================
# STEP 10: VISUALIZE RESULTS
# ============================================================================

# Plot trajectories
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Position vs time
axes[0, 0].plot(times, x1_meters, label='Joint (x1)', color='blue')
axes[0, 0].plot(times, x2_meters, label='End (x2)', color='red')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('X Position (m)')
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 0].set_title('Horizontal Position vs Time')

axes[0, 1].plot(times, y1_meters, label='Joint (y1)', color='blue')
axes[0, 1].plot(times, y2_meters, label='End (y2)', color='red')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Y Position (m)')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].set_title('Vertical Position vs Time')

# Angles vs time
axes[1, 0].plot(times, theta1_deg, label='θ1 (first arm)', color='blue')
axes[1, 0].plot(times, theta2_deg, label='θ2 (second arm)', color='red')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Angle (degrees)')
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].set_title('Angles vs Time')

# Trajectory of end marker
axes[1, 1].plot(x2_meters, y2_meters, color='red', linewidth=0.5)
axes[1, 1].scatter(x2_meters[0], y2_meters[0], color='green', s=100, label='Start')
axes[1, 1].scatter(x2_meters[-1], y2_meters[-1], color='black', s=100, label='End')
axes[1, 1].set_xlabel('X Position (m)')
axes[1, 1].set_ylabel('Y Position (m)')
axes[1, 1].set_aspect('equal')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].set_title('End Marker Trajectory')
axes[1, 1].invert_yaxis()  # Flip Y axis so down is positive (like simulation)

plt.tight_layout()
plt.savefig('pendulum_tracking_results.png', dpi=150)
plt.show()

print("\nPlots saved to 'pendulum_tracking_results.png'")
print("\nYou can now compare this data with your simulation!")
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io

st.set_page_config(layout="wide", page_title="Corrugated Cardboard Analyzer")

st.title("Corrugated Cardboard Production Analyzer")
st.subheader("Sensorminds Pvt Ltd.")

st.markdown("Upload a video of your corrugated cardboard production line for quality and production analysis.")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Main sidebar for settings
with st.sidebar:
    st.header("Analysis Settings")
    analysis_types = st.multiselect(
        "Select Measurements to Perform",
        [
            "Edge Alignment",
            "Warping Detection",
            "Color Consistency",
            "Flute Damage",
            "Crease Quality",
            "Surface Defects",
            "Moisture Analysis",
            "Thickness Uniformity",
            "Sheet Count",
            "Dimensional Analysis",
            "Production Rate",
            "Stack Height",
            "Cut Accuracy",
            "Sheet Spacing",
            "Line Speed",
            "Material Waste"
        ],
        default=["Edge Alignment", "Surface Defects", "Sheet Count", "Dimensional Analysis"]
    )

    # Add brown color range selection sliders
    st.subheader("Brown Cardboard Color Range")
    hue_min = st.slider("Hue Min", 0, 180, 10)
    hue_max = st.slider("Hue Max", 0, 180, 30)
    sat_min = st.slider("Saturation Min", 0, 255, 60)
    sat_max = st.slider("Saturation Max", 0, 255, 255)
    val_min = st.slider("Value Min", 0, 255, 100)
    val_max = st.slider("Value Max", 0, 255, 255)

    sensitivity = st.slider("Detection Sensitivity", 0.0, 1.0, 0.5)
    frame_sample_rate = st.slider("Frame Sample Rate", 1, 30, 5, help="Process 1 frame every N frames")

color_params = {
    'hue_min': hue_min,
    'hue_max': hue_max,
    'sat_min': sat_min,
    'sat_max': sat_max,
    'val_min': val_min,
    'val_max': val_max
}

def create_brown_mask(frame, hue_min=10, hue_max=30, sat_min=60, sat_max=255, val_min=100, val_max=255):
    """Create a mask that isolates brown cardboard"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for brown color range
    lower_brown = np.array([hue_min, sat_min, val_min])
    upper_brown = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Apply morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


# ------------------------------------------------------------------------------
# QUALITY MEASUREMENT FUNCTIONS
# ------------------------------------------------------------------------------

def detect_edge_alignment(frame, color_params):
    """Detect if sheets have proper edge alignment"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to find sheets
    sheet_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

    # Draw rectangles around sheets
    result_frame = frame.copy()
    alignment_scores = []

    for cnt in sheet_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Calculate alignment score based on how rectangular the contour is
        rect_area = w * h
        contour_area = cv2.contourArea(cnt)
        if rect_area > 0:
            alignment_score = contour_area / rect_area
            alignment_scores.append(alignment_score)

            # Color code based on alignment (green for good, red for bad)
            color = (0, 255, 0) if alignment_score > 0.90 else (0, 0, 255)
            cv2.drawContours(result_frame, [box], 0, color, 2)

    avg_score = np.mean(alignment_scores) if alignment_scores else 0
    return result_frame, {"alignment_score": avg_score,
                          "aligned_sheets": sum([1 for s in alignment_scores if s > 0.90])}


def detect_warping(frame, color_params):
    """Detect warping/bending in sheets"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Apply mask to the original image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Edge detection on the masked image
    edges = cv2.Canny(gray, 50, 150)

    # Hough Line Transform to detect straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    result_frame = frame.copy()
    warping_detected = False
    warp_score = 0

    if lines is not None:
        # Group lines by their orientation (horizontal/vertical)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < abs(x2 - x1):  # Horizontal-ish line
                horizontal_lines.append(line[0])
                cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Check for parallelism among horizontal lines
        if len(horizontal_lines) >= 2:
            slopes = []
            for x1, y1, x2, y2 in horizontal_lines:
                if x2 != x1:  # Avoid division by zero
                    slope = (y2 - y1) / (x2 - x1)
                    slopes.append(slope)

            # Calculate variance of slopes, higher variance = more warping
            if slopes:
                warp_score = np.var(slopes) * 1000  # Scale up for visibility
                warping_detected = warp_score > 0.5

    return result_frame, {"warp_score": warp_score, "warping_detected": warping_detected}


def analyze_color_consistency(frame, color_params):
    """Analyze color consistency across sheets"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find contours of cardboard sheets in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    sheet_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

    result_frame = frame.copy()
    color_stats = []

    for cnt in sheet_contours:
        # Create mask for this contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [cnt], 0, 255, -1)

        # Get color stats for this sheet
        h, s, v = cv2.split(hsv)
        mean_h = cv2.mean(h, mask=contour_mask)[0]
        mean_s = cv2.mean(s, mask=contour_mask)[0]
        mean_v = cv2.mean(v, mask=contour_mask)[0]
        std_h = np.std(h[contour_mask > 0]) if np.sum(contour_mask > 0) > 0 else 0

        color_stats.append((mean_h, mean_s, mean_v, std_h))

        # Draw contour with color based on value
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_frame, f"V:{mean_v:.1f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calculate overall color consistency
    consistency_score = 0
    if color_stats:
        # Calculate variance of values across sheets
        values = [stats[2] for stats in color_stats]
        consistency_score = 1.0 - min(1.0, np.std(values) / 20.0)  # Normalize

    return result_frame, {"color_consistency": consistency_score, "sheet_colors": len(color_stats)}


def detect_flute_damage(frame, color_params):
    """Detect damage to flutes (corrugation)"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Apply mask to the original image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Apply Gabor filter to enhance texture patterns (flutes)
    ksize = 15
    sigma = 4.0
    theta = 0  # 0 radians = horizontal
    lambd = 10.0
    gamma = 0.5

    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)

    # Threshold to find anomalies
    _, thresh = cv2.threshold(filtered, 150, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours of potential flute damage
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size (flute damage would be smaller)
    damage_contours = [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 1000]

    result_frame = frame.copy()

    # Draw rectangles around potential damage
    for cnt in damage_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    damage_score = len(damage_contours) / 10.0  # Normalize
    damage_score = min(1.0, damage_score)

    return result_frame, {"flute_damage_score": damage_score, "damage_count": len(damage_contours)}


def detect_crease_quality(frame, color_params):
    """Detect quality of creases/fold lines"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Apply mask to the original image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Edge detection to find creases
    edges = cv2.Canny(gray, 50, 150)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    result_frame = frame.copy()
    crease_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line length and angle
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Filter horizontal or vertical lines (likely creases)
            if (0 <= angle <= 5) or (85 <= angle <= 95) or (175 <= angle <= 180):
                crease_lines.append((x1, y1, x2, y2, length, angle))
                cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate crease quality score based on straightness and number
    crease_quality = 0
    if crease_lines:
        # Group by orientation (horizontal/vertical)
        h_lines = [line for line in crease_lines if (0 <= line[5] <= 5) or (175 <= line[5] <= 180)]
        v_lines = [line for line in crease_lines if (85 <= line[5] <= 95)]

        # Check parallelism within groups
        if h_lines:
            h_y_coords = [0.5 * (line[1] + line[3]) for line in h_lines]
            h_parallelism = 1.0 - min(1.0, np.std(h_y_coords) / 10.0)
        else:
            h_parallelism = 0

        if v_lines:
            v_x_coords = [0.5 * (line[0] + line[2]) for line in v_lines]
            v_parallelism = 1.0 - min(1.0, np.std(v_x_coords) / 10.0)
        else:
            v_parallelism = 0

        crease_quality = 0.5 * (h_parallelism + v_parallelism)

    return result_frame, {"crease_quality": crease_quality, "crease_line_count": len(crease_lines)}


def detect_surface_defects(frame, color_params):
    """Detect surface defects like tears, holes, or stains"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Apply mask to the original image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to find local anomalies
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours of potential defects
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    defect_contours = [cnt for cnt in contours if 50 < cv2.contourArea(cnt) < 500]

    result_frame = frame.copy()

    # Draw rectangles around potential defects
    for cnt in defect_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    defect_score = len(defect_contours) / 20.0  # Normalize
    defect_score = min(1.0, defect_score)

    return result_frame, {"surface_defect_score": defect_score, "defect_count": len(defect_contours)}


def analyze_moisture(frame, color_params):
    """Analyze moisture content through color/texture variance"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Convert to HSV - moisture often affects value/saturation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Analyze saturation variation (moisture affects saturation)
    h, s, v = cv2.split(hsv)
    sat_mean = cv2.mean(s, mask=mask)[0]
    sat_std = np.std(s[mask > 0]) if np.sum(mask > 0) > 0 else 0

    # Higher saturation variance often indicates moisture issues
    moisture_score = min(1.0, sat_std / 30.0)

    # Create heatmap visualization
    result_frame = frame.copy()

    # Apply the heatmap only to the cardboard areas
    heatmap = cv2.applyColorMap(s, cv2.COLORMAP_JET)

    # Blend heatmap with original image where cardboard is detected
    alpha = 0.7
    mask_3ch = cv2.merge([mask, mask, mask])
    mask_3ch = mask_3ch.astype(float) / 255

    blended = cv2.addWeighted(result_frame, 1 - alpha, heatmap, alpha, 0)
    np.copyto(result_frame, blended, where=mask_3ch.astype(bool))

    return result_frame, {"moisture_score": moisture_score, "saturation_mean": sat_mean}


def analyze_thickness(frame, color_params):
    """Detect variations in board thickness"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Apply mask to the original image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter to enhance edges (which indicate thickness changes)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Normalize to 0-255 range
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold to find areas of thickness change
    _, thresh = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    # Create a heatmap visualization
    heatmap = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)

    # Calculate thickness uniformity score
    if np.sum(mask > 0) > 0:
        uniformity_score = 1.0 - (np.sum(thresh & mask) / (np.sum(mask)))
    else:
        uniformity_score = 0

    # Blend with original image, but only on cardboard areas
    result_frame = frame.copy()
    mask_3ch = cv2.merge([mask, mask, mask])
    mask_3ch = mask_3ch.astype(float) / 255

    blended = cv2.addWeighted(result_frame, 0.7, heatmap, 0.3, 0)
    np.copyto(result_frame, blended, where=mask_3ch.astype(bool))

    return result_frame, {"thickness_uniformity": uniformity_score, "variation_percent": (1 - uniformity_score) * 100}


# ------------------------------------------------------------------------------
# PRODUCTION MEASUREMENT FUNCTIONS
# ------------------------------------------------------------------------------

def count_sheets(frame, color_params):
    """Count the number of sheets in the frame"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to find sheets
    sheet_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

    # Draw and count sheets
    result_frame = frame.copy()

    for i, cnt in enumerate(sheet_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_frame, f"#{i + 1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result_frame, {"sheet_count": len(sheet_contours)}


def analyze_dimensions(frame, color_params):
    """Measure exact sheet dimensions"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to find sheets
    sheet_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

    result_frame = frame.copy()
    dimensions = []

    # Assuming a pixel-to-cm ratio (this would need calibration in real-world)
    pixel_to_cm = 0.1

    for i, cnt in enumerate(sheet_contours):
        # Get dimensions using minimum area rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Get width and height
        width = rect[1][0] * pixel_to_cm
        height = rect[1][1] * pixel_to_cm

        dimensions.append((width, height))

        # Draw rectangle
        cv2.drawContours(result_frame, [box], 0, (0, 255, 0), 2)

        # Display dimensions
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(result_frame, f"{width:.1f}x{height:.1f}cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calculate dimensional consistency
    consistency = 0
    if dimensions:
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        width_var = np.var(widths) if len(widths) > 1 else 0
        height_var = np.var(heights) if len(heights) > 1 else 0
        consistency = 1.0 - min(1.0, (width_var + height_var) / 10.0)

    avg_width = np.mean([d[0] for d in dimensions]) if dimensions else 0
    avg_height = np.mean([d[1] for d in dimensions]) if dimensions else 0

    return result_frame, {"dimension_consistency": consistency,
                          "avg_width_cm": avg_width,
                          "avg_height_cm": avg_height}


def calculate_production_rate(frames, frame_times, color_params):
    """Calculate production rate based on sheet movement over time"""
    if len(frames) < 2:
        return frames[0] if frames else None, {"production_rate": 0, "sheets_per_minute": 0}

    # Get first and last frame
    first_frame = frames[0]
    last_frame = frames[-1]

    # Create masks for brown cardboard
    first_mask = create_brown_mask(first_frame, **color_params)
    last_mask = create_brown_mask(last_frame, **color_params)

    # Count sheets in first and last frame
    first_contours, _ = cv2.findContours(first_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    first_sheets = [cnt for cnt in first_contours if cv2.contourArea(cnt) > 5000]

    last_contours, _ = cv2.findContours(last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    last_sheets = [cnt for cnt in last_contours if cv2.contourArea(cnt) > 5000]

    # Calculate time difference
    time_diff = frame_times[-1] - frame_times[0]
    time_diff_minutes = time_diff / 60.0

    # Estimate sheets processed
    sheets_processed = abs(len(last_sheets) - len(first_sheets))
    if sheets_processed == 0:
        # If sheet count didn't change, estimate based on motion
        # This would need optical flow or more sophisticated tracking in real application
        sheets_processed = 1  # Placeholder

    # Calculate rate
    rate = sheets_processed / time_diff_minutes if time_diff_minutes > 0 else 0

    # Create visualization
    result_frame = last_frame.copy()
    cv2.putText(result_frame, f"Rate: {rate:.1f} sheets/min", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return result_frame, {"production_rate": rate, "time_elapsed_sec": time_diff}


def monitor_stack_height(frame, color_params):
    """Monitor stack heights for consistency"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to find stacks
    stack_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]

    result_frame = frame.copy()
    heights = []

    # Pixel to cm conversion (would need calibration)
    pixel_to_cm = 0.1

    for i, cnt in enumerate(stack_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        height_cm = h * pixel_to_cm
        heights.append(height_cm)

        # Draw rectangle
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display height
        cv2.putText(result_frame, f"H: {height_cm:.1f}cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Calculate height consistency
    consistency = 0
    if len(heights) > 1:
        height_var = np.var(heights)
        consistency = 1.0 - min(1.0, height_var / 5.0)

    avg_height = np.mean(heights) if heights else 0

    return result_frame, {"stack_height_consistency": consistency, "avg_height_cm": avg_height}


def verify_cut_accuracy(frame, color_params):
    """Verify if sheets are cut to specification"""
    # Create mask for brown cardboard
    mask = create_brown_mask(frame, **color_params)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to find sheets
    sheet_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

    result_frame = frame.copy()
    rectangularity_scores = []

    for cnt in sheet_contours:
        # Calculate rectangularity (how rectangular the shape is)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        contour_area = cv2.contourArea(cnt)

        if rect_area > 0:
            rectangularity = contour_area / rect_area
            rectangularity_scores.append(rectangularity)

            # Color based on score
            color = (0, 255, 0) if rectangularity > 0.95 else \
                (0, 255, 255) if rectangularity > 0.90 else (0, 0, 255)

            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_frame, f"{rectangularity:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Overall cut accuracy
    cut_accuracy = np.mean(rectangularity_scores) if rectangularity_scores else 0
    
    return result_frame, {"cut_accuracy": cut_accuracy, "sheet_count": len(sheet_contours)}

def measure_sheet_spacing(frame):
    """Measure spacing between sheets"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to find sheets
    sheet_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
    
    result_frame = frame.copy()
    spacings = []
    
    # Sort contours by x-coordinate
    bounding_rects = [cv2.boundingRect(cnt) for cnt in sheet_contours]
    sorted_rects = sorted(bounding_rects, key=lambda x: x[0])
    
    # Calculate spacings between adjacent sheets
    for i in range(len(sorted_rects) - 1):
        x1, y1, w1, h1 = sorted_rects[i]
        x2, y2, w2, h2 = sorted_rects[i+1]
        
        spacing = x2 - (x1 + w1)
        if spacing > 0:
            spacings.append(spacing)
            
            # Draw spacing
            start_point = (x1 + w1, int((y1 + y2) / 2))
            end_point = (x2, int((y1 + y2) / 2))
            cv2.line(result_frame, start_point, end_point, (0, 255, 255), 2)
            
            # Display spacing
            mid_x = (x1 + w1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(result_frame, f"{spacing}px", (mid_x - 20, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw all sheet bounding boxes
    for x, y, w, h in sorted_rects:
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Calculate consistency of spacing
    consistency = 0
    if len(spacings) > 1:
        spacing_var = np.var(spacings)
        consistency = 1.0 - min(1.0, spacing_var / 100.0)
    
    avg_spacing = np.mean(spacings) if spacings else 0
    
    return result_frame, {"avg_spacing" : avg_spacing}

def analyze_line_speed(frames, frame_times):
    """Analyze throughput in relation to quality to optimize line speed"""
    if len(frames) < 2:
        return frames[0] if frames else None, {"line_speed": 0, "sheets_per_minute": 0}

    # Count sheets in first and last frame
    first_frame, last_frame = frames[0], frames[-1]
    first_sheets = count_sheets(first_frame,color_params)[1]["sheet_count"]
    last_sheets = count_sheets(last_frame,color_params)[1]["sheet_count"]

    # Time elapsed in minutes
    time_diff = (frame_times[-1] - frame_times[0]) / 60.0
    sheet_speed = (last_sheets - first_sheets) / time_diff if time_diff > 0 else 0

    # Create visualization
    result_frame = last_frame.copy()
    cv2.putText(result_frame, f"Speed: {sheet_speed:.1f} sheets/min", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return result_frame, {"line_speed": sheet_speed}

def detect_material_waste(frame):
    """Identify irregularly sized pieces of material waste"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small irregular shapes (likely waste)
    waste_contours = [cnt for cnt in contours if 500 < cv2.contourArea(cnt) < 5000]
    result_frame = frame.copy()

    for cnt in waste_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    waste_ratio = len(waste_contours) / max(1, len(contours))  # Normalize waste proportion

    return result_frame, {"waste_ratio": waste_ratio, "waste_count": len(waste_contours)}

# if uploaded_file:
#     temp_file = tempfile.NamedTemporaryFile(delete=False)
#     temp_file.write(uploaded_file.read())
#
#     cap = cv2.VideoCapture(temp_file.name)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#
#     output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
#
#     frame_count = 0
#     processed_frames = []
#     frame_times = []
#     analysis_results = []
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_count += 1
#         if frame_count % frame_sample_rate != 0:
#             continue  # Skip frames based on the sample rate
#
#         frame_times.append(time.time())
#
#         results = {}
#         processed_frame = frame.copy()
#
#         # **Quality Measurements**
#         if "Edge Alignment" in analysis_types:
#             processed_frame, results["Edge Alignment"] = detect_edge_alignment(processed_frame)
#         if "Warping Detection" in analysis_types:
#             processed_frame, results["Warping Detection"] = detect_warping(processed_frame)
#         if "Color Consistency" in analysis_types:
#             processed_frame, results["Color Consistency"] = analyze_color_consistency(processed_frame)
#         if "Flute Damage" in analysis_types:
#             processed_frame, results["Flute Damage"] = detect_flute_damage(processed_frame)
#         if "Crease Quality" in analysis_types:
#             processed_frame, results["Crease Quality"] = detect_crease_quality(processed_frame)
#         if "Surface Defects" in analysis_types:
#             processed_frame, results["Surface Defects"] = detect_surface_defects(processed_frame)
#         if "Moisture Analysis" in analysis_types:
#             processed_frame, results["Moisture Analysis"] = analyze_moisture(processed_frame)
#         if "Thickness Uniformity" in analysis_types:
#             processed_frame, results["Thickness Uniformity"] = analyze_thickness(processed_frame)
#
#         # **Production Measurements**
#         if "Sheet Count" in analysis_types:
#             processed_frame, results["Sheet Count"] = count_sheets(processed_frame)
#         if "Dimensional Analysis" in analysis_types:
#             processed_frame, results["Dimensional Analysis"] = analyze_dimensions(processed_frame)
#         if "Production Rate" in analysis_types and len(processed_frames) > 1:
#             processed_frame, results["Production Rate"] = calculate_production_rate(processed_frames, frame_times)
#         if "Stack Height" in analysis_types:
#             processed_frame, results["Stack Height"] = monitor_stack_height(processed_frame)
#         if "Cut Accuracy" in analysis_types:
#             processed_frame, results["Cut Accuracy"] = verify_cut_accuracy(processed_frame)
#         if "Sheet Spacing" in analysis_types:
#             processed_frame, results["Sheet Spacing"] = measure_sheet_spacing(processed_frame)
#         if "Line Speed" in analysis_types and len(processed_frames) > 1:
#             processed_frame, results["Line Speed"] = analyze_line_speed(processed_frames, frame_times)
#         if "Material Waste" in analysis_types:
#             processed_frame, results["Material Waste"] = detect_material_waste(processed_frame)
#
#         analysis_results.append(results)
#         processed_frames.append(processed_frame)
#
#         # Write annotated frame to video
#         out.write(processed_frame)
#
#     cap.release()
#     out.release()
#
#     # Display the processed video
#     st.video(output_video_path)
#
#
#
def plot_results(data, title, ylabel):
    """Generate a plot for analysis trends."""
    fig, ax = plt.subplots()
    ax.plot(range(len(data)), data, marker="o", linestyle="-", color="blue")
    ax.set_title(title)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

if 'analysis_results' not in locals():
    analysis_results = []
    
if 'analysis_results' in locals() and analysis_results:
    st.markdown("## 📊 Analysis Results")

    # Extract data for visualization
    metrics = {
        "Edge Alignment": "alignment_score",
        "Warping Detection": "warp_score",
        "Color Consistency": "color_consistency",
        "Flute Damage": "flute_damage_score",
        "Crease Quality": "crease_quality",
        "Surface Defects": "surface_defect_score",
        "Moisture Analysis": "moisture_score",
        "Thickness Uniformity": "thickness_uniformity",
        "Sheet Count": "sheet_count",
        "Dimensional Analysis": "dimension_consistency",
        "Production Rate": "production_rate",
        "Stack Height": "stack_height_consistency",
        "Cut Accuracy": "cut_accuracy",
        "Sheet Spacing": "avg_spacing",
        "Line Speed": "line_speed",
        "Material Waste": "waste_ratio"
    }

    col1, col2 = st.columns(2)

    for i, (key, metric) in enumerate(metrics.items()):
        values = [r[key][metric] for r in analysis_results if key in r]

        if values:
            with (col1 if i % 2 == 0 else col2):
                plot_results(values, f"{key} Over Time", metric.replace("_", " ").title())

    # Summary Table
    st.markdown("### 📋 Summary of Key Metrics")
    summary_df = pd.DataFrame(analysis_results).dropna(axis=1, how="all")
    st.dataframe(summary_df)
else:
    st.warning("No analysis results available. Please upload a video and select analysis options.")

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    processed_frames = []
    frame_times = []
    analysis_results = []

    video_placeholder = st.empty()  # For displaying the video
    stats_placeholder = st.empty()  # For displaying statistics dynamically

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_sample_rate != 0:
            continue  # Skip frames based on the sample rate

        frame_times.append(time.time())

        results = {}
        processed_frame = frame.copy()

        # color_params = {
        #     'hue_min': hue_min,
        #     'hue_max': hue_max,
        #     'sat_min': sat_min,
        #     'sat_max': sat_max,
        #     'val_min': val_min,
        #     'val_max': val_max
        # }

        # **Quality Measurements**
        if "Edge Alignment" in analysis_types:
            processed_frame, results["Edge Alignment"] = detect_edge_alignment(processed_frame, color_params)
        if "Warping Detection" in analysis_types:
            processed_frame, results["Warping Detection"] = detect_warping(processed_frame, color_params)
        if "Color Consistency" in analysis_types:
            processed_frame, results["Color Consistency"] = analyze_color_consistency(processed_frame, color_params)
        if "Flute Damage" in analysis_types:
            processed_frame, results["Flute Damage"] = detect_flute_damage(processed_frame, color_params)
        if "Crease Quality" in analysis_types:
            processed_frame, results["Crease Quality"] = detect_crease_quality(processed_frame, color_params)
        if "Surface Defects" in analysis_types:
            processed_frame, results["Surface Defects"] = detect_surface_defects(processed_frame, color_params)
        if "Moisture Analysis" in analysis_types:
            processed_frame, results["Moisture Analysis"] = analyze_moisture(processed_frame, color_params)
        if "Thickness Uniformity" in analysis_types:
            processed_frame, results["Thickness Uniformity"] = analyze_thickness(processed_frame, color_params)

        # **Production Measurements**
        if "Sheet Count" in analysis_types:
            processed_frame, results["Sheet Count"] = count_sheets(processed_frame, color_params)
        if "Dimensional Analysis" in analysis_types:
            processed_frame, results["Dimensional Analysis"] = analyze_dimensions(processed_frame, color_params)
        if "Production Rate" in analysis_types and len(processed_frames) > 1:
            processed_frame, results["Production Rate"] = calculate_production_rate(processed_frames, frame_times, color_params)
        if "Stack Height" in analysis_types:
            processed_frame, results["Stack Height"] = monitor_stack_height(processed_frame, color_params)
        if "Cut Accuracy" in analysis_types:
            processed_frame, results["Cut Accuracy"] = verify_cut_accuracy(processed_frame, color_params)
        if "Sheet Spacing" in analysis_types:
            processed_frame, results["Sheet Spacing"] = measure_sheet_spacing(processed_frame)
        if "Line Speed" in analysis_types and len(processed_frames) > 1:
            processed_frame, results["Line Speed"] = analyze_line_speed(processed_frames, frame_times)
        if "Material Waste" in analysis_types:
            processed_frame, results["Material Waste"] = detect_material_waste(processed_frame)

        analysis_results.append(results)
        processed_frames.append(processed_frame)

        # Convert frame to bytes and display in Streamlit
        _, buffer = cv2.imencode(".jpg", processed_frame)
        # video_placeholder.image(buffer.tobytes(), channels="BGR", use_column_width=True)
        video_placeholder.image(buffer.tobytes(), channels="BGR", use_container_width=True)

        # Dynamically update statistics
        with stats_placeholder.container():
            st.markdown("## 📊 Live Statistics (Up to Current Frame)")
            metrics = {
                "Edge Alignment": "alignment_score",
                "Warping Detection": "warp_score",
                "Color Consistency": "color_consistency",
                "Flute Damage": "flute_damage_score",
                "Crease Quality": "crease_quality",
                "Surface Defects": "surface_defect_score",
                "Moisture Analysis": "moisture_score",
                "Thickness Uniformity": "thickness_uniformity",
                "Sheet Count": "sheet_count",
                "Dimensional Analysis": "dimension_consistency",
                "Production Rate": "production_rate",
                "Stack Height": "stack_height_consistency",
                "Cut Accuracy": "cut_accuracy",
                "Sheet Spacing": "avg_spacing",
                "Line Speed": "line_speed",
                "Material Waste": "waste_ratio"
            }

            col1, col2 = st.columns(2)
            for i, (key, metric) in enumerate(metrics.items()):
                values = [r[key][metric] for r in analysis_results if key in r]

                if values:
                    with (col1 if i % 2 == 0 else col2):
                        plot_results(values, f"{key} Over Time", metric.replace("_", " ").title())

            # Summary Table
            st.markdown("### 📋 Current Metrics Table")
            summary_df = pd.DataFrame(analysis_results).dropna(axis=1, how="all")
            st.dataframe(summary_df)

    cap.release()
import cv2
import numpy as np
import time
import os
import sys

# Add the current directory to path for reliable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from eye_tracker import EyeSpeakTracker
from semantic_engine import SemanticEngine
from ui_grid import UIGrid, DwellSelector

def main():
    # Set up data path for icons
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    icons_path = os.path.join(project_dir, 'data', 'singapore_icons.json')
    print(f"Using icons from: {icons_path}")

    # Initialize components
    tracker = EyeSpeakTracker()
    semantic_engine = SemanticEngine(icons_path=icons_path)
    ui_grid = UIGrid(width=1024, height=768, grid_size=3)
    dwell_selector = DwellSelector(dwell_time=1.5)

    # Initial icon clusters
    ranked_icons = semantic_engine.cluster_icons()

    # Set up windows
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("EYE-SPEAK+ Interface", cv2.WINDOW_NORMAL)

    # Make sure the UI window is properly sized
    cv2.resizeWindow("EYE-SPEAK+ Interface", 1024, 768)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("EYE-SPEAK+ system initialized successfully")

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Get gaze position
        gaze_result = tracker.get_gaze_position(frame)

        # Extract gaze coordinates if available
        gaze_pos = None
        if gaze_result is not None:
            if isinstance(gaze_result, tuple) and len(gaze_result) >= 2:
                gaze_pos = (gaze_result[0], gaze_result[1])

        # Map gaze to UI grid
        cell_idx = ui_grid.map_gaze_to_grid(gaze_pos)

        # Check for dwell selection
        selection, dwell_progress = dwell_selector.update(cell_idx)

        # Create UI frame with current icons
        ui_frame = ui_grid.create_grid(ranked_icons)

        # Highlight current selection
        if cell_idx is not None and cell_idx < len(ranked_icons):
            ui_frame = ui_grid.highlight_selection(ui_frame, cell_idx, ranked_icons, False)

        # Draw dwell progress bar
        if dwell_progress > 0:
            # Get cell coordinates
            row, col = cell_idx // ui_grid.grid_size, cell_idx % ui_grid.grid_size
            x, y = col * ui_grid.cell_width, row * ui_grid.cell_height

            # Draw progress bar at bottom of cell
            progress_width = int(ui_grid.cell_width * dwell_progress)
            cv2.rectangle(ui_frame,
                        (x, y + ui_grid.cell_height - 10),
                        (x + progress_width, y + ui_grid.cell_height),
                        (0, 255, 0), -1)

        # Handle selection
        if selection is not None and selection < len(ranked_icons):
            selected_icon = ranked_icons[selection][0]
            print(f"Selected: {selected_icon['label']}")

            # Update semantic engine context
            semantic_engine.update_context(selected_icon)

            # Update icons based on new context
            ranked_icons = semantic_engine.cluster_icons()

            # Set feedback message
            ui_grid.set_feedback(f"Selected: {selected_icon['label']}")

        # Show frames
        cv2.imshow("Webcam", frame)
        cv2.imshow("EYE-SPEAK+ Interface", ui_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

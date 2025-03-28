# ui_grid.py
import numpy as np
import time
import cv2

class UIGrid:
    def __init__(self, width=800, height=600, grid_size=3):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
    def map_gaze_to_grid(self, gaze_pos):
        """Map gaze coordinates to grid cell index"""
        if gaze_pos is None:
            return None
            
        x, y = gaze_pos
        cell_width = self.width / self.grid_size
        cell_height = self.height / self.grid_size
        
        col = min(int(x / cell_width), self.grid_size - 1)
        row = min(int(y / cell_height), self.grid_size - 1)
        
        return row * self.grid_size + col
        
    def create_grid(self, ranked_icons):
        """Create styled UI frame with icons"""
        # Create gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (self.width, self.height), (30, 30, 60), -1)
        
        # Calculate cell dimensions
        self.cell_width = int(self.width / self.grid_size)
        self.cell_height = int(self.height / self.grid_size)
        
        # Draw grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_width
                y1 = i * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                
                # Draw cell background
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 80), 2)
                cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (70, 70, 100), -1)
                
                # Add icon label if available
                idx = i * self.grid_size + j
                if idx < len(ranked_icons):
                    icon = ranked_icons[idx][0]
                    text = icon['label']
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = x1 + (self.cell_width - text_size[0]) // 2
                    text_y = y1 + (self.cell_height + text_size[1]) // 2
                    cv2.putText(frame, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
        
    def highlight_selection(self, frame, cell_idx, ranked_icons, is_selected):
        """Highlight the selected cell in the UI frame"""
        if cell_idx is None or cell_idx >= len(ranked_icons):
            return frame
            
        # Calculate cell position
        row = cell_idx // self.grid_size
        col = cell_idx % self.grid_size
        x1 = int(col * self.cell_width)
        y1 = int(row * self.cell_height)
        x2 = int((col + 1) * self.cell_width)
        y2 = int((row + 1) * self.cell_height)
        
        # Draw rectangle around cell
        color = (0, 255, 0) if is_selected else (255, 0, 0)  # Green for selected, red for hover
        thickness = 3 if is_selected else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw feedback message if set
        if hasattr(self, 'feedback_message'):
            cv2.putText(frame, self.feedback_message,
                       (10, self.height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
        
        return frame
        
    def set_feedback(self, message):
        """Set feedback message to display on UI"""
        self.feedback_message = message

class DwellSelector:
    def __init__(self, dwell_time=1.5):
        self.dwell_time = dwell_time
        self.current_cell = None
        self.dwell_start = None
        self.dwell_progress = 0
        
    def update(self, cell_idx):
        """Update dwell selection state and return (selection, progress)"""
        selection = None
        now = time.time()
        
        # Reset progress if cell changed
        if cell_idx != self.current_cell:
            self.current_cell = cell_idx
            self.dwell_start = now
            self.dwell_progress = 0
            
        # Calculate progress if dwelling on a cell
        if cell_idx is not None:
            elapsed = now - self.dwell_start
            self.dwell_progress = min(elapsed / self.dwell_time, 1.0)
            
            # Check if dwell time completed
            if elapsed >= self.dwell_time:
                selection = cell_idx
                self.dwell_start = now  # Reset timer
                self.dwell_progress = 0
                
        return selection, self.dwell_progress

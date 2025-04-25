import cv2
import numpy as np
import time
import threading
from queue import Queue
from datetime import datetime  # Added import for timestamp

class MotionFaceDetector:
    def __init__(self):
        # Configuration
        self.motion_threshold = 1000  # Motion detection threshold
        self.face_scale_factor = 1.1  # Scale factor for face detection
        self.face_min_neighbors = 5   # Minimum neighbors for face detection
        self.face_min_size = (30, 30) # Minimum face size
        self.blur_kernel = (21, 21)   # Blur kernel size
        
        # Initialization
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.prev_blurred = None  # Initialize prev_blurred
        
    def start_camera(self):
        # Try different camera sources
        for camera_index in [0, 1, -1]:
            print(f"Trying camera {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                print(f"Unable to open camera {camera_index}")
                self.cap.release()
                continue
                
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"Unable to read from camera {camera_index}")
                self.cap.release()
                continue
                
            print(f"Camera {camera_index} successfully connected!")
            # Initialize prev_blurred with the first frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
            return True
            
        return False
    
    def capture_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
    
    def process_frame(self, frame):
        # Flip the image horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale once
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(
            gray,  # Use the already calculated grayscale image
            scaleFactor=self.face_scale_factor,
            minNeighbors=self.face_min_neighbors,
            minSize=self.face_min_size
        )
        
        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Get current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Display information
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, current_time, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return frame, blurred
    
    def run(self):
        if not self.start_camera():
            print("No camera could be opened")
            return
        
        self.running = True
        capture_thread = threading.Thread(target=self.capture_thread)
        capture_thread.start()
        
        while self.running:
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
            
            # Get a new frame
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Process the frame
                processed_frame, curr_blurred = self.process_frame(frame)
                
                # Motion detection
                if self.prev_blurred is not None:  # Check if prev_blurred exists
                    diff = cv2.absdiff(self.prev_blurred, curr_blurred)
                    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) < self.motion_threshold:
                            continue
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Motion and Face Detection', processed_frame)
                
                # Update previous image
                self.prev_blurred = curr_blurred
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.running = False
        capture_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = MotionFaceDetector()
    detector.run() 
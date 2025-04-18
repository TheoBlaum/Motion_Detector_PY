# Motion and Face Detector

This program is a real-time motion and face detection system using a webcam. It combines motion detection with facial recognition to provide comprehensive visual surveillance.

## Features

- Real-time motion detection
- Face detection
- FPS (frames per second) display
- Image timestamping
- Real-time visual interface with information overlay

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. Ensure Python is installed on your system
2. Install the required dependencies:
```bash
pip install opencv-python numpy
```

## Usage

1. Run the program:
```bash
python motion_detector.py
```

2. Controls:
- Press 'q' to quit the program

## Display

The program shows:
- Number of detected faces
- Real-time FPS counter
- Current date and time
- Blue rectangles around detected faces
- Green rectangles around motion areas

## Configuration

The following parameters can be adjusted in the code:
- `motion_threshold`: Motion detection threshold
- `face_scale_factor`: Scale factor for face detection
- `face_min_neighbors`: Minimum neighbors for face detection
- `face_min_size`: Minimum face size
- `blur_kernel`: Blur kernel size

## Code Structure

The program uses a multi-threaded architecture for:
- Continuous video capture
- Image processing
- Motion and face detection
- Results display

## Notes

- The program automatically tries different camera sources (0, 1, -1)
- Performance may vary depending on your computer's capabilities
- Face detection uses OpenCV's Haar Cascade classifier
- The camera feed is mirrored horizontally for a more natural viewing experience 
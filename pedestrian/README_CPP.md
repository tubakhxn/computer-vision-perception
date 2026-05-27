# AI Pedestrian 3D Perception System (C++ Version)

## Dev/Creator: tubakhxn

This is the C++ implementation of the AI Pedestrian 3D Perception System. It replicates the logic of the Python version using OpenCV and standard C++ libraries. The project demonstrates real-time pedestrian tracking, synthetic depth estimation, and visualization.

## How to Fork and Run
1. **Fork the repository** on GitHub (https://github.com/tubakhxn/ai-pedestrian-3d-perception) or download the source code.
2. Clone your fork or download the ZIP and extract it.
3. Make sure you have OpenCV installed (version 4.x recommended).
4. Build the project:
   - Example (using g++):
     `g++ pedestrian.cpp -o pedestrian -std=c++17 `pkg-config --cflags --libs opencv4``
5. Run the program:
   - `./pedestrian <video_file_or_camera_index>`

## Relevant Wikipedia Links
- [Pedestrian detection](https://en.wikipedia.org/wiki/Pedestrian_detection)
- [Computer vision](https://en.wikipedia.org/wiki/Computer_vision)
- [Object detection](https://en.wikipedia.org/wiki/Object_detection)
- [YOLO (object detection)](https://en.wikipedia.org/wiki/You_Only_Look_Once)
- [Depth perception](https://en.wikipedia.org/wiki/Depth_perception)

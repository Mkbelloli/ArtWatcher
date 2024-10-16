
### DESCRIPTION

The project aims to identify people in a video and map them onto an external map. 
It is capable of operating both with live camera video streams and MP4 videos.

It consists of two components:
1. *Object Finder*: identifies people in the video (YOLOv3) and tracks their 
movement (Lucas-Kanade Optical Flow). The position, mapped onto the physical 
dimensions of the environment, is sent to a Viewer.
2. *Map Viewer*: displays the points received from the previous block.

### RUN
To run this project follow these steps:
1. run *UI server*: python main.py
2. run *Object Finder*: python Viewer.py


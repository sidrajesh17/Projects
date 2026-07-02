#Problem Statement
The lack of reliable, non-invasive, and efficient real-time systems for monitoring mental well-being leads to undetected stress and poor behavioural patterns, creating the need for an AI-driven, webcam-based multimodal solution that provides accurate and continuous stress detection.

#Architecture
Initially, the system comes with real-time data acquisition wherein OpenCV is used to capture the live video frames.  These frames are normalized, and for face detection, MTCNN is used. After that, the face region is cropped from the frame. The facial pixels are fed into two separate pipelines simultaneously: CNN-based FER which infers the emotion from facial expression and rPPG which calculates the heart rate from subtle color variations of the face. The fusion engine, which applies rule-based or adaptive logic to determine stress levels, combines the outputs of these two. The output of the system is displayed in real-time on a Flet-powered dashboard, where the user can monitor their current mood, heart rate, and stress level. 

#Tech Stack
Programming Language: Python 3.10 or above.
Frameworks and Libraries:
• Flet for GUI and user interface
• OpenCV for video processing and facial detection
• OpenCV for video processing and facial detection
• NumPy, SciPy for signal and data processing
• PyVHR (optional) for advanced rPPG analysis
Development Environment: Visual Studio Code / Jupyter Notebook
Operating System: Windows 10 / 11 (64-bit)

#Dataset
FER

#Run Instructions


#Screenshots of demo


<img width="680" height="347" alt="Screenshot 2026-07-02 203612" src="https://github.com/user-attachments/assets/1bc3aa11-bed2-4ec3-b5a5-c2394d6425f0" />

#Limitations
No personalization
Not much accessibility 
Performance not that good under low light



# Duke Ignite Hackathon🔥: Live Glasses Filtering with Facial/Gesture Recognition in OpenCV

## Info
This is a Flask web application that uses OpenCV to apply virtual glasses filtering to a live video feed from a user's webcam, using facial recognition and gesture recognition.

## Members
- Developers: 1. [Minghui Zhu](https://github.com/zhuminghui17) ; 2. [Congcong Ma](https://github.com/Donis666); 3. [Thomas Barker](https://github.com/ThomasBarker220).
- Mentor: JJ Je. (Shout out to Duke Ignite organizers and mentors!)

<p align="center">
  <img src="https://user-images.githubusercontent.com/68854273/229389468-129eddcb-d1a9-40c9-94c7-63e8b863eaf9.jpg" alt="alt text" width="300" height="225" />
</p>
<p align="center">Developers Selfie with Glasses Filtering😎.</p>

## Features
- Uses OpenCV to track the user's face in real-time, detecting facial landmarks such as the eyes.
- Applies virtual glasses filtering to the user's face in the video feed, based on the detected eyes coordinates.
- Uses handing tracking and gesture recognition to allow the user to switch between different types of glasses by making specific hand gestures in front of the webcam: 
  - fingers up: keep changing glasses; 
  - fingers down: stop.
- Supports multiple types of glasses, each with their own unique style and design.
- Resizes the video feed to half the original size to improve performance.
- Allows mutliple faces and fits multiple glasses. 

## Installation
- Clone the repository to your local machine.
- Install the required Python packages.
- Download the pre-trained facial landmark detection model from dlib here and place it in the root directory of the project.
- Run the application by executing python app.py.

## Usage
- Open a web browser and navigate to http://localhost:5050.
- Allow the application to access your webcam.
- The live video feed should now appear in the browser window with virtual glasses applied to your face.
- Make specific hand gestures in front of the webcam to switch between different types of glasses.

## Future Improvements
- Improve the accuracy of the gesture landmark detection when the background and the hands are in the same color.
- Handle the issues it run slowly on some laptops. 
- Deploy it.

## References:
- [OpenCV](https://opencv.org/)
- [ignite-cv-workshop](https://github.com/bharat-krishnan/ignite-cv-workshop)
- [How to make your own Instagram filter with facial recognition from scratch using python](https://github.com/mitkrieg/live-image-face-filter-blog)


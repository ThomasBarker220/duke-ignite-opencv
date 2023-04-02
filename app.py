from flask import Flask, render_template, request, Response
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import base64
import numpy as np

app = Flask(__name__)

camera = None
detector = HandDetector(detectionCon= .8, maxHands = 1)
fdetector = FaceDetector()
video_feed_active = False
index_ = 0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
glasses_orig = cv2.imread('glass.png', -1)
eye_cache = None

images = ['glass.png', 'glasses.png', 'swirlyglasses.png']

def transparentOverlay(src, overlay, pos=(0,0), scale=1):
    overlay = cv2.resize(overlay, (0,0), fx=scale, fy=scale)
    h, w, _ = overlay.shape #size of foreground
    rows, cols, _ = src.shape #size of background image
    y, x = pos[0], pos[1] #position of foreground/overlay image

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0) #read alpha channel
            src[x+i][y+j] = alpha * overlay[i][j][:3] + (1-alpha) * src[x+i][y+j]
    return src










def process_frame(frame):
    # Your frame processing logic here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 100, 200)

    # index = 0
    # length = 10
    # hairs = [0,1,2,3,4,5,6,7,8,9]
    
    height, width, layers = frame.shape

    frame = cv2.resize(frame, ( width//2, height//2))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 6, 0, (120,120), (350,350))
    frame_cp = frame.copy()
    img, bboxs = fdetector.findFaces(frame)

    # hands, image = detector.findHands(frame)
    if len(bboxs) == 0:
        return frame
    for bbox in bboxs:
        x, y, w, h = bbox["bbox"]
        if h > 0 and w > 0:
            glasses_ymin = int(y + 1 * h / 5)
            glasses_ymax = int(y + 2.5 * h / 5)
            height_glasses = glasses_ymax - glasses_ymin

            face_glasses_roi_color = frame_cp[glasses_ymin:glasses_ymax, x:x + w]


            face_glasses_roi_color = transparentOverlay(face_glasses_roi_color, glasses)
            frame_cp[glasses_ymin:glasses_ymax, x:x + w] = face_glasses_roi_color
    # if hands:
    #     lmList = hands[0]

    #     fingerUp = detector.fingersUp(lmList)
        
    #     global index_
    #     # hair = hairs[index]
    #     if fingerUp[1] == 1:
    #         # hair = hairs[index]

    #             # hair = hairs[index]
    #             index_ = (index_ + 1) % 10
    #             cv2.putText(frame, str(index_), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

                
    #     else:
    #         cv2.putText(frame, str(index_), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    
    return frame_cp


def gen_frames():
    global camera, video_feed_active
    while video_feed_active:
        success, frame = camera.read()
        if not success:
            break
        processed_frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        result = base64.b64encode(buffer).decode('utf-8')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera, video_feed_active
    if not video_feed_active:
        camera = cv2.VideoCapture(0)
        video_feed_active = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/close-video')
# def close_video():
#     global video_feed_active
#     video_feed_active = False
#     return "ok"

@app.route('/stop_video_feed')
def stop_video_feed():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Video feed stopped"


if __name__ == '__main__':
    app.run(debug=True)



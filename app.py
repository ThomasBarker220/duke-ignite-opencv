# # app.py
# from flask import Flask, render_template, request, Response
# from cvzone.HandTrackingModule import HandDetector
# import cv2
# import base64

# app = Flask(__name__)

# camera = cv2.VideoCapture(0)
# detector = HandDetector(detectionCon= .8, maxHands = 1)
# video_feed_active = False

# def process_frame(frame):
#     # Your frame processing logic here
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # edges = cv2.Canny(gray, 100, 200)
    
#     index = 0
#     length = 10
#     hairs = [0,1,2,3,4,5,6,7,8,9]
    
#     height, width, layers = frame.shape

#     frame = cv2.resize(frame, ( width//2, height//2))

#     hands, image = detector.findHands(frame)
#     if hands:
#         lmList = hands[0]

#         fingerUp = detector.fingersUp(lmList)
        
#         hair = hairs[index]
#         if fingerUp[1] == 1:
#             # hair = hairs[index]
            
#                 # hair = hairs[index]
#             cv2.putText(frame, str(hair), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                
#         else:
#             cv2.putText(frame, str(hair), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    
#     return image


# def gen_frames():
#     global camera, video_feed_active
#     while video_feed_active:
#         success, frame = camera.read()
#         if not success:
#             break
#         processed_frame = process_frame(frame)
#         _, buffer = cv2.imencode('.jpg', processed_frame)
#         result = base64.b64encode(buffer).decode('utf-8')
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
#     camera.release()
#     cv2.destroyAllWindows()
    
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     global camera, video_feed_active
#     if not video_feed_active:
#         camera = cv2.VideoCapture(0)
#         video_feed_active = True
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/close-video')
# def close_video():
#     global video_feed_active
#     video_feed_active = False
#     return "ok"



# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, Response
from cvzone.HandTrackingModule import HandDetector
import cv2
import base64

app = Flask(__name__)

camera = None
detector = HandDetector(detectionCon= .8, maxHands = 1)
video_feed_active = False
index_ = 0

def process_frame(frame):
    # Your frame processing logic here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 100, 200)

    # index = 0
    # length = 10
    # hairs = [0,1,2,3,4,5,6,7,8,9]
    
    height, width, layers = frame.shape

    frame = cv2.resize(frame, ( width//2, height//2))

    hands, image = detector.findHands(frame)
    if hands:
        lmList = hands[0]

        fingerUp = detector.fingersUp(lmList)
        
        global index_
        # hair = hairs[index]
        if fingerUp[1] == 1:
            # hair = hairs[index]

                # hair = hairs[index]
            # print(index_)
            index_ = (index_ + 1) % 10
            # print(index_)
            cv2.putText(frame, str(index_), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            
        else:
            cv2.putText(frame, str(index_), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    
    return image


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



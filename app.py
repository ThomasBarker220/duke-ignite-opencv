from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFilter
import io
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import base64
import numpy as np

app = Flask(__name__)
detector = HandDetector(detectionCon=0.8, maxHands=1)
fdetector = FaceDetector()
index_ = 0

numOfGlasses = 9
images = [f'./glasses/glasses{i}.png' for i in range(0, numOfGlasses)]


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
    height, width, layers = frame.shape
    frame = cv2.resize(frame, (width//2, height//2))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_cp = frame.copy()
    
    hands, image = detector.findHands(frame)
    if hands:
        lmList = hands[0]
        fingerUp = detector.fingersUp(lmList)
        
        global index_
        if fingerUp[1] == 1:
            index_ = (index_ + 1) % numOfGlasses
            cv2.putText(frame_cp, str(index_), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame_cp, str(index_), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            
    img, bboxs = fdetector.findFaces(frame)
    if len(bboxs) == 0:
        return frame_cp
    for bbox in bboxs:
        x, y, w, h = bbox["bbox"]
        if h > 0 and w > 0:
            glasses_ymin = int(y + 0.7 * h / 5)
            glasses_ymax = int(y + 2.5 * h / 5)
            height_glasses = glasses_ymax - glasses_ymin
            face_glasses_roi_color = frame_cp[glasses_ymin:glasses_ymax, x:x + w]

            glasses_orig = cv2.imread(images[index_], -1)
            glasses = cv2.resize(glasses_orig, (w, height_glasses), interpolation=cv2.INTER_CUBIC)
            face_glasses_roi_color = transparentOverlay(face_glasses_roi_color, glasses)
            frame_cp[glasses_ymin:glasses_ymax, x:x + w] = face_glasses_roi_color
    return frame_cp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        image_data = request.form['image']
        image_bytes = io.BytesIO(base64.b64decode(image_data.split(',')[1]))
        image = Image.open(image_bytes)# Process the image here
        processed_image = process_frame(np.array(image))
        # Convert processed image to JPEG format and encode it in base64
        buffer = io.BytesIO()
        Image.fromarray(processed_image).save(buffer, format="JPEG")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode('ascii')
        return jsonify({'result': encoded_image})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5050, debug=True)
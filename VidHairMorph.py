import cv2
import numpy as np


images = ['glass.png', 'glasses.png', 'swirlyglasses.png']
#get facial classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

eye_cache = None

cap = cv2.VideoCapture(0)

def runVideo(index):

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 2:
                eye_cache = eyes

            elif eye_cache is not None:
                eyes = eye_cache

        img = cv2.imread(images[index], -1)
        img_h = img.shape[0]
        img_w = img.shape[1]

        src_mat = np.array([[0,0], [img_w, 0], [img_w, img_h], [0, img_h]])
        if eyes[0][0] < eyes[1][0]:
            dst_mat = np.array([
                [x + eyes[0][0], y + eyes[0][1]],
                [x + eyes[1][0] + eyes[1][2], y + eyes[1][2]],
                [x + eyes[1][0] + eyes[1][2], y + eyes[1][1] + eyes[1][3]],
                [x + eyes[0][0], y + eyes[0][1] + eyes[0][3]]
            ])
        else:
            dst_mat = np.array([
                [x + eyes[1][0], y + eyes[1][1]],
                [x + eyes[0][0] + eyes[0][2], y + eyes[0][2]],
                [x + eyes[0][0] + eyes[0][2], y + eyes[0][1] + eyes[1][3]],
                [x + eyes[1][0], y + eyes[1][1] + eyes[1][3]]
            ])
        face_h = frame.shape[0]
        face_w = frame.shape[1]

        hom = cv2.findHomography(src_mat, dst_mat)[0]

        warped = cv2.warpPerspective(img, hom, (face_w, face_h))

        mask = warped[:,:,3]

        mask_scale = mask.copy() / 255.0
        mask_scale = np.dstack([mask_scale] * 3)

        warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)

        warped_multiplied = cv2.multiply(mask_scale, warped.astype('float'))
        image_multiplied = cv2.multiply(frame.astype(float), 1.0 - mask_scale)
        output = cv2.add(warped_multiplied, image_multiplied)
        output = output.astype('uint8')

        cv2.imshow('glasses', output)

        if cv2.waitKey(60) & 0xff == ord('q'):
            break
    
runVideo(1)
cap.release()
cv2.destroyAllWindows()
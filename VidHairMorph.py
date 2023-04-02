import cv2

#get facial classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

glasses_orig = cv2.imread('glass.png', -1)

# original_hair_h,original_hair_w,hair_channels = glasses.shape

# glasses_gray = cv2.cvtColor(glasses, cv2.COLOR_BGR2GRAY)

# ret, original_mask = cv2.threshold(hair_gray, 10, 255, cv2.THRESH_BINARY_INV)
# original_mask_inv = cv2.bitwise_not(original_mask)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


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

while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 6, 0, (120,120), (350,350))

    for (x,y,w,h) in faces:
        if h > 0 and w > 0:
            glasses_ymin = int(y + 1.5 * h / 5)
            glasses_ymax = int(y + 2.5 * h / 5)
            height_glasses = glasses_ymax - glasses_ymin

            face_glasses_roi_color = img[glasses_ymin:glasses_ymax, x:x + w]

            glasses = cv2.resize(glasses_orig, (w, height_glasses), interpolation=cv2.INTER_CUBIC)

            transparentOverlay(face_glasses_roi_color, glasses)
        
    cv2.imshow('Glasses', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# face_w = w
#         face_h = h
#         face_x1 = x
#         face_x2 = face_x1 + face_w
#         face_y1 = y
#         face_y2 = face_y1 + face_h

#         hair_width = int(1.2*face_w)
#         hair_height = int(hair_width * original_hair_h / original_hair_w)

#         hair_x1 = face_x1
#         hair_x2 = hair_x1 + hair_width
#         hair_y1 = int(face_y1 * 0.5)
#         hair_y2 = hair_y1 + hair_height

#         if hair_x1 < 0:
#             hair_x1 = 0
#         if hair_y1 < 0:
#             hair_y1 = 0
#         if hair_x2 > img_w:
#             hair_x2 = img_w
#         if hair_y2 > img_h:
#             hair_y2 = img_h

#         hair_width = hair_x2 - hair_x1
#         hair_height = hair_y2 - hair_y1

#         roi = img[hair_y1:hair_y2, hair_x1:hair_x2]

#         hair = cv2.resize(hair, (hair_width, hair_height), interpolation = cv2.INTER_AREA)


#         mask = cv2.resize(original_mask, (hair_width, hair_height), interpolation = cv2.INTER_AREA)

#         mask_inv = cv2.resize(original_mask_inv, (hair_width, hair_height), interpolation = cv2.INTER_AREA)





#         roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
#         roi_fg = cv2.bitwise_and(hair, hair, mask=mask_inv)
#         dst = cv2.add(roi_bg,roi_fg)

#         img[hair_y1:hair_y2, hair_x1:hair_x2] = dst

#         break

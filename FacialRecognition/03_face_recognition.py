import cv2
import numpy as np

# =========================
# Load Trained Recognizer
# =========================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# =========================
# Load Haar Cascades
# =========================

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

eyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_eye.xml'
)

smileCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_smile.xml'
)
# =========================
# Font
# =========================
font = cv2.FONT_HERSHEY_SIMPLEX

# =========================
# Names Corresponding to IDs
# =========================
names = [
    'None',
    'ayu',
    'bhu'
]

# =========================
# Initialize Webcam
# =========================
cam = cv2.VideoCapture(0)

# HD Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Webcam Brightness & Contrast
cam.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cam.set(cv2.CAP_PROP_CONTRAST, 150)

# Minimum face size
minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("\n[INFO] Face Recognition Started")
print("[INFO] Press ESC to Exit")

# =========================
# Main Loop
# =========================
while True:

    ret, img = cam.read()

    if not ret:
        print("[ERROR] Webcam not accessible")
        break

    # Mirror Effect
    img = cv2.flip(img, 1)

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =========================
    # Detect Faces
    # =========================
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    # =========================
    # Face Recognition
    # =========================
    for (x, y, w, h) in faces:

        # Draw Face Rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # =========================
        # Predict Face
        # =========================
        id, confidence = recognizer.predict(
            roi_gray
        )

        if confidence < 85:

            name = names[id]

            accuracy = round(100 - confidence)

        else:

            name = "Unknown"

            accuracy = round(100 - confidence)

        # =========================
        # Display Name
        # =========================
        cv2.putText(
            img,
            str(name),
            (x + 5, y - 10),
            font,
            1,
            (255, 255, 255),
            2
        )

        # =========================
        # Display Confidence
        # =========================
        cv2.putText(
            img,
            f"{accuracy}%",
            (x + 5, y + h - 5),
            font,
            0.8,
            (0, 255, 255),
            2
        )

        # =========================
        # Detect Eyes
        # =========================
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

        # =========================
        # Detect Smile
        # =========================
        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 25)
        )

        for (sx, sy, sw, sh) in smiles:

            cv2.rectangle(
                roi_color,
                (sx, sy),
                (sx + sw, sy + sh),
                (0, 255, 255),
                2
            )

            cv2.putText(
                img,
                "Smile",
                (x + sx, y + sy - 10),
                font,
                0.7,
                (0, 255, 255),
                2
            )

    # =========================
    # Show Window
    # =========================
    cv2.imshow('Advanced Face Recognition', img)

    # ESC Key to Exit
    k = cv2.waitKey(10) & 0xff

    if k == 27:
        break

# =========================
# Cleanup
# =========================
print("\n[INFO] Exiting Program")

cam.release()
cv2.destroyAllWindows()
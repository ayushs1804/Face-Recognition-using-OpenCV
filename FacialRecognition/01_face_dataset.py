import cv2
import os

# =========================
# Create Dataset Folder
# =========================
dataset_path = "dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# =========================
# Load Haar Cascades
# =========================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_eye.xml'
)

smile_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_smile.xml'
)

# =========================
# User Input
# =========================
face_id = input(
    '\nEnter User ID (numeric) ==> '
)

person_name = input(
    '\nEnter Person Name ==> '
)

# =========================
# Initialize Webcam
# =========================
cam = cv2.VideoCapture(0)

# HD Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Brightness & Contrast
cam.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cam.set(cv2.CAP_PROP_CONTRAST, 150)

print("\n[INFO] Initializing Face Dataset Collection...")
print("[INFO] Look at the camera")
print("[INFO] Smile, blink, change expressions")
print("[INFO] Press ESC to stop")

# =========================
# Counter
# =========================
count = 0

# =========================
# Main Loop
# =========================
while True:

    ret, img = cam.read()

    if not ret:
        print("[ERROR] Webcam access failed")
        break

    # Mirror effect
    img = cv2.flip(img, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =========================
    # Detect Faces
    # =========================
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # =========================
        # Eye Detection
        # =========================
        eyes = eye_detector.detectMultiScale(
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
        # Smile Detection
        # =========================
        smiles = smile_detector.detectMultiScale(
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
                "Smile Detected",
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # =========================
        # Save Face Samples
        # =========================
        count += 1

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (200, 200))

        file_name = (
            f"dataset/User.{face_id}.{count}.jpg"
        )

        cv2.imwrite(file_name, face)

        # =========================
        # Display Information
        # =========================
        cv2.putText(
            img,
            f"Samples: {count}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            img,
            person_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    # =========================
    # Show Webcam
    # =========================
    cv2.imshow(
        'Advanced Face Dataset Collection',
        img
    )

    # ESC to Exit
    k = cv2.waitKey(100) & 0xff

    if k == 27:
        break

    # Stop after 50 samples
    elif count >= 50:
        break

# =========================
# Cleanup
# =========================
print("\n[INFO] Dataset Collection Complete")
print(f"[INFO] Total Samples Collected: {count}")

cam.release()
cv2.destroyAllWindows()
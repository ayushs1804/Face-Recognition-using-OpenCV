import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Function to get images and labels
def getImagesAndLabels(path):

    imagePaths = [
        os.path.join(path, f)
        for f in os.listdir(path)
    ]

    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        try:

            PIL_img = Image.open(imagePath).convert('L')

            img_numpy = np.array(PIL_img, 'uint8')

            id = int(
                os.path.split(imagePath)[-1].split(".")[1]
            )

            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:

                faceSamples.append(
                    img_numpy[y:y+h, x:x+w]
                )

                ids.append(id)

        except Exception as e:

            print(f"Skipping {imagePath}: {e}")

    return faceSamples, ids

print("\n[INFO] Training faces. Please wait ...")

faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids))

# Save trained model
recognizer.write('trainer/trainer.yml')

print(f"\n[INFO] {len(np.unique(ids))} faces trained.")

print("\n[INFO] Model saved at trainer/trainer.yml")

print("\nTraining Complete!")
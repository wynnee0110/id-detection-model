import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("student_id_modelV2.h5")

IMG_SIZE = 128
THRESHOLD = 0.8

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img, verbose=0)[0][0]

    if pred > THRESHOLD:
        label = f"Student ID Detected ({pred*100:.1f}%)"
        color = (0, 255, 0)
    else:
        label = f"No ID ({(1-pred)*100:.1f}%)"
        color = (0, 0, 255)

    # Show result
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Student ID Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

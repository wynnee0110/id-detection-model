import cv2
import os

# Create folders if they don't exist
os.makedirs("dataset/student_id", exist_ok=True)
os.makedirs("dataset/no_id", exist_ok=True)

cap = cv2.VideoCapture(0)

student_count = len(os.listdir("dataset/student_id"))
no_id_count = len(os.listdir("dataset/no_id"))

print("Press 's' to save Student ID")
print("Press 'n' to save No ID")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # UI text
    cv2.putText(frame, "S: Student ID | N: No ID | Q: Quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Student ID images: {student_count}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"No ID images: {no_id_count}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save Student ID image
    if key == ord('s'):
        img_path = f"dataset/student_id/sid_{student_count}.jpg"
        cv2.imwrite(img_path, frame)
        student_count += 1
        print(f"Saved {img_path}")

    # Save No ID image
    elif key == ord('n'):
        img_path = f"dataset/no_id/noid_{no_id_count}.jpg"
        cv2.imwrite(img_path, frame)
        no_id_count += 1
        print(f"Saved {img_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

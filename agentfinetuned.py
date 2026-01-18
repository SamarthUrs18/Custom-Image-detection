from ultralytics import YOLO
import cv2

print("Loading CoreML model... (this may take a moment)")
model = YOLO('best.mlpackage') 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Sunscreen Detection Agent Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(frame, conf=0.5, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Vaseline Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
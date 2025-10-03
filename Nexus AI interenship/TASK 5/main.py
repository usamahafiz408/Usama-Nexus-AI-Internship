import cv2
from transformers import pipeline

# Load Hugging Face object detection model
pipe = pipeline("object-detection", model="PekingU/rtdetr_r50vd")

# Open video file (or use 0 for webcam)
cap = cv2.VideoCapture("input.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame BGR->RGB for HF model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    results = pipe(rgb_frame)

    # Draw boxes
    for r in results:
        score = r["score"]
        if score < 0.3:  # confidence threshold
            continue
        box = r["box"]
        label = r["label"]

        x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Processing complete! Output saved as output.mp4")

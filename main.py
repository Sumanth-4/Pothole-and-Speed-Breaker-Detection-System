import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/road_detection_model.h5', compile=False)
labels = ['normal', 'pothole', 'speedbreaker']
CONF_THRESHOLD = 0.60  # minimum confidence

last_spoken_label = None

cap = cv2.VideoCapture(0)
print("Camera started. Press 'q' to quit.")

def draw_colored_label(frame, text, bg_color, text_color=(0,0,0), font_scale=1.0, thickness=2):
    """Draw centered label with colored background at top area."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 20, 12
    rect_w = tw + pad_x * 2
    rect_h = th + pad_y * 2
    # position at top center
    x1 = int((w - rect_w) / 2)
    y1 = 20
    x2 = x1 + rect_w
    y2 = y1 + rect_h
    # draw filled rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    # draw text centered inside rectangle
    text_x = x1 + pad_x
    text_y = y1 + pad_y + th
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(img.astype('float32') / 255.0, axis=0)

    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred)); conf = float(pred[idx]); label = labels[idx]

    # Decide label to show and background color
    if label == 'normal' or conf < CONF_THRESHOLD:
        show_text = "Normal Road"
        bg = (0, 200, 0)      # green background (BGR)
        txt = (0, 0, 0)       # black text
    elif label == 'pothole':
        show_text = "POTHOLE"
        bg = (0, 0, 200)      # red background (BGR)
        txt = (0, 0, 0)
    else:  # speedbreaker
        show_text = "SPEEDBREAKER"
        bg = (0, 220, 220)    # yellow-ish background (BGR) -> (B,G,R)
        txt = (0, 0, 0)

    # Draw UI
    draw_colored_label(frame, f"{show_text}  {conf:.2f}", bg, txt, font_scale=1.0, thickness=2)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")

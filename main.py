import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



model_path = "blaze_face_short_range.tflite"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)

detector = vision.FaceDetector.create_from_options(options)


def process_img(img, detector):

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:

        bbox = detection.bounding_box

        x1 = bbox.origin_x
        y1 = bbox.origin_y
        w = bbox.width
        h = bbox.height

        x2 = min(W, x1 + w)
        y2 = min(H, y1 + h)

        img[y1:y2, x1:x2] = cv2.blur(img[y1:y2, x1:x2], (30, 30))

    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default="webcam")
args.add_argument("--filePath", default=None)
args = args.parse_args()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)


if args.mode == "image":

    img = cv2.imread(args.filePath)
    img = process_img(img, detector)
    cv2.imwrite(os.path.join(output_dir, "output.png"), img)

elif args.mode == "video":

    cap = cv2.VideoCapture(args.filePath)
    ret, frame = cap.read()

    output_video = cv2.VideoWriter(
        os.path.join(output_dir, "output.mp4"),
        cv2.VideoWriter_fourcc(*"MP4V"),
        25,
        (frame.shape[1], frame.shape[0])
    )

    while ret:
        frame = process_img(frame, detector)
        output_video.write(frame)
        ret, frame = cap.read()

    cap.release()
    output_video.release()

elif args.mode == "webcam":

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_img(frame, detector)
        cv2.imshow("Face Anonymizer", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

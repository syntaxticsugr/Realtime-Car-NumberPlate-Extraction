import cv2
from screeninfo import get_monitors
from ultralytics import YOLO



def detect_number_plate(frame, model):

    detections = model(frame)[0].boxes.data.tolist()

    for detection in detections:

        x1, y1, x2, y2, _, _ = detection

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)



def main(model_path, video_path):

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    window_width = screen_width // 2
    window_height = screen_height // 2

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        detect_number_plate(frame, model)

        frame = cv2.resize(frame, (window_width, window_height))

        cv2.imshow('Press Q To Exit', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    model_path = '../runs/detect/train3/weights/last.pt'

    video_path = '../test_videos/test_video_3840_2160_30fps.mp4'

    main(model_path, video_path)

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator

model = YOLO("yolov8s.pt")
names = model.model.names

# print(names)

cam = cv2.VideoCapture("vid2.mp4")

centerPoint = (10, 10)

while True:
    success, frame = cam.read()
    if not success:
        break
    
    frame = cv2.resize(frame, (1300, 700))
    results = model.predict(frame)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    # print(boxes)
    # print(clss)
    annotator = Annotator(frame)

    for box, cls in zip(boxes, clss):
        annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))
        annotator.visioneye(box, centerPoint)

    cv2.imshow("Vision Eye Ultralytics", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
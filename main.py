import cv2
dectect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = video.read()
    video_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = dectect.detectMultiScale(
        video_gray,
        scaleFactor=1.3,
        minNeighbors=5, )
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(100) == ord("a"):
        break
video.release()
cv2.destroyAllWindows()

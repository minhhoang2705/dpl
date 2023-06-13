import cv2 
import matplotlib.pyplot as plt
img = cv2.imread("348359356_1283334348930519_8013613579673235108_n.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = detector.detectMultiScale(
    gray, 
    scaleFactor= 1.05,
    minNeighbors=7,
    minSize=(40,40)
)
print("Found {} faces".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("my dectection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
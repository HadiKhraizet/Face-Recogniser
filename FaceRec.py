import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


known_image_1 = cv2.imread('smiling.jpeg', cv2.IMREAD_GRAYSCALE)
known_image_2 = cv2.imread('serious.jpeg', cv2.IMREAD_GRAYSCALE)

faces_1 = face_cascade.detectMultiScale(known_image_1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
faces_2 = face_cascade.detectMultiScale(known_image_2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

(x1, y1, w1, h1) = faces_1[0]
known_face_1 = known_image_1[y1:y1+h1, x1:x1+w1]

(x2, y2, w2, h2) = faces_2[0]
known_face_2 = known_image_2[y2:y2+h2, x2:x2+w2]

known_face_1 = cv2.resize(known_face_1, (known_face_2.shape[1], known_face_2.shape[0]))

mse = ((known_face_1 - known_face_2) ** 2).mean()

threshold = 100

known_image_1_display = cv2.cvtColor(known_image_1, cv2.COLOR_GRAY2BGR)
known_image_2_display = cv2.cvtColor(known_image_2, cv2.COLOR_GRAY2BGR)
known_face_1_display = cv2.cvtColor(known_face_1, cv2.COLOR_GRAY2BGR)
known_face_2_display = cv2.cvtColor(known_face_2, cv2.COLOR_GRAY2BGR)

cv2.rectangle(known_image_1_display, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
cv2.rectangle(known_image_2_display, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

if mse > threshold:
    label = "The two faces match"
else:
    label = "The 2 faces dont match"

print(label)
print(mse)

cv2.imshow("Detected face 1", known_image_1_display)
cv2.imshow("Detected face 2", known_image_2_display)
cv2.imshow("Extracted face 1", known_face_1_display)
cv2.imshow("Extracted face 2", known_face_2_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

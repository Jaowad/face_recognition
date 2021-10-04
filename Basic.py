import cv2
import numpy as np
import face_recognition


imgMe = face_recognition.load_image_file('ImagesBasic/bill gates.jpg')
imgMe = cv2.cvtColor(imgMe, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Bill gates test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgMe)[0]
encodeMe = face_recognition.face_encodings(imgMe)[0]
cv2.rectangle(imgMe, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeMe], encodeTest)
facedis = face_recognition.face_distance([encodeMe], encodeTest)


print(results, facedis)
cv2.putText(imgTest, f'{results}{round(facedis[0],2)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

cv2.imshow('BIll', imgMe)
cv2.imshow('BillTest', imgTest)
cv2.waitKey(0)
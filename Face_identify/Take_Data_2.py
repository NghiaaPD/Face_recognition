import cv2
import os


face_cascade = cv2.CascadeClassifier("C:/codeN/Python/Python Homework/Project Python/Face_identify/haarcascade_frontalface_default.xml")

face_id = input("Nhập ID của khuôn mặt:")

count = 1

cam = cv2.VideoCapture(0)

while True:
    
    ret , frame = cam.read()

    # Chuyển đổi hình ảnh sang đen trắng
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong hình ảnh đen trắng
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Vẽ hình chữ nhật quanh các khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    cv2.imshow("Chương trình nhận diện khuôn mặt", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.waitKey(1) & 0xFF == ord("s"):
        while True:
            cv2.imwrite("C:/codeN/Python/Python Homework/Project Python/Face_identify/Data/Face." + str(face_id) + "." + str(count) + ".jpg", gray_frame[y:y+h,x:x+w])
            count += 1
            if count > 30:
                print("Lấy dữu liệu của ID {} hoàn tất".format(face_id))
                break 


cam.release()
cv2.destroyAllWindows()


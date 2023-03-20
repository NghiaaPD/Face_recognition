import cv2
import os



face_detector = cv2.CascadeClassifier("path to haarcascade_frontalface_default.xml")

face_id = input("Nhập ID của khuôn mặt:")

count = 0

cam = cv2.VideoCapture(0)

while True:
    
    ret, frame = cam.read()  # Đọc hình ảnh từ webcam
   
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển đổi hình ảnh sang đen trắng
    
    faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)  # Phát hiện khuôn mặt trong hình ảnh đen trắng

    
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Vẽ hình chữ nhật quanh các khuôn mặt được phát hiện
        count += 1

        cv2.imwrite("Face_identify/Data/Face." + str(face_id) + "." + str(count) + ".jpg", gray_frame[y:y+h,x:x+w])    
    
        cv2.imshow("Webcam", frame)  # Hiển thị hình ảnh với các hình chữ nhật được vẽ quanh các khuôn mặt

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    elif count >= 20:
        break
    
if count >= 20:
    print("[INFO] Hoàn tất lấy giữ liệu")

cam.release()
cv2.destroyAllWindows()











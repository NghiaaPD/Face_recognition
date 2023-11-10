import cv2

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

face_id = input("Nhập ID của khuôn mặt:")
name = input(f"Nhập tên của khuôn mặt có ID {face_id}:")
with open('Name_list/Name.txt', 'a') as f:
    f.write(f"{name},")

count = 1

cam = cv2.VideoCapture(0)

while True:

    ret, frame = cam.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow("Chương trình nhận diện khuôn mặt", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.waitKey(1) & 0xFF == ord("s"):
        for i in range(30):
            cv2.imwrite("./Data/Face." + str(face_id) + "." + str(count) + ".jpg", gray_frame[y:y+h,x:x+w])
            count += 1
            if count > 30:
                print("Lấy dữu liệu của ID {} hoàn tất".format(face_id))
                break


cam.release()
cv2.destroyAllWindows()


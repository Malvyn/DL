import cv2
#初始化摄像头
cap = cv2.VideoCapture(0)
# 创建 haar 级联
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# 循环捕获每一帧
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	print("发现{0} 人脸!".format(len(faces)))

	# 画出框的位置
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


	# 显示图像
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): #按键q停止显示
		break

# 关闭
cap.release()
cv2.destroyAllWindows()

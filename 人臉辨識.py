import cv2

img=cv2.resize(cv2.imread("faces.jpg"),(0,0),fx=0.65,fy=0.65)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 載入人臉辨識模型
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)
# 辨識 新變數=變數.detectMultiScale(變數,每次縮小倍數,相鄰框最少要有幾個) 一框一框偵測,偵測完縮小並重偵測直到偵測到的次數到達第三個參數
# 第二個參數越大電腦運作速度較快但越容易沒偵測到 第三個參數越大越嚴謹
Face=faceCascade.detectMultiScale(gray,1.1,3) # 偵測出為矩形
print(len(Face))

for (x,y,w,h) in Face:
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow("img",img)
cv2.waitKey(0)

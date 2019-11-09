## 파이썬의 opencv, pytesseract을 이용해서 글자 인식하기

------

* 목차

  ------

  ## 사진 촬영 & 찍은 사진 로드하기

> ```
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> import pytesseract
> from PIL import Image
> ```

시작하기전 cv2, numpy, PIL, matplotlib, pytesseract 묘듈을 import해준다.

또한 나는 matplotlib에서 이미지를 보여줄떄 배경을 검은색으로 설정해주었다.

> `plt.style.use('dark_background')`

사진을 먼저 인식시키려면 사진을 불러와야한다. 나는 웹캠에서 사진을 불러오는 방식을 사용하려고한다.

> `capture = cv2.VideoCapture(0)
> capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
> capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)`

먼저 VideoCapture를 이용해서 웹캠의 영상을 capture에 저장한다. 그리고 넓이와 높이 크기를 지정해준다.

> `while True:
>     ret, frame = capture.read()
>     cv2.imshow("VideoFrame", frame)
>     if cv2.waitKey(1) > 0: break
>
> capture.release()
> cv2.destroyAllWindows()
> CAM_ID = 0
> def capture(camid = CAM_ID):
>     cam = cv2.VideoCapture(camid)
>     if cam.isOpened() == False:
>         print ('cant open the cam (%d)' % camid)
>         return None
>
>     ret, frame = cam.read()
>     if frame is None:
>         print ('frame is not exist')
>         return None
>     
>     # png로 압축 없이 영상 저장 
>     cv2.imwrite('1.jpg',frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
>     cam.release()
>
> if __name__ == '__main__':
>     capture()`

사진을 1.jpg로 저장한다.



> `img_ori = cv2.imread('1.jpg') `
>
> `height, width, channel = img_ori.shape`
>
> `plt.figure(figsize=(12, 10))
> plt.imshow(img_ori, cmap='gray')`

img_ori로 1.jpg를 불러온다.

불러온 이미지의 너비, 높이, 채널을 저장한다.



## 	불러온 사진을 회색으로 바꾸기

이미지 프로세싱을 더욱 쉽게하기 위해서 불러온 사진을 BGR에서 GRAY로 바꿔주려고한다.

>```
>gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
>```

gray라는 이름으로  회색으로 바뀐 사진을 저장한다.

------

## 	노이즈 제거 & 검정색 또는 흰색으로 분류하기

>```
>img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
>```

위 구문을 통해 이지미의 노이즈를 제거해준다.

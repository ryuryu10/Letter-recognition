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

또한 이미지를 좀더 구분하기 쉽게 만들기위해 검정색또는 흰색으로 분류를 해볼것이다.

  ![그림](file:///C:\Users\ryury\AppData\Local\Temp\tmp594F.jpg)  

특정값 기준으로 아래면 하얀색을 기준값보다 크면 검정색으로 처리할것이다.

>```
>img_thresh = cv2.adaptiveThreshold(
>    img_blurred, 
>    maxValue=255.0, 
>    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
>    thresholdType=cv2.THRESH_BINARY_INV, 
>    blockSize=19, 
>    C=9
>)
>```

​	<img src="C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109170658834.png" style="zoom:25%;" /> 

 이렇게 회색으로 바꾼 이미지의 노이즈를 제거하고 색갈을 분류하게되면

<img src="C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109170753673.png" alt="image-20191109170753673" style="zoom:25%;" />

이렇게 검정색또는 하얀색으로 분류가된다.

------

## 	컨투어 찾기 & 윤곽선 찾기

해당 이미지를 바로 pytesseract에 돌린다면 어느정도 정확한 글자가나올수도있지만 배경같은 다양한 환경에 의해 인식률이 낮아질수있어 그점을 보완하기 위해서 앞에서 처리한 이지미를 바로 pytesseract를 통해 인식하지않고 약간의 검증을 한뒤 인식시킬것이다.

먼저 cv2.findContours()를 통해 해당이미지의 윤곽선을 그릴것이다.

>```
>contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
>temp_result = np.zeros((height, width, channel), dtype=np.uint8)
>```

그리고 cv2.drawContours를 통해 컨투어를 그릴것이다.

>```
>cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
>```

<img src="C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109171619720.png" alt="image-20191109171619720" style="zoom:50%;" /> 이미지가 이렇게 변한것을 볼수있다.

------

## 	컨투어 영역을 사각형으로 표시하기

사진속 그려진 컨투어가 글자인지 아닌지 구분하기 앞서 그려진 영역을 사각형으로 표시할것이다.

>```
>temp_result = np.zeros((height, width, channel), dtype=np.uint8)
>
>contours_dict = []
>
>for contour in contours:
>    x, y, w, h = cv2.boundingRect(contour)
>    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
>    contours_dict.append({
>        'contour': contour,
>        'x': x,
>        'y': y,
>        'w': w,
>        'h': h,
>        'cx': x + (w / 2),
>        'cy': y + (h / 2)
>    })
>```

**cv2.boundingRect** 를통해 그러진 컨투어의 사각형 범위를 구한다. 또한

**cv2.rectangle** 를 통해 구한 사각형을 그린다.

**contours_dict.append**에서 컨투어의 x,y,w,h,줌심좌표 x,y를 저장한다.

이과정을 거치게되면 이미지가 이렇게 변한것을 볼수있다.

![image-20191109172550831](C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109172550831.png)

------

## 	분류하기 - 1

>```
>MIN_AREA = 80 #
>MIN_WIDTH, MIN_HEIGHT = 2, 8
>MIN_RATIO, MAX_RATIO = 0.25, 1.0
>possible_contours = []
>cnt = 0
>```

먼저 5개의 값을 사전에 정의해주자, MIN_AREA는 boundingRect의 최소넓이를 80이라서 설정했다.

MIN_WIDTH, MIN_HEIGHT는 최소넓이, 높이는 각각 2와 8로 설정했다

MIN_RATIO, MAX_RATIO는 boundingRect의 가로 대비 세로비율을 0.25와 1.0으로 설정했다.

>```
>for d in contours_dict:
>    area = d['w'] * d['h']
>    ratio = d['w'] / d['h']
>    
>    if area > MIN_AREA \
>    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
>    and MIN_RATIO < ratio < MAX_RATIO:
>        d['idx'] = cnt
>        cnt += 1
>        possible_contours.append(d)
>```

먼저 area에는 넓이를 저장하고 ratio에는 가로 대비 세로 비율을 저장한다.

5번쨰줄부터 위에 5개의 정의된값을 가지고 비교를 하기 시작한다.

조건을 모두 만족한다면 글자인 가능성이 있다고 다판단하고  possible_contours에 저장한다.

또한 인덱스의 값도 같이 저장한다.

>```
>temp_result = np.zeros((height, width, channel), dtype=np.uint8)
>
>for d in possible_contours: 
>    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
>```

가능성이 있다고 판단해  possible_contours에 저장한 값을 이용해 사각형을 그린다. 

결과는 이렇게 나왔다.

![image-20191109174622581](C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109174622581.png)

------

## 	분류하기 - 2

>```
>MAX_DIAG_MULTIPLYER = 5
>MAX_ANGLE_DIFF = 12.0 
>MAX_AREA_DIFF = 0.5 
>MAX_WIDTH_DIFF = 0.8 
>MAX_HEIGHT_DIFF = 0.2
>MIN_N_MATCHED = 3 
>```

시작하기전 6개의 파라미터값을 지정해주자.

### ●  MAX_DIAG_MULTIPLYER = 5

​	![image-20191109180602476](C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109180602476.png)

첫번쨰 컨투어 사각형과 두번째 컨투어 사각형의 각각 줌점의 길이를 5로 설정했다. 

### ● MAX_ANGLE_DIFF = 12.0 



![image-20191109181207249](C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109181207249.png) 



첫번째 컨투어와 두번째 컨투어의 중점을 기준으로 그림처럼 직각삼각형을 그린다음

빨간색으로 표시한 부분의 각도를 12로 설정

### ●  MAX_AREA_DIFF = 0.5 

![image-20191109181421931](C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109181421931.png) 

첫번째 컨투어와 두번째 컨투어 사각형 면적 차이를 0.5로 설정

### ●  MAX_WIDTH_DIFF = 0.8 

![image-20191109181749430](C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109181749430.png) 

첫번째 컨투어와 두번째 컨투어 사각형의 너비차이를 0.8로 설정

### ●  MAX_HEIGHT_DIFF = 0.2

![image-20191109181941041](C:\Users\ryury\AppData\Roaming\Typora\typora-user-images\image-20191109181941041.png)

첫번째 사각형과 두번째 사각형의 높이차이가 0.2로 설정

### ● MIN_N_MATCHED = 3 

위 파라미터값을 모두 만족한 그룹안에 컨투어가 3개로 가정.

사각형이 3개 미만일떄 그 그룹을 제외하기위해 사용한다.



>```
>def find_chars(contour_list):
>    matched_result_idx = [] 
>    for d1 in contour_list:
>        matched_contours_idx = []
>        for d2 in contour_list:
>            if d1['idx'] == d2['idx']:
>                continue
>            dx = abs(d1['cx'] - d2['cx'])
>            dy = abs(d1['cy'] - d2['cy'])
>            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
>            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))          
>            if dx == 0:
>                angle_diff = 90
>            else:
>                angle_diff = np.degrees(np.arctan(dy / dx))
>            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
>            width_diff = abs(d1['w'] - d2['w']) / d1['w']
>            height_diff = abs(d1['h'] - d2['h']) / d1['h']
>            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
>            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
>            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
>                matched_contours_idx.append(d2['idx'])
>        matched_contours_idx.append(d1['idx'])
>        if len(matched_contours_idx) < MIN_N_MATCHED:
>            continue
>        matched_result_idx.append(matched_contours_idx)
>        unmatched_contour_idx = []
>        for d4 in contour_list:
>            if d4['idx'] not in matched_contours_idx:
>        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)      
>        recursive_contour_list = find_chars(unmatched_contour)      
>        for idx in recursive_contour_list:
>            matched_result_idx.append(idx)
>        break
>    return matched_result_idx
>```

나중에 Recursive(재귀적) 방식으로 문자 후보를 찾기때문에 먼저 find_chars이름을 가진 함수를 만들어주자.

남는 결과물을 저장하기 위해 matched_result_idx란 이름을 가진 리스트를 생성해준다.

**for d1 in contour_list: ** 에는 첫번째 컨투어를 **for d2 in contour_list:**  에는 두번째 컨투어를 가져온다.

 만약 첫번째 컨투어와 두번째 컨투어가같다면 비교할 필요가 없기때문에 

> ```
> if d1['idx'] == d2['idx']:
>                 continue
> ```

를 통해 continue해준다.

**dx**와**dy**에는 밑변과 높이를 구해 저장해준다. 그리고

> ```
> diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
> ```

diagonal_length1에다가 첫번째 사각형의 대각선 길이를 구해 저장한다.

> ```
> distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
> ```

 **np.linalg.norm()**를 통해 백터 a와 백터b 사이의 거리를 구한다.

이를 이용해서 첫번째 사각형과 두번째 사각형 중점의 거리를 구해 distance에 저장한다.

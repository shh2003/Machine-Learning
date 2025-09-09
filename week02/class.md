# 슬라이딩 윈도우

슬라이딩 윈도우는 윈도우를 만들어 이동하며 시계열 데이터를 자르는 방법입니다.  
활동 인식(WISDM 등)에서 가장 안정적이고 정확도가 높은 방식으로 평가됩니다.

---

## 1. 개념
- `window_size`: 한 윈도우의 길이(고정 구간)
- `step`: 윈도우가 이동하는 간격 → **step < window_size**로 설정해 일부 겹치도록 만듦.
- 보통 **50% 겹침** → `step = window_size / 2`

---

## 2. 예시
원본 데이터 길이 = 10  
데이터 = `[1,2,3,4,5,6,7,8,9,10]`  
`window_size = 4`, `step = 2 (50% 겹침)`
윈도우1 → [1,2,3,4]
윈도우2 → [3,4,5,6]
윈도우3 → [5,6,7,8]
윈도우4 → [7,8,9,10]

### **시각화**
데이터 인덱스: 0 1 2 3 4 5 6 7 8 9
윈도우1: [0 1 2 3]
윈도우2: [2 3 4 5]
윈도우3: [4 5 6 7]
윈도우4: [6 7 8 9]


> 각 윈도우가 2개의 데이터를 서로 **겹치며** 이동 → 경계 구간을 놓치지 않음.

---

## 3. WISDM 데이터에 적용
WISDM은 약 **100Hz**로 수집되므로,  
- `window_size = 200` → **약 2초 길이의 데이터**  
- `step = 100` → **1초 간격으로 다음 윈도우 시작**

예시:
윈도우1: [0 ~ 199]
윈도우2: [100 ~ 299]
윈도우3: [200 ~ 399]
...

---

## 4. 코드 예시
```python
import numpy as np
import pandas as pd

# WISDM 데이터 예시
data = {
    'user': [1]*10,
    'label': [0]*10,
    'x': [1,2,3,4,5,6,7,8,9,10],
    'y': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'z': [9.7,9.6,9.5,9.4,9.3,9.2,9.1,9.0,8.9,8.8]
}
df = pd.DataFrame(data)

WINDOW_SIZE = 4
STEP = 2  # 50% 겹침

def make_windows_partial_overlap(df_part, window_size=WINDOW_SIZE, step=STEP):
    X_list, y_list = [], []
    arr = df_part[["x", "y", "z"]].values  # (T, 3)
    labels = df_part["label"].values       # 라벨

    for start in range(0, len(arr) - window_size + 1, step):
        seg = arr[start:start + window_size]  # (window_size, 3)
        X_list.append(seg.T)                  # CNN 입력형태 (3, window_size)

        # 윈도우 라벨: 다수결로 결정
        window_label = np.bincount(labels[start:start + window_size]).argmax()
        y_list.append(window_label)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

X, y = make_windows_partial_overlap(df)
print("X shape:", X.shape)  # (윈도우 개수, 3, window_size)
print("y:", y)
```
X shape: (4, 3, 4)
y: [0 0 0 0]


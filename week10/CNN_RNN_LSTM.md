RNN의 문제 -> 기울기 소실 성능을 높이고 싶으면 층을 쌓고 그러면 기울기가 점점작아져 0에 가까워짐.
LSTM 셀을 이용해 장기기역 그래도 기울기 소실을 생김 근데 너무 복잡함.
GRU 

---

# CNN 아키텍처 비교: ResNet, SENet, Xception

---

## 🔹 1. ResNet (Residual Network)

**핵심 개념:** **Skip Connection (잔차 연결, Shortcut Connection)**

- 기존의 CNN은 층이 깊어질수록 **기울기 소실(Gradient Vanishing)** 문제가 심함.  
- ResNet은 입력 `x`를 다음 층의 출력 `F(x)`에 **더하는(skip)** 구조로 해결.  

### 공식  
\[
y = F(x) + x
\]

이 구조 덕분에 모델은 “입력에서 바뀐 정도(Residual)”만 학습하면 되므로,  
**학습 안정성**이 높아지고 **깊은 네트워크** 구성이 가능함.

> **요약:** “ResNet = Skip Connection을 통한 깊은 학습 안정화”

---

## 🔹 2. SENet (Squeeze-and-Excitation Network)

**핵심 개념:** **Skip + 채널별 중요도 조절(Channel Attention)**  

- ResNet의 skip 구조를 유지하면서,  
- 추가로 “각 채널이 얼마나 중요한지”를 학습하는 **채널 주의(attention)** 구조를 추가.

### 동작 과정
1. **Squeeze:** Global Average Pooling으로 각 채널의 전체 정보를 하나의 값으로 압축.  
2. **Excitation:** 이 값들을 통해 채널별 중요도를 계산 (MLP 사용).  
3. **Reweight:** 입력 feature map의 각 채널에 이 가중치를 곱함.  

> **요약:** “SENet = ResNet + 채널별 주의(attention) 가중치 학습”  
> 중요 채널은 강화하고, 덜 중요한 채널은 약화시킴.

---

## 🔹 3. Xception (Extreme Inception)

**핵심 개념:** **Depthwise Separable Convolution → 채널별로 필터 분리 처리**

- Inception 구조를 더 극단적으로 확장한 버전.  
- 일반 합성곱(convolution)은 **공간 + 채널**을 동시에 처리하지만,  
  Xception은 두 단계를 **완전히 분리**함.

### 과정
1. **Depthwise Convolution:**  
   각 채널마다 **하나의 필터**를 적용 → 채널 간 혼합 없음.  
2. **Pointwise Convolution (1×1 Conv):**  
   채널들을 다시 합쳐서 새로운 조합으로 만듦.  

이 방식은 **연산량을 크게 줄이면서도 성능을 유지**함.

> **요약:** “Xception = 채널별 필터를 따로 적용 + 1×1 합성곱으로 통합”

---

## 🔸 한눈에 비교 정리

| 모델 | 핵심 개념 | 특징 |
|------|------------|------|
| **ResNet** | Skip Connection | 입력을 다음 층에 더함 → 기울기 소실 해결 |
| **SENet** | Skip + Channel Attention | 채널 중요도(가중치) 학습 |
| **Xception** | Depthwise + Pointwise Conv | 채널별 필터 분리로 효율적 계산 |





---



seq to seq model
- 번역기
- 

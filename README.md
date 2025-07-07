# 🏥 ONNX 기반 피부 질환 진단 시스템

H5 모델을 8비트 양자화하고 ONNX/TFLite로 변환하여 더 빠르고 효율적인 피부 질환 진단 시스템입니다.

## 📋 시스템 구성

```
📁 onnx_skin_diagnosis/
├── convert_h5_to_onnx.py          # ✅ H5 → ONNX 변환 스크립트
├── camera_h5_diagnosis.py         # 📱 H5 모델 카메라 진단 프로그램
├── camera_onnx_diagnosis.py       # 📱 기본 ONNX 카메라 진단 프로그램
├── camera_onnx_optimized.py       # 🚀 최적화된 ONNX 카메라 진단 프로그램
├── README.md                      # 📖 프로젝트 설명서
├── README_WINDOW.md               # 📖 Windows 버전 설명서
├── OPTIMIZED_option.md            # 📖 최적화 옵션 설명서
├── requirements.txt               # 📋 필요한 패키지 목록
├── 📁 model/                      # 🧠 모델 저장소
│   ├── skin_model.h5              # 원본 Keras 모델 (11.4MB)
│   ├── skin_model.onnx            # 변환된 ONNX 모델 (9.6MB)
│   ├── skin_model_quantized.tflite # 양자화 TFLite 모델 (2.9MB)
│   ├── skin_model_quantized_dynamic.onnx # 동적 양자화 ONNX (2.6MB)
│   └── skin_model_quantized_static.onnx  # 정적 양자화 ONNX (2.6MB)
└── 📁 captures/                   # 📸 진단 이미지 저장 폴더
```

## 🚀 설치 및 설정



<details>
<summary> # window </summary>
<div markdown="1">

### 1. 필요한 패키지 설치

```bash
# 기본 패키지
pip install -r requirements.txt

## 📂 사용 방법

### 1단계: 모델 변환

먼저 H5 모델을 ONNX/TFLite로 변환합니다:

```bash
python3 convert_h5_to_onnx.py
```

**변환 결과:**
- ✅ `skin_model.onnx` - ONNX 모델 생성
- ✅ `skin_model_quantized.tflite` - 8비트 양자화 TFLite 모델 생성
- ✅ `skin_model_quantized_dynamic.onxx` - 8비트 동적 양자화 onxx 모델 생성
- ✅ `skin_model_static_dynamic.onxx` - 8비트 정적 양자화 onxx 모델 생성
 

### 2단계: 진단 프로그램 실행

```bash
# ollama 실행 (터미널 하나 더 열어서 진행행)
ollama run gemma3:1b

```

```bash
# h5 기본
python camera_h5_diagnosis.py

# onxx 기본
python camera_onnx_diagnosis.py

# onxx runtime 적용
python camera_onnx_optimized.py

```
</div>
</details>


<details>
<summary> # Linux </summary>
<div markdown="1">


### 1. 필요한 패키지 설치

```bash
# 기본 패키지 설치
pip install -r requirements.txt

# Pillow 최신 버전 업그레이드 (텍스트 렌더링 오류 방지용)
pip install --upgrade pillow

# 리눅스(Ubuntu) 환경에서 한글 폰트가 깨질 경우 아래 명령어로 나눔글꼴 설치
sudo apt update
sudo apt install fonts-nanum

💡fonts-nanum은 한글을 깨지지 않게 표시하기 위해 필요합니다. 설치 후 코드에서 다음과 같이 경로를 설정하세요:
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# Ollama 설치 (Snap 기반)
sudo snap install ollama

## 📂 사용 방법

### 1단계: 모델 변환

먼저 H5 모델을 ONNX/TFLite로 변환합니다:


```bash
python3 convert_h5_to_onnx.py
```


**변환 결과:**
- ✅ `skin_model.onnx` - ONNX 모델 생성
- ✅ `skin_model_quantized.tflite` - 8비트 양자화 TFLite 모델 생성
- ✅ `skin_model_quantized_dynamic.onxx` - 8비트 동적 양자화 onxx 모델 생성
- ✅ `skin_model_static_dynamic.onxx` - 8비트 정적 양자화 onxx 모델 생성
 


### 2단계: 진단 프로그램 실행

```bash
# ollama 실행 (터미널 하나 더 열어서 진행행)
ollama run gemma3:1b

```

```bash
# h5 기본
python3 camera_h5_diagnosis.py

# onxx 기본
python3 camera_onnx_diagnosis.py

# onxx runtime 적용
python3 camera_onnx_optimized.py

```

</div>
</details>




## 🎯 모델 우선순위

프로그램은 다음 순서로 모델을 로드합니다:

1. **ONNX 모델** (최우선) - 가장 빠른 추론 속도
2. **TFLite 모델** (8비트 양자화) - 작은 파일 크기, 빠른 속도
3. **원본 H5 모델** (백업) - 변환 실패 시 사용

## 📊 진단 클래스 (7개)

1. **기저세포암** - 가장 흔한 피부암
2. **표피낭종** - 양성 낭종
3. **혈관종** - 혈관 증식 병변 
4. **비립종** - 작은 각질 주머니
5. **정상피부** - 건강한 피부
6. **편평세포암** - 두 번째 흔한 피부암
7. **사마귀** - HPV 감염

## 🎮 조작 방법

### 실시간 모드
- **카메라 화면**: 실시간 예측 결과 표시
- **모델 정보**: 사용 중인 모델 타입 (ONNX/TFLite/H5) 표시

### 진단 모드
- **'c' 키**: 5초간 연속 촬영하여 정확한 진단 시작
- **진단 과정**: 5번 촬영 → 결과 일치 확인 → 최종 진단
- **AI 조언**: Ollama Gemma3를 통한 개인맞춤 건강 조언

### 기타
- **'q' 키**: 프로그램 종료

## 🔧 성능 최적화

### 모델 크기 비교
- **원본 H5**: ~50MB
- **ONNX**: ~45MB (10% 감소)
- **TFLite 양자화**: ~12MB (75% 감소)

### 추론 속도
- **ONNX**: 가장 빠름 (CPU 최적화)
- **TFLite**: 빠름 (메모리 효율적)
- **H5**: 보통 (TensorFlow 오버헤드)

## 🛠️ 문제 해결

### 모델 로드 실패
```bash
❌ ONNX 모델 로드 실패: ...
❌ TFLite 모델 로드 실패: ...
✅ 원본 H5 모델을 사용합니다.
```
→ 변환 과정을 다시 실행하세요.

### 카메라 접근 실패
```bash
❌ 카메라를 열 수 없습니다.
```
→ 다른 프로그램이 카메라를 사용 중인지 확인하세요.

### Ollama 연결 실패
```bash
Ollama 모델을 호출하는 중 오류가 발생했습니다...
```
→ `ollama run gemma3` 명령으로 모델을 실행하세요.

## 📈 추가 기능

### 이미지 저장
- 진단 시 모든 캡처 이미지가 `captures/` 폴더에 저장됩니다
- 파일명: `capture_YYYYMMDD_HHMMSS_N.png`

### AI 건강 조언
- Ollama Gemma3 모델을 통한 실시간 건강 조언 제공
- 간결하고 실용적인 정보 제공 (200자 내외)

## ⚠️ 주의사항

1. **의학적 조언 아님**: 이 시스템은 참고용이며, 정확한 진단은 전문의와 상담하세요.
2. **조명 조건**: 충분한 조명에서 사용하세요.
3. **카메라 위치**: 진단 부위를 화면 중앙에 위치시키세요.
4. **정확성**: 5번 연속 촬영에서 같은 결과가 나와야 신뢰할 수 있습니다.

## 📞 기술 지원

문제가 발생하면 다음 사항을 확인해주세요:

1. **Python 버전**: 3.8 이상 권장
2. **패키지 버전**: 최신 버전 사용 권장
3. **모델 파일**: 변환된 모델 파일이 정상적으로 생성되었는지 확인
4. **하드웨어**: 충분한 RAM과 CPU 성능 필요

---

**© 2025 ONNX 기반 피부 질환 진단 시스템** 

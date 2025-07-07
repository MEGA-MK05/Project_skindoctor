import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import time
import os
import ollama
import onnxruntime as ort

# --- 설정 ---
CAPTURE_INTERVAL = 1  # 캡처 간격 (초)
CAPTURE_COUNT = 5     # 캡처 횟수
CAPTURE_FOLDER = "captures" # 캡처 이미지 저장 폴더
OLLAMA_MODEL = "gemma3" # 사용할 Ollama 모델

DISPLAY_UPDATE_INTERVAL_MS = 400 # 화면에 표시되는 예측 결과 업데이트 주기 (밀리초)

# 모델 경로 설정
ONNX_MODEL_PATH = "./model/skin_model.onnx"
TFLITE_MODEL_PATH = "./model/skin_model_quantized.tflite"

# --- 클래스 및 모델 설정 ---
# 클래스명 (7개 클래스)
class_names_kr = [
    '기저세포암',
    '표피낭종',
    '혈관종',
    '비립종',
    '정상피부',
    '편평세포암',
    '사마귀'
]

# --- ONNX 모델 클래스 ---
class ONNXModel:
    def __init__(self, model_path):
        """ONNX 모델 로드"""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"ONNX 모델 로드 성공: {model_path}")
    
    def predict(self, input_data):
        """ONNX 모델 예측"""
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return result[0]

# --- TFLite 모델 클래스 ---
class TFLiteModel:
    def __init__(self, model_path):
        """TFLite 모델 로드"""
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 입력 및 출력 텐서 정보
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"TFLite 모델 로드 성공: {model_path}")
    
    def predict(self, input_data):
        """TFLite 모델 예측"""
        # 입력 데이터 설정
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # 추론 실행
        self.interpreter.invoke()
        
        # 출력 데이터 가져오기
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data


def get_solution_from_gemma(disease_name):
    """
    로컬 Ollama의 Gemma3 모델에게 피부 질환에 대한 간단한 가이드 요청.
    응답은 사용자가 이해하기 쉽게 5단계로 요약되며, 200자 내외로 제한됨.
    """

    prompt = f"""
당신은 피부 건강 전문 AI 어시스턴트입니다. 아래 피부 질환에 대해 200자 내외로 간결하게 안내해주세요.

피부 질환명: {disease_name}

아래 형식에 따라 한국어로 정확하고 간단명료하게 작성하세요:

1. 질환 설명: 일반인이 이해할 수 있도록 간단히
2. 즉시 조치사항: 응급성 여부 포함
3. 가정 관리 방법: 손쉽게 실천 가능한 팁
4. 전문 치료 방법: 병원에서 받을 수 있는 치료
5. 주의사항: 재발, 감염, 자가 치료 경고 등


각 항목은 줄바꿈으로 구분하여 제시하세요.
답변은 200자 내외로 간결하게 작성해주세요.
    """.strip()

    if disease_name in sample_responses:
        return sample_responses[disease_name]
        

    print(f"\n[{OLLAMA_MODEL} 모델에게 조언을 요청합니다... 잠시만 기다려 주세요.]")

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()

    except Exception as e:
        return f"[오류] Ollama 모델을 호출하는 중 문제가 발생했습니다: {e}\nOllama 서버가 실행 중인지 확인하세요."


# --- 모델 초기화 함수 ---
def initialize_model():
    """사용 가능한 모델을 초기화합니다."""
    model = None
    model_type = None
    
    # 1. ONNX 모델 시도
    if os.path.exists(ONNX_MODEL_PATH):
        try:
            model = ONNXModel(ONNX_MODEL_PATH)
            model_type = "ONNX"
            print("ONNX 모델을 사용합니다.")
        except Exception as e:
            print(f"ONNX 모델 로드 실패: {e}")
    
    # 2. TFLite 모델 시도 (ONNX 실패 시)
    if model is None and os.path.exists(TFLITE_MODEL_PATH):
        try:
            model = TFLiteModel(TFLITE_MODEL_PATH)
            model_type = "TFLite"
            print("TFLite 모델을 사용합니다.")
        except Exception as e:
            print(f"TFLite 모델 로드 실패: {e}")
    
    # 3. 원본 H5 모델 시도 (둘 다 실패 시)
    if model is None:
        try:
            import tensorflow as tf
            from tensorflow import keras
            h5_model_path = "C:/Users/kccistc/project/onnx_skin_diagnosis/model/skin_model.h5"
            model = keras.models.load_model(h5_model_path)
            model_type = "H5"
            print("원본 H5 모델을 사용합니다.")
        except Exception as e:
            print(f"H5 모델 로드 실패: {e}")
    
    return model, model_type

# --- 메인 로직 ---
def main():
    print("ONNX Skin Diagnosis System")
    print("=" * 50)
    
    # 캡처 폴더 생성
    if not os.path.exists(CAPTURE_FOLDER):
        os.makedirs(CAPTURE_FOLDER)
    
    # 모델 초기화
    model, model_type = initialize_model()
    if model is None:
        print("사용 가능한 모델이 없습니다.")
        print("먼저 convert_h5_to_onnx.py를 실행하여 모델을 변환하세요.")
        return
    
    print(f"사용 중인 모델: {model_type}")
    
    # 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        print(f"오류: 폰트 파일을 찾을 수 없습니다: {font_path}. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    # 화면 표시 업데이트를 위한 변수
    last_display_update_time = time.time()
    current_display_label = ""

    # 카메라 설정
    cap = cv2.VideoCapture(1) # 외부 웹캠
    if not cap.isOpened():
        cap = cv2.VideoCapture(0) # 내장 웹캠
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

    print("카메라가 준비되었습니다.")
    print("화면을 보며 진단할 부위를 중앙에 위치시키세요.")
    print("키보드 'c'를 누르면 5초간 연속으로 촬영하여 진단합니다.")
    print("키보드 'q'를 누르면 프로그램을 종료합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 중앙 1:1 영역 crop
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        crop_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

        # --- 실시간 예측 ---
        # 이미지 전처리
        img_array = cv2.resize(crop_frame, (96, 96))
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
        img_array = img_array.astype(np.float32) / 255.0  # 정규화

        # 모델 타입에 따른 예측
        if model_type == "H5":
            predictions = model.predict(img_array, verbose=0)
        else:
            predictions = model.predict(img_array)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]

        # 결과 텍스트 생성
        current_label = f"{class_names_kr[predicted_class_idx]} ({confidence*100:.1f}%)"
        
        # 화면 표시 업데이트 주기 제어
        current_time = time.time()
        if (current_time - last_display_update_time) * 1000 >= DISPLAY_UPDATE_INTERVAL_MS:
            current_display_label = current_label
            last_display_update_time = current_time

        # 화면에 표시 (Pillow 사용)
        img_pil = Image.fromarray(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), f"실시간 예측 ({model_type}):", font=font, fill=(0, 255, 0))
        draw.text((10, 35), current_display_label, font=font, fill=(0, 255, 0))
        display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('ONNX Skin Disease Diagnosis', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # --- 'c' 키를 눌러 연속 캡처 및 진단 ---
        if key == ord('c'):
            # 화면을 검게 만들고 "의사의 답변 준비중..." 메시지 표시
            black_screen = np.zeros_like(display_frame)
            
            # Pillow를 사용하여 텍스트 추가
            img_pil_black = Image.fromarray(cv2.cvtColor(black_screen, cv2.COLOR_BGR2RGB))
            draw_black = ImageDraw.Draw(img_pil_black)
            
            text = "의사의 답변 준비중..."
            
            # 텍스트 크기 계산
            try:
                # Pillow 10.0.0 이상
                text_bbox = draw_black.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                # 이전 버전의 Pillow
                text_width, text_height = draw_black.textsize(text, font=font)

            text_x = (black_screen.shape[1] - text_width) // 2
            text_y = (black_screen.shape[0] - text_height) // 2
            
            draw_black.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
            
            # OpenCV 형식으로 다시 변환하여 표시
            black_screen_with_text = cv2.cvtColor(np.array(img_pil_black), cv2.COLOR_RGB2BGR)
            cv2.imshow('ONNX Skin Disease Diagnosis', black_screen_with_text)
            cv2.waitKey(1) # 화면을 즉시 업데이트

            print("\n" + "="*40)
            print(f"진단을 시작합니다. {CAPTURE_COUNT}초 동안 {CAPTURE_COUNT}번 촬영합니다.")
            print("="*40)
            
            captured_classes = []
            
            for i in range(CAPTURE_COUNT):
                time.sleep(CAPTURE_INTERVAL)
                
                # 현재 프레임(crop_frame)으로 예측
                current_img_array = cv2.resize(crop_frame, (96, 96))
                current_img_array = np.expand_dims(current_img_array, axis=0)
                current_img_array = current_img_array.astype(np.float32) / 255.0

                # 모델 타입에 따른 예측
                if model_type == "H5":
                    current_predictions = model.predict(current_img_array, verbose=0)
                else:
                    current_predictions = model.predict(current_img_array)
                
                current_predicted_idx = np.argmax(current_predictions[0])
                
                predicted_name = class_names_kr[current_predicted_idx]
                captured_classes.append(predicted_name)
                
                # 캡처 이미지 저장
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                capture_path = os.path.join(CAPTURE_FOLDER, f"capture_{timestamp}_{i+1}.png")
                cv2.imwrite(capture_path, crop_frame)
                
                print(f"촬영 {i+1}/5... 예측: {predicted_name} (이미지 저장: {capture_path})")

            # --- 최종 진단 ---
            print("\n" + "-"*40)
            if len(set(captured_classes)) == 1:
                final_diagnosis = captured_classes[0]
                print(f"최종 진단 결과: **{final_diagnosis}**")
                print(f"사용 모델: {model_type}")
                print("-"*40)
                
                # Gemma3 해결책 요청
                solution = get_solution_from_gemma(final_diagnosis)
                print("\n[Ollama Gemma3의 건강 조언]")
                print(solution)
                print("\n(주의: 이 정보는 참고용이며, 정확한 진단과 치료를 위해 반드시 전문 의료기관을 방문하세요.)")
                
            else:
                print("진단 실패: 예측 결과가 일치하지 않습니다.")
                print(f"지난 {CAPTURE_COUNT}번의 예측: {captured_classes}")
            
            print("="*40)
            print("\n다시 진단하려면 'c'를, 종료하려면 'q'를 누르세요.")

        # --- 'q' 키를 눌러 종료 ---
        elif key == ord('q'):
            print("프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
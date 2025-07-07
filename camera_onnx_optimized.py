import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import time
import os
import platform
import ollama
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import psutil
import threading
import queue

# --- 설정 ---
CAPTURE_INTERVAL = 1  # 캡처 간격 (초)
CAPTURE_COUNT = 5     # 캡처 횟수
CAPTURE_FOLDER = "captures" # 캡처 이미지 저장 폴더
OLLAMA_MODEL = "gemma3:1b" # 사용할 Ollama 모델

# 카메라 설정 (라즈베리 파이 5 최적화를 위해 조정 가능)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

PREDICTION_SMOOTHING_WINDOW_SIZE = 5 # 예측 결과 스무딩을 위한 프레임 수 (5~10 정도 권장)
DISPLAY_UPDATE_INTERVAL_MS = 400 # 화면에 표시되는 예측 결과 업데이트 주기 (밀리초)

# 모델 경로 설정

ONNX_MODEL_PATH = "./model/skin_model.onnx"
ONNX_OPTIMIZED_PATH = "./model/skin_model_quantized.onnx" # 자의적으로 바꿔서 최적화 모델 경로 설정
ONNX_QUANTIZED_PATH = "./model/skin_model_quantized.onnx"
TFLITE_MODEL_PATH = "./model/skin_model_quantized.tflite"

# --- OS별 설정 함수 ---
def get_system_font_path():
    """OS별 시스템 폰트 경로 반환"""
    system = platform.system()
    
    if system == "Windows":
        return "C:/Windows/Fonts/malgun.ttf"
    elif system == "Linux":
        # Ubuntu/Debian 계열
        linux_fonts = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/NanumGothic.ttf",  # Arch Linux
            "/System/Library/Fonts/Helvetica.ttc"    # macOS backup
        ]
        for font in linux_fonts:
            if os.path.exists(font):
                return font
    elif system == "Darwin":  # macOS
        return "/System/Library/Fonts/AppleGothic.ttf"
    
    # 기본값 (폰트가 없는 경우)
    return None

def get_backup_model_path():
    """백업 H5 모델 경로 반환 (OS 무관)"""
    possible_paths = [
        "./model/jaehong_skin_model.h5",  # 상대 경로 (우선)
        "../pth/jaehong_skin_model.h5",   # 상위 폴더
        "./jaehong_skin_model.h5",       # 현재 폴더
        "C:/Users/kccistc/project/pth/jaehong_skin_model.h5",  # Windows 절대 경로
        "/home/kccistc/project/pth/jaehong_skin_model.h5"       # Linux 절대 경로
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

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

# --- 최적화된 ONNX 모델 클래스 ---
class OptimizedONNXModel:
    def __init__(self, model_path, optimization_level="all", use_gpu=False):
        """
        최적화된 ONNX 모델 로드
        
        Args:
            model_path: 모델 경로
            optimization_level: 최적화 레벨 ("disable", "basic", "extended", "all")
            use_gpu: GPU 사용 여부
        """
        self.model_path = model_path
        self.optimization_level = optimization_level
        self.use_gpu = use_gpu
        
        # 세션 옵션 설정
        self.session_options = ort.SessionOptions()
        
        # 최적화 레벨 설정
        if optimization_level == "disable":
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif optimization_level == "basic":
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif optimization_level == "extended":
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:  # "all"
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 병렬 처리 설정
        cpu_count = psutil.cpu_count(logical=False)
        self.session_options.intra_op_num_threads = cpu_count
        self.session_options.inter_op_num_threads = cpu_count
        
        # 메모리 패턴 최적화
        self.session_options.enable_mem_pattern = True
        self.session_options.enable_cpu_mem_arena = True
        
        # 실행 제공자 설정
        providers = self._get_providers()
        
        try:
            # ONNX 세션 생성
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=self.session_options,
                providers=providers
            )
            
            # 입출력 정보
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # 모델 정보 출력
            print(f"✅ 최적화된 ONNX 모델 로드 성공: {model_path}")
            print(f"   🔧 최적화 레벨: {optimization_level}")
            print(f"   🧵 Intra-op threads: {self.session_options.intra_op_num_threads}")
            print(f"   🧵 Inter-op threads: {self.session_options.inter_op_num_threads}")
            print(f"   💻 사용 중인 Providers: {self.session.get_providers()}")
            
        except Exception as e:
            print(f"❌ 최적화된 ONNX 모델 로드 실패: {e}")
            raise
    
    def _get_providers(self):
        """사용 가능한 실행 제공자 반환"""
        providers = []
        available_providers = ort.get_available_providers()
        system = platform.system()
        
        # GPU 사용 시도
        if self.use_gpu:
            # DirectML (Windows만)
            if system == "Windows" and 'DmlExecutionProvider' in available_providers:
                providers.append('DmlExecutionProvider')
                print("🎮 DirectML Provider 사용 (Windows GPU)")
            
            # CUDA (NVIDIA - 모든 OS)
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                print("🚀 CUDA Provider 사용 (NVIDIA GPU)")
            
            # ROCm (AMD - Linux)
            if system == "Linux" and 'ROCMExecutionProvider' in available_providers:
                providers.append('ROCMExecutionProvider')
                print("🔥 ROCm Provider 사용 (AMD GPU)")
            
            # OpenVINO (Intel - 모든 OS)
            if 'OpenVINOExecutionProvider' in available_providers:
                providers.append('OpenVINOExecutionProvider')
                print("⚡ OpenVINO Provider 사용 (Intel GPU)")
            
            # TensorRT (NVIDIA - Linux 주로)
            if 'TensorrtExecutionProvider' in available_providers:
                providers.append('TensorrtExecutionProvider')
                print("🏎️ TensorRT Provider 사용 (NVIDIA GPU)")
        
        # CPU는 항상 백업으로 추가
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def predict(self, input_data):
        """최적화된 예측"""
        try:
            result = self.session.run([self.output_name], {self.input_name: input_data})
            return result[0]
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return None

# --- 성능 벤치마킹 함수 ---
def benchmark_model(model, test_data, num_runs=100):
    """모델 성능 벤치마킹"""
    print(f"🏃 성능 벤치마킹 시작 ({num_runs}회 실행)...")
    
    # 워밍업
    for _ in range(10):
        model.predict(test_data)
    
    # 실제 벤치마킹
    start_time = time.time()
    for _ in range(num_runs):
        model.predict(test_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    fps = 1 / (avg_time / 1000)
    
    print(f"   ⚡ 평균 추론 시간: {avg_time:.2f}ms")
    print(f"   🎯 초당 프레임: {fps:.1f} FPS")
    
    return avg_time, fps

# --- 비동기 예측 클래스 ---
class AsyncPredictor:
    def __init__(self, model):
        self.model = model
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.prediction_thread = threading.Thread(target=self._prediction_worker)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        self.last_prediction = None
    
    def _prediction_worker(self):
        """백그라운드 예측 작업자"""
        while True:
            try:
                input_data = self.input_queue.get(timeout=0.1)
                result = self.model.predict(input_data)
                
                # 결과 큐가 가득 찬 경우 이전 결과 제거
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put(result)
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 비동기 예측 오류: {e}")
    
    def predict_async(self, input_data):
        """비동기 예측 요청"""
        # 입력 큐가 가득 찬 경우 이전 요청 제거
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.input_queue.put_nowait(input_data)
        except queue.Full:
            pass
    
    def get_prediction(self):
        """예측 결과 가져오기"""
        try:
            result = self.output_queue.get_nowait()
            self.last_prediction = result
            return result
        except queue.Empty:
            return self.last_prediction

# --- 모델 초기화 함수 ---
def initialize_optimized_model():
    """최적화된 모델 초기화"""
    print("🚀 최적화된 ONNX 모델 초기화 시작...")
    
    # 1. 원본 ONNX 모델 확인
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ 원본 ONNX 모델이 없습니다: {ONNX_MODEL_PATH}")
        print("💡 먼저 convert_h5_to_onnx.py를 실행하여 모델을 변환하세요.")
        return None, None
    
    # 2. 최적화된 모델 로드 시도 (우선순위대로)
    models_to_try = [
        (ONNX_QUANTIZED_PATH, "동적 양자화 ONNX"),
        (ONNX_MODEL_PATH, "기본 ONNX")
    ]
    
    for model_path, description in models_to_try:
        if os.path.exists(model_path):
            try:
                # GPU 사용 가능 여부 확인
                use_gpu = len([p for p in ort.get_available_providers() 
                              if p in ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']]) > 0
                
                model = OptimizedONNXModel(
                    model_path, 
                    optimization_level="all",
                    use_gpu=use_gpu
                )
                
                print(f"✅ {description} 모델 로드 성공")
                return model, description
                
            except Exception as e:
                print(f"❌ {description} 모델 로드 실패: {e}")
                continue
    
    # 4. 백업으로 H5 모델 시도
    try:
        import tensorflow as tf
        from tensorflow import keras
        h5_model_path = get_backup_model_path()
        if h5_model_path and os.path.exists(h5_model_path):
            model = keras.models.load_model(h5_model_path)
            print(f"✅ 백업 H5 모델 로드 성공: {h5_model_path}")
            return model, "H5 백업"
        else:
            print("❌ 백업 H5 모델 파일을 찾을 수 없습니다")
    except Exception as e:
        print(f"❌ 백업 H5 모델 로드 실패: {e}")
    
    return None, None

# --- Ollama Gemma3 함수 ---
# 클래스 이름 (한글 → 영어 변환용, 또는 UI 표기용)
class_names_kr = [
    '기저세포암',
    '표피낭종',
    '혈관종',
    '비립종',
    '정상피부',
    '편평세포암',
    '사마귀'
]

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

    print(f"\n[{OLLAMA_MODEL} 모델에게 조언을 요청합니다... 잠시만 기다려 주세요.]")

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()

    except Exception as e:
        return f"[오류] Ollama 모델을 호출하는 중 문제가 발생했습니다: {e}\nOllama 서버가 실행 중인지 확인하세요."

# --- 메인 로직 ---
def main():
    print("최적화된 ONNX 기반 피부 질환 진단 시스템")
    print("=" * 55)
    
    # 시스템 정보 출력
    print(f"💻 CPU 코어: {psutil.cpu_count(logical=False)} 물리 / {psutil.cpu_count(logical=True)} 논리")
    print(f"🧠 사용 가능한 메모리: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"⚡ 사용 가능한 ONNX Providers: {ort.get_available_providers()}")
    print("=" * 55)
    
    # 캡처 폴더 생성
    if not os.path.exists(CAPTURE_FOLDER):
        os.makedirs(CAPTURE_FOLDER)
    
    # 최적화된 모델 초기화
    model, model_type = initialize_optimized_model()
    if model is None:
        print("❌ 사용 가능한 모델이 없습니다.")
        return
    
    print(f"📊 사용 중인 모델: {model_type}")
    
    # 성능 벤치마킹 (ONNX 모델의 경우)
    if model_type and ("ONNX" in model_type or "최적화" in model_type):
        test_data = np.random.random((1, 96, 96, 3)).astype(np.float32)
        avg_time, fps = benchmark_model(model, test_data)
        
        # 비동기 예측기 초기화
        async_predictor = AsyncPredictor(model)
        use_async = True
        print("🔄 비동기 예측 모드 활성화")
    else:
        use_async = False
        print("⏳ 동기 예측 모드 사용")
    
    # 폰트 설정 (OS별 대응)
    font_path = get_system_font_path()
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 20)
            small_font = ImageFont.truetype(font_path, 14)
            print(f"✅ 폰트 로드 성공: {font_path}")
        else:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            print("⚠️ 시스템 폰트를 찾을 수 없어 기본 폰트를 사용합니다")
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        print("⚠️ 폰트 로드 실패, 기본 폰트를 사용합니다")

    # 카메라 설정 (여러 인덱스 시도)
    cap = None
    camera_indices = [0, 1, 2]  # 리눅스에서는 보통 0, 윈도우에서는 1이 주로 사용됨
    
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"✅ 카메라 {idx}번 연결 성공")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        print("❌ 사용 가능한 카메라를 찾을 수 없습니다.")
        print("💡 다음을 확인해보세요:")
        print("   - 카메라가 연결되어 있는지")
        print("   - 다른 프로그램에서 카메라를 사용 중인지")
        print("   - 카메라 권한이 있는지")
        return

    # 카메라 해상도 최적화
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    print("📷 카메라가 준비되었습니다.")
    print("화면을 보며 진단할 부위를 중앙에 위치시키세요.")
    print("키보드 'c'를 누르면 5초간 연속으로 촬영하여 진단합니다.")
    print("키보드 'q'를 누르면 프로그램을 종료합니다.")
    print("키보드 'b'를 누르면 벤치마킹을 다시 실행합니다.")

    # 성능 측정 변수
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0 # FPS 값을 저장할 변수 초기화
    last_display_update_time = time.time() # 마지막 디스플레이 업데이트 시간
    current_display_label = ""

    # 예측 스무딩을 위한 리스트
    recent_predictions = []
    
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

        # 이미지 전처리
        img_array = cv2.resize(crop_frame, (96, 96))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype(np.float32) / 255.0

        # 예측 수행
        if use_async:
            # 비동기 예측
            async_predictor.predict_async(img_array)
            predictions = async_predictor.get_prediction()
            
            if predictions is not None:
                current_predicted_class_idx = np.argmax(predictions[0])
                current_confidence = predictions[0][current_predicted_class_idx]
            else:
                current_predicted_class_idx = 0
                current_confidence = 0.0
        else:
            # 동기 예측
            if model_type and ("ONNX" in model_type or "최적화" in model_type):
                predictions = model.predict(img_array)
                if predictions is not None:
                    current_predicted_class_idx = np.argmax(predictions[0])
                    current_confidence = predictions[0][current_predicted_class_idx]
                else:
                    current_predicted_class_idx = 0
                    current_confidence = 0.0
            else:
                predictions = model.predict(img_array, verbose=0)
                current_predicted_class_idx = np.argmax(predictions[0])
                current_confidence = predictions[0][current_predicted_class_idx]

        # 예측 결과 스무딩
        recent_predictions.append((current_predicted_class_idx, current_confidence))
        if len(recent_predictions) > PREDICTION_SMOOTHING_WINDOW_SIZE:
            recent_predictions.pop(0) # 가장 오래된 예측 제거

        # 스무딩된 예측 결과 계산
        if recent_predictions:
            # 각 클래스별로 등장 횟수 계산
            class_counts = {}
            for idx, _ in recent_predictions:
                class_counts[idx] = class_counts.get(idx, 0) + 1
            
            # 가장 많이 등장한 클래스 선택
            smoothed_predicted_class_idx = max(class_counts, key=class_counts.get)
            
            # 해당 클래스의 평균 신뢰도 계산
            smoothed_confidence_sum = sum([conf for idx, conf in recent_predictions if idx == smoothed_predicted_class_idx])
            smoothed_confidence_count = class_counts[smoothed_predicted_class_idx]
            smoothed_confidence = smoothed_confidence_sum / smoothed_confidence_count
        else:
            smoothed_predicted_class_idx = 0
            smoothed_confidence = 0.0

        # 화면 표시 업데이트 주기 제어
        current_time = time.time()
        if (current_time - last_display_update_time) * 1000 >= DISPLAY_UPDATE_INTERVAL_MS:
            current_display_label = f"{class_names_kr[smoothed_predicted_class_idx]} ({smoothed_confidence*100:.1f}%)"
            last_display_update_time = current_time

        # FPS 계산
        frame_count += 1
        if frame_count % 30 == 0:
            fps_end_time = time.time()
            current_fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time

        # 화면에 표시
        img_pil = Image.fromarray(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 메인 정보
        draw.text((10, 10), f"🔬 실시간 예측 ({model_type}):", font=font, fill=(0, 255, 0))
        draw.text((10, 35), current_display_label, font=font, fill=(0, 255, 0))
        
        # 성능 정보 (FPS는 항상 표시)
        draw.text((10, 65), f"⚡ FPS: {current_fps:.1f}", font=small_font, fill=(255, 255, 0))
        
        # 사용 중인 Provider 정보 (ONNX 모델의 경우)
        if hasattr(model, 'session'):
            provider_info = model.session.get_providers()[0]
            draw.text((10, 85), f"💻 Provider: {provider_info.replace('ExecutionProvider', '')}", font=small_font, fill=(255, 255, 0))
        
        display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow('최적화된 ONNX 피부 진단', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 'b' 키로 벤치마킹 실행
        if key == ord('b') and ("ONNX" in model_type or "최적화" in model_type):
            print("\n" + "="*50)
            print("🏃 실시간 벤치마킹 실행")
            print("="*50)
            avg_time, fps = benchmark_model(model, img_array)

        # 'c' 키로 진단 실행
        elif key == ord('c'):
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
            cv2.imshow('최적화된 ONNX 피부 진단', black_screen_with_text)
            cv2.waitKey(1) # 화면을 즉시 업데이트

            # 진단 로직 (기존과 동일)
            print("\n" + "="*40)
            print(f"진단을 시작합니다. {CAPTURE_COUNT}초 동안 {CAPTURE_COUNT}번 촬영합니다.")
            print("="*40)
            
            captured_classes = []
            
            for i in range(CAPTURE_COUNT):
                time.sleep(CAPTURE_INTERVAL)
                
                # 현재 프레임으로 예측
                if "ONNX" in model_type or "최적화" in model_type:
                    current_predictions = model.predict(img_array)
                    if current_predictions is not None:
                        current_predicted_idx = np.argmax(current_predictions[0])
                    else:
                        current_predicted_idx = 0
                else:
                    current_predictions = model.predict(img_array, verbose=0)
                    current_predicted_idx = np.argmax(current_predictions[0])
                
                predicted_name = class_names_kr[current_predicted_idx]
                captured_classes.append(predicted_name)
                
                # 캡처 이미지 저장
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                capture_path = os.path.join(CAPTURE_FOLDER, f"capture_{timestamp}_{i+1}.png")
                cv2.imwrite(capture_path, crop_frame)
                
                print(f"촬영 {i+1}/5... 예측: {predicted_name}")

            # 최종 진단
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
            print("\n다시 진단하려면 'c'를, 벤치마킹은 'b'를, 종료하려면 'q'를 누르세요.")

        # 'q' 키로 종료
        elif key == ord('q'):
            print("프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
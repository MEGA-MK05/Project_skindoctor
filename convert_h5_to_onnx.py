import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import numpy as np
import os
import shutil
# ONNX 양자화를 위한 추가 import
try:
    from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantType
    ONNX_QUANTIZATION_AVAILABLE = True
except ImportError:
    ONNX_QUANTIZATION_AVAILABLE = False
    print("경고: onnxruntime.quantization을 사용할 수 없습니다. 'pip install onnxruntime' 설치 후 재시도하세요.")

def convert_h5_to_onnx(h5_model_path, onnx_output_path):
    """
    H5 모델을 ONNX 형식으로 변환합니다.
    """
    print("=" * 60)
    print("H5 모델 → ONNX 변환 시작")
    print("=" * 60)
    
    # 1. H5 모델 로드
    print(f"1. H5 모델 로드: {h5_model_path}")
    try:
        model = keras.models.load_model(h5_model_path)
        print(f"   [성공] 모델 로드 성공")
        print(f"   [정보] 모델 구조: {model.input_shape} -> {model.output_shape}")
    except Exception as e:
        print(f"   [실패] 모델 로드 실패: {e}")
        return False
    
    # 2. ONNX 변환
    print("2. ONNX 변환 중...")
    temp_saved_model_path = "./temp_saved_model"
    
    try:
        onnx_model = None
        
        # 방법 1: 직접 Keras 모델에서 ONNX 변환 시도
        try:
            print("   - 방법 1: 직접 Keras → ONNX 변환 시도...")
            
            # 모델 준비
            dummy_input = tf.zeros((1, 96, 96, 3), dtype=tf.float32)
            model(dummy_input)  # 모델 호출하여 그래프 빌드
            
            # 입력 서명 설정
            input_signature = [tf.TensorSpec(shape=[None, 96, 96, 3], dtype=tf.float32)]
            
            # ONNX 변환
            onnx_model, _ = tf2onnx.convert.from_keras(
                model, 
                input_signature=input_signature,
                opset=11
            )
            print("   - 방법 1 성공: 직접 변환 완료")
            
        except Exception as e1:
            print(f"   - 방법 1 실패: {e1}")
            
            # 방법 2: SavedModel 경유 변환 시도
            try:
                print("   - 방법 2: SavedModel 경유 변환 시도...")
                
                # 모델에 call 메서드 명시적 정의
                @tf.function
                def model_func(x):
                    return model(x)
                
                # 구체적인 입력으로 트레이스
                concrete_func = model_func.get_concrete_function(
                    tf.TensorSpec(shape=[1, 96, 96, 3], dtype=tf.float32)
                )
                
                # SavedModel로 저장
                tf.saved_model.save(model, temp_saved_model_path, signatures=concrete_func)
                print("   - SavedModel 변환 완료")
                
                # SavedModel을 ONNX로 변환
                onnx_model, _ = tf2onnx.convert.from_saved_model(temp_saved_model_path)
                print("   - 방법 2 성공: SavedModel 경유 변환 완료")
                
            except Exception as e2:
                print(f"   - 방법 2 실패: {e2}")
                
                # 방법 3: 모델 재구성 후 변환 시도
                try:
                    print("   - 방법 3: 모델 재구성 후 변환 시도...")
                    
                    # 모델 구조 분석
                    input_shape = model.input_shape[1:]  # (96, 96, 3)
                    
                    # 새로운 입력 레이어 생성
                    new_input = tf.keras.layers.Input(shape=input_shape, name='input')
                    
                    # 기존 모델의 레이어들을 순차적으로 적용
                    x = new_input
                    for i, layer in enumerate(model.layers):
                        # 레이어 이름 설정
                        layer_name = f"{layer.__class__.__name__}_{i}"
                        
                        # 레이어 복사 및 적용
                        if hasattr(layer, 'get_config'):
                            layer_config = layer.get_config()
                            layer_config['name'] = layer_name
                            
                            # 레이어 재생성
                            new_layer = layer.__class__.from_config(layer_config)
                            new_layer.set_weights(layer.get_weights())
                            x = new_layer(x)
                        else:
                            # 설정이 없는 레이어의 경우 직접 적용
                            x = layer(x)
                    
                    # 새로운 함수형 모델 생성
                    functional_model = tf.keras.Model(inputs=new_input, outputs=x, name='functional_model')
                    
                    print("   - 모델 재구성 완료")
                    
                    # 재구성된 모델로 ONNX 변환
                    input_signature = [tf.TensorSpec(shape=[None] + list(input_shape), dtype=tf.float32)]
                    onnx_model, _ = tf2onnx.convert.from_keras(
                        functional_model,
                        input_signature=input_signature,
                        opset=11
                    )
                    print("   - 방법 3 성공: 모델 재구성 후 변환 완료")
                    
                except Exception as e3:
                    print(f"   - 방법 3 실패: {e3}")
                    
                    # 방법 4: 단순 클론 방법
                    try:
                        print("   - 방법 4: 단순 클론 방법 시도...")
                        
                        # 모델 클론 생성
                        cloned_model = tf.keras.models.clone_model(model)
                        cloned_model.set_weights(model.get_weights())
                        
                        # 명시적으로 모델 빌드
                        cloned_model.build((None, 96, 96, 3))
                        
                        # 더미 데이터로 모델 호출
                        dummy_input = tf.random.normal((1, 96, 96, 3))
                        _ = cloned_model(dummy_input)
                        
                        # 입력/출력 이름 명시적 설정
                        cloned_model.input_names = ['input']
                        cloned_model.output_names = ['output']
                        
                        # ONNX 변환
                        input_signature = [tf.TensorSpec(shape=[None, 96, 96, 3], dtype=tf.float32, name='input')]
                        onnx_model, _ = tf2onnx.convert.from_keras(
                            cloned_model,
                            input_signature=input_signature,
                            opset=11
                        )
                        print("   - 방법 4 성공: 단순 클론 방법 완료")
                        
                    except Exception as e4:
                        print(f"   - 방법 4 실패: {e4}")
                        raise Exception(f"모든 변환 방법 실패. 방법1: {e1}, 방법2: {e2}, 방법3: {e3}, 방법4: {e4}")
        
        if onnx_model is None:
            raise Exception("ONNX 모델 변환 실패")

        # ONNX 모델 파일로 저장
        with open(onnx_output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"   [성공] ONNX 모델 생성 성공: {onnx_output_path}")
        
        # 모델 크기 비교
        original_size = os.path.getsize(h5_model_path) / (1024 * 1024)  # MB
        onnx_size = os.path.getsize(onnx_output_path) / (1024 * 1024)  # MB
        print(f"   [정보] 크기 비교: {original_size:.2f}MB -> {onnx_size:.2f}MB")
        
        # 임시 SavedModel 폴더 삭제
        if os.path.exists(temp_saved_model_path):
            shutil.rmtree(temp_saved_model_path)
            print("   - 임시 SavedModel 폴더 삭제 완료")
        
    except Exception as e:
        print(f"   [실패] ONNX 변환 실패: {e}")
        # 임시 SavedModel 폴더 삭제 (실패 시에도)
        if os.path.exists(temp_saved_model_path):
            shutil.rmtree(temp_saved_model_path)
            print("   - 임시 SavedModel 폴더 삭제 완료")
        return False
    
    # 3. ONNX 모델 검증
    print("3. ONNX 모델 검증")
    try:
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("   [성공] ONNX 모델 검증 성공")
    except Exception as e:
        print(f"   [실패] ONNX 모델 검증 실패: {e}")
        return False
    
    print("=" * 60)
    print("[성공] 변환 완료!")
    print(f"📁 ONNX 모델 저장 경로: {onnx_output_path}")
    print("=" * 60)
    
    return True

class SkinModelCalibrationDataReader(CalibrationDataReader):
    """
    피부 질환 진단 모델을 위한 캘리브레이션 데이터 리더
    """
    def __init__(self, model_input_shape=(96, 96, 3), num_samples=100, input_name='input'):
        self.model_input_shape = model_input_shape
        self.num_samples = num_samples
        self.current_index = 0
        self.input_name = input_name
        
    def get_next(self):
        if self.current_index >= self.num_samples:
            return None
            
        # 96x96x3 크기의 랜덤 이미지 생성 (0-1 범위로 정규화)
        input_data = np.random.random((1, *self.model_input_shape)).astype(np.float32)
        self.current_index += 1
        
        return {self.input_name: input_data}

def create_quantized_onnx_dynamic(onnx_model_path, quantized_onnx_path):
    """
    ONNX 모델을 동적 양자화(Dynamic Quantization)로 변환합니다.
    가장 빠르고 간단한 방법이지만 정확도가 약간 떨어질 수 있습니다.
    """
    print("=" * 60)
    print("ONNX 모델 → 동적 양자화 ONNX 변환 시작")
    print("=" * 60)
    
    if not ONNX_QUANTIZATION_AVAILABLE:
        print("   [실패] onnxruntime.quantization을 사용할 수 없습니다.")
        return False
    
    try:
        print(f"1. 원본 ONNX 모델 로드: {onnx_model_path}")
        
        # 동적 양자화 실행
        print("2. 동적 양자화 실행 중...")
        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=quantized_onnx_path,
            weight_type=QuantType.QUInt8  # 8비트 unsigned int 사용
        )
        
        print(f"   [성공] 동적 양자화 ONNX 모델 생성: {quantized_onnx_path}")
        
        # 파일 크기 비교
        original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(quantized_onnx_path) / (1024 * 1024)  # MB
        print(f"   [정보] 크기 비교: {original_size:.2f}MB -> {quantized_size:.2f}MB ({quantized_size/original_size*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"   [실패] 동적 양자화 실패: {e}")
        return False

def create_quantized_onnx_static(onnx_model_path, quantized_onnx_path):
    """
    ONNX 모델을 정적 양자화(Static Quantization)로 변환합니다.
    캘리브레이션 데이터를 사용하여 더 정확한 양자화를 수행합니다.
    """
    print("=" * 60)
    print("ONNX 모델 → 정적 양자화 ONNX 변환 시작")
    print("=" * 60)
    
    if not ONNX_QUANTIZATION_AVAILABLE:
        print("   [실패] onnxruntime.quantization을 사용할 수 없습니다.")
        return False
    
    try:
        print(f"1. 원본 ONNX 모델 로드: {onnx_model_path}")
        
        # ONNX 모델 정보 확인
        onnx_model = onnx.load(onnx_model_path)
        input_name = onnx_model.graph.input[0].name
        print(f"   [정보] 모델 입력 이름: {input_name}")
        
        # 캘리브레이션 데이터 리더 생성
        print("2. 캘리브레이션 데이터 준비 중...")
        calibration_data_reader = SkinModelCalibrationDataReader(input_name=input_name)
        
        # 정적 양자화 실행
        print("3. 정적 양자화 실행 중...")
        quantize_static(
            model_input=onnx_model_path,
            model_output=quantized_onnx_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=QuantType.QUInt8  # 8비트 unsigned int 사용
        )
        
        print(f"   [성공] 정적 양자화 ONNX 모델 생성: {quantized_onnx_path}")
        
        # 파일 크기 비교
        original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(quantized_onnx_path) / (1024 * 1024)  # MB
        print(f"   [정보] 크기 비교: {original_size:.2f}MB -> {quantized_size:.2f}MB ({quantized_size/original_size*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"   [실패] 정적 양자화 실패: {e}")
        return False

def create_quantized_onnx(h5_model_path, tflite_output_path):
    """
    H5 모델을 8비트 양자화된 TFLite로 변환합니다.
    """
    print("=" * 60)
    print("H5 모델 → 8비트 양자화 TFLite 변환 시작")
    print("=" * 60)
    
    # 1. H5 모델 로드
    print(f"1. H5 모델 로드: {h5_model_path}")
    try:
        model = keras.models.load_model(h5_model_path)
        print(f"   [성공] 모델 로드 성공")
    except Exception as e:
        print(f"   [실패] 모델 로드 실패: {e}")
        return False
    
    # 2. 양자화를 위한 대표 데이터셋 생성
    print("2. 양자화를 위한 대표 데이터셋 생성")
    def representative_dataset():
        for _ in range(100):
            # 96x96x3 크기의 랜덤 이미지 생성 (0-1 범위로 정규화)
            data = np.random.random((1, 96, 96, 3)).astype(np.float32)
            yield [data]
    
    # 3. TensorFlow Lite 변환기 설정 (8비트 양자화)
    print("3. TensorFlow Lite 변환기 설정 (8비트 양자화)")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # 4. TFLite 모델 변환
    print("4. TFLite 모델 변환 중...")
    try:
        tflite_model = converter.convert()
        
        # TFLite 파일 저장
        with open(tflite_output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"   [성공] TFLite 모델 생성 성공: {tflite_output_path}")
        
        # 파일 크기 비교
        original_size = os.path.getsize(h5_model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(tflite_output_path) / (1024 * 1024)  # MB
        print(f"   [정보] 크기 비교: {original_size:.2f}MB -> {tflite_size:.2f}MB ({tflite_size/original_size*100:.1f}%)")
        
    except Exception as e:
        print(f"   [실패] TFLite 변환 실패: {e}")
        return False
    
    print("=" * 60)
    print("[성공] 8비트 양자화 TFLite 변환 완료!")
    print(f"[저장] TFLite 모델 저장 경로: {tflite_output_path}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    # 경로 설정
    h5_model_path = "./model/skin_model.h5"
    onnx_output_path = "./model/skin_model.onnx"
    onnx_quantized_dynamic_path = "./model/skin_model_quantized_dynamic.onnx"
    onnx_quantized_static_path = "./model/skin_model_quantized_static.onnx"
    tflite_output_path = "./model/skin_model_quantized.tflite"
    
    # model 폴더가 없으면 생성
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 출력 폴더 생성: {model_dir}")
    
    print("[시작] 피부 질환 진단 모델 변환 시작!")
    print("=" * 60)
    
    # 1. H5 → ONNX 변환
    print("[작업 1] H5 → ONNX 변환")
    onnx_success = convert_h5_to_onnx(h5_model_path, onnx_output_path)
    
    print("\n" + "=" * 60)
    
    # 2. ONNX → 동적 양자화 ONNX 변환
    onnx_dynamic_success = False
    if onnx_success:
        print("[작업 2] ONNX → 동적 양자화 ONNX 변환")
        onnx_dynamic_success = create_quantized_onnx_dynamic(onnx_output_path, onnx_quantized_dynamic_path)
    else:
        print("[작업 2] ONNX → 동적 양자화 ONNX 변환 (건너뜀: ONNX 변환 실패)")
    
    print("\n" + "=" * 60)
    
    # 3. ONNX → 정적 양자화 ONNX 변환
    onnx_static_success = False
    if onnx_success:
        print("[작업 3] ONNX → 정적 양자화 ONNX 변환")
        onnx_static_success = create_quantized_onnx_static(onnx_output_path, onnx_quantized_static_path)
    else:
        print("[작업 3] ONNX → 정적 양자화 ONNX 변환 (건너뜀: ONNX 변환 실패)")
    
    print("\n" + "=" * 60)
    
    # 4. H5 → 8비트 양자화 TFLite 변환
    print("[작업 4] H5 → 8비트 양자화 TFLite 변환")
    tflite_success = create_quantized_onnx(h5_model_path, tflite_output_path)
    
    print("\n" + "=" * 60)
    print("[요약] 변환 결과 요약")
    print("=" * 60)
    
    if onnx_success:
        print("[성공] ONNX 변환 성공")
    else:
        print("[실패] ONNX 변환 실패")
    
    if onnx_dynamic_success:
        print("[성공] ONNX 동적 양자화 변환 성공")
    else:
        print("[실패] ONNX 동적 양자화 변환 실패")
    
    if onnx_static_success:
        print("[성공] ONNX 정적 양자화 변환 성공")
    else:
        print("[실패] ONNX 정적 양자화 변환 실패")
    
    if tflite_success:
        print("[성공] 8비트 양자화 TFLite 변환 성공")
    else:
        print("[실패] 8비트 양자화 TFLite 변환 실패")
    
    success_count = sum([onnx_success, onnx_dynamic_success, onnx_static_success, tflite_success])
    
    if success_count > 0:
        print(f"[결과] {success_count}/4 변환이 성공했습니다!")
        print("[안내] 이제 카메라 진단 프로그램을 실행할 수 있습니다.")
        print("[모델 파일]")
        if onnx_success:
            print(f"  - 원본 ONNX: {onnx_output_path}")
        if onnx_dynamic_success:
            print(f"  - 동적 양자화 ONNX: {onnx_quantized_dynamic_path}")
        if onnx_static_success:
            print(f"  - 정적 양자화 ONNX: {onnx_quantized_static_path}")
        if tflite_success:
            print(f"  - 양자화 TFLite: {tflite_output_path}")
    else:
        print("[결과] 모든 변환이 실패했습니다.")
        print("[안내] 필요한 패키지 설치:")
        print("   pip install tf2onnx onnx onnxruntime tensorflow") 
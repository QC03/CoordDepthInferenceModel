# 실시간 2D-to-3D 루트 상대 깊이 추정

rtmlib RTMPose와 ResNet-18 기반 모델로 15개 관절의 **골반 기준 카메라 Z
상대 깊이**를 추정합니다. 절대 위치나 전체 XYZ를 복원하는 모델은 아닙니다.

## 프로젝트 구성

```text
src/
├─ config/       모델 설정 및 관절 매핑
├─ data/         데이터셋, 포즈 검출, 좌표 변환
├─ model/        PyTorch 모델
├─ training/     학습, 손실, 평가 지표
└─ inference/    ONNX export/검증 및 실시간 추론

train/           Human3.6M 영상 및 3D 라벨
model.pth        학습된 PyTorch 체크포인트
depth_lifting.onnx
                실시간 추론용 ONNX 모델
```

`train/`, `.venv/`, `model.pth`, `depth_lifting.onnx`는 `.gitignore`에 의해
버전 관리에서 제외됩니다. 각각 로컬 데이터, 가상환경, 학습/추론 산출물이므로
실행에는 필요합니다.

## 설치

기본 설치:

```cmd
python -m pip install -r requirements.txt
```

NVIDIA GPU를 사용하려면 프로젝트 가상환경을 활성화한 뒤 GPU 의존성을 설치합니다.

```cmd
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade --force-reinstall -r requirements-gpu.txt
```

GPU 설치 확인:

```cmd
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`True`가 출력되어야 CUDA 학습을 사용할 수 있습니다. rtmlib 포즈 모델은 첫
실행 시 자동으로 다운로드됩니다.

## 학습

`train/<subject>` 아래의 영상을 읽고 같은 action의 `joints3d_25` JSON을 정답으로
사용합니다. 각 프레임의 2D 관절은 rtmlib로 검출합니다.

### 빠른 동작 확인

아래 명령은 파이프라인 확인용이며 깊이 품질을 보장하지 않습니다.

```cmd
python -m src.training.train ^
  --subjects s01 s02 ^
  --epochs 1 ^
  --frame-stride 10 ^
  --max-samples 100 ^
  --eval-max-samples 100 ^
  --batch-size 8 ^
  --num-workers 0 ^
  --pose-mode lightweight ^
  --pose-device cuda ^
  --device cuda
```

### 실제 학습 권장값

`--max-samples`는 학습 데이터 수이고, `--eval-max-samples`는 학습 종료 후 MAE를
계산할 검증 데이터 수입니다. 검증은 기본적으로 최대 100개로 제한됩니다.

```cmd
python -m src.training.train ^
  --subjects s01 s02 ^
  --epochs 20 ^
  --frame-stride 10 ^
  --max-samples 5000 ^
  --eval-max-samples 100 ^
  --batch-size 8 ^
  --num-workers 0 ^
  --pose-mode lightweight ^
  --pose-device cuda ^
  --device cuda
```

Windows에서 CUDA 포즈 검출기를 사용할 때는 `--num-workers 0`을 권장합니다.
여러 워커가 H.264 디코더와 CUDA 포즈 모델을 동시에 만들면 메모리 부족이나
codec 초기화 오류가 발생할 수 있습니다.

학습이 끝나면 다음 파일이 덮어써집니다.

```text
model.pth
depth_lifting.onnx
```

카메라 좌표 변환이나 관절 매핑을 변경한 뒤에는 기존 두 파일을 사용하지 말고
반드시 새로 학습해야 합니다.

## ONNX 검증

PyTorch와 ONNX 출력이 일치하는지 확인합니다.

```cmd
python -m src.inference.compare_onnx ^
  depth_lifting.onnx ^
  --checkpoint model.pth
```

`max_abs_error`가 작은 값이어야 합니다. 이 검사는 깊이 정확도가 아니라
PyTorch와 ONNX 변환 결과의 일치 여부를 확인합니다.

## 실시간 추론

영상 파일:

```cmd
python -m src.inference.realtime ^
  input.mp4 ^
  --onnx depth_lifting.onnx ^
  --pose-device cuda
```

웹캠 0번:

```cmd
python -m src.inference.realtime ^
  --onnx depth_lifting.onnx ^
  --pose-device cuda
```

다른 웹캠은 `--camera-index`로 선택합니다. 종료하려면 `q`를 누릅니다.

```cmd
python -m src.inference.realtime ^
  --onnx depth_lifting.onnx ^
  --camera-index 1 ^
  --pose-device cuda
```

처리 순서는 다음과 같습니다.

```text
영상 프레임
→ rtmlib 2D 관절 검출
→ 이미지와 2D 관절 전처리
→ ONNX 깊이 추론
→ 골반 기준 root-relative depth
→ One-Euro temporal filter
→ X-Y / Y-Z skeleton 표시
```

## 좌표와 관절

루트는 좌·우 골반의 평균입니다.

```text
root-relative depth = Z_joint - (Z_left_hip + Z_right_hip) / 2
```

Human3.6M 카메라 파라미터의 `T`는 이동 벡터가 아니라 월드 좌표계의 카메라
중심입니다. 따라서 카메라 좌표는 다음처럼 계산합니다.

```text
P_camera = R @ (P_world - T)
```

현재 모델의 출력 순서는 다음 15개입니다.

```text
nose,
left/right shoulder,
left/right elbow,
left/right wrist,
left/right hip,
left/right knee,
left/right ankle,
left/right foot
```

현재 데이터와 모델에는 손가락·손끝 관절이 없습니다. 따라서 손 전체의 깊이는
예측하지 않고 손목 깊이만 예측합니다. 손가락까지 필요하면 손 3D 라벨이 포함된
whole-body 데이터셋, 손 관절을 검출하는 2D 모델, 새 관절 매핑과 출력 헤드가
필요합니다. `--pose-mode performance`는 기존 관절의 검출 방식만 바꾸며 손가락을
추가하지 않습니다.

## 평가 해석

`mean_mae_mm`는 root-relative depth의 평균 절대 오차(mm)입니다. 절대 깊이,
전체 XYZ 오차, 손가락 깊이 정확도를 의미하지 않습니다. `--max-samples 100`과
`--epochs 1`은 smoke test일 뿐이며, 실제 깊이 품질은 더 많은 샘플과 epoch로
학습한 뒤 확인해야 합니다.

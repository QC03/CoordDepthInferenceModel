# 실시간 2D-to-3D 루트 상대 깊이 추정

MediaPipe Pose Landmarker와 ResNet-18 기반 모델을 사용해 15개 관절의
루트 상대 카메라 Z 깊이를 추정하는 프로젝트입니다. 모델은 절대 3D XYZ가
아닌 골반 기준 상대 깊이를 출력합니다.

## 1. 설치

```cmd
python -m pip install -r requirements.txt
```

NVIDIA GPU로 학습하려면 CUDA가 포함된 PyTorch 빌드를 설치합니다. RTX 5070을
포함한 NVIDIA GPU 환경에서는 다음 명령을 사용하세요.

```cmd
python -m pip install -r requirements-gpu.txt
```

학습은 기본적으로 CUDA가 가능하면 `cuda`, 아니면 `cpu`를 자동 선택합니다.
강제로 지정하려면 `--device cuda` 또는 `--device cpu`를 사용할 수 있습니다.

필요한 구성 요소:

- Human3.6M 영상 및 `joints3d_25` 정답
- MediaPipe Tasks `.task` 포즈 모델
- PyTorch: 학습용
- ONNX Runtime: 추론용

소스 코드는 기능별로 다음과 같이 `src` 아래에 구성되어 있습니다.

```text
src/
├─ config/       설정 및 관절 매핑
├─ data/         데이터셋 및 좌표 변환
├─ model/        PyTorch 모델
├─ training/     학습, 손실, 평가 지표
└─ inference/    ONNX export/검증 및 실시간 추론
```

첫 실행 시 Pose Landmarker `.task` 파일을 자동으로 다운로드합니다. 직접 지정하려면
`--pose-model` 옵션을 사용합니다.

## 2. 학습

`train/<subject>` 아래의 영상을 읽고, 같은 action의 `joints3d_25` JSON을 정답으로
사용합니다. 2D 랜드마크는 각 영상 프레임에서 `.task` 모델로 검출합니다.

```cmd
python -m src.training.train ^
  --subjects s01 s02 ^
  --epochs 1 ^
  --frame-stride 10 ^
  --max-samples 100
```

학습 중 `tqdm`으로 진행률, epoch, loss를 표시합니다. 학습 종료 시 다음 파일이
자동 생성됩니다.

CUDA를 사용할 수 있는 PyTorch 환경이면 학습 장치로 CUDA를 자동 선택하고, 그렇지
않으면 CPU를 사용합니다. 시작 시 `device=cuda` 또는 `device=cpu`로 선택 결과를
표시합니다.

```text
model.pth
depth_lifting.onnx
```

전체 데이터를 사용하려면 `--frame-stride`와 `--max-samples`를 생략합니다.
데이터 위치가 기본값 `train`과 다르면 다음처럼 지정합니다.

```cmd
python -m src.training.train ^
  --data-root D:\path\to\train ^
  --subjects s01 s02
```

`.task` 파일을 직접 지정하는 예:

```cmd
python -m src.training.train ^
  --subjects s01 ^
  --pose-model D:\models\pose_landmarker_lite.task
```

## 3. ONNX 확인

학습이 끝나면 별도 export 없이 생성된 ONNX 파일을 확인할 수 있습니다.

```cmd
python -m src.inference.compare_onnx ^
  depth_lifting.onnx ^
  --checkpoint model.pth
```

ONNX 입력은 `B=1`, `224x224`로 고정됩니다.

## 4. 실시간 추론

실시간 추론은 PyTorch나 `model.pth`가 아니라 ONNX 파일을 사용합니다.

영상 파일을 지정하는 경우:

```cmd
python -m src.inference.realtime ^
  input.mp4 ^
  --onnx depth_lifting.onnx
```

영상 인자를 생략하거나 파일이 없으면 기본 웹캠(카메라 `0`)을 사용합니다.
웹캠 창을 종료하려면 `q` 키를 누릅니다.

```cmd
python -m src.inference.realtime ^
  --onnx depth_lifting.onnx
```

다른 웹캠을 사용하려면:

```cmd
python -m src.inference.realtime ^
  --onnx depth_lifting.onnx ^
  --camera-index 1
```

로컬 `.task` 모델을 사용할 때:

```cmd
python -m src.inference.realtime ^
  input.mp4 ^
  --onnx depth_lifting.onnx ^
  --pose-model D:\models\pose_landmarker_lite.task
```

프레임 처리 순서:

```text
영상 프레임 읽기
→ MediaPipe Tasks로 2D 랜드마크 검출
→ 이미지와 2D 좌표 전처리
→ ONNX Runtime 추론
→ 골반 기준 root-relative 깊이 계산
→ 관절별 One-Euro 필터 적용
```

추론 창은 왼쪽에 원본 영상을 표시하고, 오른쪽에 X-Y 및 Y-Z 축 기준 스켈레톤을
나란히 표시합니다. Y-Z 패널의 Z축은 모델이 추정한 root-relative depth입니다.
`q` 키를 누르면 종료하며, 결과를 별도 파일로 저장하지는 않습니다.

## 5. 좌표와 평가

루트는 왼쪽·오른쪽 골반의 평균입니다.

```text
root-relative depth = Z_joint - (Z_left_hip + Z_right_hip) / 2
```

평가 지표는 root-relative depth MAE(mm)이며, XYZ 모델의 MPJPE와 다릅니다.

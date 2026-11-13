# 실시간 2D-to-3D 루트 상대 깊이 추정

rtmlib RTMPose와 ResNet-18 기반 모델을 사용해 15개 관절의
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

기존 CPU 전용 PyTorch가 이미 설치되어 있다면 반드시 다음처럼 강제 재설치해야
합니다.

```cmd
python -m pip install --upgrade --force-reinstall -r requirements-gpu.txt
```

학습은 기본적으로 CUDA가 가능하면 `cuda`, 아니면 `cpu`를 자동 선택합니다.
강제로 지정하려면 `--device cuda` 또는 `--device cpu`를 사용할 수 있습니다.

필요한 구성 요소:

- Human3.6M 영상 및 `joints3d_25` 정답
- rtmlib 및 ONNX Runtime 포즈 모델
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

첫 실행 시 rtmlib의 경량 RTMPose 모델을 자동으로 다운로드합니다. Human3.6M이
단일 인물 영상이므로 별도 사람 검출 단계를 생략하고 전체 프레임에서 RTMPose를
실행합니다. CUDA 환경에서는 포즈 검출도 GPU에서 실행됩니다.

## 2. 학습

`train/<subject>` 아래의 영상을 읽고, 같은 action의 `joints3d_25` JSON을 정답으로
사용합니다. 2D 랜드마크는 각 영상 프레임에서 rtmlib RTMPose로 검출합니다.

```cmd
python -m src.training.train ^
  --subjects s01 s02 ^
  --epochs 1 ^
  --frame-stride 10 ^
  --max-samples 100 ^
  --batch-size 8 ^
  --num-workers 2 ^
  --pose-mode lightweight ^
  --pose-device cuda
```

학습 중 `tqdm`으로 진행률, epoch, loss를 표시합니다. 학습 종료 시 다음 파일이
자동 생성됩니다.

`--batch-size`는 GPU 메모리에 맞춰 늘리면 학습 처리량을 높일 수 있습니다. 영상에서
프레임을 읽고 포즈를 검출하는 작업은 `--num-workers`개의 프로세스에서 병렬로
수행되며, 메모리가 부족하거나 CPU 사용률이 너무 높으면 `0` 또는 더 작은 값으로
낮출 수 있습니다. CUDA 사용 시 AMP와 pinned memory도 자동으로 활성화됩니다.

CUDA를 사용할 수 있는 PyTorch 환경이면 학습 장치로 CUDA를 자동 선택하고, 그렇지
않으면 CPU를 사용합니다. 시작 시 `device=cuda` 또는 `device=cpu`로 선택 결과를
표시합니다.

```text
model.pth
depth_lifting.onnx
```

전체 데이터를 사용하려면 `--frame-stride`와 `--max-samples`를 생략합니다.
`--frame-stride`가 클수록 프레임 간격이 커져 샘플 수가 줄어듭니다.
`--max-samples`를 지정하면 전체 데이터에 걸쳐 샘플을 균등하게 선택합니다.
데이터 위치가 기본값 `train`과 다르면 다음처럼 지정합니다.

```cmd
python -m src.training.train ^
  --data-root D:\path\to\train ^
  --subjects s01 s02
```

포즈 검출 모델의 속도/정확도 모드:

```cmd
python -m src.training.train ^
  --subjects s01 ^
  --pose-mode lightweight ^
  --pose-device cuda
```

`--pose-mode`는 `lightweight`, `balanced`, `performance` 중 하나입니다. GPU 메모리
사용량이 높으면 `lightweight`를 사용하고, 포즈 정확도가 더 중요하면 `balanced`를
사용합니다. CUDA 검출기를 DataLoader 워커에서 실행할 때 문제가 있으면
`--num-workers 0`으로 한 프로세스에서 실행합니다.

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

CPU에서 포즈 검출을 실행할 때:

```cmd
python -m src.inference.realtime ^
  input.mp4 ^
  --onnx depth_lifting.onnx ^
  --pose-device cpu
```

프레임 처리 순서:

```text
영상 프레임 읽기
→ rtmlib RTMPose로 2D 랜드마크 검출
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

학습 레이블은 각 영상의 카메라 ID에 대응하는 `camera_parameters`의 외부 파라미터를
사용해 월드 좌표를 카메라 좌표로 변환한 뒤 생성합니다. 따라서 카메라별 촬영 방향이
달라도 카메라 Z축 기준 깊이를 학습합니다. 실시간 Y-Z 시각화의 깊이 축은 미터 단위
고정 범위를 사용하며, 작은 노이즈를 화면 전체로 자동 확대하지 않습니다.

3D 정답 관절은 Human3.6M 원본 순서에서 모델의 rtmlib 15관절 순서로 재배열한
뒤 사용합니다. rtmlib Halpe26에서 코, 어깨, 팔꿈치, 손목, 골반, 무릎, 발목,
좌우 엄지발가락을 선택하며, 이 순서를 바꾸면 좌우 골반, 무릎, 발목의 깊이 레이블이 서로 다른
관절에 연결되므로 기존 `model.pth`와 `depth_lifting.onnx`는 폐기하고 새 코드로
다시 학습해야 합니다.

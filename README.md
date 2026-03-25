# DimCam — Low-Light Stereo Image Enhancement

저조도 환경에서 촬영된 스테레오 이미지 쌍의 밝기와 디테일을 복원하는 딥러닝 모델입니다.

## Architecture

```
Input (Stereo L/R)
    │
    ▼
┌─────────────┐
│  DPCE-Net   │  Zero-DCE 기반 gamma map 생성
│  (backbone) │  → dpce_enhanced 초기 향상
└──────┬──────┘
       │  [img, dpce_enhanced, gamma_map] (9ch)
       ▼
┌─────────────┐
│  NAFBlocks  │  Self-refinement (L/R 독립)
│  ×N blocks  │  LayerNorm + DW-Conv + SimpleGate + SCA
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    SCAM     │  Stereo Cross Attention Module
│  (windowed) │  L↔R 양방향 어텐션 (disparity range 내)
└──────┬──────┘
       │
       ▼
  Gamma Delta  →  Illumination-aware fusion
       │           (dark_mask 기반 적응적 보정)
       ▼
  Final Enhanced Stereo Pair
```

**핵심 구성 요소:**

| 모듈 | 출처 | 역할 |
|------|------|------|
| DPCE-Net | Zero-DCE 변형 | Gamma map 생성 (초기 밝기 향상) |
| NAFBlock | NAFSSR | 자기 정제 (채널 어텐션 + 게이트) |
| SCAM | NAFSSR | 스테레오 크로스 어텐션 (L↔R 정보 교환) |
| DimCamLoss | Custom | 6종 손실 함수 통합 (stereo, depth, light, sfp, gamma, color) |

## Project Structure

```
├── model.py              # DimCamEnhancer 모델 (NAFBlock + SCAM)
├── Myloss.py             # 통합 손실 함수 (DimCamLoss)
├── train.py              # 단일 스테이지 학습
├── train_2stage.py       # 2-Stage 학습 (권장)
├── test.py               # 추론 및 평가
├── dataloader.py         # LunarStereoDataset 로더
├── local_arch.py         # TiledInferenceWrapper
├── utils.py              # 공통 유틸리티 (str2bool)
├── DPCE2/                # DPCE-Net backbone
│   ├── model.py          # enhance_net_nopool, gamma_enhance
│   └── snapshots/        # Pretrained DPCE 가중치
├── data/                 # 캘리브레이션 파일
└── snapshots_*/          # 학습된 체크포인트
```

## Setup

```bash
# conda 환경 생성
conda env create -f dimcam_env_environment.yml
conda activate dimcam_env

# 또는 핵심 패키지만 설치
pip install torch torchvision kornia tensorboard tqdm
```

**요구 사항:** Python 3.10+, PyTorch 2.0+, CUDA GPU (권장)

## Usage

### Training (2-Stage, 권장)

```bash
# Stage 1: DPCE frozen, Transformer만 학습 (lr=1e-4)
# Stage 2: 전체 모델 fine-tuning (lr=1e-5)
python train_2stage.py \
    --data_path ~/Downloads/lunardataset2/ \
    --dpce_weights_path DPCE2/snapshots/Epoch_200_original.pth \
    --embed_dim 64 \
    --num_blocks 5 \
    --epochs_stage1 30 \
    --epochs_stage2 20
```

### Training (Single Stage)

```bash
python train.py \
    --data_path ~/Downloads/lunardataset2/ \
    --dpce_weights_path DPCE2/snapshots/Epoch_200_original.pth \
    --embed_dim 64 \
    --num_blocks 5 \
    --epochs 50
```

### Inference

```bash
python test.py \
    --data_path ~/Downloads/lunardataset2/ \
    --weights_path snapshots_dimcam_2stage/dimcam_enhancer_best.pth \
    --embed_dim 64 \
    --num_blocks 5
```

## Loss Functions

| 손실 | 수식 | 역할 |
|------|------|------|
| L_stereo | Gradient consistency (homography warp) | 스테레오 쌍 일관성 |
| L_depth | Gradient consistency (MiDaS) | 깊이 구조 보존 |
| L_light | Global + Local patch L2 | 목표 밝기(0.6) 달성 |
| L_sfp | MSE(VGG(dpce_enhanced), VGG(final)) | 구조/텍스처 보존 |
| L_gamma | Edge-aware smoothness | Gamma map 평활화 (엣지 보존) |
| L_color | Color ratio preservation | 원본 색상 비율 유지 |

### Default Weights

| 파라미터 | train.py | train_2stage.py |
|---------|----------|-----------------|
| lambda_stereo | 2.0 | 2.0 |
| lambda_depth | 0.1 | 0.1 |
| w_light | 0.1 | 0.1 |
| w_sfp | 0.2 | 0.2 |
| w_gamma | 0.001 | 0.001 |
| w_color | 5.0 | 0.5 |

## Key Design Decisions

- **9-channel input (A1):** Transformer가 `[원본, DPCE향상결과, gamma_map]`을 모두 참조하여 "어디를 더 보정할지" 학습
- **Illumination-aware delta (A3):** 어두운 영역에 더 큰 gamma 보정 적용 (`dark_mask = 1 - luminance`)
- **Windowed attention (A4):** 전체 W×W 대신 ±64px 디스패리티 범위 내만 attend → 메모리 ~75% 절감
- **Edge-aware smoothness (B2):** 이미지 엣지에서는 gamma 변화를 허용하여 halo artifact 감소
- **VGG reference fix (B1):** 어두운 원본 대신 dpce_enhanced를 VGG 기준으로 사용 (dead ReLU 문제 해결)

## License

Private repository.

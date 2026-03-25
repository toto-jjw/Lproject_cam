# Conda Environments Summary
> 문서화 날짜: 2026-02-24

전체 conda 환경 목록 (`conda env list`):

| 환경 이름 | 경로 | 재현용 파일 |
|-----------|------|-------------|
| `.conda` (로컬) | `/home/jaewon/Lproject_sim/.conda` | `conda_environment.yml` |
| `dimcam_env` | `/home/jaewon/anaconda3/envs/dimcam_env` | `dimcam_env_environment.yml` |
| `DPCE` | `/home/jaewon/anaconda3/envs/DPCE` | `DPCE_environment.yml` |
| `DimCam-final` | `/home/jaewon/anaconda3/envs/DimCam-final` | - |
| `base` | `/home/jaewon/anaconda3` | - |

---

## 1. `dimcam_env`

### 기본 정보

| 항목 | 값 |
|------|-----|
| Python 버전 | **3.11.14** |
| 채널 | `defaults` |
| 경로 | `/home/jaewon/anaconda3/envs/dimcam_env` |
| 재현용 파일 | `dimcam_env_environment.yml` |

### 주요 사용처

- `run_enhancer.sh` — DimCam 영상 향상 모델 실행
- `Lproject_cam/` — DimCam 모델 학습/추론

### 주요 패키지

#### 딥러닝 프레임워크
| 패키지 | 버전 |
|--------|------|
| `torch` | 2.8.0+cu128 (CUDA 12.8) |
| `torchvision` | 0.23.0+cu128 |
| `torchaudio` | 2.8.0+cu128 |
| `timm` | 1.0.21 |
| `kornia` | 0.8.1 |

#### 영상처리 / 데이터
| 패키지 | 버전 |
|--------|------|
| `opencv-python` | 4.12.0.88 |
| `pillow` | 11.3.0 |
| `scikit-image` | 0.26.0 |
| `numpy` | 2.2.6 |
| `pandas` | 3.0.0 |
| `scikit-learn` | 1.8.0 |
| `scipy` | 1.17.0 |

#### 시각화 / 3D
| 패키지 | 버전 |
|--------|------|
| `matplotlib` | 3.10.8 |
| `seaborn` | - |
| `plotly` | 6.5.2 |
| `open3d` | 0.19.0 |
| `trimesh` | 4.11.1 |

#### 학습 도구
| 패키지 | 버전 |
|--------|------|
| `tensorboard` | 2.20.0 |
| `omegaconf` | 2.3.0 |
| `einops` | 0.8.1 |
| `safetensors` | 0.6.2 |
| `huggingface-hub` | 1.0.1 |
| `tqdm` | 4.67.1 |

#### 개발 도구
| 패키지 | 버전 |
|--------|------|
| `ipython` | 9.9.0 |
| `ipywidgets` | 8.1.8 |
| `dash` | 3.4.0 |
| `jupyter-core` | 5.9.1 |

### 환경 복원

```bash
conda env create -f dimcam_env_environment.yml
conda activate dimcam_env
```

---

## 2. `DPCE`

### 기본 정보

| 항목 | 값 |
|------|-----|
| Python 버전 | **3.13.9** |
| 채널 | `pytorch`, `nvidia`, `conda-forge`, `defaults` |
| 경로 | `/home/jaewon/anaconda3/envs/DPCE` |
| 재현용 파일 | `DPCE_environment.yml` |

### 주요 사용처

- `Lproject_cam/model.py` — `DPCE.model`에서 `enhance_net_nopool`, `gamma_enhance` 임포트
- `data/visualize_dem.ipynb` — Jupyter 커널로 사용
- `run_denoiser.sh` / `Lproject_cam/DPCE/` — DPCE 기반 저조도 영상 향상 모듈

### 주요 패키지

#### 딥러닝 프레임워크
| 패키지 | 버전 |
|--------|------|
| `pytorch` | 2.8.0 (conda, CPU+MKL) |
| `torchvision` | 0.23.0 (pip) |
| `timm` | 1.0.21 |
| `kornia` | 0.8.1 |

#### CUDA 관련 (pip)
| 패키지 | 버전 |
|--------|------|
| `nvidia-cuda-runtime-cu12` | 12.8.90 |
| `nvidia-cudnn-cu12` | 9.10.2.21 |
| `nvidia-cublas-cu12` | 12.8.4.1 |
| `triton` | 3.4.0 |

#### 영상처리 / 컴퓨터 비전
| 패키지 | 버전 |
|--------|------|
| `opencv` (conda) | 4.12.0 |
| `pillow` | 12.0.0 |
| `numpy` | 2.3.3 (pip) |
| `lmdb` | 0.9.31 |

#### 3D / 시각화
| 패키지 | 버전 |
|--------|------|
| `matplotlib` | 3.10.8 |
| `open3d` | - |

#### 학습 도구
| 패키지 | 버전 |
|--------|------|
| `tensorboard` | 2.20.0 |
| `safetensors` | 0.6.2 |
| `huggingface_hub` | 1.0.1 |
| `omegaconf` | - |

#### 개발 도구
| 패키지 | 버전 |
|--------|------|
| `ipython` | 9.9.0 |
| `ipykernel` | 7.1.0 |
| `jupyter_client` | 8.8.0 |

### 환경 복원

```bash
conda env create -f DPCE_environment.yml
conda activate DPCE
```

---

## 환경별 용도 요약

| 환경 | 용도 | Python | GPU |
|------|------|--------|-----|
| `.conda` (로컬) | `data/` Jupyter 노트북 데이터 분석 | 3.14.2 | ❌ |
| `dimcam_env` | DimCam 모델 학습/추론, 영상 향상 | 3.11.14 | ✅ CUDA 12.8 |
| `DPCE` | DPCE 저조도 향상, DEM 시각화 노트북 | 3.13.9 | ✅ CUDA 12.1 |

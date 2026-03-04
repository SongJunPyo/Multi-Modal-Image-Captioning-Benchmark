# 📸 Image Captioning: From Vision Encoders to LLMs

본 프로젝트는 이미지에서 시각적 정보를 추출하여 자연어 문장으로 설명하는 **이미지 캡셔닝(Image Captioning)** 태스크를 수행하기 위해, 전통적인 Vision Encoder 기반 모델부터 최신 LLM 및 VLM 활용 방식까지 단계적으로 구현하고 비교 분석한 연구입니다.

---

## 🚀 프로젝트 개요
이미지 캡셔닝 연구 흐름에 따라 총 3가지 접근 방향을 설정하여 실험을 진행했습니다:

1.  **Approach (1): Vision Encoder 기반 캡셔닝**
    * ResNet + LSTM (Baseline) 
    * ViT + LSTM 
    * **ViT + GPT-2 (최종 선정 모델)**
2.  **Approach (2): Encoder + LLM 기반 캡셔닝**
    * ResNet101 + MLP Projector + Qwen2 
    * CLIP ViT + GPT-OSS-20B (LoRA 미세조정)
3.  **Approach (3): 사전 학습 VLM 활용**
    * Qwen2-VL-2B-Instruct 기반 제로샷/프롬프트 실험 

---

## 🏆 최종 모델: ViT-Base + GPT-2
다양한 실험 결과, **Vision Transformer(ViT)**의 전역적 특징 추출 능력과 사전 학습된 **GPT-2**의 풍부한 언어 지식을 결합한 모델이 가장 우수한 성능을 보였습니다.

### 모델 구조 
* **Vision Encoder:** ViT-Base (가중치 고정)를 통한 196개 패치 토큰 추출
* **Projection Layer:** 시각 특징을 GPT-2 임베딩 공간으로 의미적 정렬
* **Language Decoder:** GPT-2 Small을 사용하여 이미지를 Prefix로 입력받아 캡션 생성

### 성능 지표 (정량적 결과)
| 모델 | BLEU-4 | METEOR | CIDEr-D |
| :--- | :---: | :---: | :---: |
| ResNet + LSTM (Baseline) | 0.1630 | 0.3765 | 1.0854 |
| ViT + LSTM | 0.1699 | **0.3911** | 1.1189 |
| **ViT + GPT-2 (Final)** | **0.1930** | 0.3801 | **1.2176** |

---

## 💻 설치 및 실행 방법

### 1. 환경 설정
```bash
pip install torch torchvision transformers pillow pandas
```

### 2. 데이터 구조
데이터셋은 다음과 같은 구조로 배치되어야 합니다.
```text
data/
  ├── TRAIN/
  │    ├── images/
  │    └── train.csv
  └── VAL/
       ├── images/
       └── val.csv
```

### 3. 학습 및 추론
최종 모델인 ViT + GPT-2를 실행하려면 다음 스크립트를 사용합니다.
```bash
python main_ViT_GPT2.py
```

---

## 📊 가설 검증 결과

* H1: ViT + Transformer 디코더가 CNN+LSTM보다 우수하다 (채택)
* H2: LLM 결합 모델이 더 자연스러운 문장을 생성한다 (부분 채택) - 의미 전달 품질은 향상되었으나 정량 지표 개선은 제한적
* H3: LoRA 기반 미세조정이 효율적인 성능 향상을 보장한다 (기각) - 본 실험 환경에서는 기대치 미달

---

## 📜 참고 문헌
* Deep Residual Learning for Image Recognition (ResNet)
* Show, Attend and Tell: Visual Attention 기반 캡션 생성
* LLaVA: Visual Instruction Tuning
* LoRA: Low-Rank Adaptation of Large Language Models

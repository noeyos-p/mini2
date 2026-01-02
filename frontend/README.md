# 🏞️ 시각 도우미 (Vision Assistant)

시각장애인을 위한 **이미지 & 비디오 설명** 웹 서비스입니다.  
사진을 찍거나 영상을 녹화하면 AI가 자세히 설명해줍니다.

## ✨ 주요 기능

- 📷 **카메라 촬영** - 실시간으로 사진 찍기
- 🎬 **비디오 녹화** - 최대 10초 영상 녹화
- 🖼️ **이미지 업로드** - 기존 사진 불러오기
- 📁 **비디오 업로드** - 기존 영상 불러오기
- ❓ **자유 질문** - "뭐가 보여?", "뭐하고 있어?" 등
- 📝 **전체 설명** - 이미지/영상 전체 설명
- 🎤 **음성 입력 (STT)** - 말로 질문하기
- 🔊 **음성 출력 (TTS)** - 답변 읽어주기
- ⚡ **스트리밍 응답** - 실시간으로 답변 출력
- 🌙 **고대비 다크모드** - 접근성 강화 UI

## 🛠️ 기술 스택

### Frontend
- React 18 + TypeScript
- Vite
- Web Speech API (STT + TTS)
- CSS (접근성 강화)

### Backend
- FastAPI (Python)
- **Qwen2-VL-2B** - 이미지 → 영어 설명
- **M2M100** - 영어 → 한국어 번역
- PyTorch + CUDA
- Hugging Face Transformers

### 음성 처리 (브라우저 내장)
- **STT**: Web Speech API (Chrome/Edge 권장)
- **TTS**: Web Speech API

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │ 카메라  │  │ 갤러리  │  │   STT   │  │      TTS        │ │
│  │  촬영   │  │ 업로드  │  │ 음성입력│  │    음성출력     │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
│       └────────────┴────────────┴────────────────┘          │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │ API (SSE 스트리밍)
┌───────────────────────────┼──────────────────────────────────┐
│                        Backend                               │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Qwen2-VL-2B (이미지 인식)                  ││
│  │                   영어 설명 생성                        ││
│  └─────────────────────────┬───────────────────────────────┘│
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              M2M100 (번역 모델)                         ││
│  │                 영어 → 한국어                           ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

## 📋 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| Python | 3.10+ | 3.10 |
| Node.js | 18+ | 20+ |
| CUDA | 11.8+ | 12.6 |
| VRAM | 6GB | 8GB+ |
| RAM | 16GB | 32GB |

### GPU VRAM 사용량
- **Qwen2-VL-2B**: ~5GB
- **M2M100**: ~1.5GB
- **총합**: ~6GB

## 🚀 설치 및 실행

### 1. Backend 설정

```bash
# 1. Conda 환경 생성 및 활성화
conda create -n v1 python=3.10 -y
conda activate v1

# 2. CUDA 버전 확인
nvidia-smi

# 3. PyTorch 설치 (CUDA 12.6 예시)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. 나머지 패키지 설치
cd backend
pip install -r requirements.txt

# 5. 서버 실행
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

> ⚠️ 첫 실행 시 모델 다운로드 (~6.5GB)가 진행됩니다.

### 2. Frontend 설정

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

프론트엔드가 `http://localhost:5173`에서 실행됩니다.

## 💬 사용 예시

### 풍경 질문 예시

| 질문 | 답변 예시 |
|------|----------|
| "뭐가 보여?" | "푸른 호수가 펼쳐져 있고, 멀리 눈 덮인 산이 보여요." |
| "하늘 어때?" | "맑고 파란 하늘에 흰 구름이 떠 있어요." |
| "노을이야?" | "네, 해가 지고 있어서 하늘이 주황색과 분홍색으로 물들었어요." |
| "산 보여?" | "멀리 높은 산이 보여요. 정상에 눈이 덮여 있어요." |
| "분위기 어때?" | "고요하고 평화로운 분위기예요." |

### 음성 사용법

1. 🎤 버튼 클릭
2. "하늘 색깔이 어때?" 말하기
3. 텍스트로 자동 입력됨
4. 📤 보내기 또는 자동 전송
5. 🔊 답변을 음성으로 읽어줌

## ⌨️ 키보드 단축키

| 단축키 | 기능 |
|--------|------|
| `Ctrl + Enter` | 질문 전송 |
| `Escape` | 음성/생성 중지 |

## 🌐 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 헬스 체크 |
| POST | `/api/ask` | 이미지 질문 (일반) |
| POST | `/api/ask-stream` | 이미지 질문 (스트리밍) |
| POST | `/api/describe-stream` | 이미지 전체 설명 (스트리밍) |
| POST | `/api/ask-video` | 비디오 질문 (일반) |
| POST | `/api/ask-video-stream` | 비디오 질문 (스트리밍) |
| POST | `/api/describe-video-stream` | 비디오 전체 설명 (스트리밍) |

### 이미지 요청 예시

```json
POST /api/ask-stream
{
  "image_base64": "data:image/jpeg;base64,...",
  "question": "뭐가 보여?",
  "language": "ko"
}
```

### 비디오 요청 예시

```json
POST /api/ask-video-stream
{
  "video_base64": "data:video/webm;base64,...",
  "question": "뭐하고 있어?",
  "language": "ko"
}
```

## 📁 프로젝트 구조

```
vision-assistant/
├── backend/
│   ├── main.py              # FastAPI 서버 + VL 모델 + 번역
│   └── requirements.txt     # Python 의존성
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # 메인 React 컴포넌트 (STT/TTS 포함)
│   │   ├── App.css          # 스타일 (접근성 강화)
│   │   ├── main.tsx         # React 진입점
│   │   └── vite-env.d.ts    # 타입 정의
│   ├── index.html           # HTML 템플릿
│   ├── package.json         # npm 의존성
│   ├── tsconfig.json        # TypeScript 설정
│   └── vite.config.ts       # Vite 설정
└── README.md
```

## 🔧 트러블슈팅

### CUDA 관련 오류
```bash
# CUDA 버전 확인
nvidia-smi

# 적절한 PyTorch 버전 설치 (CUDA 버전에 맞게)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 메모리 부족 (OOM)
- 다른 GPU 사용 프로그램 종료
- 이미지 해상도 줄이기

### 음성 입력 안 됨
- Chrome 또는 Edge 브라우저 사용
- 마이크 권한 허용
- HTTPS 또는 localhost에서만 작동

### 번역이 이상함
- 영어 답변 확인 (콘솔 로그)
- 질문 매핑 추가 필요할 수 있음

## 🎯 지원하는 질문 패턴

### 풍경 요소
```
산, 바다, 호수, 강, 하늘, 구름, 나무, 숲, 꽃, 
해, 달, 별, 노을, 해변, 폭포, 초원, 다리, 등대
```

### 질문 유형
```
뭐가 있어, 뭐가 보여, 앞에 뭐, 하늘 어때, 
분위기 어때, 날씨 어때, 색깔이 뭐야, 
아침이야, 저녁이야, 맑아, 흐려
```

## 📊 사용 모델 정보

| 모델 | 역할 | HuggingFace |
|------|------|-------------|
| Qwen2-VL-2B-Instruct | 이미지 인식 | [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| M2M100-418M | 영→한 번역 | [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) |

## 📄 라이선스

MIT License

## 🙏 크레딧

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) - Vision-Language Model
- [M2M100](https://github.com/facebookresearch/fairseq) - Multilingual Translation
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
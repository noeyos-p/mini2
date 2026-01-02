"""
시각장애인을 위한 이미지 질문-응답 API
Qwen2-VL-2B (영어) + M2M100 번역 (한국어)
개선: 질문 매핑 우선순위, 복합 질문 처리, 번역 후처리
"""

import io
import base64
import re
import asyncio
import gc
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from threading import Thread

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image

# 전역 변수
vl_model = None
vl_processor = None
translator = None
translator_tokenizer = None
device = None

# 모델 설정
VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
TRANSLATOR_MODEL_ID = "facebook/m2m100_418M"  # 다시 M2M100 (더 빠르고 안정적)


# =============================================
# 질문 매핑 (우선순위: 긴 패턴부터)
# =============================================

# 주어 (대상) 매핑 - 풍경 요소
SUBJECT_MAP = {
    # 자연
    "산": "the mountain",
    "산이": "the mountain",
    "바다": "the ocean",
    "바다가": "the ocean",
    "호수": "the lake",
    "호수가": "the lake",
    "강": "the river",
    "강이": "the river",
    "하늘": "the sky",
    "하늘이": "the sky",
    "구름": "the clouds",
    "구름이": "the clouds",
    "나무": "the trees",
    "나무가": "the trees",
    "숲": "the forest",
    "숲이": "the forest",
    "꽃": "the flowers",
    "꽃이": "the flowers",
    "해": "the sun",
    "해가": "the sun",
    "달": "the moon",
    "달이": "the moon",
    "별": "the stars",
    "별이": "the stars",
    "눈": "the snow",
    "비": "the rain",
    "안개": "the fog",
    "노을": "the sunset",
    "일출": "the sunrise",
    
    # 지형
    "언덕": "the hill",
    "절벽": "the cliff",
    "해변": "the beach",
    "모래": "the sand",
    "바위": "the rocks",
    "폭포": "the waterfall",
    "들판": "the field",
    "초원": "the meadow",
    
    # 인공물
    "건물": "the building",
    "집": "the house",
    "다리": "the bridge",
    "길": "the road",
    "배": "the boat",
    "등대": "the lighthouse",
}

# 질문 유형 매핑 (풍경 최적화)
QUESTION_MAP = [
    # 전체 풍경
    ("뭐가 있어", "Describe this landscape scene."),
    ("뭐가 보여", "Describe this landscape scene."),
    ("앞에 뭐", "What is in the foreground?"),
    ("뭐 있어", "Describe this scene."),
    ("설명해", "Describe this landscape in detail."),
    
    # 위치별
    ("앞쪽", "What is in the foreground?"),
    ("뒤쪽", "What is in the background?"),
    ("가운데", "What is in the center?"),
    ("멀리", "What is in the distance?"),
    ("가까이", "What is nearby?"),
    
    # 자연 요소
    ("하늘", "Describe the sky."),
    ("구름", "Describe the clouds."),
    ("산", "Describe the mountains."),
    ("바다", "Describe the ocean or sea."),
    ("호수", "Describe the lake."),
    ("강", "Describe the river."),
    ("나무", "Describe the trees."),
    ("숲", "Describe the forest."),
    ("꽃", "Describe the flowers."),
    ("해", "Describe the sun."),
    ("달", "Describe the moon."),
    
    # 분위기/시간
    ("분위기", "What is the mood or atmosphere?"),
    ("느낌", "What is the mood or atmosphere?"),
    ("날씨", "What is the weather like?"),
    ("시간", "What time of day is it?"),
    ("계절", "What season does it look like?"),
    ("아침", "Is this morning or sunrise?"),
    ("저녁", "Is this evening or sunset?"),
    ("낮", "Is this daytime?"),
    ("밤", "Is this nighttime?"),
    
    # 색상
    ("무슨 색", "What colors do you see?"),
    ("무슨색", "What colors do you see?"),
    ("색깔", "What are the main colors?"),
    ("색이", "What colors are there?"),
    
    # 날씨
    ("맑", "Is it clear or sunny?"),
    ("흐", "Is it cloudy?"),
    ("비", "Is it raining?"),
    ("눈", "Is it snowing?"),
    ("안개", "Is there fog or mist?"),
    
    # 건물/인공물
    ("건물", "Are there any buildings?"),
    ("집", "Are there any houses?"),
    ("다리", "Is there a bridge?"),
    ("길", "Is there a road or path?"),
    
    # 일반
    ("어디", "What place is this?"),
    ("장소", "What kind of place is this?"),
    ("어때", "How does this scene look?"),
    ("예뻐", "Is this beautiful?"),
]


def convert_question_to_english(question: str) -> str:
    """한국어 질문을 영어로 변환 (복합 질문 처리)"""
    
    # 1. 주어(대상) 찾기
    subject = "it"
    found_subject_ko = None
    for ko, en in SUBJECT_MAP.items():
        if ko in question:
            subject = en
            found_subject_ko = ko
            break
    
    # 2. 질문 유형 찾기 (긴 패턴부터 체크)
    for pattern, en_template in QUESTION_MAP:
        if pattern in question:
            en_question = en_template.replace("{subject}", subject)
            return en_question
    
    # 3. 주어만 있고 질문 유형이 없으면 → 설명 요청
    if found_subject_ko:
        return f"Describe {subject}."
    
    # 4. 기본값 - 풍경 설명
    return "Describe this landscape scene."


# =============================================
# 번역 후처리
# =============================================

def clean_translation(text: str) -> str:
    """번역 결과 정리 (자연스러운 대화체)"""
    if not text:
        return ""
    
    # 마크다운 형식 제거
    text = re.sub(r'#{1,6}\s*', '', text)  # ### 헤더 제거
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # **bold** 제거
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # - 리스트 제거
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # 1. 숫자 리스트 제거
    text = re.sub(r'`([^`]+)`', r'\1', text)  # `code` 제거
    
    # 불필요한 라벨 제거
    text = re.sub(r'장면\s*설명\s*:?\s*', '', text)
    text = re.sub(r'배경\s*요소\s*:?\s*', '', text)
    text = re.sub(r'산\s*범위\s*:?\s*', '산이 ', text)
    text = re.sub(r':\s*:', ':', text)
    
    # 중복 단어 제거
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    
    # 이상한 문자 제거
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    
    # 불필요한 번역투 표현 제거
    remove_patterns = [
        r'이미지에서\s*',
        r'사진에서\s*',
        r'그림은\s*',
        r'다음과\s*같은\s*',
        r'기능을\s*가진\s*',
        r'보여줍니다\s*',
        r'그것은\s*',
        r'이것은\s*',
        r'^예[,.\s]*',
        r'^네[,.\s]*',
        r'조건을\s*',
        r'두\s*개의\s*동물이\s*',
        r'서로\s*옆에\s*',
        r'프레임의\s*',
        r'화면의\s*',
        r'이\s*동물\s*두\s*마리.*?요\.',
        r'자연\s*환경의\s*한가운데.*',
        r'예를\s*들어.*',
        r'근처에.*요\.',
        r'그들은\s*서로.*',
        r'\(.*?\)',
        r'것처럼.*?있습니다.*',
        r'더\s*자세한.*',
        r'그\s*위에.*?있네요\.',
        r'투입되어.*',
    ]
    for pattern in remove_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 자연스러운 대화체로 변환 (~네요 + ~해요 혼합)
    replacements = [
        # 관찰/발견 표현 → ~네요
        ('있습니다.', '있네요.'),
        ('있어요.', '있네요.'),
        ('보입니다.', '보이네요.'),
        ('보여요.', '보이네요.'),
        ('있다.', '있네요.'),
        ('보인다.', '보이네요.'),
        
        # 상태/설명 표현 → ~해요/~이에요
        ('입니다.', '이에요.'),
        ('없습니다.', '없어요.'),
        ('됩니다.', '돼요.'),
        ('합니다.', '해요.'),
        ('습니다.', '어요.'),
        ('ㅂ니다.', '요.'),
        ('이다.', '이에요.'),
        ('없다.', '없어요.'),
        ('한다.', '해요.'),
        ('된다.', '돼요.'),
        
        # 형용사 자연스럽게
        ('맑습니다.', '맑아요.'),
        ('푸릅니다.', '푸르네요.'),
        ('밝습니다.', '밝아요.'),
        ('어둡습니다.', '어두워요.'),
        
        # ⭐ 번역 오류 수정
        ('표지판', '무늬'),
        ('냄비', '털'),
        ('프레임', '화면'),
        ('코트', '털'),
        ('마킹', '무늬'),
        ('패치', '무늬'),
        ('벽돌', '무늬'),
        ('스팟', '점'),
        ('콘크리트 바닥', '바닥'),
        ('야외 환경을', ''),
        ('야외 조건을', ''),
        ('환경을', ''),
        ('구멍', '털'),
        ('다른 옷', '다른 한 마리'),
        ('다른 옷은', '다른 한 마리는'),
        ('녹색 근처', '잔디 위'),
        ('초록색 근처', '잔디 위'),
        ('녹색에서', '잔디에서'),
        ('그림을 포함', '무늬가 있'),
        ('두 개의 고양이', '고양이 두 마리'),
        ('두 개의 동물', '동물 두 마리'),
        ('두 개의 개', '강아지 두 마리'),
        ('하나는', '한 마리는'),
        ('과 같은 그림', ' 무늬'),
        ('자연 환경의', ''),
        ('한가운데', ''),
        ('예를 들어', ''),
        ('습도 노출', ''),
        ('로 인해', ''),
        ('융합', ''),
        ('모시로 보이는', ''),
        ('비가 흐르는', ''),
        ('물의 융합', ''),
        ('점과 같은', ''),
        
        # 추가 번역 오류 수정
        ('무더운 땅 표면', '잔디'),
        ('땅 표면', '땅'),
        ('투입되어', '드리워져'),
        ('그림자가 드리워져 있네요.', ''),
        ('관심을 끌고있는', ''),
        ('관심을 기울이고', ''),
        ('무언가를 관찰하는', ''),
        ('카메라 밖에서', ''),
        ('더 자세한 설명이 필요하지 않습니다', ''),
        ('서로 평온하지만', '평화롭게'),
        ('것처럼 서로', ''),
        
        # 번역투 수정
        ('나타나며', '보이고'),
        ('던집니다', '드리우고'),
        ('구성되어', '이루어져'),
        ('착용하는', '쓴'),
        ('가진', '있는'),
        ('거대한', '큰'),
        ('범위', ''),
        ('요소', ''),
        ('눈에 띄는', ''),
        ('열려있는', '넓은'),
        ('위치하고', ''),
        ('앉아 있네요. 이', '앉아 있고'),
        
        # 불필요한 표현
        ('의 왼쪽에는', ' 왼쪽에'),
        ('의 오른쪽에는', ' 오른쪽에'),
        ('주위에', '에'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    
    # 어색한 조사 수정
    text = re.sub(r'를\s*\.', '를요.', text)
    text = re.sub(r'을\s*\.', '을요.', text)
    
    # 문장 정리
    text = text.strip()
    text = re.sub(r'^[.,:\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 문장 끝 자연스럽게
    if text and not text[-1] in '.!?요':
        if text.endswith('다'):
            text = text[:-1] + '네요.'
        elif text.endswith('음') or text.endswith('임'):
            text = text + '요.'
        else:
            text += '요.'
    
    # 마지막 정리 - 연속된 조사 정리
    text = re.sub(r'\s+', ' ', text)
    
    # 세미콜론을 마침표로 변환
    text = text.replace(';', '.')
    
    # 문장이 너무 길면 2문장만 유지
    sentences = re.split(r'(?<=[.!?요])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 2:
        text = ' '.join(sentences[:2])
    
    # 문법 오류 수정
    text = re.sub(r'마리은', '마리는', text)
    text = re.sub(r'있해요', '있어요', text)
    text = re.sub(r'없해요', '없어요', text)
    text = re.sub(r'요\.', '요.', text)
    
    return text.strip()


# =============================================
# 모델 로드
# =============================================

def load_models():
    """VL 모델 + 번역 모델 로드"""
    global vl_model, vl_processor, translator, translator_tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    if device == "cuda":
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # 1. Vision-Language 모델 로드
        print(f"📦 Loading VL model: {VL_MODEL_ID}")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        vl_processor = AutoProcessor.from_pretrained(
            VL_MODEL_ID, 
            trust_remote_code=True,
            min_pixels=256*28*28,
            max_pixels=512*28*28,
        )
        vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            VL_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        print("✅ VL model loaded!")
        
        # 2. 번역 모델 로드 (M2M100)
        print(f"📦 Loading translator: {TRANSLATOR_MODEL_ID}")
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        
        translator_tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATOR_MODEL_ID)
        translator = M2M100ForConditionalGeneration.from_pretrained(TRANSLATOR_MODEL_ID)
        
        if device == "cuda":
            translator = translator.to(device)
        
        print("✅ Translator loaded!")
        print("🚀 All models ready!")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise e


def translate_to_korean(text: str) -> str:
    """영어를 한국어로 번역 (M2M100)"""
    global translator, translator_tokenizer, device
    
    if not text.strip():
        return ""
    
    try:
        # M2M100: 소스 언어 설정
        translator_tokenizer.src_lang = "en"
        
        inputs = translator_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = translator.generate(
                **inputs,
                forced_bos_token_id=translator_tokenizer.get_lang_id("ko"),
                max_length=128,
                num_beams=3,
            )
        
        translated = translator_tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0]
        
        # 후처리
        return clean_translation(translated)
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# =============================================
# FastAPI 앱
# =============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    # 정리
    global vl_model, vl_processor, translator, translator_tokenizer
    del vl_model, vl_processor, translator, translator_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


app = FastAPI(
    title="Vision Assistant API",
    description="시각장애인을 위한 이미지 질문-응답 API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    image_base64: str
    question: str
    language: str = "ko"


class AnswerResponse(BaseModel):
    answer: str
    success: bool
    error: Optional[str] = None


def process_image(image_data: str) -> Image.Image:
    """Base64 이미지 디코딩"""
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"이미지 처리 실패: {e}")


# 영어 시스템 프롬프트 (극도로 짧게)
SYSTEM_PROMPT = """Say what you see in ONE simple sentence. 10 words maximum.
Example: "A cat and dog sitting on grass."
Do NOT describe colors, positions, shadows, or background."""


def clean_english_answer(text: str) -> str:
    """영어 답변 정리 (번역 전)"""
    if not text:
        return ""
    
    # 괄호 안 내용 제거
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 불필요한 표현 제거
    remove_phrases = [
        r'which\s+.*?[,.]',
        r'with\s+shadows?\s+.*?[,.]',
        r'casting\s+.*?[,.]',
        r'observing\s+.*?[,.]',
        r'looking\s+at\s+.*?[,.]',
        r'paying\s+attention\s+.*?[,.]',
        r'outside\s+the\s+(frame|camera).*?[,.]',
        r'off[\s-]camera.*?[,.]',
        r'in\s+the\s+background.*?[,.]',
        r'more\s+details?\s+.*',
        r'no\s+further\s+.*',
    ]
    for pattern in remove_phrases:
        text = re.sub(pattern, '.', text, flags=re.IGNORECASE)
    
    # 첫 문장만 추출
    sentences = re.split(r'[.!?;]', text)
    if sentences:
        text = sentences[0].strip()
    
    # 마침표 추가
    if text and not text.endswith('.'):
        text += '.'
    
    return text.strip()


def generate_english_answer(image: Image.Image, question: str) -> str:
    """영어로 이미지 설명 생성"""
    global vl_model, vl_processor, device
    
    from qwen_vl_utils import process_vision_info
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = vl_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = vl_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    if device == "cuda":
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        generated_ids = vl_model.generate(
            **inputs,
            max_new_tokens=30,  # 매우 짧게
            do_sample=False,
            repetition_penalty=1.5,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = vl_processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return answer.strip()


def generate_answer(image: Image.Image, question: str, language: str = "ko") -> str:
    """이미지 설명 생성"""
    
    # 한국어 질문 → 영어 변환
    if language == "ko":
        en_question = convert_question_to_english(question)
        print(f"🔄 질문 변환: '{question}' → '{en_question}'")
    else:
        en_question = question
    
    # 영어로 답변 생성
    english_answer = generate_english_answer(image, en_question)
    print(f"🇺🇸 영어 원본: {english_answer}")
    
    # 영어 답변 정리 (번역 전)
    english_answer = clean_english_answer(english_answer)
    print(f"🇺🇸 영어 정리: {english_answer}")
    
    # 한국어로 번역
    if language == "ko":
        korean_answer = translate_to_korean(english_answer)
        print(f"🇰🇷 한국어 답변: {korean_answer}")
        return korean_answer
    
    return english_answer


async def generate_stream(image: Image.Image, question: str, language: str = "ko") -> AsyncGenerator[str, None]:
    """스트리밍 답변 생성"""
    
    # 한국어 질문 → 영어 변환
    if language == "ko":
        en_question = convert_question_to_english(question)
    else:
        en_question = question
    
    # 영어 답변 생성
    english_answer = generate_english_answer(image, en_question)
    
    # 영어 답변 정리 (번역 전)
    english_answer = clean_english_answer(english_answer)
    
    # 한국어 번역
    if language == "ko":
        final_answer = translate_to_korean(english_answer)
    else:
        final_answer = english_answer
    
    # 한 글자씩 스트리밍
    for char in final_answer:
        yield char
        await asyncio.sleep(0.02)


# =============================================
# API 엔드포인트
# =============================================

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "running",
        "version": "2.2.0",
        "vl_model": VL_MODEL_ID,
        "translator": TRANSLATOR_MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "features": ["영어 답변 정리", "자연스러운 대화체", "빠른 응답"]
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """이미지 질문 답변"""
    try:
        image = process_image(request.image_base64)
        answer = generate_answer(image, request.question, request.language)
        return AnswerResponse(answer=answer, success=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return AnswerResponse(answer="오류가 발생했어요. 다시 시도해주세요.", success=False, error=str(e))


@app.post("/api/ask-stream")
async def ask_question_stream(request: QuestionRequest):
    """이미지 질문 답변 (스트리밍)"""
    try:
        image = process_image(request.image_base64)
        
        async def event_generator():
            async for chunk in generate_stream(image, request.question, request.language):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))


@app.post("/api/describe-stream")
async def describe_image_stream(request: QuestionRequest):
    """이미지 전체 설명 (스트리밍)"""
    try:
        image = process_image(request.image_base64)
        question = "Describe this landscape scene in detail."
        
        async def event_generator():
            async for chunk in generate_stream(image, question, request.language):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))


@app.post("/api/describe", response_model=AnswerResponse)
async def describe_image(request: QuestionRequest):
    """이미지 전체 설명"""
    try:
        image = process_image(request.image_base64)
        question = "Describe this landscape scene in detail."
        answer = generate_answer(image, question, request.language)
        return AnswerResponse(answer=answer, success=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
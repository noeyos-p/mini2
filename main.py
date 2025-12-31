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
TRANSLATOR_MODEL_ID = "facebook/m2m100_418M"


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
    """번역 결과 정리 (풍경 설명 최적화)"""
    if not text:
        return ""
    
    # 중복 단어 제거
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    
    # 이상한 문자 제거
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 불필요한 번역투 표현 제거
    remove_patterns = [
        r'이미지에서\s*',
        r'사진에서\s*',
        r'그것은\s*',
        r'이것은\s*',
        r'^예[,.\s]*',
        r'^네[,.\s]*',
    ]
    for pattern in remove_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 자연스러운 한국어로 변환
    replacements = [
        # 존댓말 변환
        ('입니다.', '이에요.'),
        ('있습니다.', '있어요.'),
        ('없습니다.', '없어요.'),
        ('됩니다.', '돼요.'),
        ('합니다.', '해요.'),
        ('봅니다.', '봐요.'),
        ('습니다.', '어요.'),
        ('ㅂ니다.', '요.'),
        ('이다.', '이에요.'),
        ('있다.', '있어요.'),
        ('없다.', '없어요.'),
        ('한다.', '해요.'),
        ('보인다.', '보여요.'),
        ('된다.', '돼요.'),
        
        # 풍경 관련 자연스러운 표현
        ('반사', '비치고'),
        ('반영', '비치고'),
        ('위치하고', '있고'),
        ('존재하고', '있고'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    
    # 문장 정리
    text = text.strip()
    text = re.sub(r'^[.,\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 문장 끝 자연스럽게
    if text and not text[-1] in '.!?요':
        if text[-1] in '다':
            text = text[:-1] + '요.'
        else:
            text += '요.'
    
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
        
        # 2. 번역 모델 로드
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
        # 소스 언어 설정 (영어)
        translator_tokenizer.src_lang = "en"
        
        # 한 번에 번역 (문장 분리 X → 더 자연스러움)
        inputs = translator_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        )
        
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = translator.generate(
                **inputs,
                forced_bos_token_id=translator_tokenizer.get_lang_id("ko"),
                max_length=256,
                num_beams=3,  # 빔 서치로 품질 향상
            )
        
        translated = translator_tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0]
        
        # 후처리
        return clean_translation(translated)
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # 번역 실패 시 원문 반환


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


# 영어 시스템 프롬프트 (풍경 설명 최적화)
SYSTEM_PROMPT = """You are describing a scenic landscape for a visually impaired person.

Describe in this order:
1. Overall scene (mountain, ocean, forest, city, etc.)
2. Sky and lighting (sunrise, sunset, cloudy, clear)
3. Main elements from front to back
4. Colors and atmosphere
5. Any notable details

Be vivid but natural. Around 2-3 sentences."""


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
            max_new_tokens=80,  # 풍경 설명은 좀 더 길게
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
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
    print(f"🇺🇸 영어 답변: {english_answer}")
    
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
        "version": "2.0.0",
        "vl_model": VL_MODEL_ID,
        "translator": TRANSLATOR_MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "features": ["질문 매핑 개선", "복합 질문 처리", "번역 후처리", "친근한 말투"]
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
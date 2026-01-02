"""
시각장애인을 위한 이미지/비디오 질문-응답 API
Qwen2-VL-2B (영어) + M2M100 번역 (한국어)
지원: 이미지 + 비디오 (10초 이내 권장)
"""

import io
import base64
import re
import asyncio
import gc
import tempfile
import os
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
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

# 비디오 설정
MAX_VIDEO_DURATION = 10  # 최대 10초
VIDEO_FPS = 1.0  # 1초당 1프레임 추출


# =============================================
# 질문 매핑 (우선순위: 긴 패턴부터)
# =============================================

SUBJECT_MAP = {
    "산": "the mountain", "바다": "the ocean", "호수": "the lake",
    "강": "the river", "하늘": "the sky", "구름": "the clouds",
    "나무": "the trees", "숲": "the forest", "꽃": "the flowers",
    "해": "the sun", "달": "the moon", "별": "the stars",
    "눈": "the snow", "비": "the rain", "안개": "the fog",
    "노을": "the sunset", "일출": "the sunrise",
    "언덕": "the hill", "절벽": "the cliff", "해변": "the beach",
    "바위": "the rocks", "폭포": "the waterfall", "들판": "the field",
    "건물": "the building", "집": "the house", "다리": "the bridge",
    "길": "the road", "배": "the boat", "등대": "the lighthouse",
    "사람": "the person", "고양이": "the cat", "강아지": "the dog",
    "개": "the dog", "새": "the bird", "차": "the car",
}

QUESTION_MAP = [
    ("뭐가 있어", "Describe this scene."),
    ("뭐가 보여", "Describe this scene."),
    ("뭐 있어", "Describe this scene."),
    ("설명해", "Describe this scene in detail."),
    ("뭐 해", "What is happening?"),
    ("뭐해", "What is happening?"),
    ("뭐하고 있어", "What is happening?"),
    ("무슨 일", "What is happening?"),
    ("움직", "What is moving?"),
    ("하늘", "Describe the sky."),
    ("날씨", "What is the weather like?"),
    ("분위기", "What is the mood or atmosphere?"),
    ("무슨 색", "What colors do you see?"),
    ("색깔", "What are the main colors?"),
    ("몇 명", "How many people are there?"),
    ("몇 마리", "How many animals are there?"),
    ("어디", "What place is this?"),
    ("어때", "How does this look?"),
]


def convert_question_to_english(question: str) -> str:
    """한국어 질문을 영어로 변환"""
    subject = "it"
    found_subject_ko = None
    for ko, en in SUBJECT_MAP.items():
        if ko in question:
            subject = en
            found_subject_ko = ko
            break
    
    for pattern, en_template in QUESTION_MAP:
        if pattern in question:
            return en_template.replace("{subject}", subject)
    
    if found_subject_ko:
        return f"Describe {subject}."
    
    return "Describe this scene."


# =============================================
# 번역 후처리
# =============================================

def clean_translation(text: str) -> str:
    """번역 결과 정리"""
    if not text:
        return ""
    
    # 마크다운 제거
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # 불필요한 표현 제거
    remove_patterns = [
        r'이미지에서\s*', r'사진에서\s*', r'그림은\s*',
        r'비디오에서\s*', r'영상에서\s*',
        r'그것은\s*', r'이것은\s*',
        r'\(.*?\)', r'프레임의\s*',
    ]
    for pattern in remove_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 번역 오류 수정
    replacements = [
        ('있습니다.', '있네요.'), ('보입니다.', '보이네요.'),
        ('입니다.', '이에요.'), ('합니다.', '해요.'),
        ('표지판', '무늬'), ('냄비', '털'), ('벽돌', '무늬'),
        ('두 개의 고양이', '고양이 두 마리'),
        ('두 개의', '두 '),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    
    # 정리
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(';', '.')
    
    # 2문장 제한
    sentences = re.split(r'(?<=[.!?요])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 2:
        text = ' '.join(sentences[:2])
    
    # 문장 끝 처리
    if text and not text[-1] in '.!?요':
        text += '요.'
    
    return text.strip()


# =============================================
# 모델 로드
# =============================================

def load_models():
    """모델 로드"""
    global vl_model, vl_processor, translator, translator_tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # VL 모델
        print(f"📦 Loading VL model: {VL_MODEL_ID}")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        vl_processor = AutoProcessor.from_pretrained(
            VL_MODEL_ID, trust_remote_code=True,
            min_pixels=256*28*28, max_pixels=512*28*28,
        )
        vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            VL_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        print("✅ VL model loaded!")
        
        # 번역 모델
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
    """영어를 한국어로 번역"""
    global translator, translator_tokenizer, device
    
    if not text.strip():
        return ""
    
    try:
        translator_tokenizer.src_lang = "en"
        inputs = translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = translator.generate(
                **inputs,
                forced_bos_token_id=translator_tokenizer.get_lang_id("ko"),
                max_length=128, num_beams=3,
            )
        
        translated = translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return clean_translation(translated)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# =============================================
# FastAPI
# =============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    global vl_model, vl_processor, translator, translator_tokenizer
    del vl_model, vl_processor, translator, translator_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


app = FastAPI(title="Vision Assistant API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class QuestionRequest(BaseModel):
    image_base64: str
    question: str
    language: str = "ko"


class VideoQuestionRequest(BaseModel):
    video_base64: str
    question: str
    language: str = "ko"


class AnswerResponse(BaseModel):
    answer: str
    success: bool
    error: Optional[str] = None


def process_image(image_data: str) -> Image.Image:
    """Base64 이미지 디코딩"""
    if "," in image_data:
        image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def process_video(video_data: str) -> str:
    """Base64 비디오를 임시 파일로 저장"""
    if "," in video_data:
        video_data = video_data.split(",")[1]
    video_bytes = base64.b64decode(video_data)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        return f.name


# 시스템 프롬프트
IMAGE_PROMPT = """Say what you see in ONE simple sentence. 10 words maximum.
Example: "A cat and dog sitting on grass."
Do NOT describe colors, positions, shadows, or background."""

VIDEO_PROMPT = """Describe what is happening in this video in 1-2 simple sentences.
Focus on the main action or movement. 20 words maximum."""


def clean_english_answer(text: str) -> str:
    """영어 답변 정리"""
    if not text:
        return ""
    
    text = re.sub(r'\([^)]*\)', '', text)
    remove_phrases = [
        r'which\s+.*?[,.]', r'with\s+shadows?\s+.*?[,.]',
        r'in\s+the\s+background.*?[,.]', r'off[\s-]camera.*?[,.]',
    ]
    for pattern in remove_phrases:
        text = re.sub(pattern, '.', text, flags=re.IGNORECASE)
    
    sentences = re.split(r'[.!?;]', text)
    if sentences:
        text = sentences[0].strip()
    
    if text and not text.endswith('.'):
        text += '.'
    
    return text.strip()


def generate_english_answer(image: Image.Image, question: str) -> str:
    """이미지 설명 생성"""
    global vl_model, vl_processor, device
    from qwen_vl_utils import process_vision_info
    
    messages = [
        {"role": "system", "content": IMAGE_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}
    ]
    
    text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vl_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    if device == "cuda":
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        generated_ids = vl_model.generate(**inputs, max_new_tokens=30, do_sample=False, repetition_penalty=1.5)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    answer = vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer.strip()


def generate_english_answer_video(video_path: str, question: str) -> str:
    """비디오 설명 생성"""
    global vl_model, vl_processor, device
    from qwen_vl_utils import process_vision_info
    
    messages = [
        {"role": "system", "content": VIDEO_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": VIDEO_FPS},
            {"type": "text", "text": question},
        ]}
    ]
    
    text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vl_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    if device == "cuda":
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        generated_ids = vl_model.generate(**inputs, max_new_tokens=50, do_sample=False, repetition_penalty=1.5)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    answer = vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer.strip()


def generate_answer(image: Image.Image, question: str, language: str = "ko") -> str:
    """이미지 답변 생성"""
    if language == "ko":
        en_question = convert_question_to_english(question)
        print(f"🔄 질문: '{question}' → '{en_question}'")
    else:
        en_question = question
    
    english_answer = generate_english_answer(image, en_question)
    print(f"🇺🇸 영어: {english_answer}")
    
    english_answer = clean_english_answer(english_answer)
    
    if language == "ko":
        korean_answer = translate_to_korean(english_answer)
        print(f"🇰🇷 한국어: {korean_answer}")
        return korean_answer
    return english_answer


def generate_answer_video(video_path: str, question: str, language: str = "ko") -> str:
    """비디오 답변 생성"""
    if language == "ko":
        en_question = convert_question_to_english(question)
        print(f"🔄 질문: '{question}' → '{en_question}'")
    else:
        en_question = question
    
    english_answer = generate_english_answer_video(video_path, en_question)
    print(f"🇺🇸 영어: {english_answer}")
    
    english_answer = clean_english_answer(english_answer)
    
    if language == "ko":
        korean_answer = translate_to_korean(english_answer)
        print(f"🇰🇷 한국어: {korean_answer}")
        return korean_answer
    return english_answer


async def generate_stream(image: Image.Image, question: str, language: str = "ko") -> AsyncGenerator[str, None]:
    """이미지 스트리밍"""
    if language == "ko":
        en_question = convert_question_to_english(question)
    else:
        en_question = question
    
    english_answer = generate_english_answer(image, en_question)
    english_answer = clean_english_answer(english_answer)
    final_answer = translate_to_korean(english_answer) if language == "ko" else english_answer
    
    for char in final_answer:
        yield char
        await asyncio.sleep(0.02)


async def generate_stream_video(video_path: str, question: str, language: str = "ko") -> AsyncGenerator[str, None]:
    """비디오 스트리밍"""
    if language == "ko":
        en_question = convert_question_to_english(question)
    else:
        en_question = question
    
    english_answer = generate_english_answer_video(video_path, en_question)
    english_answer = clean_english_answer(english_answer)
    final_answer = translate_to_korean(english_answer) if language == "ko" else english_answer
    
    for char in final_answer:
        yield char
        await asyncio.sleep(0.02)


# =============================================
# API 엔드포인트
# =============================================

@app.get("/")
async def root():
    return {
        "status": "running", "version": "3.0.0",
        "vl_model": VL_MODEL_ID, "translator": TRANSLATOR_MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "features": ["이미지 인식", "비디오 인식", "스트리밍 응답"]
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        image = process_image(request.image_base64)
        answer = generate_answer(image, request.question, request.language)
        return AnswerResponse(answer=answer, success=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))


@app.post("/api/ask-stream")
async def ask_question_stream(request: QuestionRequest):
    try:
        image = process_image(request.image_base64)
        
        async def event_generator():
            async for chunk in generate_stream(image, request.question, request.language):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    except Exception as e:
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))


@app.post("/api/describe-stream")
async def describe_image_stream(request: QuestionRequest):
    try:
        image = process_image(request.image_base64)
        
        async def event_generator():
            async for chunk in generate_stream(image, "Describe this scene.", request.language):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    except Exception as e:
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))


# 비디오 API

@app.post("/api/ask-video", response_model=AnswerResponse)
async def ask_video_question(request: VideoQuestionRequest):
    video_path = None
    try:
        video_path = process_video(request.video_base64)
        answer = generate_answer_video(video_path, request.question, request.language)
        return AnswerResponse(answer=answer, success=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


@app.post("/api/ask-video-stream")
async def ask_video_question_stream(request: VideoQuestionRequest):
    video_path = None
    try:
        video_path = process_video(request.video_base64)
        
        async def event_generator():
            async for chunk in generate_stream_video(video_path, request.question, request.language):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
        
        return StreamingResponse(event_generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    except Exception as e:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))


@app.post("/api/describe-video-stream")
async def describe_video_stream(request: VideoQuestionRequest):
    video_path = None
    try:
        video_path = process_video(request.video_base64)
        
        async def event_generator():
            async for chunk in generate_stream_video(video_path, "What is happening in this video?", request.language):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
        
        return StreamingResponse(event_generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    except Exception as e:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        return AnswerResponse(answer="오류가 발생했어요.", success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
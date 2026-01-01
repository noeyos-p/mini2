import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

// Web Speech API 타입 정의
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}

interface SpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: ((this: SpeechRecognition, ev: Event) => void) | null;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
  onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => void) | null;
  onend: ((this: SpeechRecognition, ev: Event) => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}

// API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ConversationItem {
  type: 'question' | 'answer';
  text: string;
  timestamp: Date;
  isStreaming?: boolean;
}

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [cameraActive, setCameraActive] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [isListening, setIsListening] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const questionInputRef = useRef<HTMLTextAreaElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // 대화 끝으로 스크롤
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation, streamingText]);

  // TTS 음성 출력
  const speak = useCallback((text: string) => {
    if (!ttsEnabled || !('speechSynthesis' in window)) return;

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ko-KR';
    utterance.rate = 0.9;
    utterance.pitch = 1;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, [ttsEnabled]);

  // TTS 중지
  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  // STT 초기화
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = 'ko-KR';
      recognition.continuous = false;
      recognition.interimResults = true;

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('');

        setQuestion(transcript);

        // 최종 결과면 자동 전송 (선택적)
        if (event.results[event.results.length - 1].isFinal) {
          // 자동 전송 원하면 아래 주석 해제
          // setTimeout(() => handleSubmit(), 500);
        }
      };

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('STT Error:', event.error);
        setIsListening(false);
        if (event.error === 'not-allowed') {
          setError('마이크 권한이 필요합니다. 브라우저 설정에서 허용해주세요.');
          speak('마이크 권한이 필요합니다.');
        }
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, [speak]);

  // 음성 인식 시작/중지
  const toggleListening = useCallback(() => {
    if (!recognitionRef.current) {
      setError('이 브라우저는 음성 인식을 지원하지 않습니다.');
      speak('음성 인식을 지원하지 않는 브라우저입니다.');
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      try {
        recognitionRef.current.start();
        speak('말씀하세요.');
      } catch (err) {
        console.error('STT start error:', err);
      }
    }
  }, [isListening, speak]);

  // 스트리밍 중지
  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  // 카메라 시작
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });

      streamRef.current = stream;
      setCameraActive(true);  // 먼저 true로!

    } catch (err) {
      console.error('Camera error:', err);
      setError('카메라를 사용할 수 없습니다.');
      speak('카메라를 사용할 수 없습니다.');
    }
  }, [speak]);

  // 카메라 활성화되면 stream 연결
  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(console.error);
      speak('카메라가 활성화되었습니다.');
    }
  }, [cameraActive, speak]);

  // 카메라 중지
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, []);

  // 사진 촬영
  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0);
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      setImage(imageData);
      setConversation([]);
      stopCamera();
      speak('사진이 촬영되었습니다. 질문을 입력해주세요.');

      setTimeout(() => questionInputRef.current?.focus(), 100);
    }
  }, [stopCamera, speak]);

  // 파일 업로드
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('이미지 파일만 업로드할 수 있습니다.');
      speak('이미지 파일만 업로드할 수 있습니다.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setImage(result);
      setConversation([]);
      setError(null);
      speak('이미지가 업로드되었습니다. 질문을 입력해주세요.');

      setTimeout(() => questionInputRef.current?.focus(), 100);
    };
    reader.readAsDataURL(file);
  }, [speak]);

  // 스트리밍 질문 제출
  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();

    if (!image) {
      setError('먼저 이미지를 촬영하거나 업로드해주세요.');
      speak('먼저 이미지를 촬영하거나 업로드해주세요.');
      return;
    }

    if (!question.trim()) {
      setError('질문을 입력해주세요.');
      speak('질문을 입력해주세요.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setStreamingText('');
    speak('답변을 생성 중입니다.');

    // 질문 추가
    const newQuestion: ConversationItem = {
      type: 'question',
      text: question,
      timestamp: new Date(),
    };
    setConversation(prev => [...prev, newQuestion]);
    const currentQuestion = question;
    setQuestion('');

    // AbortController 설정
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_URL}/api/ask-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: image,
          question: currentQuestion,
          language: 'ko',
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error('서버 응답 오류');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullText = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') {
                // 스트리밍 완료
                break;
              }
              fullText += data;
              setStreamingText(fullText);
            }
          }
        }
      }

      // 스트리밍 완료 후 대화에 추가
      const newAnswer: ConversationItem = {
        type: 'answer',
        text: fullText,
        timestamp: new Date(),
      };
      setConversation(prev => [...prev, newAnswer]);
      setStreamingText('');
      speak(fullText);

    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        console.log('스트리밍 중단됨');
      } else {
        console.error('API error:', err);
        const errorMessage = '서버 연결에 실패했습니다. 다시 시도해주세요.';
        setError(errorMessage);
        speak(errorMessage);
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [image, question, speak]);

  // 이미지 전체 설명 요청 (스트리밍)
  const handleDescribeImage = useCallback(async () => {
    if (!image) {
      speak('먼저 이미지를 촬영하거나 업로드해주세요.');
      return;
    }

    setIsLoading(true);
    setStreamingText('');
    speak('이미지를 분석 중입니다.');

    const describeQuestion: ConversationItem = {
      type: 'question',
      text: '이 이미지를 전체적으로 설명해주세요.',
      timestamp: new Date(),
    };
    setConversation(prev => [...prev, describeQuestion]);

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_URL}/api/describe-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: image,
          question: '',
          language: 'ko',
        }),
        signal: abortControllerRef.current.signal,
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullText = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') break;
              fullText += data;
              setStreamingText(fullText);
            }
          }
        }
      }

      const newAnswer: ConversationItem = {
        type: 'answer',
        text: fullText,
        timestamp: new Date(),
      };
      setConversation(prev => [...prev, newAnswer]);
      setStreamingText('');
      speak(fullText);

    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        console.error('API error:', err);
        speak('이미지 설명을 가져올 수 없습니다.');
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [image, speak]);

  // 새 이미지 시작
  const handleReset = useCallback(() => {
    setImage(null);
    setConversation([]);
    setQuestion('');
    setError(null);
    setStreamingText('');
    stopCamera();
    stopStreaming();
    speak('새로운 이미지를 촬영하거나 업로드해주세요.');
  }, [stopCamera, stopStreaming, speak]);

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      stopCamera();
      stopSpeaking();
      stopStreaming();
    };
  }, [stopCamera, stopSpeaking, stopStreaming]);

  // 키보드 단축키
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'Enter') {
        handleSubmit();
      }
      if (e.key === 'Escape') {
        stopSpeaking();
        stopStreaming();
      }
      if (e.ctrlKey && e.key === 'd') {
        e.preventDefault();
        handleDescribeImage();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSubmit, stopSpeaking, stopStreaming, handleDescribeImage]);

  return (
    <div className="app" role="application" aria-label="시각 도우미">
      <header className="header">
        <h1>👁️ 시각 도우미</h1>
        <p className="subtitle">이미지에 대해 무엇이든 물어보세요</p>

        <div className="tts-toggle">
          <label htmlFor="tts-checkbox" className="tts-label">
            <input
              id="tts-checkbox"
              type="checkbox"
              checked={ttsEnabled}
              onChange={(e) => setTtsEnabled(e.target.checked)}
              aria-label="음성 안내 켜기/끄기"
            />
            <span>🔊 음성 안내 {ttsEnabled ? '켜짐' : '꺼짐'}</span>
          </label>
          {isSpeaking && (
            <button
              onClick={stopSpeaking}
              className="stop-speaking-btn"
              aria-label="음성 중지"
            >
              ⏹️ 음성 중지
            </button>
          )}
        </div>
      </header>

      <main className="main-content">
        {/* 이미지 영역 */}
        <section className="image-section" aria-label="이미지 영역">
          {!image && !cameraActive && (
            <div className="image-input-area">
              <button
                onClick={startCamera}
                className="btn btn-primary btn-large"
                aria-label="카메라로 사진 촬영"
              >
                📷 카메라로 촬영
              </button>

              <span className="or-divider">또는</span>

              <button
                onClick={() => fileInputRef.current?.click()}
                className="btn btn-secondary btn-large"
                aria-label="갤러리에서 이미지 선택"
              >
                🖼️ 갤러리에서 선택
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden-input"
                aria-hidden="true"
              />
            </div>
          )}

          {cameraActive && (
            <div className="camera-area">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="camera-preview"
                aria-label="카메라 미리보기"
              />
              <div className="camera-controls">
                <button
                  onClick={capturePhoto}
                  className="btn btn-capture"
                  aria-label="사진 촬영"
                >
                  📸 촬영
                </button>
                <button
                  onClick={stopCamera}
                  className="btn btn-cancel"
                  aria-label="취소"
                >
                  ❌ 취소
                </button>
              </div>
            </div>
          )}

          {image && (
            <div className="image-preview-area">
              <img
                src={image}
                alt="업로드된 이미지"
                className="image-preview"
              />
              <div className="image-actions">
                <button
                  onClick={handleDescribeImage}
                  className="btn btn-describe"
                  disabled={isLoading}
                  aria-label="이미지 전체 설명 듣기"
                >
                  📝 전체 설명 듣기
                </button>
                <button
                  onClick={handleReset}
                  className="btn btn-reset"
                  aria-label="새 이미지로 시작"
                >
                  🔄 새 이미지
                </button>
              </div>
            </div>
          )}

          <canvas ref={canvasRef} className="hidden-canvas" aria-hidden="true" />
        </section>

        {/* 대화 영역 */}
        {image && (
          <section className="conversation-section" aria-label="대화 영역">
            <div
              className="conversation-list"
              role="log"
              aria-live="polite"
              aria-label="질문과 답변 목록"
            >
              {conversation.length === 0 && !streamingText && (
                <p className="empty-message">
                  이미지에 대해 궁금한 점을 질문해보세요.
                  <br />
                  예: "사람이 몇 명 있어?", "날씨가 어때?", "어떤 색이 보여?"
                </p>
              )}

              {conversation.map((item, index) => (
                <div
                  key={index}
                  className={`message ${item.type}`}
                  role={item.type === 'answer' ? 'status' : undefined}
                >
                  <span className="message-icon">
                    {item.type === 'question' ? '❓' : '💬'}
                  </span>
                  <p className="message-text">{item.text}</p>
                  {item.type === 'answer' && (
                    <button
                      onClick={() => speak(item.text)}
                      className="btn-speak"
                      aria-label="이 답변 다시 듣기"
                    >
                      🔊
                    </button>
                  )}
                </div>
              ))}

              {/* 스트리밍 중인 답변 */}
              {streamingText && (
                <div className="message answer streaming">
                  <span className="message-icon">💬</span>
                  <p className="message-text">
                    {streamingText}
                    <span className="cursor">▌</span>
                  </p>
                </div>
              )}

              {isLoading && !streamingText && (
                <div className="message answer loading" aria-live="assertive">
                  <span className="loading-spinner" aria-hidden="true">⏳</span>
                  <p>답변을 생성하고 있습니다...</p>
                </div>
              )}

              <div ref={conversationEndRef} />
            </div>

            {/* 질문 입력 */}
            <form onSubmit={handleSubmit} className="question-form">
              <label htmlFor="question-input" className="sr-only">
                질문 입력
              </label>
              <div className="input-wrapper">
                <textarea
                  ref={questionInputRef}
                  id="question-input"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="이미지에 대해 질문하세요... (마이크 버튼으로 음성 입력)"
                  className="question-input"
                  disabled={isLoading}
                  rows={2}
                  aria-describedby="question-help"
                />
                <button
                  type="button"
                  onClick={toggleListening}
                  className={`btn btn-mic ${isListening ? 'listening' : ''}`}
                  disabled={isLoading}
                  aria-label={isListening ? '음성 인식 중지' : '음성으로 질문하기'}
                >
                  {isListening ? '🔴' : '🎤'}
                </button>
              </div>
              <p id="question-help" className="sr-only">
                질문을 입력하고 Ctrl+Enter 또는 보내기 버튼을 눌러 전송하세요.
              </p>
              <div className="form-buttons">
                <button
                  type="submit"
                  className="btn btn-send"
                  disabled={isLoading || !question.trim()}
                  aria-label="질문 보내기"
                >
                  {isLoading ? '⏳' : '📤'} 보내기
                </button>
                {isLoading && (
                  <button
                    type="button"
                    onClick={stopStreaming}
                    className="btn btn-stop"
                    aria-label="생성 중지"
                  >
                    ⏹️ 중지
                  </button>
                )}
              </div>
            </form>
          </section>
        )}

        {/* 에러 메시지 */}
        {error && (
          <div
            className="error-message"
            role="alert"
            aria-live="assertive"
          >
            ⚠️ {error}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          🎤 마이크 버튼으로 음성 질문 | Ctrl+Enter (전송) | ESC (중지)
        </p>
      </footer>
    </div>
  );
}

export default App;
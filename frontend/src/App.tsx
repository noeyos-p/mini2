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

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ConversationItem {
  type: 'question' | 'answer';
  text: string;
  timestamp: Date;
}

function App() {
  // 미디어 상태
  const [image, setImage] = useState<string | null>(null);
  const [video, setVideo] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<'image' | 'video'>('image');
  
  // UI 상태
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState('');
  
  // 음성 상태
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [isListening, setIsListening] = useState(false);
  
  // 카메라/녹화 상태
  const [cameraActive, setCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoFileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const questionInputRef = useRef<HTMLTextAreaElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const recordingTimerRef = useRef<number | null>(null);

  // 스크롤
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation, streamingText]);

  // TTS
  const speak = useCallback((text: string) => {
    if (!ttsEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ko-KR';
    utterance.rate = 0.9;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }, [ttsEnabled]);

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
      recognition.onstart = () => setIsListening(true);
      recognition.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = Array.from(event.results).map(result => result[0].transcript).join('');
        setQuestion(transcript);
      };
      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('STT Error:', event.error);
        setIsListening(false);
        if (event.error === 'not-allowed') {
          setError('마이크 권한이 필요합니다.');
          speak('마이크 권한이 필요합니다.');
        }
      };
      recognition.onend = () => setIsListening(false);
      recognitionRef.current = recognition;
    }
    return () => { if (recognitionRef.current) recognitionRef.current.abort(); };
  }, [speak]);

  const toggleListening = useCallback(() => {
    if (!recognitionRef.current) {
      setError('음성 인식을 지원하지 않는 브라우저입니다.');
      return;
    }
    if (isListening) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
      speak('말씀하세요.');
    }
  }, [isListening, speak]);

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
        audio: true, // 비디오 녹화용
      });
      streamRef.current = stream;
      setCameraActive(true);
    } catch (err) {
      console.error('Camera error:', err);
      setError('카메라를 사용할 수 없습니다.');
      speak('카메라를 사용할 수 없습니다.');
    }
  }, [speak]);

  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(console.error);
      speak('카메라가 활성화되었습니다.');
    }
  }, [cameraActive, speak]);

  const stopCamera = useCallback(() => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsRecording(false);
    setRecordingTime(0);
  }, [isRecording]);

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
      setVideo(null);
      setMediaType('image');
      setConversation([]);
      stopCamera();
      speak('사진이 촬영되었습니다.');
      setTimeout(() => questionInputRef.current?.focus(), 100);
    }
  }, [stopCamera, speak]);

// 1️⃣ 녹화 중지 (먼저 선언)
const stopRecording = useCallback((): void => {
  if (!isRecording) return;

  if (recordingTimerRef.current) {
    clearInterval(recordingTimerRef.current);
    recordingTimerRef.current = null;
  }

  mediaRecorderRef.current?.stop();
  setIsRecording(false);
}, [isRecording]);


// 2️⃣ 녹화 시작
const startRecording = useCallback((): void => {
  if (!streamRef.current) return;

  recordedChunksRef.current = [];

  const mediaRecorder = new MediaRecorder(streamRef.current, {
    mimeType: 'video/webm',
  });

  mediaRecorder.ondataavailable = (event: BlobEvent) => {
    if (event.data.size > 0) {
      recordedChunksRef.current.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
    const reader = new FileReader();

    reader.onloadend = () => {
      const base64 = reader.result as string;
      setVideo(base64);
      setImage(null);
      setMediaType('video');
      setConversation([]);
      stopCamera();
      speak('영상이 녹화되었습니다.');
      setTimeout(() => questionInputRef.current?.focus(), 100);
    };

    reader.readAsDataURL(blob);
  };

  mediaRecorderRef.current = mediaRecorder;
  mediaRecorder.start();

  setIsRecording(true);
  setRecordingTime(0);

  // ⏱️ 10초 자동 중지
  recordingTimerRef.current = window.setInterval(() => {
    setRecordingTime(prev => {
      if (prev >= 9) {
        stopRecording(); // ✅ 자동 중지 + 버튼 상태 복구
        return 10;
      }
      return prev + 1;
    });
  }, 1000);

  speak('녹화를 시작합니다. 최대 10초입니다.');
}, [speak, stopCamera, stopRecording]);


// 3️⃣ 녹화 토글 버튼용
const toggleRecording = useCallback((): void => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}, [isRecording, startRecording, stopRecording]);



  // 이미지 파일 업로드
  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('이미지 파일만 업로드할 수 있습니다.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
      setVideo(null);
      setMediaType('image');
      setConversation([]);
      setError(null);
      speak('이미지가 업로드되었습니다.');
      setTimeout(() => questionInputRef.current?.focus(), 100);
    };
    reader.readAsDataURL(file);
  }, [speak]);

  // 비디오 파일 업로드
  const handleVideoUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      setError('비디오 파일만 업로드할 수 있습니다.');
      return;
    }
    // 파일 크기 제한 (50MB)
    if (file.size > 50 * 1024 * 1024) {
      setError('비디오 파일이 너무 큽니다. (최대 50MB)');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      setVideo(e.target?.result as string);
      setImage(null);
      setMediaType('video');
      setConversation([]);
      setError(null);
      speak('영상이 업로드되었습니다.');
      setTimeout(() => questionInputRef.current?.focus(), 100);
    };
    reader.readAsDataURL(file);
  }, [speak]);

  // 질문 제출
  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();

    if (!image && !video) {
      setError('먼저 이미지나 영상을 준비해주세요.');
      speak('먼저 이미지나 영상을 준비해주세요.');
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

    const newQuestion: ConversationItem = {
      type: 'question',
      text: question,
      timestamp: new Date(),
    };
    setConversation(prev => [...prev, newQuestion]);
    const currentQuestion = question;
    setQuestion('');

    abortControllerRef.current = new AbortController();

    try {
      const endpoint = mediaType === 'video' ? '/api/ask-video-stream' : '/api/ask-stream';
      const body = mediaType === 'video' 
        ? { video_base64: video, question: currentQuestion, language: 'ko' }
        : { image_base64: image, question: currentQuestion, language: 'ko' };

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) throw new Error('서버 응답 오류');

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
        setError('서버 연결에 실패했습니다.');
        speak('서버 연결에 실패했습니다.');
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [image, video, mediaType, question, speak]);

  // 전체 설명
  const handleDescribe = useCallback(async () => {
    if (!image && !video) {
      speak('먼저 이미지나 영상을 준비해주세요.');
      return;
    }

    setIsLoading(true);
    setStreamingText('');
    speak(mediaType === 'video' ? '영상을 분석 중입니다.' : '이미지를 분석 중입니다.');

    const describeQuestion: ConversationItem = {
      type: 'question',
      text: mediaType === 'video' ? '이 영상을 설명해주세요.' : '이 이미지를 설명해주세요.',
      timestamp: new Date(),
    };
    setConversation(prev => [...prev, describeQuestion]);

    abortControllerRef.current = new AbortController();

    try {
      const endpoint = mediaType === 'video' ? '/api/describe-video-stream' : '/api/describe-stream';
      const body = mediaType === 'video'
        ? { video_base64: video, question: '', language: 'ko' }
        : { image_base64: image, question: '', language: 'ko' };

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
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
        speak('설명을 가져올 수 없습니다.');
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [image, video, mediaType, speak]);

  // 리셋
  const handleReset = useCallback(() => {
    setImage(null);
    setVideo(null);
    setMediaType('image');
    setConversation([]);
    setQuestion('');
    setError(null);
    setStreamingText('');
    stopCamera();
    stopStreaming();
    speak('새로운 이미지나 영상을 준비해주세요.');
  }, [stopCamera, stopStreaming, speak]);

  // 정리
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
      if (e.ctrlKey && e.key === 'Enter') handleSubmit();
      if (e.key === 'Escape') { stopSpeaking(); stopStreaming(); }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSubmit, stopSpeaking, stopStreaming]);

  return (
    <div className="app" role="application" aria-label="시각 도우미">
      <header className="header">
        <h1>👁️ 시각 도우미</h1>
        <p className="subtitle">이미지와 영상에 대해 물어보세요</p>

        <div className="tts-toggle">
          <label htmlFor="tts-checkbox" className="tts-label">
            <input
              id="tts-checkbox"
              type="checkbox"
              checked={ttsEnabled}
              onChange={(e) => setTtsEnabled(e.target.checked)}
            />
            <span>🔊 음성 안내 {ttsEnabled ? '켜짐' : '꺼짐'}</span>
          </label>
          {isSpeaking && (
            <button onClick={stopSpeaking} className="stop-speaking-btn">
              ⏹️ 음성 중지
            </button>
          )}
        </div>
      </header>

      <main className="main-content">
        {/* 미디어 영역 */}
        <section className="image-section" aria-label="미디어 영역">
          {!image && !video && !cameraActive && (
            <div className="image-input-area">
              <div className="media-buttons">
                <button onClick={startCamera} className="btn btn-primary btn-large">
                  📷 카메라
                </button>
                <button onClick={() => fileInputRef.current?.click()} className="btn btn-secondary btn-large">
                  🖼️ 이미지
                </button>
                <button onClick={() => videoFileInputRef.current?.click()} className="btn btn-secondary btn-large">
                  🎬 영상
                </button>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden-input"
              />
              <input
                ref={videoFileInputRef}
                type="file"
                accept="video/*"
                onChange={handleVideoUpload}
                className="hidden-input"
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
    />

    {isRecording && (
      <div className="recording-indicator">
        🔴 녹화 중 {recordingTime}초 / 10초
      </div>
    )}

    <div className="camera-controls">
      <button
        onClick={capturePhoto}
        className="btn btn-capture"
        disabled={isRecording}
      >
        📸 사진
      </button>

      {/* ✅ 여기: 녹화 토글 버튼 */}
      <button
        type="button"
        onClick={toggleRecording}
        className={`btn ${isRecording ? 'btn-stop-record' : 'btn-record'}`}
      >
        {isRecording ? '⏹️ 녹화 중지' : '🎬 녹화 시작'}
      </button>

      <button onClick={stopCamera} className="btn btn-cancel">
        ❌ 취소
      </button>
    </div>
  </div>
)}


          {image && (
            <div className="image-preview-area">
              <div className="media-badge">📷 이미지</div>
              <img src={image} alt="업로드된 이미지" className="image-preview" />
              <div className="image-actions">
                <button onClick={handleDescribe} className="btn btn-describe" disabled={isLoading}>
                  📝 전체 설명
                </button>
                <button onClick={handleReset} className="btn btn-reset">
                  🔄 새로 시작
                </button>
              </div>
            </div>
          )}

          {video && (
            <div className="image-preview-area">
              <div className="media-badge">🎬 영상</div>
              <video src={video} controls className="video-preview" />
              <div className="image-actions">
                <button onClick={handleDescribe} className="btn btn-describe" disabled={isLoading}>
                  📝 전체 설명
                </button>
                <button onClick={handleReset} className="btn btn-reset">
                  🔄 새로 시작
                </button>
              </div>
            </div>
          )}

          <canvas ref={canvasRef} className="hidden-canvas" />
        </section>

        {/* 대화 영역 */}
        {(image || video) && (
          <section className="conversation-section" aria-label="대화 영역">
            <div className="conversation-list" role="log" aria-live="polite">
              {conversation.length === 0 && !streamingText && (
                <p className="empty-message">
                  {mediaType === 'video' ? '영상' : '이미지'}에 대해 질문해보세요.
                  <br />
                  예: "뭐가 보여?", "뭐하고 있어?", "날씨가 어때?"
                </p>
              )}

              {conversation.map((item, index) => (
                <div key={index} className={`message ${item.type}`}>
                  <span className="message-icon">{item.type === 'question' ? '❓' : '💬'}</span>
                  <p className="message-text">{item.text}</p>
                  {item.type === 'answer' && (
                    <button onClick={() => speak(item.text)} className="btn-speak">🔊</button>
                  )}
                </div>
              ))}

              {streamingText && (
                <div className="message answer streaming">
                  <span className="message-icon">💬</span>
                  <p className="message-text">
                    {streamingText}<span className="cursor">▌</span>
                  </p>
                </div>
              )}

              {isLoading && !streamingText && (
                <div className="message answer loading">
                  <span className="loading-spinner">⏳</span>
                  <p>답변을 생성하고 있습니다...</p>
                </div>
              )}

              <div ref={conversationEndRef} />
            </div>

            <form onSubmit={handleSubmit} className="question-form">
              <div className="input-wrapper">
                <textarea
                  ref={questionInputRef}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder={`${mediaType === 'video' ? '영상' : '이미지'}에 대해 질문하세요...`}
                  className="question-input"
                  disabled={isLoading}
                  rows={2}
                />
                <button
                  type="button"
                  onClick={toggleListening}
                  className={`btn btn-mic ${isListening ? 'listening' : ''}`}
                  disabled={isLoading}
                >
                  {isListening ? '🔴' : '🎤'}
                </button>
              </div>
              <div className="form-buttons">
                <button type="submit" className="btn btn-send" disabled={isLoading || !question.trim()}>
                  {isLoading ? '⏳' : '📤'} 보내기
                </button>
                {isLoading && (
                  <button type="button" onClick={stopStreaming} className="btn btn-stop">
                    ⏹️ 중지
                  </button>
                )}
              </div>
            </form>
          </section>
        )}

        {error && (
          <div className="error-message" role="alert">
            ⚠️ {error}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          🎤 마이크로 음성 질문 | 🎬 10초 영상 녹화 | Ctrl+Enter (전송) | ESC (중지)
        </p>
      </footer>
    </div>
  );
}

export default App;
import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { WaveformVisualizer } from './components/WaveformVisualizer';
import { AudioInputControls } from './components/AudioInputControls';
import { TranscriptIntelligence } from './components/TranscriptIntelligence';
import { TelemetryDrawer } from './components/TelemetryDrawer';
import { SessionHistory } from './components/SessionHistory';
import {
  fetchAIHealth,
  fetchRecentSessions,
  fetchSampleAudios,
  uploadAudioFile,
  processSpeechAudio,
} from './services/api';
import { AIHealthStatus, SampleAudio, SpeechProcessResult, StreamMessage, StreamSegment } from './types';
import { AlertTriangle, Radio } from 'lucide-react';

export const App: React.FC = () => {
  const [health, setHealth] = useState<AIHealthStatus | null>(null);
  const [samples, setSamples] = useState<SampleAudio[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [activeMode, setActiveMode] = useState<'stream' | 'batch'>('stream');
  const [selectedLanguage, setSelectedLanguage] = useState<string>('');

  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<SpeechProcessResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Streaming WebSocket state
  const [streamingSegments, setStreamingSegments] = useState<StreamSegment[]>([]);
  const [streamingTranscript, setStreamingTranscript] = useState('');
  const [streamingTranslation, setStreamingTranslation] = useState('');
  const [currentSessionId, setCurrentSessionId] = useState<string>('');

  const websocketRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const timerIntervalRef = useRef<any>(null);

  useEffect(() => {
    loadSystemData();
  }, []);

  const loadSystemData = async () => {
    try {
      const [h, s, sess] = await Promise.all([
        fetchAIHealth().catch(() => null),
        fetchSampleAudios().catch(() => []),
        fetchRecentSessions().catch(() => []),
      ]);
      if (h) setHealth(h);
      if (s) setSamples(s);
      if (sess) setSessions(sess);
    } catch (err) {
      console.warn('System data initial fetch:', err);
    }
  };

  const handleStartRecording = async () => {
    try {
      setErrorMessage(null);
      setStreamingSegments([]);
      setStreamingTranscript('');
      setStreamingTranslation('');
      setResult(null);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      mediaStreamRef.current = stream;

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/api/v1/speech/stream`;
      const ws = new WebSocket(wsUrl);
      websocketRef.current = ws;

      ws.onopen = () => {
        console.log('[PRISM Stream] WebSocket connected.');
        ws.send(
          JSON.stringify({
            type: 'start_session',
            language_hint: selectedLanguage || 'te',
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const msg: StreamMessage = JSON.parse(event.data);
          const msgType = (msg.type || '').toLowerCase();

          if (msg.session_id) {
            setCurrentSessionId(msg.session_id);
          }

          if (msgType === 'partial_transcript') {
            const partialText = msg.transcript || msg.text || '';
            setStreamingTranscript(partialText);

            // Update partial hypothesis in segment list without creating duplicate rows
            setStreamingSegments((prev) => {
              const segId = String(msg.segment_id || 'seg_partial');
              const filtered = prev.filter((s) => s.status !== 'partial' && s.segment_id !== segId);
              return [
                ...filtered,
                {
                  segment_id: segId,
                  start_time: msg.start_time || 0.0,
                  end_time: msg.end_time || 0.0,
                  status: 'partial',
                  language: msg.language || 'te',
                  transcript: partialText,
                  translation: '',
                  confidence: msg.confidence || 0.80,
                },
              ];
            });
          } else if (msgType === 'final_transcript' || msgType === 'final_segment') {
            const finalText = msg.transcript || msg.text || '';
            const finalTrans = msg.translation || '';
            const segId = String(msg.segment_id || `seg_${Date.now()}`);

            setStreamingSegments((prev) => {
              const filtered = prev.filter((s) => s.status !== 'partial' && s.segment_id !== segId);
              const newSegment: StreamSegment = {
                segment_id: segId,
                start_time: msg.start_time || 0.0,
                end_time: msg.end_time || 0.0,
                status: 'final',
                language: msg.language || 'te',
                transcript: finalText,
                translation: finalTrans,
                confidence: msg.confidence || 0.95,
              };
              const updated = [...filtered, newSegment];
              setStreamingTranscript(updated.map((s) => s.transcript).join(' '));
              setStreamingTranslation(updated.map((s) => s.translation).join(' '));
              return updated;
            });
          } else if (msgType === 'translation') {
            if (msg.segment_id && msg.translation) {
              const segId = String(msg.segment_id);
              setStreamingSegments((prev) =>
                prev.map((s) => (s.segment_id === segId ? { ...s, translation: msg.translation! } : s))
              );
            }
          } else if (msgType === 'session_completed') {
            if (msg.final_transcript) setStreamingTranscript(msg.final_transcript);
            if (msg.final_english_translation) setStreamingTranslation(msg.final_english_translation);
            loadSystemData();
          } else if (msgType === 'error') {
            setErrorMessage(msg.error || 'Streaming error');
          }
        } catch (e) {
          console.warn('WS parse error:', e);
        }
      };

      ws.onerror = (e) => {
        console.warn('WebSocket stream error:', e);
      };

      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = async (e) => {
        if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          const reader = new FileReader();
          reader.onload = () => {
            const base64Audio = (reader.result as string).split(',')[1];
            ws.send(
              JSON.stringify({
                type: 'audio_chunk',
                audio: base64Audio,
              })
            );
          };
          reader.readAsDataURL(e.data);
        }
      };

      mediaRecorder.start(400);
      setIsRecording(true);
      setRecordingDuration(0);

      timerIntervalRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      console.error('Microphone error:', err);
      setErrorMessage(
        'Unable to access microphone. Please check browser permissions or select a golden test sample below.'
      );
    }
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }

    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({ type: 'stop_session' }));
    }
  };

  const handleFileUpload = async (file: File) => {
    try {
      setIsProcessing(true);
      setErrorMessage(null);
      setStreamingSegments([]);
      setStreamingTranscript('');
      setStreamingTranslation('');

      const res = await uploadAudioFile(file, selectedLanguage);
      setResult(res);
      setCurrentSessionId(res.session_id);
      loadSystemData();
    } catch (err: any) {
      setErrorMessage(err.message || 'Error processing audio file.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSelectSample = async (sample: SampleAudio) => {
    try {
      setIsProcessing(true);
      setErrorMessage(null);
      setStreamingSegments([]);
      setSelectedLanguage(sample.language);

      const audioUrl = `/api/v1/speech/sample/${sample.filename}`;
      const response = await fetch(audioUrl);
      const blob = await response.blob();
      const file = new File([blob], sample.filename, { type: 'audio/wav' });

      const res = await uploadAudioFile(file, sample.language);
      setResult(res);
      setCurrentSessionId(res.session_id);
      loadSystemData();
    } catch (err: any) {
      setErrorMessage(err.message || 'Error running test sample inference.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between tactical-grid">
      <Header
        health={health}
        activeMode={activeMode}
        onModeChange={(mode) => setActiveMode(mode)}
      />

      <main className="max-w-7xl mx-auto px-4 py-6 w-full space-y-6 flex-1">
        {errorMessage && (
          <div className="p-4 rounded-xl bg-red-950/80 border border-red-500/50 flex items-center justify-between gap-3 text-red-200 text-sm">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-xs font-mono text-red-300 hover:text-white"
            >
              [DISMISS]
            </button>
          </div>
        )}

        <WaveformVisualizer
          isRecording={isRecording}
          audioStream={mediaStreamRef.current}
          isPlaying={isProcessing}
        />

        <AudioInputControls
          isRecording={isRecording}
          isProcessing={isProcessing}
          onStartRecording={handleStartRecording}
          onStopRecording={handleStopRecording}
          onFileUpload={handleFileUpload}
          onSelectSample={handleSelectSample}
          samples={samples}
          selectedLanguage={selectedLanguage}
          onLanguageChange={(lang) => setSelectedLanguage(lang)}
          recordingDuration={recordingDuration}
        />

        {isProcessing && (
          <div className="glass-panel p-4 flex items-center justify-center gap-3 font-mono text-sm text-amber-300 border-amber-500/40 animate-pulse">
            <Radio className="w-5 h-5 animate-spin" />
            <span>PROCESSING SPEECH PIPELINE (ASR + CODE-SWITCH + NMT)...</span>
          </div>
        )}

        <TranscriptIntelligence
          result={result}
          streamingSegments={streamingSegments}
          streamingTranscript={streamingTranscript}
          streamingTranslation={streamingTranslation}
          isStreaming={isRecording}
          sessionId={currentSessionId}
        />

        <TelemetryDrawer result={result} />

        <SessionHistory
          sessions={sessions}
          onSelectSession={(sess) => {
            if (sess.metadata && sess.metadata.timing) {
              setResult(sess.metadata);
            }
          }}
        />
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/90 py-4 text-center text-xs font-mono text-slate-500">
        PRISM (Police Investigation System Management) // 100% Offline Self-Hosted AI Architecture // Restricted Police Use Only
      </footer>
    </div>
  );
};

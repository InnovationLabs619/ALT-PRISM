import { AIHealthStatus, SampleAudio, SpeechProcessResult } from '../types';

const API_BASE = '/api/v1';

export async function fetchAIHealth(): Promise<AIHealthStatus> {
  const res = await fetch(`${API_BASE}/health/ai`);
  if (!res.ok) throw new Error('Failed to fetch AI Engine health');
  return res.json();
}

export async function fetchSampleAudios(): Promise<SampleAudio[]> {
  const res = await fetch(`${API_BASE}/speech/samples`);
  if (!res.ok) throw new Error('Failed to load golden test samples');
  return res.json();
}

export async function uploadAudioFile(
  file: File,
  languageHint?: string,
  targetLanguage: string = 'en'
): Promise<SpeechProcessResult> {
  const formData = new FormData();
  formData.append('file', file);
  if (languageHint) {
    formData.append('language_hint', languageHint);
  }
  formData.append('target_language', targetLanguage);

  const res = await fetch(`${API_BASE}/speech/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || 'Audio file upload failed');
  }

  return res.json();
}

export async function processSpeechAudio(
  audioData: File | Blob,
  languageHint?: string,
  targetLanguage: string = 'en'
): Promise<SpeechProcessResult> {
  const formData = new FormData();
  formData.append('file', audioData, 'audio_recording.wav');
  if (languageHint) {
    formData.append('language_hint', languageHint);
  }
  formData.append('target_language', targetLanguage);

  const res = await fetch(`${API_BASE}/speech/process`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || 'Audio processing failed');
  }

  return res.json();
}

export async function saveSessionDetails(
  sessionId: str,
  transcript?: string,
  englishTranslation?: string,
  officerNotes?: string
) {
  const res = await fetch(`${API_BASE}/speech/session/${sessionId}/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      transcript,
      english_translation: englishTranslation,
      officer_notes: officerNotes,
    }),
  });

  if (!res.ok) {
    throw new Error('Failed to save session updates');
  }
  return res.json();
}

export function getExportTxtUrl(sessionId: string): string {
  return `${API_BASE}/speech/session/${sessionId}/export/txt`;
}

export function getExportDocxUrl(sessionId: string): string {
  return `${API_BASE}/speech/session/${sessionId}/export/docx`;
}

export async function fetchRecentSessions() {
  const res = await fetch(`${API_BASE}/sessions`);
  if (!res.ok) return [];
  return res.json();
}

export interface TimingMetrics {
  audio_duration: number;
  preprocessing_latency: number;
  asr_latency: number;
  code_switch_latency: number;
  translation_latency: number;
  total_latency: number;
}

export interface AudioQuality {
  snr_db: number;
  quality_score: number;
  channels: number;
  sample_rate: number;
}

export interface ASRSegment {
  segment_id: number;
  start_sec: number;
  end_sec: number;
  text: string;
  confidence: number;
  language: string;
}

export interface SpeechProcessResult {
  session_id: string;
  language: string;
  language_confidence: number;
  is_code_switched: boolean;
  languages: string[];
  transcript: string;
  english_translation: string;
  segments: ASRSegment[];
  timing: TimingMetrics;
  models: {
    asr: string;
    asr_version?: string;
    translation: string;
    translation_version?: string;
    pipeline_version?: string;
  };
  audio_quality: AudioQuality;
  status: string;
  error_message?: string | null;
}

export interface SampleAudio {
  audio_id: string;
  filename: string;
  language: string;
  language_name: string;
  duration_sec: number;
  incident_category: string;
  expected_transcript: string;
  expected_translation: string;
  is_code_switched: boolean;
}

export interface AIHealthStatus {
  status: string;
  device: string;
  vram_allocated_mb?: number | null;
  asr: {
    status: string;
    model: string;
    version: string;
  };
  translation: {
    status: string;
    model: string;
    version: string;
  };
  metrics: {
    inferences_count: number;
    audio_seconds_processed: number;
    uptime_seconds: number;
  };
}

export interface OfficerProfile {
  username: string;
  full_name: string;
  badge_number: string;
  department: string;
  role: string;
}

export interface StreamSegment {
  segment_id: string;
  start_time: number;
  end_time: number;
  status: 'partial' | 'final';
  language: string;
  transcript: string;
  translation: string;
  confidence: number;
}

export interface StreamMessage {
  type: string;
  session_id?: string;
  segment_id?: string | number;
  start_time?: number;
  end_time?: number;
  status?: string;
  text?: string;
  transcript?: string;
  translation?: string;
  language?: string;
  language_name?: string;
  is_code_switched?: boolean;
  languages?: string[];
  confidence?: number;
  is_partial?: boolean;
  final_transcript?: string;
  final_english_translation?: string;
  error?: string;
}


import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Upload, Play, Sparkles, Volume2, Globe, FileAudio } from 'lucide-react';
import { SampleAudio } from '../types';

interface ControlsProps {
  isRecording: boolean;
  isProcessing: boolean;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onFileUpload: (file: File) => void;
  onSelectSample: (sample: SampleAudio) => void;
  samples: SampleAudio[];
  selectedLanguage: string;
  onLanguageChange: (lang: string) => void;
  recordingDuration: number;
}

export const AudioInputControls: React.FC<ControlsProps> = ({
  isRecording,
  isProcessing,
  onStartRecording,
  onStopRecording,
  onFileUpload,
  onSelectSample,
  samples,
  selectedLanguage,
  onLanguageChange,
  recordingDuration,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [playingSampleId, setPlayingSampleId] = useState<string | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  const formatTimer = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${mins.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const handlePlaySampleAudio = (sample: SampleAudio, e: React.MouseEvent) => {
    e.stopPropagation();
    if (playingSampleId === sample.audio_id) {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause();
      }
      setPlayingSampleId(null);
      return;
    }

    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
    }

    const audioUrl = `/api/v1/speech/sample/${sample.filename}`;
    const audio = new Audio(audioUrl);
    audioPlayerRef.current = audio;
    setPlayingSampleId(sample.audio_id);

    audio.play().catch((err) => console.warn('Playback error:', err));
    audio.onended = () => setPlayingSampleId(null);
  };

  return (
    <div className="space-y-4">
      {/* Primary Action Card */}
      <div className="glass-panel p-5 space-y-4">
        
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Globe className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-mono font-semibold text-slate-300">
              LANGUAGE CONTEXT HINT:
            </span>
          </div>

          <select
            value={selectedLanguage}
            onChange={(e) => onLanguageChange(e.target.value)}
            disabled={isRecording || isProcessing}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs font-mono rounded-lg px-3 py-1.5 focus:outline-none focus:border-amber-500 transition-colors"
          >
            <option value="">AUTO-DETECT (Indic Multilingual)</option>
            <option value="te">Telugu (తెలుగు)</option>
            <option value="hi">Hindi (हिन्दी)</option>
            <option value="ta">Tamil (தமிழ்)</option>
            <option value="kn">Kannada (ಕನ್ನಡ)</option>
            <option value="mr">Marathi (मराठी)</option>
            <option value="bn">Bengali (বাংলা)</option>
            <option value="ml">Malayalam (മലയാളം)</option>
            <option value="en">English</option>
          </select>
        </div>

        {/* Live Capture & File Drop Buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          
          {/* Microphone Record Button */}
          <button
            onClick={isRecording ? onStopRecording : onStartRecording}
            disabled={isProcessing}
            className={`flex items-center justify-center gap-3 p-4 rounded-xl font-mono text-sm font-bold tracking-wide transition-all border ${
              isRecording
                ? 'bg-red-500/20 border-red-500 text-red-300 shadow-lg shadow-red-500/20 animate-pulse'
                : 'bg-gradient-to-r from-amber-500/20 to-amber-600/30 border-amber-500/40 text-amber-300 hover:border-amber-400 hover:shadow-lg hover:shadow-amber-500/10'
            }`}
          >
            {isRecording ? (
              <>
                <MicOff className="w-5 h-5 text-red-400" />
                <span>STOP RECORDING [{formatTimer(recordingDuration)}]</span>
              </>
            ) : (
              <>
                <Mic className="w-5 h-5 text-amber-400" />
                <span>START LIVE CAPTURE</span>
              </>
            )}
          </button>

          {/* Upload Audio File */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isRecording || isProcessing}
            className="flex items-center justify-center gap-3 p-4 rounded-xl font-mono text-sm font-bold bg-slate-900/80 border border-slate-700 text-slate-300 hover:border-cyan-500 hover:text-cyan-300 transition-all shadow-md"
          >
            <Upload className="w-5 h-5 text-cyan-400" />
            <span>UPLOAD AUDIO FILE (.WAV/.MP3)</span>
          </button>

          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="audio/*"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFileUpload(file);
            }}
          />

        </div>

      </div>

      {/* 1-Click Golden Test Dataset Playground */}
      <div className="glass-panel p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-mono font-bold text-slate-200">
              POLICE DOMAIN GOLDEN TEST SAMPLES (1-CLICK INFERENCE)
            </span>
          </div>
          <span className="text-[10px] font-mono text-slate-500">
            {samples.length} LOADED
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2.5">
          {samples.map((sample) => (
            <div
              key={sample.audio_id}
              onClick={() => onSelectSample(sample)}
              className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 hover:border-amber-500/50 cursor-pointer transition-all hover:scale-[1.01] flex flex-col justify-between gap-2 group"
            >
              <div className="flex items-center justify-between">
                <span className="badge-mono bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  {sample.language_name.split(' ')[0]}
                </span>
                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                  {sample.incident_category.replace('_', ' ')}
                </span>
              </div>

              <p className="text-xs text-slate-300 line-clamp-2 italic font-serif">
                "{sample.expected_transcript}"
              </p>

              <div className="flex items-center justify-between pt-1 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
                <span>{sample.duration_sec}s</span>
                <button
                  onClick={(e) => handlePlaySampleAudio(sample, e)}
                  className="flex items-center gap-1 text-cyan-400 hover:text-cyan-300 transition-colors"
                >
                  <Volume2 className="w-3.5 h-3.5" />
                  <span>{playingSampleId === sample.audio_id ? 'Pause' : 'Play'}</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

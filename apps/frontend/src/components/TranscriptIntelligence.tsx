import React, { useState, useEffect } from 'react';
import { Copy, Check, Sparkles, FileText, Save, Download, Edit3, ShieldAlert } from 'lucide-react';
import { SpeechProcessResult, StreamSegment } from '../types';
import { saveSessionDetails, getExportTxtUrl, getExportDocxUrl } from '../services/api';

interface TranscriptProps {
  result: SpeechProcessResult | null;
  streamingSegments?: StreamSegment[];
  streamingTranscript?: string;
  streamingTranslation?: string;
  isStreaming?: boolean;
  sessionId?: string;
}

export const TranscriptIntelligence: React.FC<TranscriptProps> = ({
  result,
  streamingSegments = [],
  streamingTranscript,
  streamingTranslation,
  isStreaming,
  sessionId,
}) => {
  const [copiedOriginal, setCopiedOriginal] = useState(false);
  const [copiedEnglish, setCopiedEnglish] = useState(false);
  
  // Editable Document State
  const [isEditing, setIsEditing] = useState(false);
  const [editedTranscript, setEditedTranscript] = useState('');
  const [editedTranslation, setEditedTranslation] = useState('');
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  // Synchronize initial text into editable state
  useEffect(() => {
    if (result) {
      setEditedTranscript(result.transcript || '');
      setEditedTranslation(result.english_translation || '');
    } else if (!isStreaming) {
      setEditedTranscript('');
      setEditedTranslation('');
    }
  }, [result, isStreaming]);

  const displaySessionId = result?.session_id || sessionId || 'PRISM-PENDING';

  // Compute live streaming text from segments
  const finalStreamTranscript = streamingSegments.map(s => s.transcript).join(' ');
  const finalStreamTranslation = streamingSegments.map(s => s.translation).join(' ');

  const currentOriginalText = isEditing
    ? editedTranscript
    : isStreaming
    ? (streamingTranscript || finalStreamTranscript || 'Listening for speech...')
    : (result?.transcript || 'No statement recorded.');

  const currentEnglishText = isEditing
    ? editedTranslation
    : isStreaming
    ? (streamingTranslation || finalStreamTranslation || 'Generating translation...')
    : (result?.english_translation || 'No statement recorded.');

  const handleSaveEdits = async () => {
    if (!result?.session_id && !sessionId) return;
    const targetId = result?.session_id || sessionId;
    try {
      setSaveStatus('Saving officer edits...');
      await saveSessionDetails(targetId!, editedTranscript, editedTranslation);
      setSaveStatus('Statement saved successfully!');
      setIsEditing(false);
      setTimeout(() => setSaveStatus(null), 3000);
    } catch (err: any) {
      setSaveStatus('Failed to save edits');
    }
  };

  const handleExportTxt = () => {
    const targetId = result?.session_id || sessionId;
    if (!targetId) return;
    window.open(getExportTxtUrl(targetId), '_blank');
  };

  const handleExportDocx = () => {
    const targetId = result?.session_id || sessionId;
    if (!targetId) return;
    window.open(getExportDocxUrl(targetId), '_blank');
  };

  const handleCopyOriginal = () => {
    navigator.clipboard.writeText(currentOriginalText);
    setCopiedOriginal(true);
    setTimeout(() => setCopiedOriginal(false), 2000);
  };

  const handleCopyEnglish = () => {
    navigator.clipboard.writeText(currentEnglishText);
    setCopiedEnglish(true);
    setTimeout(() => setCopiedEnglish(false), 2000);
  };

  return (
    <div className="space-y-4">
      
      {/* Document Action Toolbar */}
      <div className="glass-panel p-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 font-mono text-xs text-slate-300">
          <ShieldAlert className="w-4 h-4 text-amber-400" />
          <span>INVESTIGATION STATEMENT DOCUMENT</span>
          <span className="text-slate-500">|</span>
          <span className="text-amber-400 font-bold">{displaySessionId}</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {saveStatus && (
            <span className="text-xs font-mono text-emerald-400 font-semibold animate-pulse">
              {saveStatus}
            </span>
          )}

          <button
            onClick={() => setIsEditing(!isEditing)}
            disabled={isStreaming}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold flex items-center gap-1.5 transition-all border ${
              isEditing
                ? 'bg-amber-500 text-slate-950 border-amber-400'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-amber-500'
            }`}
          >
            <Edit3 className="w-3.5 h-3.5" />
            <span>{isEditing ? 'EXIT EDIT MODE' : 'EDIT STATEMENT'}</span>
          </button>

          {isEditing && (
            <button
              onClick={handleSaveEdits}
              className="px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 flex items-center gap-1.5"
            >
              <Save className="w-3.5 h-3.5 text-emerald-400" />
              <span>SAVE STATEMENT</span>
            </button>
          )}

          <button
            onClick={handleExportTxt}
            disabled={!result?.session_id && !sessionId}
            className="px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-slate-900 border border-slate-700 text-cyan-400 hover:border-cyan-500 hover:bg-cyan-950/40 flex items-center gap-1.5 disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            <span>EXPORT TXT</span>
          </button>

          <button
            onClick={handleExportDocx}
            disabled={!result?.session_id && !sessionId}
            className="px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-slate-900 border border-slate-700 text-amber-300 hover:border-amber-500 hover:bg-amber-950/40 flex items-center gap-1.5 disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5" />
            <span>EXPORT DOCX</span>
          </button>
        </div>
      </div>

      {/* Side-by-Side Dual Display Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        
        {/* LEFT: Original Spoken Transcript */}
        <div className="glass-panel p-5 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-cyan-400" />
                <h3 className="font-mono text-xs font-bold text-slate-200 tracking-wider">
                  ORIGINAL TRANSCRIPT
                </h3>
              </div>

              <div className="flex items-center gap-2">
                {result && (
                  <span className="badge-mono bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                    LANG: {result.language.toUpperCase()}
                  </span>
                )}
                {isStreaming && (
                  <span className="badge-mono bg-red-500/20 text-red-300 border border-red-500/40 animate-pulse">
                    RECORDING LIVE
                  </span>
                )}
              </div>
            </div>

            {/* Editable or Display View */}
            {isEditing ? (
              <textarea
                value={editedTranscript}
                onChange={(e) => setEditedTranscript(e.target.value)}
                rows={8}
                className="w-full p-4 rounded-lg bg-slate-950 border border-amber-500/50 text-slate-100 font-sans text-sm focus:outline-none focus:border-amber-400"
              />
            ) : (
              <div className="min-h-[140px] p-4 rounded-lg bg-slate-900/90 border border-slate-800/80 font-sans text-sm text-slate-100 leading-relaxed whitespace-pre-wrap">
                {currentOriginalText}
              </div>
            )}

            {/* Timed Segments View */}
            {streamingSegments.length > 0 && !isEditing && (
              <div className="mt-4 space-y-2">
                <span className="text-[11px] font-mono font-semibold text-slate-400">
                  TIMED STATEMENT SEGMENTS:
                </span>
                <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
                  {streamingSegments.map((seg) => (
                    <div
                      key={seg.segment_id}
                      className="p-2 rounded bg-slate-950/80 border border-slate-800 text-xs font-mono flex items-center gap-3"
                    >
                      <span className="text-amber-400 font-semibold shrink-0">
                        {formatSec(seg.start_time)} → {formatSec(seg.end_time)}
                      </span>
                      <span className="text-slate-200 flex-1">{seg.transcript}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between pt-3 border-t border-slate-800 text-xs font-mono text-slate-400">
            <span>{result?.timing ? `ASR LATENCY: ${(result.timing.asr_latency * 1000).toFixed(0)}ms` : ''}</span>
            <button
              onClick={handleCopyOriginal}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors"
            >
              {copiedOriginal ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copiedOriginal ? 'Copied' : 'Copy Original'}</span>
            </button>
          </div>
        </div>

        {/* RIGHT: English Translation Column */}
        <div className="glass-panel-amber p-5 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-400" />
                <h3 className="font-mono text-xs font-bold text-amber-300 tracking-wider">
                  ENGLISH TRANSLATION
                </h3>
              </div>
              <span className="badge-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                TARGET: EN
              </span>
            </div>

            {isEditing ? (
              <textarea
                value={editedTranslation}
                onChange={(e) => setEditedTranslation(e.target.value)}
                rows={8}
                className="w-full p-4 rounded-lg bg-slate-950 border border-amber-500/50 text-slate-100 font-sans text-sm focus:outline-none focus:border-amber-400"
              />
            ) : (
              <div className="min-h-[140px] p-4 rounded-lg bg-slate-900/90 border border-amber-500/20 font-sans text-sm text-slate-100 leading-relaxed whitespace-pre-wrap">
                {currentEnglishText}
              </div>
            )}

            {streamingSegments.length > 0 && !isEditing && (
              <div className="mt-4 space-y-2">
                <span className="text-[11px] font-mono font-semibold text-slate-400">
                  ENGLISH TIMED TRANSLATIONS:
                </span>
                <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
                  {streamingSegments.map((seg) => (
                    <div
                      key={`trans_${seg.segment_id}`}
                      className="p-2 rounded bg-slate-950/80 border border-amber-500/20 text-xs font-mono flex items-center gap-3"
                    >
                      <span className="text-cyan-400 font-semibold shrink-0">
                        {formatSec(seg.start_time)} → {formatSec(seg.end_time)}
                      </span>
                      <span className="text-slate-200 flex-1">{seg.translation}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between pt-3 border-t border-slate-800 text-xs font-mono text-slate-400">
            <span>{result?.timing ? `NMT LATENCY: ${(result.timing.translation_latency * 1000).toFixed(0)}ms` : ''}</span>
            <button
              onClick={handleCopyEnglish}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 transition-colors"
            >
              {copiedEnglish ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copiedEnglish ? 'Copied' : 'Copy English Translation'}</span>
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

function formatSec(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

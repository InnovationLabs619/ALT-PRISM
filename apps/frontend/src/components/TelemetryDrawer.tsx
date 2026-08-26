import React, { useState } from 'react';
import {
  Activity,
  Cpu,
  Download,
  FileCode,
  Printer,
  Zap,
  CheckCircle2,
  ShieldAlert,
} from 'lucide-react';
import { SpeechProcessResult } from '../types';

interface TelemetryProps {
  result: SpeechProcessResult | null;
}

export const TelemetryDrawer: React.FC<TelemetryProps> = ({ result }) => {
  const [showJsonModal, setShowJsonModal] = useState(false);

  if (!result) return null;

  const { timing, audio_quality, models } = result;
  const rtf = timing.audio_duration > 0
    ? (timing.total_latency / timing.audio_duration).toFixed(3)
    : '0.000';

  const downloadPoliceReport = () => {
    const reportContent = `
================================================================================
PRISM POLICE INVESTIGATION STATEMENT REPORT
DEPARTMENT: Law & Order - Cyberabad
OFFICER IN CHARGE: Inspector K. Rajesh (AP-POL-10492)
DATE / TIME: ${new Date().toISOString()}
SESSION ID: ${result.session_id}
================================================================================

PRIMARY LANGUAGE DETECTED: ${result.language.toUpperCase()}
CODE-SWITCHED: ${result.is_code_switched ? 'YES' : 'NO'} (${result.languages.join(', ').toUpperCase()})
LANGUAGE CONFIDENCE: ${(result.language_confidence * 100).toFixed(1)}%

AUDIO METRICS:
- Duration: ${timing.audio_duration.toFixed(2)} seconds
- Signal-to-Noise Ratio (SNR): ${audio_quality.snr_db.toFixed(1)} dB
- Quality Score: ${(audio_quality.quality_score * 100).toFixed(0)}%

--------------------------------------------------------------------------------
1. ORIGINAL SPOKEN STATEMENT (TRANSCRIPTION):
--------------------------------------------------------------------------------
${result.transcript}

--------------------------------------------------------------------------------
2. OFFICIAL POLICE ENGLISH TRANSLATION:
--------------------------------------------------------------------------------
${result.english_translation}

--------------------------------------------------------------------------------
3. ML PIPELINE TELEMETRY & SIGNATURES:
--------------------------------------------------------------------------------
- ASR Model: ${models.asr} (v${models.asr_version || '1.0'})
- Translation Model: ${models.translation} (v${models.translation_version || '1.0'})
- Total Latency: ${(timing.total_latency * 1000).toFixed(0)} ms (RTF: ${rtf})
- Host: Local Self-Hosted GPU/CPU (Offline Isolated Node)

================================================================================
VERIFIED BY INVESTIGATING OFFICER: ___________________________
================================================================================
    `.trim();

    const blob = new Blob([reportContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PRISM_REPORT_${result.session_id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="glass-panel p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h3 className="font-mono text-xs font-bold text-slate-200 tracking-wider">
            INVESTIGATION TELEMETRY & ML PERFORMANCE
          </h3>
        </div>

        {/* Action Export Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowJsonModal(true)}
            className="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-900 border border-slate-700 hover:border-slate-500 text-[11px] font-mono text-slate-300 transition-colors"
          >
            <FileCode className="w-3.5 h-3.5 text-cyan-400" />
            JSON
          </button>
          <button
            onClick={downloadPoliceReport}
            className="flex items-center gap-1 px-2.5 py-1 rounded bg-amber-500/20 border border-amber-500/40 hover:bg-amber-500/30 text-[11px] font-mono font-semibold text-amber-300 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            EXPORT FIR REPORT
          </button>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        
        {/* Latency Card */}
        <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1">
          <span className="text-[10px] font-mono text-slate-400 uppercase">Total Latency</span>
          <div className="text-lg font-bold font-mono text-amber-400">
            {(timing.total_latency * 1000).toFixed(0)} <span className="text-xs text-slate-400">ms</span>
          </div>
          <span className="text-[10px] font-mono text-slate-500">
            Duration: {timing.audio_duration.toFixed(1)}s
          </span>
        </div>

        {/* RTF Card */}
        <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1">
          <span className="text-[10px] font-mono text-slate-400 uppercase">Real-Time Factor (RTF)</span>
          <div className="text-lg font-bold font-mono text-emerald-400">
            {rtf}x
          </div>
          <span className="text-[10px] font-mono text-emerald-500">
            {Number(rtf) <= 1.0 ? '✓ Faster than real-time' : 'Standard'}
          </span>
        </div>

        {/* SNR Card */}
        <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1">
          <span className="text-[10px] font-mono text-slate-400 uppercase">Acoustic SNR</span>
          <div className="text-lg font-bold font-mono text-cyan-400">
            {audio_quality.snr_db.toFixed(1)} <span className="text-xs text-slate-400">dB</span>
          </div>
          <span className="text-[10px] font-mono text-slate-500">
            Quality: {(audio_quality.quality_score * 100).toFixed(0)}%
          </span>
        </div>

        {/* AI Models Card */}
        <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1">
          <span className="text-[10px] font-mono text-slate-400 uppercase">Self-Hosted Engine</span>
          <div className="text-xs font-mono font-semibold text-slate-200 truncate">
            {models.asr.split('/').pop()}
          </div>
          <div className="text-[10px] font-mono text-slate-400 truncate">
            {models.translation.split('/').pop()}
          </div>
        </div>

      </div>

      {/* Latency Waterfall Bar */}
      <div className="space-y-1.5 pt-2">
        <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>PIPELINE LATENCY WATERFALL BREAKDOWN:</span>
          <span>
            Pre: {(timing.preprocessing_latency * 1000).toFixed(0)}ms | ASR: {(timing.asr_latency * 1000).toFixed(0)}ms | CS: {(timing.code_switch_latency * 1000).toFixed(0)}ms | NMT: {(timing.translation_latency * 1000).toFixed(0)}ms
          </span>
        </div>
        <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden flex border border-slate-800">
          <div
            style={{ width: `${Math.max(5, (timing.preprocessing_latency / timing.total_latency) * 100)}%` }}
            className="bg-cyan-500 h-full"
            title="Preprocessing"
          />
          <div
            style={{ width: `${Math.max(15, (timing.asr_latency / timing.total_latency) * 100)}%` }}
            className="bg-amber-500 h-full"
            title="ASR"
          />
          <div
            style={{ width: `${Math.max(5, (timing.code_switch_latency / timing.total_latency) * 100)}%` }}
            className="bg-purple-500 h-full"
            title="Code-Switch"
          />
          <div
            style={{ width: `${Math.max(15, (timing.translation_latency / timing.total_latency) * 100)}%` }}
            className="bg-emerald-500 h-full"
            title="Translation"
          />
        </div>
      </div>

      {/* JSON Viewer Modal */}
      {showJsonModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl max-w-2xl w-full p-5 space-y-4 max-h-[85vh] flex flex-col shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h4 className="font-mono text-sm font-bold text-amber-400">
                RAW TELEMETRY RESPONSE [{result.session_id}]
              </h4>
              <button
                onClick={() => setShowJsonModal(false)}
                className="text-slate-400 hover:text-slate-200 font-mono text-xs"
              >
                [CLOSE]
              </button>
            </div>
            <pre className="flex-1 overflow-auto bg-slate-900 p-4 rounded text-xs font-mono text-emerald-400">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        </div>
      )}

    </div>
  );
};

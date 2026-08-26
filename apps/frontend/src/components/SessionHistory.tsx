import React from 'react';
import { History, Clock, FileText, Globe } from 'lucide-react';

interface SessionItem {
  session_id: string;
  officer_badge: string;
  status: string;
  created_at: string;
  duration_seconds: number;
  primary_language?: string | null;
  is_code_switched?: boolean | null;
  transcript?: string | null;
  english_translation?: string | null;
  metadata?: any;
}

interface HistoryProps {
  sessions: SessionItem[];
  onSelectSession: (session: SessionItem) => void;
}

export const SessionHistory: React.FC<HistoryProps> = ({ sessions, onSelectSession }) => {
  if (sessions.length === 0) {
    return (
      <div className="glass-panel p-6 text-center text-slate-500 font-mono text-xs">
        NO RECENT INVESTIGATION SESSIONS FOUND
      </div>
    );
  }

  return (
    <div className="glass-panel p-5 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-amber-400" />
          <h3 className="font-mono text-xs font-bold text-slate-200 tracking-wider">
            RECENT INVESTIGATION STATEMENTS LOG
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-500">
          {sessions.length} SESSIONS
        </span>
      </div>

      <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
        {sessions.map((sess) => (
          <div
            key={sess.session_id}
            onClick={() => onSelectSession(sess)}
            className="p-3 rounded-lg bg-slate-900/80 border border-slate-800/80 hover:border-amber-500/40 cursor-pointer transition-all hover:bg-slate-900 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 group"
          >
            <div className="space-y-1 flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-xs font-bold text-amber-400 group-hover:text-amber-300">
                  {sess.session_id}
                </span>
                {sess.primary_language && (
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                    {sess.primary_language.toUpperCase()}
                  </span>
                )}
                {sess.is_code_switched && (
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    CODE-SWITCH
                  </span>
                )}
                <span className="text-[10px] font-mono text-slate-500 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {new Date(sess.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>

              <p className="text-xs text-slate-300 truncate font-sans">
                {sess.english_translation || sess.transcript || 'Processing statement...'}
              </p>
            </div>

            <div className="flex items-center gap-2 self-end sm:self-center">
              <span className="text-[10px] font-mono text-emerald-400 px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                {sess.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

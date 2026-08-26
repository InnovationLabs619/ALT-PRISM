import React from 'react';
import { Shield, Cpu, Activity, UserCheck, HardDrive } from 'lucide-react';
import { AIHealthStatus } from '../types';

interface HeaderProps {
  health: AIHealthStatus | null;
  activeMode: 'stream' | 'batch';
  onModeChange: (mode: 'stream' | 'batch') => void;
}

export const Header: React.FC<HeaderProps> = ({ health, activeMode, onModeChange }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
        
        {/* Brand & Badge */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500/20 to-amber-600/30 border border-amber-500/40 flex items-center justify-center shadow-lg shadow-amber-500/10">
            <Shield className="w-6 h-6 text-amber-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg tracking-wider text-slate-100 font-mono">
                PRISM
              </span>
              <span className="text-xs px-2 py-0.5 rounded font-mono font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                v1.0-MVP
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium">
              POLICE INVESTIGATION SYSTEM MANAGEMENT
            </p>
          </div>
        </div>

        {/* Tactical Mode Selector */}
        <div className="flex items-center bg-slate-900/90 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => onModeChange('stream')}
            className={`px-3 py-1.5 rounded-md text-xs font-mono font-medium transition-all ${
              activeMode === 'stream'
                ? 'bg-amber-500 text-slate-950 font-bold shadow-md shadow-amber-500/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            LIVE STREAM (MIC)
          </button>
          <button
            onClick={() => onModeChange('batch')}
            className={`px-3 py-1.5 rounded-md text-xs font-mono font-medium transition-all ${
              activeMode === 'batch'
                ? 'bg-amber-500 text-slate-950 font-bold shadow-md shadow-amber-500/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            AUDIO FILE / PLAYGROUND
          </button>
        </div>

        {/* AI Hardware & Officer Status */}
        <div className="flex items-center gap-3">
          {/* AI Engine Status Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-emerald-500/30 text-xs font-mono">
            <div className="pulse-dot" />
            <span className="text-emerald-400 font-semibold">
              LOCAL AI {health?.device ? `[${health.device.toUpperCase()}]` : '[CPU]'}
            </span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400 flex items-center gap-1">
              <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
              100% OFFLINE
            </span>
          </div>

          {/* Officer Profile Badge */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs">
            <UserCheck className="w-4 h-4 text-amber-400" />
            <div>
              <div className="font-semibold text-slate-200">Insp. K. Rajesh</div>
              <div className="text-[10px] text-slate-400 font-mono">AP-POL-10492</div>
            </div>
          </div>

        </div>

      </div>
    </header>
  );
};

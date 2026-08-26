import React, { useEffect, useRef } from 'react';

interface WaveformProps {
  isRecording: boolean;
  audioStream?: MediaStream | null;
  isPlaying?: boolean;
}

export const WaveformVisualizer: React.FC<WaveformProps> = ({
  isRecording,
  audioStream,
  isPlaying,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (isRecording && audioStream) {
      // Connect live Web Audio API Analyser
      try {
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 128;
        const source = audioCtx.createMediaStreamSource(audioStream);
        source.connect(analyser);

        audioContextRef.current = audioCtx;
        analyserRef.current = analyser;
        sourceRef.current = source;
      } catch (e) {
        console.warn('Web Audio API init error:', e);
      }
    }

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      // Draw tactical background grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();

      const numBars = 48;
      const barWidth = width / numBars - 2;

      let dataArray = new Uint8Array(numBars);

      if (isRecording && analyserRef.current) {
        const bufferLength = analyserRef.current.frequencyBinCount;
        const tempArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(tempArray);
        for (let i = 0; i < numBars; i++) {
          dataArray[i] = tempArray[Math.floor((i / numBars) * bufferLength)] || 0;
        }
      } else if (isRecording || isPlaying) {
        // Simulated responsive acoustic wave when stream is active
        const time = Date.now() / 150;
        for (let i = 0; i < numBars; i++) {
          dataArray[i] = Math.floor(
            Math.sin(i * 0.3 + time) * 40 + Math.cos(i * 0.5 - time) * 30 + 90
          );
        }
      } else {
        // Idle baseline calm pulse
        const time = Date.now() / 600;
        for (let i = 0; i < numBars; i++) {
          dataArray[i] = Math.floor(Math.sin(i * 0.2 + time) * 8 + 14);
        }
      }

      // Draw tactical gradient bars
      for (let i = 0; i < numBars; i++) {
        const value = dataArray[i];
        const percent = Math.min(1.0, value / 255);
        const barHeight = Math.max(4, percent * (height - 12));

        const x = i * (barWidth + 2);
        const y = (height - barHeight) / 2;

        const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
        if (isRecording) {
          gradient.addColorStop(0, '#f59e0b'); // Neon Amber
          gradient.addColorStop(0.5, '#ef4444'); // Crimson
          gradient.addColorStop(1, '#f59e0b');
        } else {
          gradient.addColorStop(0, '#06b6d4'); // Cyan
          gradient.addColorStop(0.5, '#3b82f6'); // Blue
          gradient.addColorStop(1, '#06b6d4');
        }

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 2);
        ctx.fill();
      }

      animationFrameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, [isRecording, audioStream, isPlaying]);

  return (
    <div className="w-full relative rounded-lg overflow-hidden bg-slate-950/90 border border-slate-800 p-2 shadow-inner">
      <div className="flex items-center justify-between px-2 pb-1 text-[11px] font-mono text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${isRecording ? 'bg-amber-400 animate-ping' : 'bg-cyan-400'}`} />
          {isRecording ? 'LIVE AUDIO CAPTURE [16kHz MONO]' : 'ACOUSTIC SPECTRUM'}
        </span>
        <span className="text-slate-500">ENERGY SPECTRUM (FFT 128)</span>
      </div>
      <canvas
        ref={canvasRef}
        width={720}
        height={90}
        className="w-full h-24 block rounded"
      />
    </div>
  );
};

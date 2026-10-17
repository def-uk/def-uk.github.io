/**
 * 홈 히어로 — 코드 ↔ 캔버스.
 *
 * 같은 자리에 "코드"와 "그 코드가 그린 그림"이 겹쳐 있고, 경계를 밀면 코드가 걷히며 그림이 드러난다.
 * 코드 속 숫자(petals·radius·speed)는 좌우로 끌면 값이 바뀌고 그림이 즉시 따라온다.
 *
 * - 첫 로드: 전부 코드(100%) → 반(50%)까지 걷히는 연출. 모션 줄이기면 바로 50%.
 * - 화면 밖에선 그리기를 멈춘다. 모션 줄이기면 애니메이션 없이 정지 프레임만 그린다.
 * - 색은 --color-accent 를 읽는다(테마·페이지 포인트 색을 따라감).
 * - JS 가 없으면 서버 렌더된 코드 화면만 보인다.
 */
import { useCallback, useEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import './CodeCanvas.css';

type Key = 'petals' | 'radius' | 'speed';
type Params = Record<Key, number>;

const RANGE: Record<Key, { min: number; max: number; step: number; px: number }> = {
  // px: 값 한 칸(step)을 바꾸는 데 필요한 드래그 거리
  petals: { min: 3, max: 36, step: 1, px: 8 },
  radius: { min: 40, max: 150, step: 1, px: 2 },
  speed: { min: 0, max: 3, step: 0.1, px: 10 },
};
const DEFAULTS: Params = { petals: 12, radius: 110, speed: 0.6 };
const INTRO_FROM = 100;
const INTRO_TO = 50;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const snap = (key: Key, v: number) => {
  const { min, max, step } = RANGE[key];
  const n = Math.round(clamp(v, min, max) / step) * step;
  return Number(n.toFixed(2));
};
const fmt = (key: Key, v: number) => (RANGE[key].step < 1 ? v.toFixed(1) : String(v));

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export default function CodeCanvas() {
  const [params, setParams] = useState<Params>(DEFAULTS);
  const [split, setSplit] = useState(INTRO_FROM);
  const [active, setActive] = useState<Key | null>(null);

  const rootRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const redrawRef = useRef<() => void>(() => {});
  const introRef = useRef<number | null>(null);
  const draggingRef = useRef(false);

  /* ---------- 캔버스 ---------- */
  useEffect(() => {
    const canvas = canvasRef.current;
    const root = rootRef.current;
    if (!canvas || !root) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let color = '#c2410c';
    let raf = 0;
    let visible = true;
    let t = 0;
    let last = performance.now();
    const reduce = prefersReducedMotion();

    const readColor = () => {
      color = getComputedStyle(root).getPropertyValue('--color-accent').trim() || color;
    };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = () => {
      const { petals, radius, speed } = paramsRef.current;
      const scale = Math.min(w, h) / 320;
      const cx = w / 2;
      const cy = h / 2;
      ctx.clearRect(0, 0, w, h);
      ctx.lineCap = 'round';
      ctx.strokeStyle = color;
      ctx.lineWidth = clamp(48 / petals, 1.5, 8) * scale;
      for (let i = 0; i < petals; i++) {
        const a = (i / petals) * Math.PI * 2 + t * speed;
        const r = radius * scale * (0.6 + 0.4 * Math.sin(t * 1.5 + i * 0.7));
        const inner = radius * scale * 0.22;
        ctx.globalAlpha = 0.35 + 0.65 * (i / Math.max(1, petals - 1));
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
        ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };
    redrawRef.current = draw;

    const loop = (now: number) => {
      t += Math.min(now - last, 50) / 1000;
      last = now;
      draw();
      raf = requestAnimationFrame(loop);
    };
    const start = () => {
      if (reduce || raf || !visible) return;
      last = performance.now();
      raf = requestAnimationFrame(loop);
    };
    const stop = () => {
      cancelAnimationFrame(raf);
      raf = 0;
    };

    readColor();
    resize();
    draw();
    start();

    const ro = new ResizeObserver(() => {
      resize();
      draw();
    });
    ro.observe(canvas);

    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) start();
      else stop();
    });
    io.observe(root);

    // 테마 전환(토글·OS 설정) 시 색을 다시 읽는다.
    const scheme = window.matchMedia('(prefers-color-scheme: dark)');
    const onTheme = () => {
      readColor();
      draw();
    };
    const mo = new MutationObserver(onTheme);
    mo.observe(document.documentElement, { attributeFilter: ['data-theme'] });
    scheme.addEventListener('change', onTheme);

    return () => {
      stop();
      ro.disconnect();
      io.disconnect();
      mo.disconnect();
      scheme.removeEventListener('change', onTheme);
    };
  }, []);

  // 애니메이션이 멈춰 있을 때(모션 줄이기)도 숫자를 바꾸면 그림이 바뀌어야 한다.
  useEffect(() => redrawRef.current(), [params]);

  /* ---------- 첫 등장: 코드가 반쯤 걷힌다 ---------- */
  useEffect(() => {
    if (prefersReducedMotion()) {
      setSplit(INTRO_TO);
      return;
    }
    const begin = performance.now() + 500;
    const dur = 1100;
    const tick = (now: number) => {
      const p = clamp((now - begin) / dur, 0, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setSplit(INTRO_FROM + (INTRO_TO - INTRO_FROM) * eased);
      introRef.current = p < 1 ? requestAnimationFrame(tick) : null;
    };
    introRef.current = requestAnimationFrame(tick);
    return () => {
      if (introRef.current) cancelAnimationFrame(introRef.current);
    };
  }, []);

  const cancelIntro = () => {
    if (introRef.current) cancelAnimationFrame(introRef.current);
    introRef.current = null;
  };

  /* ---------- 경계 드래그 ---------- */
  const splitFromPointer = (clientX: number) => {
    const rect = rootRef.current?.getBoundingClientRect();
    if (!rect) return;
    setSplit(clamp(((clientX - rect.left) / rect.width) * 100, 0, 100));
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('[data-scrub]')) return;
    cancelIntro();
    draggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    splitFromPointer(e.clientX);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (draggingRef.current) splitFromPointer(e.clientX);
  };
  const onPointerUp = () => {
    draggingRef.current = false;
  };

  const onHandleKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 20 : 5;
    const next =
      e.key === 'ArrowLeft' ? split - step
      : e.key === 'ArrowRight' ? split + step
      : e.key === 'Home' ? 0
      : e.key === 'End' ? 100
      : null;
    if (next === null) return;
    e.preventDefault();
    cancelIntro();
    setSplit(clamp(next, 0, 100));
  };

  /* ---------- 숫자 스크럽 ---------- */
  const setParam = useCallback((key: Key, v: number) => {
    setParams((p) => ({ ...p, [key]: snap(key, v) }));
  }, []);

  // 컴포넌트가 아니라 함수로 부른다 — 렌더마다 새 컴포넌트 타입이 되면 span 이 다시 만들어져
  // 드래그 중 포인터 캡처가 끊긴다.
  const num = (k: Key) => {
    const onDown = (e: PointerEvent<HTMLSpanElement>) => {
      e.stopPropagation();
      e.preventDefault();
      const el = e.currentTarget;
      el.setPointerCapture(e.pointerId);
      const startX = e.clientX;
      const startV = paramsRef.current[k];
      setActive(k);
      const move = (ev: globalThis.PointerEvent) => {
        const steps = Math.round((ev.clientX - startX) / RANGE[k].px);
        setParam(k, startV + steps * RANGE[k].step);
      };
      const up = () => {
        setActive(null);
        el.removeEventListener('pointermove', move);
        el.removeEventListener('pointerup', up);
        el.removeEventListener('pointercancel', up);
      };
      el.addEventListener('pointermove', move);
      el.addEventListener('pointerup', up);
      el.addEventListener('pointercancel', up);
    };
    const onKey = (e: KeyboardEvent<HTMLSpanElement>) => {
      const dir = e.key === 'ArrowRight' || e.key === 'ArrowUp' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? -1 : 0;
      if (!dir) return;
      e.preventDefault();
      setParam(k, paramsRef.current[k] + dir * RANGE[k].step * (e.shiftKey ? 10 : 1));
    };
    return (
      <span
        data-scrub
        className={`cc__num${active === k ? ' is-active' : ''}`}
        role="slider"
        tabIndex={0}
        aria-label={k}
        aria-valuemin={RANGE[k].min}
        aria-valuemax={RANGE[k].max}
        aria-valuenow={params[k]}
        onPointerDown={onDown}
        onKeyDown={onKey}
        onDoubleClick={() => setParam(k, DEFAULTS[k])}
      >
        {fmt(k, params[k])}
      </span>
    );
  };

  const kw = (s: string) => <span className="tok-k">{s}</span>;
  const fn = (s: string) => <span className="tok-f">{s}</span>;
  const bi = (ko: string, en: string): ReactNode => (
    <>
      <span data-lang-block="ko">{ko}</span>
      <span data-lang-block="en">{en}</span>
    </>
  );

  const style = { '--split': `${split}%` } as CSSProperties;

  return (
    <div className="cc-wrap">
      <div
        ref={rootRef}
        className="cc"
        style={style}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <canvas ref={canvasRef} className="cc__canvas" aria-hidden="true" />
        <span className="cc__label cc__label--canvas" aria-hidden="true">canvas</span>

        <pre className="cc__code">
          <span className="cc__label cc__label--code" aria-hidden="true">code</span>
          <code>
            <span className="tok-c">// {bi('이 코드가 이 그림을 그립니다', 'this code draws this picture')}</span>{'\n'}
            {kw('const')} petals = {num('petals')};{'\n'}
            {kw('const')} radius = {num('radius')};{'\n'}
            {kw('const')} speed  = {num('speed')};{'\n'}
            {'\n'}
            {kw('function')} {fn('draw')}(t) {'{'}{'\n'}
            {'  '}{kw('for')} ({kw('let')} i = 0; i {'<'} petals; i++) {'{'}{'\n'}
            {'    '}{kw('const')} a = i / petals * TAU + t * speed;{'\n'}
            {'    '}{kw('const')} r = radius * (0.6 + 0.4 * {fn('sin')}(t + i));{'\n'}
            {'    '}{fn('bar')}(a, r);{'\n'}
            {'  }'}{'\n'}
            {'}'}
          </code>
        </pre>

        <div
          className="cc__handle"
          role="slider"
          tabIndex={0}
          aria-label="code ↔ canvas"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(split)}
          onKeyDown={onHandleKey}
        >
          <span className="cc__grip" aria-hidden="true">
            <svg viewBox="0 0 24 24"><path d="M9 6l-6 6 6 6M15 6l6 6-6 6" /></svg>
          </span>
        </div>
      </div>
      <p className="cc-hint">
        {bi('경계를 밀거나, 강조된 숫자를 좌우로 끌어보세요', 'Drag the divider, or scrub the highlighted numbers')}
      </p>
    </div>
  );
}

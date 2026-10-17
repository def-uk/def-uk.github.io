/**
 * 홈 히어로 — 코드 ↔ 캔버스.
 *
 * 같은 자리에 "코드"와 "그 코드가 그린 그림"이 겹쳐 있고, 경계를 밀면 코드가 걷히며 그림이 드러난다.
 * 그림은 떠다니는 도형을 탐지 박스가 따라가는 장면이다. 코드 속 숫자(objects·conf·speed)는
 * 좌우로 끌면 값이 바뀌고 그림이 즉시 따라온다. conf 보다 점수가 낮은 박스는 흐린 점선으로 남는다.
 *
 * - 첫 로드: 전부 코드(100%) → 반(50%)까지 걷히는 연출. 모션 줄이기면 바로 50%.
 * - 화면 밖에선 그리기를 멈춘다. 모션 줄이기면 애니메이션 없이 정지 프레임만 그린다.
 * - 색은 --color-accent(박스)·--color-text-muted(도형)·--color-bg(라벨 글자)를 읽는다.
 * - JS 가 없으면 서버 렌더된 코드 화면만 보인다.
 */
import { useCallback, useEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import './CodeCanvas.css';

type Key = 'objects' | 'conf' | 'speed';
type Params = Record<Key, number>;

const RANGE: Record<Key, { min: number; max: number; step: number; px: number; digits: number }> = {
  // px: 값 한 칸(step)을 바꾸는 데 필요한 드래그 거리
  objects: { min: 1, max: 8, step: 1, px: 14, digits: 0 },
  conf: { min: 0, max: 1, step: 0.05, px: 8, digits: 2 },
  speed: { min: 0, max: 3, step: 0.1, px: 10, digits: 1 },
};
const DEFAULTS: Params = { objects: 5, conf: 0.5, speed: 0.6 };
const INTRO_FROM = 100;
const INTRO_TO = 50;
// 첫 프레임(모션 줄이기면 정지 프레임)의 장면 시각 — 기본 화면 오른쪽에 도형 3개가
// 서로 떨어져 박스를 달고 있는 순간.
const START_T = 17.4;
// 예측 박스는 이 시간만큼 늦게 따라온다 — 완벽하지 않은 탐지처럼 보이게.
const LAG = 0.12;

const SHAPES = ['circle', 'square', 'triangle', 'diamond'] as const;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const snap = (key: Key, v: number) => {
  const { min, max, step } = RANGE[key];
  const n = Math.round(clamp(v, min, max) / step) * step;
  return Number(n.toFixed(2));
};
const fmt = (key: Key, v: number) => v.toFixed(RANGE[key].digits);

// 도형마다 고정된 무작위 값(0~1). 같은 i 는 언제나 같은 궤적·크기를 갖는다.
const rand = (i: number, k: number) => {
  const x = Math.sin(i * 127.1 + k * 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/** i 번째 도형의 t 시각 꼭짓점(원은 중심·반지름). 박스 계산과 그리기가 같은 값을 쓴다. */
const shapeAt = (i: number, t: number, w: number, h: number, scale: number) => {
  const type = SHAPES[i % SHAPES.length];
  const r = (16 + 12 * rand(i, 1)) * scale;
  // 기본 화면에선 왼쪽 절반을 코드가 덮으므로 오른쪽에 더 오래 머물게 한다(x 는 w*0.3 ~ 오른쪽 끝).
  const x0 = w * 0.3;
  const x1 = w - r - 30 * scale;
  const my = r + 36 * scale;
  const x = x0 + (x1 - x0) * (0.5 + 0.5 * Math.sin(t * (0.23 + 0.22 * rand(i, 2)) + rand(i, 3) * 6.28));
  const y = h / 2 + (h / 2 - my) * Math.sin(t * (0.19 + 0.2 * rand(i, 4)) + rand(i, 5) * 6.28);
  const rot = t * (rand(i, 6) - 0.5) * 1.2;
  const n = type === 'triangle' ? 3 : 4;
  const offset = type === 'square' ? Math.PI / 4 : -Math.PI / 2;
  const verts =
    type === 'circle'
      ? []
      : Array.from({ length: n }, (_, k) => {
          const a = rot + offset + (k / n) * Math.PI * 2;
          const rr = type === 'diamond' && k % 2 ? r * 0.7 : r;
          return [x + Math.cos(a) * rr, y + Math.sin(a) * rr] as const;
        });
  return { type, x, y, r, verts };
};

const bbox = (s: ReturnType<typeof shapeAt>) => {
  if (s.type === 'circle') return { x0: s.x - s.r, y0: s.y - s.r, x1: s.x + s.r, y1: s.y + s.r };
  const xs = s.verts.map((v) => v[0]);
  const ys = s.verts.map((v) => v[1]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
};

// 점수는 대부분 0.9 근처에 있다가 가끔 0.39 까지 떨어진다. 기본 conf 0.50 이면 박스가 이따금
// 꺼지고, conf 를 올릴수록 꺼지는 박스가 늘어난다.
const scoreAt = (i: number, t: number) => 0.99 - 0.6 * Math.pow(0.5 + 0.5 * Math.sin(t * 0.5 + i * 2.1), 3);

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
    let accent = '#c2410c';
    let muted = '#6b7280';
    let bg = '#ffffff';
    let raf = 0;
    let visible = true;
    const reduce = prefersReducedMotion();
    // speed 를 곱해 쌓은 "장면 시각". speed 를 바꿔도 도형이 튀지 않는다.
    let t = START_T;
    let last = performance.now();

    const readColor = () => {
      const cs = getComputedStyle(root);
      accent = cs.getPropertyValue('--color-accent').trim() || accent;
      muted = cs.getPropertyValue('--color-text-muted').trim() || muted;
      bg = cs.getPropertyValue('--color-bg').trim() || bg;
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
      const { objects, conf } = paramsRef.current;
      const scale = Math.min(w, h) / 320;
      const pad = 3 * scale;
      const font = Math.max(9, Math.round(10 * scale));
      ctx.clearRect(0, 0, w, h);
      ctx.font = `600 ${font}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.textBaseline = 'middle';

      for (let i = 0; i < objects; i++) {
        // 도형
        const s = shapeAt(i, t, w, h, scale);
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = muted;
        ctx.beginPath();
        if (s.type === 'circle') ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        else s.verts.forEach(([vx, vy], k) => (k ? ctx.lineTo(vx, vy) : ctx.moveTo(vx, vy)));
        ctx.closePath();
        ctx.fill();

        // 예측 박스: 조금 늦게 따라오는 위치의 외곽 + 여백
        const b = bbox(shapeAt(i, t - LAG, w, h, scale));
        const bx = b.x0 - pad;
        const by = b.y0 - pad;
        const bw = b.x1 - b.x0 + pad * 2;
        const bh = b.y1 - b.y0 + pad * 2;
        const score = scoreAt(i, t);
        const kept = score >= conf;

        ctx.strokeStyle = accent;
        ctx.lineWidth = Math.max(1.5, 1.75 * scale);
        ctx.setLineDash(kept ? [] : [4 * scale, 4 * scale]);
        ctx.globalAlpha = kept ? 1 : 0.22;
        ctx.strokeRect(bx, by, bw, bh);
        ctx.setLineDash([]);
        if (!kept) continue;

        // 라벨: 박스 위, 위가 모자라면 박스 안쪽 위. 오른쪽 끝을 넘으면 안쪽으로 민다.
        const text = `${s.type} ${score.toFixed(2)}`;
        const lw = ctx.measureText(text).width + 8 * scale;
        const lh = font + 6 * scale;
        const lx = Math.min(bx - ctx.lineWidth / 2, w - lw);
        const ly = by - lh >= 0 ? by - lh : by;
        ctx.globalAlpha = 1;
        ctx.fillStyle = accent;
        ctx.fillRect(lx, ly, lw, lh);
        ctx.fillStyle = bg;
        ctx.fillText(text, lx + 4 * scale, ly + lh / 2);
      }
      ctx.globalAlpha = 1;
    };
    redrawRef.current = draw;

    const loop = (now: number) => {
      t += (Math.min(now - last, 50) / 1000) * paramsRef.current.speed;
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

  // 애니메이션이 멈춰 있을 때(모션 줄이기·speed 0)도 숫자를 바꾸면 그림이 바뀌어야 한다.
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
            <span className="tok-c">// {bi('움직이는 도형을 찾아 박스를 칩니다', 'find the moving shapes and box them')}</span>{'\n'}
            {kw('const')} objects = {num('objects')};{'\n'}
            {kw('const')} conf    = {num('conf')};{'\n'}
            {kw('const')} speed   = {num('speed')};{'\n'}
            {'\n'}
            {kw('function')} {fn('draw')}(t) {'{'}{'\n'}
            {'  '}{kw('for')} ({kw('const')} o {kw('of')} {fn('scene')}(objects, t * speed)) {'{'}{'\n'}
            {'    '}{kw('const')} {'{'} box, label, score {'}'} = {fn('detect')}(o);{'\n'}
            {'    '}{kw('if')} (score {'<'} conf) {kw('continue')};{'\n'}
            {'    '}{fn('rect')}(box, label, score);{'\n'}
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

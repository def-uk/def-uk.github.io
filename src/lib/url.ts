/** base 경로를 붙여 안전한 절대 링크를 만든다(유저 사이트는 base '/'). */
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${base}${p}` || '/';
}

/**
 * 크로스 문서 View Transition 이름. 목록 카드와 상세 페이지의 미디어가 같은 이름을 가지면
 * 페이지를 넘어갈 때 카드가 상세 화면으로 모핑된다(global.css 의 @view-transition).
 * 이름은 한 페이지 안에서 유일해야 하고 CSS 식별자여야 한다.
 */
export function vtName(kind: 'work' | 'lab', path: string): string {
  return `vt-${kind}-${path.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
}

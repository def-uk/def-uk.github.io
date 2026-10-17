export const LOCALES = ['ko', 'en'] as const;
export type Locale = (typeof LOCALES)[number];
export const DEFAULT_LOCALE: Locale = 'ko';

export function isLocale(value: string): value is Locale {
  return (LOCALES as readonly string[]).includes(value);
}

/** 상단 고정 네비 + UI 문구 (한/영 동시 보유, 토글로 전환) */
export const UI = {
  home: { ko: '홈', en: 'Home' },
  work: { ko: 'Work', en: 'Work' },
  lab: { ko: 'Lab', en: 'Lab' },
  log: { ko: 'Log', en: 'Log' },
  about: { ko: 'About', en: 'About' },
  workIntro: { ko: '이야기가 있는 프로젝트들', en: 'Projects with a story' },
  labIntro: { ko: '작품과 실험, 결과물 하나씩', en: 'Pieces and experiments, one at a time' },
  logIntro: { ko: '기술 글, TIL, 생각 기록', en: 'Tech notes, TILs and essays' },
  featuredWork: { ko: '대표 작업', en: 'Selected work' },
  latestLab: { ko: '최근 Lab', en: 'Latest from the lab' },
  viewAll: { ko: '전체 보기', en: 'View all' },
  role: { ko: '역할', en: 'Role' },
  period: { ko: '기간', en: 'Period' },
  tools: { ko: '도구', en: 'Tools' },
  relatedWork: { ko: '관련 프로젝트', en: 'Related project' },
  themeToggle: { ko: '테마 전환', en: 'Toggle theme' },
  // 토글 버튼에 보이는 '전환 대상' 라벨: ko 활성 → 'EN' 노출, en 활성 → '한국어' 노출
  switchTo: { ko: 'EN', en: '한국어' },
  translationMissing: {
    ko: '이 글의 한국어 버전이 아직 없습니다.',
    en: 'The English version of this post is not available yet.',
  },
  readMore: { ko: '자세히 보기', en: 'Read more' },
  all: { ko: '전체', en: 'All' },
  categories: { ko: '카테고리', en: 'Categories' },
  recent: { ko: '최근 Log', en: 'Recent log' },
  noContent: { ko: '아직 콘텐츠가 없습니다.', en: 'No content yet.' },
  backToList: { ko: '목록으로', en: 'Back to list' },
  backToHome: { ko: '홈으로', en: 'Back to home' },
  // 단일 페이지(cv 등)가 draft: true 일 때 — 배포에서 본문 대신 보이는 안내
  comingSoon: { ko: '아직 준비중입니다', en: 'Coming soon' },
  comingSoonBody: {
    ko: '내용을 더 다듬고 있습니다. 정리되는 대로 공개할게요.',
    en: 'This page is still being polished. It will be published once it is ready.',
  },
  // 같은 페이지를 dev 서버에서 볼 때 본문 위에 붙는 배너
  draftNotice: {
    ko: '초안 미리보기 — 배포된 사이트에서는 이 페이지에 준비중 안내만 보입니다.',
    en: 'Draft preview — the deployed site shows only a coming-soon notice here.',
  },
  toc: { ko: '목차', en: 'Contents' },
  tocAside: { ko: '목차', en: 'On this page' },
  comments: { ko: '댓글', en: 'Comments' },
} as const;

/** 폴더명을 사람이 읽기 좋은 라벨로 (예: 'data-structures' → 'Data structures') */
export function humanize(segment: string): string {
  const spaced = segment.replace(/[-_]+/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** kind 태그 라벨(필터 칩·배지). */
export const KIND_LABEL = {
  code: { ko: 'Code', en: 'Code' },
  design: { ko: 'Design', en: 'Design' },
  video: { ko: 'Video', en: 'Video' },
  motion: { ko: 'Motion', en: 'Motion' },
} as const;

/** tools 슬러그 → 표시 이름. 목록에 없으면 humanize 로 보여준다. */
export const TOOL_LABEL: Record<string, string> = {
  'premiere-pro': 'Premiere Pro',
  'after-effects': 'After Effects',
  higgsfield: 'Higgsfield',
  python: 'Python',
  pytorch: 'PyTorch',
  'scikit-learn': 'scikit-learn',
  'hugging-face': 'Hugging Face',
  flutter: 'Flutter',
  react: 'React',
  nextjs: 'Next.js',
};

/** 이 도구가 들어가면 AI 배지를 단다. */
export const AI_TOOLS = new Set(['higgsfield']);

export const toolLabel = (slug: string) => TOOL_LABEL[slug] ?? humanize(slug);

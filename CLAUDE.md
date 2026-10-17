# CLAUDE.md

dongwook의 개발 + 영상·모션 포트폴리오. **Astro 7** 정적 사이트, GitHub Pages 배포(`def-uk.github.io`, 레포 `def-uk/def-uk.github.io`).
기획·로드맵은 [`docs/PLAN.md`](docs/PLAN.md). 시스템 코드는 [raewoo0908.github.io](https://github.com/raewoo0908/raewoo0908.github.io)에서 작성자 허락을 받고 가져와 고친 것이다.

## 절대 규칙

- **언어 전환은 오직 토글로만.** URL에 `ko`·`en`을 절대 넣지 않는다. 한 문서 = 언어중립 URL 1개에 한/영을 둘 다 정적 렌더하고 `<html data-lang>` + CSS로 활성 언어만 보인다. `/en/…` 같은 로케일 URL 트리 금지.
- **번역 워크플로**: 사용자는 한국어만 쓴다. Work·Log는 Claude가 같은 폴더에 `en.md`를 만들어 함께 커밋한다. Lab은 `index.md` frontmatter의 `en` 값을 채운다.
- **디자인 값은 `src/styles/tokens.css` 한 곳에서.** 폰트·크기·색상·라이트/다크 팔레트. 색을 컴포넌트에 하드코딩하지 않는다(예외: kind 배지 색, 미디어 위 흰 글자).
- **모션은 `prefers-reduced-motion`을 존중한다.** 애니메이션·자동재생을 추가하면 반드시 끄는 분기를 둔다.

## 컬렉션

| 컬렉션 | 경로 | 파일 | URL |
| --- | --- | --- | --- |
| Work | `src/content/work/<slug>/` | `ko.md` + `en.md` 짝, 미디어는 `media/` | `/work/<slug>` |
| Lab | `src/content/lab/<연도>/<slug>/` | `index.md` 하나 + 미디어 | `/lab/<연도>/<slug>` |
| Log | `src/content/log/<카테고리>[/…]/<slug>/` | `ko.md` + `en.md` 짝 (폴더 = 카테고리) | `/log/…` |
| 페이지 | `src/content/pages/{home,about}/` | `ko.md` + `en.md` 짝 | `/`, `/about` |

- 스키마는 [`src/content.config.ts`](src/content.config.ts). Work는 `kind`·`tools`·`role`·`period`·`accent`·`cover`·`coverVideo`·`featured`, Lab은 `title{ko,en}`·`caption{ko,en}`·`kind`·`tools`·`media`·`poster`·`work`.
- **`kind`** 는 `code`·`design`·`video`·`motion` 중 복수 선택. 목록 필터 칩이 된다.
- **`tools`** 슬러그: `premiere-pro`·`after-effects`·`higgsfield`·`python`·`flutter`·`react`·`nextjs`(표시 이름은 `src/lib/i18n.ts`의 `TOOL_LABEL`). `higgsfield`가 있으면 AI 배지가 자동으로 붙는다.
- **Work vs Lab**: 이야기(문제 → 과정 → 결과)가 있으면 Work, 결과물 하나면 Lab.
- frontmatter 미디어 경로(`cover`·`coverVideo`·`media`·`poster`)는 **글 폴더 기준 상대경로**. `src/lib/media.ts`가 해석하고, 오타면 빌드가 실패한다. 이미지는 자동 최적화, 영상은 그대로 복사된다.
- `date`는 **게시 시각(KST)**, 시각까지 쓴다(`date +%Y-%m-%dT%H:%M:00+09:00`). 목록은 최신순. ko/en 짝은 같은 값.
  - 예외: velog에서 옮겨온 Log 글(2026-10-17 이전)은 velog 원래 발행 시각을 유지한다.
- `draft: true`는 배포 빌드에서 빠진다(dev에서는 보인다). 단일 페이지는 배포에서 "준비중" 안내로 바뀐다.
- `sample-*` 폴더는 템플릿이다. 실제 콘텐츠가 생기면 지운다.

## 인터랙션 규칙

- **인터랙티브 컴포넌트는 React 섬**(`@astrojs/react`)으로 `src/components/<영역>/*.tsx`에 둔다. 예: 홈 히어로 `src/components/hero/CodeCanvas.tsx`(`client:load`).
- **페이지 전환은 브라우저 기본 크로스 문서 View Transitions**(`global.css`의 `@view-transition`)다. Astro `<ClientRouter />`는 쓰지 않는다 — 이식한 스크립트들과 `<html data-lang>`이 페이지 교체 방식과 맞지 않는다.
- 카드 ↔ 상세 모핑은 양쪽 요소에 같은 `view-transition-name`(`vtName()` in `src/lib/url.ts`)을 준다. 한 페이지 안에서 이름이 겹치면 전환이 통째로 건너뛰어진다.
- 마크다운 플러그인(remark/rehype)은 `@astrojs/markdown-remark`가 직접 의존성으로 있어야 돈다(Astro 7.3+ 기본 처리기가 바뀜). 지우지 말 것.

## 미디어 규칙

- GitHub Pages는 **파일당 100MB 제한**이고 Git LFS 파일을 서빙하지 못한다.
- 미리보기 루프는 **3MB 이하 webm**(VP9, `-pix_fmt yuv420p`). 포스터는 jpg. 긴 본편은 YouTube/Vimeo.
- 이 환경의 ffmpeg는 webp 인코더가 없다. 포스터는 jpg로 뽑는다.

## 본문 글쓰기 도구 (레퍼런스에서 이식)

- 이미지는 글 폴더 `image/`에 둔다. 글자가 들어간 이미지는 `<이름>.ko.<ext>` / `<이름>.en.<ext>` 짝.
- 이미지 폭: `![설명](./image/a.png "w=320")`. 본문에 `<img width>`를 직접 쓰지 않는다.
- 커스텀 이모지: `src/assets/emoji/<이름>.png` → 본문에서 `:이름:`. frontmatter에 쓸 때는 값을 따옴표로 감싼다.
- 줄 끝 ` %% 2026.01 ~ 현재` → 날짜를 오른쪽 정렬.
- 본문 링크는 자동으로 새 탭에서 열린다.
- MDX에서는 `<!-- -->` 대신 `{/* */}`.

## 검사 훅

- **pre-commit**(`.githooks/pre-commit`): ko/en 짝 동기화, 이미지 언어쌍, ko/en 구조 drift(LLM 없이 무료). 1회 설치: `git config core.hooksPath .githooks`
- **Claude 세션 훅**(`.claude/settings.json`): 편집 후 짝 갱신 알림, 짝이 어긋난 채 턴을 끝내면 차단.
- 유료 LLM 의미 검사(`npm run drift`)는 꺼져 있다. 사용자가 요청할 때만 실행한다.
- 고치는 쪽은 언제나 `en.md`다(`ko.md`가 원본).

## 명령어

```bash
npm run dev         # 개발 서버 → 항상 http://localhost:4321 (백그라운드 데몬)
npm run dev:stop    # 개발 서버 내리기
npm run build       # 정적 빌드 → dist/  (dev 서버를 먼저 내릴 것)
npx astro check     # 타입 체크
npm test            # scripts/ 검사 로직 테스트
```

**dev 서버는 반드시 `npm run dev`로 띄운다.** 그냥 `astro dev`로 띄우면 데몬이 남아 포트가 4322, 4323…으로 밀린다. dev 서버가 떠 있는 채로 `npm run build`를 돌리면 본문이 빈 페이지가 조용히 생길 수 있다.

## 배포

`main`에 push하면 `.github/workflows/deploy.yml`이 빌드해 `https://def-uk.github.io`로 배포한다. 레포 Settings → Pages → Source를 **GitHub Actions**로 설정해야 한다.

## 핵심 파일

| 경로 | 역할 |
| --- | --- |
| `src/lib/content.ts` | ko/en 폴더 → 언어중립 문서, Work 메타, Lab 아이템 |
| `src/lib/media.ts` | frontmatter 상대경로 미디어 해석 |
| `src/lib/i18n.ts` | UI 문구, kind/tool 라벨 |
| `src/components/{WorkGrid,LabGrid,KindFilter,MediaView,WorkMeta}.astro` | 카드·필터·미디어 |
| `src/components/{LangToggle,ThemeToggle}.astro` + `BaseLayout.astro`(FOUC 스크립트) | 언어·테마 전환 |
| `src/pages/[collection]/[...path].astro` | Work·Log 상세, Log 카테고리 목록 |
| `src/pages/lab/[...path].astro` | Lab 상세(←/→ 이동) |

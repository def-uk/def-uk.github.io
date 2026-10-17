# def-uk.github.io 기획

> 레퍼런스: [raewoo0908.github.io](https://github.com/raewoo0908/raewoo0908.github.io) — 작성자 허락 하에 코드 재사용.
> 글쓰기 인프라(한/영 토글 · 폴더 카테고리 · 이모지 · 목차 · giscus)는 가져오고, 겉모습과 경험은 새로 만든다.

## 1. 정체성

**코드도 짜고 영상·모션도 만드는 하이브리드 메이커.**

| 영역 | 도구 |
| --- | --- |
| 영상 · 모션 | Premiere Pro, After Effects, Higgsfield(AI 영상·이미지) |
| 개발 | Python, Flutter, React, Next.js |

## 2. 컨셉 — 코드 ↔ 캔버스

레퍼런스의 대표 기능이 **한/영 토글**이라면, 이 사이트의 대표 기능은 **같은 작업을 코드로도, 비주얼로도 보는 것**이다.

- **홈 히어로**: 왼쪽은 코드, 오른쪽은 그 코드가 그린 그래픽. 가운데 경계를 마우스로 밀면 코드가 그림으로 바뀐다.
- **확장**: Work 케이스스터디에 `Dev 뷰 / Design 뷰` 탭 — 같은 프로젝트를 구현과 디자인 관점으로.
- **톤**: 플레이풀/인터랙티브. 단, `prefers-reduced-motion` 존중, 모바일은 가벼운 버전.

## 3. 정보 구조

| 메뉴 | 내용 | 화면 |
| --- | --- | --- |
| **Work** | 프로젝트 케이스스터디 (문제 → 과정 → 결과) | 카드 목록 → 상세 글 |
| **Lab** | 작품·실험 하나 (영상, 모션, 시안, 스케치) | 메이슨리 그리드 → 전체화면 뷰어 |
| **Log** | 기술 글, 튜토리얼, TIL, 에세이 (폴더 = 카테고리) | 목록 → 상세 글 |
| **About** | 소개 + CV + 경력 타임라인 | 단일 페이지 |

**Work vs Lab 기준: 이야기가 있으면 Work, 결과물 하나면 Lab.** 둘은 서로 연결된다(`work:` 필드 / `<LabItem>` 임베드).

**홈**: 히어로 → 대표 Work 3개(`featured`) → Lab 미리보기 → 최근 Log.

## 4. kind 태그 — code / design / video / motion

메뉴가 아니라 작업마다 붙이는 태그(복수 가능). Work·Lab 목록 위 필터 칩으로 거른다.

| kind | 카드 표지 | 본문 컴포넌트 |
| --- | --- | --- |
| **code** | 스크린샷 / 코드 타이핑 애니메이션 | 코드 블록 · `<Repo>` GitHub 카드 · `<Demo>` 라이브 임베드 |
| **design** | 대표 이미지, 호버 시 다른 시안 | `<Gallery>` · `<BeforeAfter>` · `<Palette>` · `<Figma>` |
| **video** | 호버 시 5~10초 무음 루프 | `<Video>` (짧은 것은 자체 호스팅, 긴 것은 YouTube/Vimeo 파사드) |
| **motion** | 화면에 보이면 자동 루프 | `<Motion>` webm · Lottie · 라이브 코드 스케치 / `<Scrubber>` 프레임 메이킹 뷰 |

### 도구별 결과물 → 사이트 표현

| 도구 | 결과물 | 표현 |
| --- | --- | --- |
| Premiere Pro | 영상 | `video` — 미리보기 루프 + 본편 YouTube |
| After Effects | 모션 | `motion` — webm 루프, 가능하면 Bodymovin → Lottie(가볍고 선명) |
| Higgsfield | AI 영상·이미지 | `video`/`design` + 자동 **AI 배지** (`tools`에 higgsfield가 있으면) |
| Python | 스크립트·분석·자동화 | `code` — 코드 블록, 결과 이미지 |
| Flutter | 앱 | `code`+`design` — 화면 녹화 루프, 가능하면 web 빌드 임베드 |
| React / Next.js | 웹 | `code` — 라이브 데모 링크·iframe |

## 5. 콘텐츠 모델

```
src/content/
├─ work/<slug>/{ko.md, en.md, media/…}         ← 긴 글: 한/영 짝 파일 (en은 Claude 번역)
├─ lab/<year>/<slug>/{index.md, loop.webm, …}  ← 짧은 캡션: 파일 하나에 한/영
├─ log/<카테고리>/…/<slug>/{ko.md, en.md}        ← 레퍼런스 posts와 동일
└─ pages/{home,about}/{ko.md, en.md}
```

**Work frontmatter** (레퍼런스 + 추가)
```yaml
title: 캐치더메모리
date: 2026-10-17T15:00:00+09:00
description: 한 줄 요약
kind: [code, design]
tools: [flutter, after-effects]
role: 기획 · 디자인 · 개발
period: 2026.03 ~ 2026.06
accent: "#ff5c39"     # 이 페이지의 테마 색
cover: ./media/cover.webp
coverVideo: ./media/cover.webm
featured: true        # 홈 노출
draft: false
```

**Lab `index.md`**
```yaml
title: { ko: "타이포 모션 실험 #3", en: "Type Motion Study #3" }
caption: { ko: "가변 폰트 굵기를 사운드에 반응시킴", en: "Variable font weight reacting to sound" }
kind: [motion]
tools: [after-effects]
date: 2026-10-17T15:00:00+09:00
media: ./loop.webm
poster: ./poster.webp
sketch: ./sketch.ts   # (선택) 있으면 영상 대신 브라우저에서 실행
work: catch-the-memory  # (선택) 연결된 Work
```

## 6. 기술 구성

- **Astro** (레퍼런스와 동일) — 정적 사이트, 인터랙티브 부분만 섬(island)으로 로드.
- **React 섬** — 인터랙티브 컴포넌트(히어로, 뷰어, ⌘K)는 익숙한 React로 작성 (`@astrojs/react`).
- **View Transitions** — Astro 내장. 카드 → 상세 전환.
- **애니메이션** — Motion(Framer Motion) 또는 GSAP. 히어로 그래픽은 Canvas 2D, 무거운 3D는 지양.
- **미디어** — GitHub Pages 파일당 100MB 제한, LFS 서빙 불가. 미리보기 루프 ≤ 3MB webm(+mp4), 긴 영상은 YouTube/Vimeo. `npm run media`(ffmpeg)로 압축본·포스터 자동 생성.
- **번역 검사** — 무료 구조 검사(짝 파일 · 이미지 언어쌍)만. 레퍼런스의 유료 LLM drift 검사는 필요해지면 추가.

## 7. 콘텐츠 수급 — "정리 안 된 작업물이 많다"

사이트보다 이게 병목이다. 사이트 개발과 병행한다.

1. **인벤토리** — `docs/inventory.md` 표에 생각나는 작업을 전부 적는다(이름 · 연도 · kind · 파일 위치 · 상태). 완성도는 따지지 않는다.
2. **Lab 먼저** — 이미 렌더된 영상·이미지는 설명 한 줄이면 올라간다. 가장 빨리 사이트가 채워진다.
3. **Work는 3개만** — 가장 자신 있는 것 3개를 골라 케이스스터디로. Claude가 인터뷰(질문 → 답변)로 초안을 쓴다.
4. **Claude 명령어**
   - `/new-lab <파일> "한 줄 설명"` → 폴더 생성 · 압축 · 포스터 · `index.md`(영문 캡션 포함)
   - `/new-work <이름>` → 인터뷰 → `ko.md` 초안 + `en.md` 번역

**공개 기준: Lab 10개 + Work 3개 + About.**

## 8. 단계

| 단계 | 범위 | 완료 기준 |
| --- | --- | --- |
| **1. 뼈대** ✅ | 레퍼런스 이식, 컬렉션 재편(work/lab/log/about), 다크 모드, 기본 목록·상세, 배포 | 글을 쓰고 배포할 수 있다 |
| **2. 시그니처** 🚧 | ✅ 코드↔캔버스 히어로 · ✅ View Transitions · ✅ kind 필터 · ✅ 프로젝트별 accent · ⬜ Lab 전체화면 뷰어 · ⬜ 미디어 컴포넌트(`<Video>`·`<Gallery>`·`<BeforeAfter>`) · ⬜ Dev/Design 뷰 탭 | "이 사람 사이트다" |
| **3. 디테일** | ⌘K 팔레트, 커서, giscus(본인 레포), 이스터에그, `/new-lab`·`/new-work` 명령어 | 완성도 |

## 9. 미정

- 사이트 표시 이름과 소개 한 줄 (현재 placeholder: "dongwook")
- 폰트 — 1단계는 레퍼런스의 A2Z 유지, 2단계에서 정체성에 맞게 재검토
- giscus용 댓글 레포 (3단계)

# def-uk.github.io

코드도 짜고 영상·모션도 만드는 사람의 작업실. [Astro](https://astro.build)로 만들고 GitHub Pages로 배포합니다.

- **Work**: 프로젝트 케이스스터디
- **Lab**: 작품과 실험
- **Log**: 기술 글, TIL, 에세이
- **About**: 소개와 이력
- 한/영 토글, 라이트/다크 테마, `code` · `design` · `video` · `motion` 필터

기획은 [`docs/PLAN.md`](docs/PLAN.md), 작성 규칙은 [`CLAUDE.md`](CLAUDE.md)를 보세요.
한/영 시스템과 글쓰기 도구는 [raewoo0908.github.io](https://github.com/raewoo0908/raewoo0908.github.io)에서 허락을 받고 가져왔습니다.

## 개발

```bash
npm install                          # 최초 1회
git config core.hooksPath .githooks  # 최초 1회: 커밋 전 ko/en 검사
npm run dev                          # http://localhost:4321
npm run build                        # dist/
```

## 올리는 법

| 무엇을 | 어디에 |
| --- | --- |
| 프로젝트 | `src/content/work/<이름>/ko.md` (+ `media/`) |
| 작품 하나 | `src/content/lab/<연도>/<이름>/index.md` (+ 영상·이미지) |
| 글 | `src/content/log/<카테고리>/<이름>/ko.md` |

`sample-*` 폴더를 복사해서 시작하면 됩니다. 한국어만 쓰면 영어는 Claude가 채웁니다.

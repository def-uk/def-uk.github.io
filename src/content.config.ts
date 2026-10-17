import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

/** 작업 유형 태그. 메뉴가 아니라 Work·Lab 목록의 필터 칩이 된다. */
export const KINDS = ['code', 'design', 'video', 'motion'] as const;
const kind = z.array(z.enum(KINDS)).default([]);

/**
 * 한/영 글이 공유하는 frontmatter 스키마.
 * 각 글은 <카테고리>/.../<slug>/ 폴더 안에 ko.md, en.md 두 파일로 존재하며,
 * 두 파일 각각이 이 스키마로 검증된다(title 등은 해당 언어값).
 */
const bilingualSchema = z.object({
  title: z.string(),
  date: z.coerce.date().optional(),
  draft: z.boolean().default(false),
  description: z.string().optional(),
  cover: z.string().optional(),
  tags: z.array(z.string()).default([]),
});

// Log — 기술 글·튜토리얼·TIL·에세이. 폴더 = 카테고리.
const log = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/log' }),
  schema: bilingualSchema,
});

// Work — 프로젝트 케이스스터디. 레퍼런스 스키마에 작업 메타데이터를 더한다.
// cover·coverVideo 는 글 폴더 기준 상대경로(./media/cover.webp) — src/lib/media.ts 가 해석한다.
const work = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/work' }),
  schema: bilingualSchema.extend({
    kind,
    tools: z.array(z.string()).default([]),
    role: z.string().optional(),
    period: z.string().optional(),
    accent: z
      .string()
      .regex(/^#[0-9a-fA-F]{6}$/, 'accent 는 #rrggbb 형식')
      .optional(),
    coverVideo: z.string().optional(),
    featured: z.boolean().default(false),
  }),
});

// Lab — 작품·실험 하나. 캡션이 짧아 짝 파일 대신 index.md 하나에 한/영을 함께 둔다.
const bi = z.object({ ko: z.string(), en: z.string() });
const lab = defineCollection({
  loader: glob({ pattern: '**/index.md', base: './src/content/lab' }),
  schema: z.object({
    title: bi,
    caption: bi.partial().optional(),
    kind,
    tools: z.array(z.string()).default([]),
    date: z.coerce.date(),
    draft: z.boolean().default(false),
    /** 대표 미디어(이미지·영상). 폴더 기준 상대경로. */
    media: z.string(),
    /** 영상일 때 첫 프레임 대신 보여줄 이미지. */
    poster: z.string().optional(),
    /** 연결된 Work 의 경로(예: 'catch-the-memory'). */
    work: z.string().optional(),
  }),
});

// 홈/About 같은 단일 페이지. 폴더당 ko.md, en.md 짝.
// draft: true 는 글 컬렉션과 달리 '배포에서 목록에서 뺀다'가 아니라
// '배포에서 본문 대신 준비중 안내를 보여준다'는 뜻이다(단일 페이지는 목록이 없으므로).
const pages = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/pages' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { log, work, lab, pages };

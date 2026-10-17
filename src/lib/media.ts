import type { ImageMetadata } from 'astro';

/**
 * Work·Lab frontmatter 의 상대경로 미디어(`./media/cover.webp`, `./loop.webm`)를
 * 빌드 산출물로 해석한다.
 *
 * frontmatter 문자열은 Astro 이미지 파이프라인을 타지 않으므로, 두 컬렉션 폴더의 미디어를
 * Vite glob 으로 미리 묶어 두고 "글 파일 위치 + 상대경로"로 찾는다.
 *   - 이미지 → ImageMetadata (astro:assets <Image> 로 리사이즈·webp 최적화)
 *   - 영상   → 해시가 붙은 URL 문자열(그대로 <video src>)
 * 오타는 조용히 빈 카드가 되지 않도록 빌드 실패로 막는다.
 */
const images = import.meta.glob<ImageMetadata>(
  '/src/content/{work,lab}/**/*.{png,jpg,jpeg,webp,avif,gif,svg}',
  { eager: true, import: 'default' },
);
const videos = import.meta.glob<string>('/src/content/{work,lab}/**/*.{webm,mp4,mov}', {
  eager: true,
  query: '?url',
  import: 'default',
});

export type Media =
  | { type: 'image'; src: ImageMetadata }
  | { type: 'video'; src: string };

/** filePath: 엔트리의 저장소 기준 경로(예: 'src/content/lab/2026/foo/index.md'). */
export function resolveMedia(filePath: string, rel: string): Media {
  const dir = '/' + filePath.replace(/^\/+/, '').replace(/\/[^/]+$/, '');
  const key = normalize(`${dir}/${rel}`);
  if (key in images) return { type: 'image', src: images[key] };
  if (key in videos) return { type: 'video', src: videos[key] };
  throw new Error(`[media] ${filePath}: "${rel}" 파일을 찾을 수 없습니다 (${key})`);
}

export function resolveMaybe(filePath: string, rel?: string): Media | undefined {
  return rel ? resolveMedia(filePath, rel) : undefined;
}

function normalize(path: string): string {
  const out: string[] = [];
  for (const part of path.split('/')) {
    if (part === '' || part === '.') continue;
    if (part === '..') out.pop();
    else out.push(part);
  }
  return '/' + out.join('/');
}

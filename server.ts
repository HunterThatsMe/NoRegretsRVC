import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { APP_VERSION, BRAND_NAME } from './src/data/defaults.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const port = Number(process.env.PORT) || 3000;
  const isProd = process.env.NODE_ENV === 'production';

  app.use(express.json());

  function isNewerVersion(candidate: string, current: string): boolean {
    const clean = (v: string) =>
      v
        .replace(/^v/i, '')
        .split('-')[0]
        .split('.')
        .map((n) => parseInt(n, 10) || 0);
    const a = clean(candidate);
    const b = clean(current);
    const len = Math.max(a.length, b.length);
    for (let i = 0; i < len; i++) {
      const av = a[i] ?? 0;
      const bv = b[i] ?? 0;
      if (av > bv) return true;
      if (av < bv) return false;
    }
    return false;
  }

  // Live GitHub release / tag / commit checker for Settings -> Check for updates
  app.get('/api/check-updates', async (_req, res) => {
    const repoOwner = 'HunterThatsMe';
    const repoName = 'NoRegretsRVC';
    const ghHeaders = {
      Accept: 'application/vnd.github+json',
      'User-Agent': `${BRAND_NAME}-UpdateChecker`,
    };

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 6000);

      const relResp = await fetch(
        `https://api.github.com/repos/${repoOwner}/${repoName}/releases/latest`,
        { headers: ghHeaders, signal: controller.signal }
      );
      clearTimeout(timer);

      let newestVersion = APP_VERSION;
      let publishedAt = '';
      let releaseUrl = `https://github.com/${repoOwner}/${repoName}/releases`;

      if (relResp.ok) {
        const relData = (await relResp.json()) as {
          tag_name?: string;
          published_at?: string;
          html_url?: string;
        };
        if (relData.tag_name) newestVersion = relData.tag_name.replace(/^v/i, '');
        if (relData.published_at) publishedAt = relData.published_at;
        if (relData.html_url) releaseUrl = relData.html_url;
      } else {
        // Fallback to git tags if a tag exists without a published GitHub Release
        const tagsResp = await fetch(
          `https://api.github.com/repos/${repoOwner}/${repoName}/tags?per_page=5`,
          { headers: ghHeaders }
        );
        if (tagsResp.ok) {
          const tags = (await tagsResp.json()) as { name?: string }[];
          const topTag = tags?.[0]?.name?.replace(/^v/i, '');
          if (topTag) {
            newestVersion = topTag;
            releaseUrl = `https://github.com/${repoOwner}/${repoName}/releases/tag/${tags[0].name}`;
          }
        }
      }

      const commitResp = await fetch(
        `https://api.github.com/repos/${repoOwner}/${repoName}/commits/main`,
        { headers: ghHeaders }
      );
      let latestCommit = '';
      let commitDate = '';
      if (commitResp.ok) {
        const commitData = (await commitResp.json()) as {
          sha?: string;
          commit?: { author?: { date?: string } };
        };
        if (commitData.sha) latestCommit = commitData.sha.slice(0, 7);
        if (commitData.commit?.author?.date) commitDate = commitData.commit.author.date;
      }

      const upToDate = !isNewerVersion(newestVersion, APP_VERSION);
      res.json({
        ok: true,
        repo: `${repoOwner}/${repoName}`,
        currentVersion: APP_VERSION,
        newestVersion,
        latestTag: newestVersion,
        latestCommit,
        publishedAt: publishedAt || commitDate,
        releaseUrl,
        upToDate,
        checkedAt: new Date().toISOString(),
      });
    } catch (err) {
      res.status(200).json({
        ok: false,
        repo: `${repoOwner}/${repoName}`,
        currentVersion: APP_VERSION,
        newestVersion: APP_VERSION,
        upToDate: true,
        releaseUrl: `https://github.com/${repoOwner}/${repoName}/releases`,
        error:
          err instanceof Error ? err.message : 'Unable to reach GitHub API right now.',
        checkedAt: new Date().toISOString(),
      });
    }
  });

  // Force file download (Content-Disposition: attachment) instead of inline browser viewing
  for (const downloadName of [
    'NoRegretsRVC-Full-Repo.txt',
    'NoRegretsRVC_Kaggle.ipynb',
    'NoRegretsRVC-Beta-0.0.0.zip',
  ]) {
    app.get(`/${downloadName}`, (_req, res, next) => {
      const distCandidate = path.resolve(__dirname, 'dist', downloadName);
      const publicCandidate = path.resolve(__dirname, 'public', downloadName);
      const target =
        isProd && fs.existsSync(distCandidate) ? distCandidate : publicCandidate;
      if (fs.existsSync(target)) {
        res.download(target, downloadName);
        return;
      }
      next();
    });
  }

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      const indexFile = path.join(distDir, 'index.html');
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send('Build output not found. Run npm run build first.');
      }
    });
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`${BRAND_NAME} server running at http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});

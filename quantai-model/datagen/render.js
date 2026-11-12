#!/usr/bin/env node
/*
 * Render QuantAI's real chat context and verify actions in its real engine.
 *
 * Loads ai.html headless (Playwright + Chromium), exposes the app's internal
 * functions for this run only, and for each job:
 *   1. loads the job's dataset (or the built-in example dataset),
 *   2. runs any earlier analyses so "ANALYSES ALREADY RUN" is realistic,
 *   3. captures agentContext() exactly as the chat sends it,
 *   4. runs each expected action through runAction() and records success/errors.
 *
 * Usage: node render.js <repo root> <jobs.jsonl> <out.jsonl>
 * The repository files are not modified; the hook is injected into the served copy.
 */
const fs = require('fs');
const http = require('http');
const path = require('path');
const { chromium } = require('playwright');

const [repo, jobsPath, outPath] = process.argv.slice(2);
if (!repo || !jobsPath || !outPath) { console.error('usage: node render.js <repo> <jobs.jsonl> <out.jsonl>'); process.exit(2); }

const TYPES = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.webmanifest': 'application/manifest+json' };
const ANCHOR = 'start();\n\n})();';
const HOOK = 'window.__QA = { loadDataset, runAction, agentContext, dataContext, resultContext, parseAgent, S, RESEARCH_KB };\n' + ANCHOR;

function serve(root) {
  return new Promise(resolve => {
    const server = http.createServer((req, res) => {
      const url = decodeURIComponent(req.url.split('?')[0]);
      const file = path.join(root, url === '/' ? 'ai.html' : url);
      if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
      let body = fs.readFileSync(file);
      if (url.endsWith('/assets/js/quantai.js')) {
        const src = body.toString('utf8');
        if (!src.includes(ANCHOR)) throw new Error('quantai.js hook anchor not found; update render.js');
        body = Buffer.from(src.replace(ANCHOR, HOOK), 'utf8');
      }
      res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
      res.end(body);
    });
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

(async () => {
  const server = await serve(path.resolve(repo));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch();
  const context = await browser.newContext({ acceptDownloads: false });
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', e => pageErrors.push(e.message));
  await page.route('**/*', r => r.request().url().startsWith(base) ? r.continue() : r.abort());
  await page.goto(base + '/ai.html', { waitUntil: 'load' });
  await page.waitForFunction(() => window.__QA && window.__QA.S && window.__QA.S.ds, null, { timeout: 60000 });
  // Keep the built-in example dataset so jobs can return to it.
  await page.evaluate(() => { const S = window.__QA.S; window.__BUILTIN = { headers: S.raw.headers, rows: S.raw.rows, source: S.dsSource }; });

  const jobs = fs.readFileSync(jobsPath, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
  const out = fs.createWriteStream(outPath);
  let done = 0;
  for (const job of jobs) {
    const result = await page.evaluate(job => {
      const Q = window.__QA;
      const describeVars = () => Q.S.ds.vars.map(v => ({ name: v.name, label: v.label, type: v.type, nMissing: v.nMissing }));
      const run = a => {
        try {
          const r = Q.runAction(JSON.parse(JSON.stringify(a)));
          if (r && r.entry) { const res = r.entry.result; return { ok: true, title: res.title, method: res.method, writeup: res.writeup || '', n: res.n, warnings: res.warnings || [] }; }
          if (r && r.forecast) return { ok: !Q.S.fc.err, forecast: true, error: Q.S.fc.err || null, value: Q.S.fc.value, date: Q.S.fc.date };
          return { ok: true, other: Object.keys(r || {}) };
        } catch (e) { return { ok: false, error: String(e && e.message || e) }; }
      };
      try {
        if (job.dataset === 'builtin') Q.loadDataset(window.__BUILTIN.headers, window.__BUILTIN.rows, window.__BUILTIN.source, true);
        else Q.loadDataset(job.dataset.headers, job.dataset.rows, job.dataset.source, false);
        Q.S.fc = { date: '', value: '', h: '12', res: null, err: null };
        const pre = (job.pre_actions || []).map(run);
        const context = Q.agentContext();
        const verify = (job.actions || []).map(run);
        return { id: job.id, vars: describeVars(), context, pre, verify, quality: (Q.S.quality || []).map(q => q.msg) };
      } catch (e) {
        return { id: job.id, fatal: String(e && e.stack || e) };
      }
    }, job);
    out.write(JSON.stringify(result) + '\n');
    if (++done % 100 === 0) process.stderr.write(`rendered ${done}/${jobs.length}\n`);
  }
  if (jobs.some(j => j.want_kb)) {
    const kb = await page.evaluate(() => (window.__QA.RESEARCH_KB || []).map(x => ({ k: x.k, t: x.t, c: x.c || null, a: x.a })));
    out.write(JSON.stringify({ id: '__kb__', kb }) + '\n');
  }
  out.end();
  await browser.close();
  server.close();
  if (pageErrors.length) { console.error('page errors:', pageErrors.slice(0, 5)); }
  process.stderr.write(`rendered ${done} jobs\n`);
})().catch(e => { console.error(e); process.exit(1); });

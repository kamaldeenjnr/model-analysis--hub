#!/usr/bin/env node
/*
 * End-to-end check of the browser ↔ local model route, in a real headless browser.
 *
 * Serves the site, sets window.QUANTAI_LOCAL_AI_URL, and answers /chat with a
 * fake local model. It then uses the page like a person would:
 *   1. sends a chat message and checks the request the browser makes
 *      (message, context, history; no raw data rows),
 *   2. checks the returned action runs in the engine and a result appears,
 *   3. opens the full result, clicks "Explain in plain language" and checks the
 *      model's reply text is shown, not raw JSON.
 * Usage: node e2e_local_ai.js [repo root]   (needs Playwright + Chromium)
 */
const fs = require('fs');
const http = require('http');
const path = require('path');
const { chromium } = require('playwright');

const repo = path.resolve(process.argv[2] || path.join(__dirname, '..'));
const LOCAL = 'http://quantai-local.test';
const TYPES = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png' };
const EXPLANATION = 'PLAIN-LANGUAGE EXPLANATION FROM THE LOCAL MODEL.';
const ACTION = { type: 'regression', outcome: 'hypertension', predictors: ['age_years', 'sex'], model: 'modpoisson', cluster: null, offset: null };

function fail(msg) { console.error('FAIL: ' + msg); process.exitCode = 1; }

(async () => {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    const file = path.join(repo, url === '/' ? 'ai.html' : url);
    if (!file.startsWith(repo) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
    res.end(fs.readFileSync(file));
  }).listen(0, '127.0.0.1');
  await new Promise(r => server.once('listening', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [], requests = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.addInitScript(url => { window.QUANTAI_LOCAL_AI_URL = url; }, LOCAL);
  await page.route('**/*', async route => {
    const url = route.request().url();
    if (url.startsWith(LOCAL)) {
      const body = JSON.parse(route.request().postData() || '{}');
      requests.push(body);
      const explaining = /RESULT TO EXPLAIN/.test(body.message);
      // While explaining, the fake model also returns an action, which the page must ignore.
      const reply = explaining ? { reply: EXPLANATION, action: ACTION } : { reply: "I'll estimate adjusted prevalence ratios.", action: ACTION };
      return route.fulfill({ status: 200, contentType: 'application/json', headers: { 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({ ...reply, model: 'fake', local_inference: true, retrieved_knowledge: [] }) });
    }
    return url.startsWith(base) ? route.continue() : route.abort();
  });
  await page.goto(base + '/ai.html', { waitUntil: 'load' });
  await page.waitForSelector('#qa-chat-input', { timeout: 30000 });

  await page.fill('#qa-chat-input', 'Estimate adjusted prevalence ratios for hypertension using age and sex');
  await page.press('#qa-chat-input', 'Enter');
  await page.waitForSelector('[data-act="chat-open"]', { timeout: 30000 }).catch(() => fail('no result appeared in the chat'));

  const chat = requests[0];
  if (!chat) fail('the browser did not call the local /chat endpoint');
  else {
    if (typeof chat.message !== 'string' || !chat.message.includes('hypertension')) fail('message field missing');
    if (!/^DATASET: /.test(chat.context) || !chat.context.includes('VARIABLES (name in code')) fail('context is not the dataset summary');
    if (!Array.isArray(chat.history)) fail('history is not a list');
    if (chat.context.length > 9000) fail('context exceeds the 9,000-character limit');
    // privacy: every context line must be one of the summary line types the app writes; a raw data row would not be.
    const allowed = [/^DATASET: /, /^VARIABLES \(/, /^- [^|]+ \| [^|]+ \| [a-z]+ \| \d+ missing \| /, /^$/, /^DATA CHECK /, /^- \[(REVIEW|note)\] /,
      /^ANALYSES ALREADY RUN/, /^\d+\. /, /^ {3}warning: /, /^FORECAST: /, /^QUANTAI TOOLS: /, /^METHODOLOGY INPUTS:$/, /^The visitor is planning/,
      /^Research question framework /, /^Design answers: /, /^Rule-based design recommendation: /, /^Sample size calculator selected: /,
      /^Test selector result: /, /^QuantAI methodology tools: /];
    const odd = chat.context.split('\n').filter(line => !allowed.some(re => re.test(line)));
    if (odd.length) fail('unexpected context lines (possible raw data): ' + JSON.stringify(odd.slice(0, 3)));
  }
  const chatText = await page.textContent('body');
  if (!chatText.includes("I'll estimate adjusted prevalence ratios.")) fail('model reply not shown in the chat');
  if (!/Modified Poisson|prevalence ratio/i.test(chatText)) fail('the engine result for the returned action is not shown');

  await page.click('[data-act="chat-open"]');
  await page.waitForSelector('[data-act="ai-explain"]', { timeout: 30000 });
  await page.click('[data-act="ai-explain"]');
  await page.waitForFunction(t => document.body.innerText.includes(t) || document.querySelector('.notice.bad'), EXPLANATION, { timeout: 30000 })
    .catch(() => fail('explanation did not appear'));
  const after = await page.textContent('body');
  if (!after.includes(EXPLANATION)) fail('explanation text not shown');
  if (after.includes('{"reply"') || after.includes('"action":')) fail('raw JSON shown to the user');
  if (requests.length !== 2) fail(`expected 2 local requests (chat, explain), saw ${requests.length}`);
  if (errors.length) fail('page errors: ' + errors.slice(0, 3).join(' | '));

  await browser.close();
  server.close();
  if (!process.exitCode) console.log('PASS: chat request, engine run and explanation all work through the local model route');
})().catch(e => { console.error(e); process.exit(1); });

/* Site assistant — answers questions about Amadu Kamal and Model Analysis Hub.
   If AI_URL is set, questions go to the AI assistant (a Cloudflare Worker that calls Groq; the key stays there).
   If it is empty, slow or unavailable, the built-in knowledge base below answers instead. */
(function () {
  'use strict';
  var AI_URL = 'https://mah-assistant.kamalamadu8.workers.dev';
  var WA = 'https://wa.me/233595586430', MAIL = 'amadukamal8@gmail.com';

  var KB = [
    { id: 'about', words: 'who amadu kamal founder about you yourself person owner behind background story bio',
      a: 'Amadu Kamal is a data analyst and software engineer from Ghana, and the founder of Model Analysis Hub. He builds statistical models, risk maps, dashboards and websites, and created the free QuantAI data tool. <a href="founder.html">More about Amadu</a>.' },
    { id: 'contact', words: 'contact reach email mail phone number call whatsapp talk message hire book meeting',
      a: 'The quickest way is WhatsApp: <a href="' + WA + '" target="_blank" rel="noopener">+233 59 558 6430</a>. You can also email <a href="mailto:' + MAIL + '">' + MAIL + '</a> or use the <a href="index.html#contact">contact form</a>.' },
    { id: 'services', words: 'service services offer do help build work with can you make need project job freelance consult consulting',
      a: 'Amadu takes on three kinds of work: <b>statistical and predictive models</b> (forecasts, risk models, validation), <b>geospatial analysis</b> (maps of where a problem is concentrated) and <b>websites, dashboards and web platforms</b> (like <a href="projects.html#sunshine-project">The Sunshine Project website</a>). Tell him what you need: <a href="index.html#contact">get in touch</a>.' },
    { id: 'price', words: 'price cost charge rate fee budget how much pay quote',
      a: 'Pricing depends on the size of the job and how clean the data is. Send a short description and a sample of the data on <a href="' + WA + '" target="_blank" rel="noopener">WhatsApp</a> or <a href="mailto:' + MAIL + '">email</a> and you will get a quote.' },
    { id: 'projects', words: 'projects portfolio work examples case studies done previous past built',
      a: 'Highlights: <a href="projects.html#sunshine-project">The Sunshine Project website</a> for a Ghanaian NGO, a <a href="projects.html#malaria-risk">malaria risk model for Ghana</a> (with risk maps, exceedance probabilities and validation), a <a href="projects.html#healthcare-dashboard">healthcare utilisation dashboard</a> and <a href="projects.html#projectflow">ProjectFlow</a>, a project management platform for university research. <a href="projects.html">See all projects</a>.' },
    { id: 'malaria', words: 'malaria ghana map maps risk disease health epidemiology geospatial spatial region exceedance prevalence',
      a: 'The malaria project estimates relative malaria risk for each region of Ghana, the probability that prevalence goes above 30%, and checks the predictions against observed prevalence. <a href="projects.html#malaria-risk">See the maps</a>.' },
    { id: 'sunshine', words: 'sunshine project ngo charity website websites site web design client clients built make build wordpress landing page',
      a: 'Yes, Amadu builds websites. His latest is for <a href="projects.html#sunshine-project">The Sunshine Project</a>, a Ghanaian NGO: a full multi-page site with an admin login so their team updates programmes, schedules and photos themselves. It is live at <a href="https://sunshine-project-website.vercel.app" target="_blank" rel="noopener">sunshine-project-website.vercel.app</a>. Want one? <a href="https://wa.me/233595586430" target="_blank" rel="noopener">Message him on WhatsApp</a>.' },
    { id: 'projectflow', words: 'projectflow project flow university student thesis research management platform phd masters undergraduate',
      a: 'ProjectFlow is a full-stack platform for managing university research projects at Undergraduate, Master\'s and PhD level, from first draft to final publication. It was completed in 2025. <a href="projects.html#projectflow">Details</a>.' },
    { id: 'quantai', words: 'quantai quant ai tool analyse analyze upload spreadsheet csv excel data statistics forecast regression correlation',
      a: '<b>QuantAI</b> is a free data tool on this site. Add a CSV or Excel file, then type a question such as "What affects sales most?" in the box that appears. It answers in plain English and backs it up with charts, correlations, a regression and a forecast. Your file stays on your device. <a href="ai.html">Open QuantAI</a>.' },
    { id: 'privacy', words: 'privacy private safe secure data upload stored store server confidential',
      a: 'QuantAI does its calculations inside your browser, so your file is never uploaded or saved. When you ask a question with AI switched on, only a summary of the numbers is sent to get the answer. You can switch AI off for built-in answers only.' },
    { id: 'skills', words: 'skills tools stack languages technologies python r javascript react sql arcgis geopandas numpy tech',
      a: 'Main tools: <b>Python</b> and <b>R</b> for statistics and modelling (NumPy, SciPy), <b>ArcGIS</b> and <b>GeoPandas</b> for maps, and <b>JavaScript</b>, <b>React</b> and <b>SQL</b> for web platforms and dashboards.' },
    { id: 'location', words: 'where location based country ghana africa city remote',
      a: 'Amadu is based in Ghana and works with clients remotely too.' },
    { id: 'education', words: 'education school university degree study studied learn learned self taught training',
      a: 'Amadu built his skills over five years through courses, mentors and a long run of real projects. <a href="founder.html">Read his background</a>.' },
    { id: 'hello', words: 'hi hello hey good morning afternoon evening greetings',
      a: 'Hello! Ask me about Amadu, his projects, the QuantAI data tool, or how to get in touch.' },
    { id: 'thanks', words: 'thanks thank you great cool nice ok okay',
      a: 'You\'re welcome. If you want to talk about a project, WhatsApp is the fastest: <a href="' + WA + '" target="_blank" rel="noopener">message Amadu</a>.' }
  ];

  var STOP = ' a an the i me my you your he his is are was do does did can could will would how what who where when why which to of in on for and or it this that with about please tell show '
  function tokens(s) { return String(s).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(function (t) { return t && STOP.indexOf(' ' + t + ' ') < 0; }); }
  function lev(a, b) {
    if (Math.abs(a.length - b.length) > 2) return 3;
    var d = []; for (var i = 0; i <= a.length; i++) { d[i] = [i]; }
    for (var j = 1; j <= b.length; j++) d[0][j] = j;
    for (i = 1; i <= a.length; i++) for (j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    return d[a.length][b.length];
  }
  function answer(q) {
    var qt = tokens(q), best = null, bestScore = 0;
    if (!qt.length) qt = String(q).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
    KB.forEach(function (k) {
      var words = k.words.split(' '), score = 0;
      qt.forEach(function (t) {
        if (t.length < 2) return;
        words.forEach(function (w) {
          if (t === w) score += 3;
          else if (t.length > 3 && (w.indexOf(t) === 0 || t.indexOf(w) === 0)) score += 2;
          else if (t.length > 4 && lev(t, w) <= 1) score += 1.5;
        });
      });
      if (score > bestScore) { bestScore = score; best = k; }
    });
    if (best && bestScore >= 2) return best.a;
    return 'I\'m not sure about that one. I can tell you about <b>Amadu</b>, his <b>projects</b>, <b>services</b>, the <b>QuantAI</b> tool, or how to <b>contact</b> him. For anything else, <a href="' + WA + '" target="_blank" rel="noopener">ask Amadu directly on WhatsApp</a>.';
  }

  var css = '' +
    '.asst-btn{position:fixed;right:20px;bottom:20px;z-index:90;display:flex;align-items:center;gap:.5rem;height:52px;padding:0 1.1rem 0 .9rem;border:0;border-radius:999px;background:#0C203B;color:#fff;font:600 .95rem "Plex Sans",system-ui,sans-serif;box-shadow:0 10px 30px -10px rgba(12,32,59,.6);cursor:pointer;transition:transform .2s}' +
    '.asst-btn:hover{transform:translateY(-2px)}.asst-btn svg{width:20px;height:20px}' +
    '.asst{position:fixed;right:20px;bottom:84px;z-index:91;width:360px;max-width:calc(100vw - 32px);height:min(520px,calc(100vh - 120px));display:flex;flex-direction:column;background:#fff;border:1px solid #DDE3EA;border-radius:16px;box-shadow:0 30px 60px -20px rgba(12,32,59,.45);overflow:hidden;font-family:"Plex Sans",system-ui,sans-serif}' +
    '.asst[hidden]{display:none}' +
    '.asst-h{display:flex;align-items:center;justify-content:space-between;gap:.75rem;padding:.9rem 1rem;background:#0C203B;color:#fff}' +
    '.asst-h b{display:block;font:800 1rem "Bricolage","Plex Sans",sans-serif}.asst-h small{display:block;color:#AFBCCD;font-size:.78rem}' +
    '.asst-h img{width:36px;height:36px;border-radius:50%;object-fit:cover;object-position:50% 20%}' +
    '.asst-x{width:36px;height:36px;border:0;border-radius:8px;background:rgba(255,255,255,.1);color:#fff;cursor:pointer;font-size:1.1rem}' +
    '.asst-log{flex:1;overflow-y:auto;padding:1rem;display:flex;flex-direction:column;gap:.6rem;background:#F5F7FA}' +
    '.asst-m{max-width:85%;padding:.6rem .8rem;border-radius:12px;font-size:.92rem;line-height:1.5;color:#0E1726;background:#fff;border:1px solid #DDE3EA;align-self:flex-start}' +
    '.asst-m a{color:#C4412B;font-weight:600}.asst-m.me{align-self:flex-end;background:#0C203B;color:#fff;border-color:#0C203B}' +
    '.asst-chips{display:flex;flex-wrap:wrap;gap:.4rem;padding:.6rem 1rem 0;background:#fff}' +
    '.asst-chips button{border:1px solid #DDE3EA;background:#fff;border-radius:999px;padding:.35rem .7rem;font-size:.8rem;cursor:pointer;color:#3A4658}.asst-chips button:hover{border-color:#0C203B;color:#0C203B}' +
    '.asst-f{display:flex;gap:.5rem;padding:.75rem 1rem 1rem;background:#fff}' +
    '.asst-f input{flex:1;min-width:0;height:44px;padding:0 .8rem;border:1.5px solid #DDE3EA;border-radius:10px;font-size:.95rem;background:#F5F7FA}' +
    '.asst-f input:focus{outline:none;border-color:#0C203B;background:#fff}' +
    '.asst-f button{width:44px;height:44px;border:0;border-radius:10px;background:#C4412B;color:#fff;cursor:pointer;display:grid;place-items:center}' +
    '.asst-f button svg{width:18px;height:18px}.asst-f button:disabled{opacity:.6;cursor:wait}' +
    '.asst-dots{display:inline-flex;gap:4px;align-items:center;height:1.2em}.asst-dots i{width:6px;height:6px;border-radius:50%;background:#8A96A8;animation:asstb 1s infinite ease-in-out}' +
    '.asst-dots i:nth-child(2){animation-delay:.15s}.asst-dots i:nth-child(3){animation-delay:.3s}' +
    '@keyframes asstb{0%,80%,100%{opacity:.3;transform:translateY(0)}40%{opacity:1;transform:translateY(-3px)}}' +
    '@media (prefers-reduced-motion:reduce){.asst-dots i{animation:none}}' +
    '@media print{.asst,.asst-btn{display:none!important}}';

  var st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
  var chatIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/></svg>';
  var sendIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>';
  var wrap = document.createElement('div');
  wrap.innerHTML =
    '<button class="asst-btn" type="button" aria-expanded="false" aria-controls="asst">' + chatIcon + '<span>' + (/ai\.html$/.test(location.pathname) ? 'Ask about Amadu' : 'Ask a question') + '</span></button>' +
    '<section class="asst" id="asst" role="dialog" aria-label="Questions about Amadu Kamal" hidden>' +
      '<div class="asst-h"><div style="display:flex;gap:.65rem;align-items:center"><img src="assets/img/amadu-kamal-480.webp" alt=""><div><b>Ask about Amadu</b><small>AI assistant · answers about his work</small></div></div><button class="asst-x" type="button" aria-label="Close">&times;</button></div>' +
      '<div class="asst-log" aria-live="polite"></div>' +
      '<div class="asst-chips"><button type="button">What services do you offer?</button><button type="button">Show me projects</button><button type="button">What is QuantAI?</button><button type="button">How do I contact you?</button></div>' +
      '<form class="asst-f"><label class="sr" for="asst-q">Your question</label><input id="asst-q" type="text" autocomplete="off" placeholder="Type a question…" maxlength="300"><button type="submit" aria-label="Send">' + sendIcon + '</button></form>' +
    '</section>';
  document.body.appendChild(wrap);

  var btn = wrap.querySelector('.asst-btn'), box = wrap.querySelector('.asst'), log = wrap.querySelector('.asst-log');
  var form = wrap.querySelector('form'), input = wrap.querySelector('input');
  function add(html, me) {
    var m = document.createElement('div'); m.className = 'asst-m' + (me ? ' me' : '');
    if (me) m.textContent = html; else m.innerHTML = html;
    log.appendChild(m); log.scrollTop = log.scrollHeight;
  }
  function open(v) {
    box.hidden = !v; btn.setAttribute('aria-expanded', String(v));
    if (v) { if (!log.children.length) add('Hi, I can answer quick questions about Amadu Kamal and Model Analysis Hub. What would you like to know?'); input.focus(); }
    else btn.focus();
  }
  /* Turn the AI's plain text into safe HTML: escape everything, then allow **bold** and links to known places only */
  function esc(t) { return t.replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function safeHref(u) {
    if (/^(index|founder|projects|ai)\.html(#[\w-]+)?$/.test(u)) return u;
    if (/^mailto:[\w.+-]+@[\w.-]+$/.test(u)) return u;
    if (/^https:\/\/(wa\.me|www\.linkedin\.com|github\.com|sunshine-project-website\.vercel\.app|modelanalysishub\.com)(\/[^\s]*)?$/.test(u)) return u;
    return null;
  }
  function linkTag(href, text) {
    var ext = /^https:/.test(href) ? ' target="_blank" rel="noopener"' : '';
    return '<a href="' + href + '"' + ext + '>' + text + '</a>';
  }
  function render(text) {
    var h = esc(text);
    h = h.replace(/\[([^\]]{1,80})\]\(([^)\s]{1,200})\)/g, function (m, t, u) {
      var href = safeHref(u.replace(/&amp;/g, '&')); return href ? linkTag(esc(href), t) : t;
    });
    h = h.replace(/(^|[\s(])(https:\/\/[^\s<)]+)/g, function (m, pre, u) {
      var clean = u.replace(/[.,;:]+$/, ''), tail = u.slice(clean.length), href = safeHref(clean.replace(/&amp;/g, '&'));
      return pre + (href ? linkTag(esc(href), clean.replace(/^https:\/\//, '')) : clean) + tail;
    });
    h = h.replace(/\*\*([^*]{1,120})\*\*/g, '<b>$1</b>');
    return h.replace(/\n{2,}/g, '<br><br>').replace(/\n/g, '<br>');
  }

  var history = [], busy = false;
  function ask(q) {
    q = q.trim(); if (!q || busy) return;
    add(q, true); input.value = '';
    history.push({ role: 'user', content: q.slice(0, 600) });
    if (!AI_URL) { setTimeout(function () { reply(answer(q), q); }, 250); return; }

    busy = true; form.querySelector('button').disabled = true;
    var typing = document.createElement('div');
    typing.className = 'asst-m'; typing.setAttribute('aria-label', 'Typing');
    typing.innerHTML = '<span class="asst-dots"><i></i><i></i><i></i></span>';
    log.appendChild(typing); log.scrollTop = log.scrollHeight;

    var ctrl = window.AbortController ? new AbortController() : null;
    var timer = setTimeout(function () { if (ctrl) ctrl.abort(); }, 15000);
    fetch(AI_URL, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history.slice(-10) }), signal: ctrl ? ctrl.signal : undefined
    }).then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (d) { if (!d || typeof d.reply !== 'string' || !d.reply) throw 0; return d.reply; })
      .then(function (text) { typing.remove(); reply(render(text), text); },
            function () { typing.remove(); reply(answer(q), q); })
      .then(function () { clearTimeout(timer); busy = false; form.querySelector('button').disabled = false; });
  }
  function reply(html, plain) {
    add(html);
    var t = document.createElement('div'); t.innerHTML = html;
    history.push({ role: 'assistant', content: (t.textContent || plain || '').slice(0, 600) });
  }
  btn.addEventListener('click', function () { open(box.hidden); });
  wrap.querySelector('.asst-x').addEventListener('click', function () { open(false); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !box.hidden) open(false); });
  form.addEventListener('submit', function (e) { e.preventDefault(); ask(input.value); });
  wrap.querySelectorAll('.asst-chips button').forEach(function (b) { b.addEventListener('click', function () { ask(b.textContent); }); });
})();

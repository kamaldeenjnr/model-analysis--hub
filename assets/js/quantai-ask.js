/* QuantAI — "Ask about your data".
   Answers plain-English questions about the loaded file.
   With AI on, a statistical summary of the data (never the file itself) goes to the Model Analysis Hub assistant.
   Built-in answers cover common questions and take over whenever the AI is off or unreachable. */
(function () {
  'use strict';
  var AI_URL = 'https://mah-assistant.kamalamadu8.workers.dev';
  var Q = window.QuantAI;
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { return String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };

  var form = $('#qa-ask-form'), input = $('#qa-ask-q'), log = $('#qa-ask-log'), chipsEl = $('#qa-ask-chips'), aiToggle = $('#qa-ask-ai');
  if (!form || !Q) return;

  var app = null, ctx = '', history = [], busy = false;

  /* ------------------------------------------------------------ data helpers */
  function prof() { return app.state.prof; }
  function nums() { return prof().cols.filter(function (c) { return c.type === 'number' && !c.isId; }); }
  function cats() { return prof().cols.filter(function (c) { return c.type === 'category' && c.unique >= 2 && c.unique <= 12; }); }
  function dateCol() { return prof().cols.filter(function (c) { return c.type === 'date'; })[0] || null; }
  function labelCol() {
    var d = dateCol(); if (d) return d;
    return prof().cols.filter(function (c) { return (c.type === 'text' || c.type === 'category') && c.unique > prof().rows * 0.6; })[0] || null;
  }
  function rowLabel(i) {
    var l = labelCol(); if (!l) return 'row ' + (i + 2);
    var v = l.values[i];
    if (v instanceof Date) return Q.fmtDate(v, l.step);
    return v === null || v === undefined ? 'row ' + (i + 2) : String(v);
  }
  function argExtreme(col, wantMax) {
    var best = -1;
    col.values.forEach(function (v, i) { if (isNaN(v)) return; if (best < 0 || (wantMax ? v > col.values[best] : v < col.values[best])) best = i; });
    return best;
  }
  function norm(s) { return String(s).toLowerCase().replace(/[_\-.]+/g, ' ').replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim(); }
  function mentioned(q) {
    var nq = ' ' + norm(q) + ' ', found = [];
    prof().cols.forEach(function (c) {
      var n = norm(c.name); if (!n) return;
      var hit = nq.indexOf(' ' + n + ' ') >= 0;
      if (!hit) {
        var words = n.split(' ').filter(function (w) { return w.length > 3 && ['total', 'number', 'count', 'average', 'value', 'distributed'].indexOf(w) < 0; });
        hit = words.some(function (w) { return nq.indexOf(' ' + w + ' ') >= 0 || nq.indexOf(' ' + w + 's ') >= 0 || (w.length > 4 && nq.indexOf(' ' + w.slice(0, -1)) >= 0); });
      }
      if (hit) found.push(c);
    });
    return found;
  }
  function trendOf(col) {
    var d = dateCol(), s = Q.series(prof(), d, col), y = s.y;
    if (y.length < 6) return null;
    var h = Math.floor(y.length / 2), a = Q.mean(y.slice(0, h)), b = Q.mean(y.slice(y.length - h));
    return { first: a, second: b, change: a ? (b - a) / Math.abs(a) : NaN, n: y.length, from: s.x[0], to: s.x[s.x.length - 1], dated: !!d };
  }
  function drivers(target) {
    return nums().filter(function (c) { return c !== target; }).map(function (c) {
      var r = Q.pearson(c.values, target.values); return { col: c, r: r.r, p: r.p, n: r.n };
    }).filter(function (d) { return isFinite(d.r); }).sort(function (a, b) { return Math.abs(b.r) - Math.abs(a.r); });
  }
  function groupMeans(cat, num) {
    var g = {};
    cat.values.forEach(function (v, i) { var x = num.values[i]; if (v === null || isNaN(x)) return; (g[v] = g[v] || []).push(x); });
    return Object.keys(g).map(function (k) { return { level: k, mean: Q.mean(g[k]), n: g[k].length }; }).sort(function (a, b) { return b.mean - a.mean; });
  }
  function forecastOf(col) {
    var d = dateCol(); if (!d) return null;
    var s = Q.series(prof(), d, col); if (s.y.length < 8) return null;
    var period = d.step && d.step.period > 1 && s.y.length >= 2 * d.step.period ? d.step.period : 1;
    try {
      var hw = Q.holtWinters(s.y, period, 3);
      return { next: hw.forecast.map(function (v, k) { return { when: Q.fmtDate(Q.addSteps(s.x[s.x.length - 1], d.step, k + 1), d.step), v: v, lo: hw.lower[k], hi: hw.upper[k] }; }), seasonal: period > 1, last: s.y[s.y.length - 1] };
    } catch (e) { return null; }
  }
  function strengthWord(r) { var a = Math.abs(r); return a >= 0.8 ? 'very strong' : a >= 0.6 ? 'strong' : a >= 0.4 ? 'moderate' : a >= 0.2 ? 'weak' : 'very weak'; }
  function F(v) { return Q.fmt(v); }

  /* ------------------------------------------------------------ summary for the AI */
  function buildContext() {
    var p = prof(), out = [];
    out.push('File: ' + app.state.name + '. ' + p.rows + ' rows, ' + p.cols.length + ' columns. Empty cells: ' + Q.pct(p.missingPct) + '. Duplicate rows: ' + p.duplicates + '.');
    out.push('COLUMNS:');
    p.cols.forEach(function (c) {
      var line = '- "' + c.name + '" (' + c.type + (c.isId ? ', looks like an ID' : '') + ', missing ' + c.missing + ')';
      if (c.type === 'number') {
        var s = c.stats, hi = argExtreme(c, true), lo = argExtreme(c, false);
        line += ': mean ' + F(s.mean) + ', median ' + F(s.median) + ', sd ' + F(s.sd) + ', min ' + F(s.min) + ' (' + rowLabel(lo) + '), max ' + F(s.max) + ' (' + rowLabel(hi) + '), total ' + F(s.sum) + ', unusual values ' + s.outliers;
        var t = trendOf(c); if (t && t.dated) line += ', second half of period averages ' + (isFinite(t.change) ? Q.pct(Math.abs(t.change)) + (t.change >= 0 ? ' higher' : ' lower') : 'n/a') + ' than first half';
      } else if (c.type === 'date') {
        line += ': from ' + Q.fmtDate(c.range[0], c.step) + ' to ' + Q.fmtDate(c.range[1], c.step) + ', every ' + (c.step ? c.step.name : 'step');
      } else if (c.freq) {
        line += ': ' + c.unique + ' distinct; most common ' + c.freq.top.slice(0, 5).map(function (t) { return '"' + String(t.value).slice(0, 30) + '" (' + t.count + ')'; }).join(', ');
      }
      out.push(line);
    });
    var ns = nums();
    if (ns.length >= 2) {
      var pairs = [];
      for (var i = 0; i < ns.length; i++) for (var j = i + 1; j < ns.length; j++) { var r = Q.pearson(ns[i].values, ns[j].values); if (isFinite(r.r)) pairs.push({ a: ns[i].name, b: ns[j].name, r: r.r, p: r.p }); }
      pairs.sort(function (x, y) { return Math.abs(y.r) - Math.abs(x.r); });
      out.push('CORRELATIONS (strongest first): ' + pairs.slice(0, 10).map(function (x) { return '"' + x.a + '" & "' + x.b + '" r=' + Q.fmt(x.r, 2) + ' (p ' + Q.fmtP(x.p) + ')'; }).join('; '));
      var target = ns[ns.length - 1], X = ns.slice(0, -1).slice(0, 8);
      var m = Q.ols(target.values, X.map(function (c) { return c.values; }), X.map(function (c) { return c.name; }));
      if (m && !m.error) out.push('REGRESSION of "' + target.name + '" on the other numeric columns: R²=' + Q.fmt(m.r2, 3) + '; ' + m.coefs.slice(1).map(function (c) { return '"' + c.name + '" coef ' + Q.fmt(c.estimate) + ' (p ' + Q.fmtP(c.p) + ')'; }).join('; '));
      var fc = forecastOf(target);
      if (fc) out.push('FORECAST of "' + target.name + '" (Holt-Winters' + (fc.seasonal ? ', seasonal' : '') + '): ' + fc.next.map(function (x) { return x.when + ' ≈ ' + F(x.v) + ' (80% range ' + F(x.lo) + '–' + F(x.hi) + ')'; }).join('; '));
    }
    cats().slice(0, 3).forEach(function (c) {
      ns.slice(-2).forEach(function (n) {
        out.push('AVERAGE "' + n.name + '" BY "' + c.name + '": ' + groupMeans(c, n).slice(0, 8).map(function (g) { return '"' + g.level + '" ' + F(g.mean) + ' (n=' + g.n + ')'; }).join(', '));
      });
    });
    try { out.push('FINDINGS: ' + Q.findings(p).map(function (f) { return f.text; }).join(' ')); } catch (e) {}
    return out.join('\n').slice(0, 7000);
  }

  /* ------------------------------------------------------------ built-in answers */
  var has = function (q, words) { return words.some(function (w) { return new RegExp('\\b' + w).test(q); }); };
  function localAnswer(question) {
    var q = question.toLowerCase(), cols = mentioned(question), ns = nums(), p = prof();
    var numCols = cols.filter(function (c) { return c.type === 'number'; });
    var catCol = cols.filter(function (c) { return c.type === 'category'; })[0];
    var target = numCols[0] || ns[ns.length - 1];
    var go = function (tab, label) { return ' <button type="button" class="qa-link" data-go="' + tab + '">' + label + '</button>'; };

    if (!ns.length && !has(q, ['row', 'column', 'missing', 'empty', 'problem', 'quality'])) return 'This file has no number columns, so there is little to calculate. The Columns tab shows what each column contains.' + go('columns', 'Open Columns');

    if (has(q, ['how many row', 'how many record', 'how many entr', 'size', 'how big', 'how many column'])) {
      return 'The file has <b>' + Q.fmtInt(p.rows) + ' rows</b> and <b>' + p.cols.length + ' columns</b>: ' + p.cols.map(function (c) { return esc(c.name); }).join(', ') + '.';
    }
    if (has(q, ['missing', 'empty', 'blank', 'problem', 'quality', 'error', 'clean', 'wrong', 'issue', 'outlier', 'unusual'])) {
      var miss = p.cols.filter(function (c) { return c.missing; }).map(function (c) { return esc(c.name) + ' (' + c.missing + ')'; });
      var outl = ns.filter(function (c) { return c.stats.outliers; }).map(function (c) { return esc(c.name) + ' (' + c.stats.outliers + ')'; });
      var parts = [];
      parts.push(miss.length ? 'Empty cells in: ' + miss.join(', ') + '.' : 'No empty cells.');
      parts.push(p.duplicates ? p.duplicates + ' duplicate rows.' : 'No duplicate rows.');
      parts.push(outl.length ? 'Unusual values worth checking in: ' + outl.join(', ') + '.' : 'No values stand far outside the rest.');
      return parts.join(' ');
    }
    if (catCol && numCols.length && has(q, ['by ', 'per ', 'each', 'compare', 'which', 'differ', 'group'])) {
      var gm = groupMeans(catCol, numCols[0]);
      return 'Average <b>' + esc(numCols[0].name) + '</b> by <b>' + esc(catCol.name) + '</b>: ' + gm.map(function (g) { return esc(g.level) + ' ' + F(g.mean); }).join(', ') + '. Highest is <b>' + esc(gm[0].level) + '</b>.';
    }
    if (has(q, ['forecast', 'predict', 'next', 'future', 'expect', 'will ', 'coming'])) {
      var fc = forecastOf(target);
      if (!fc) return 'A forecast needs a date column and at least 8 time points. ' + (dateCol() ? 'There are not enough points for "' + esc(target.name) + '".' : 'This file has no date column.');
      return 'Forecast for <b>' + esc(target.name) + '</b>' + (fc.seasonal ? ', including its repeating seasonal pattern' : '') + ': ' + fc.next.map(function (x) { return esc(x.when) + ' about <b>' + F(x.v) + '</b> (likely ' + F(x.lo) + ' to ' + F(x.hi) + ')'; }).join('; ') + '.' + go('forecast', 'Open the full forecast');
    }
    if (has(q, ['affect', 'drive', 'influence', 'impact', 'cause', 'why', 'factor', 'explain', 'depend', 'relat', 'correlat', 'link', 'connect', 'effect'])) {
      if (numCols.length >= 2) {
        var r = Q.pearson(numCols[0].values, numCols[1].values);
        return '<b>' + esc(numCols[0].name) + '</b> and <b>' + esc(numCols[1].name) + '</b> have a ' + strengthWord(r.r) + (r.r >= 0 ? ' positive' : ' negative') + ' relationship (r = ' + Q.fmt(r.r, 2) + ', ' + (r.p < 0.05 ? 'unlikely to be chance' : 'could be chance') + '). ' + (r.r >= 0 ? 'When one is high, the other tends to be high too.' : 'When one is high, the other tends to be low.') + ' This shows they move together, not that one causes the other.' + go('relations', 'See the scatter plot');
      }
      var d = drivers(target);
      if (!d.length) return 'There are no other number columns to compare with "' + esc(target.name) + '".';
      var top = d.slice(0, 3).map(function (x) { return '<b>' + esc(x.col.name) + '</b> (' + strengthWord(x.r) + ', r = ' + Q.fmt(x.r, 2) + ')'; });
      return 'The columns most closely linked to <b>' + esc(target.name) + '</b> are ' + top.join(', ') + '. Linked does not mean caused, but these are the first places to look.' + go('regression', 'Test them together in a regression');
    }
    if (has(q, ['trend', 'increas', 'decreas', 'grow', 'rise', 'rising', 'fall', 'drop', 'over time', 'going up', 'going down', 'change'])) {
      var t = trendOf(target);
      if (!t) return 'There are too few rows to judge a trend in "' + esc(target.name) + '".';
      var dir = !isFinite(t.change) || Math.abs(t.change) < 0.03 ? 'roughly flat' : t.change > 0 ? 'rising' : 'falling';
      return '<b>' + esc(target.name) + '</b> is ' + dir + (t.dated ? ' over time' : ' across the file') + ': the second half averages ' + F(t.second) + ', against ' + F(t.first) + ' in the first half' + (isFinite(t.change) ? ' (' + (t.change >= 0 ? '+' : '−') + Q.pct(Math.abs(t.change)) + ')' : '') + '.' + (dateCol() ? go('forecast', 'See the chart') : '');
    }
    var s = target.stats;
    if (has(q, ['highest', 'max', 'most', 'peak', 'top', 'largest', 'biggest', 'best'])) {
      var hi = argExtreme(target, true);
      return 'The highest <b>' + esc(target.name) + '</b> is <b>' + F(s.max) + '</b>, in ' + esc(rowLabel(hi)) + '.';
    }
    if (has(q, ['lowest', 'min', 'least', 'smallest', 'worst', 'bottom'])) {
      var lo = argExtreme(target, false);
      return 'The lowest <b>' + esc(target.name) + '</b> is <b>' + F(s.min) + '</b>, in ' + esc(rowLabel(lo)) + '.';
    }
    if (has(q, ['total', 'sum', 'altogether', 'overall', 'combined'])) return 'The total of <b>' + esc(target.name) + '</b> is <b>' + F(s.sum) + '</b> across ' + Q.fmtInt(s.n) + ' rows.';
    if (has(q, ['median', 'middle'])) return 'The median <b>' + esc(target.name) + '</b> is <b>' + F(s.median) + '</b>. Half the rows are below it and half above.';
    if (has(q, ['average', 'mean', 'typical', 'usual', 'normally'])) return 'The average <b>' + esc(target.name) + '</b> is <b>' + F(s.mean) + '</b> (median ' + F(s.median) + ', from ' + F(s.min) + ' to ' + F(s.max) + ').';
    if (has(q, ['spread', 'range', 'vary', 'variation', 'distribut', 'deviation'])) return '<b>' + esc(target.name) + '</b> runs from ' + F(s.min) + ' to ' + F(s.max) + '. Most values fall between ' + F(s.q1) + ' and ' + F(s.q3) + ' (standard deviation ' + F(s.sd) + ').' + go('columns', 'See the histogram');
    if (cols.length && numCols.length) return '<b>' + esc(target.name) + '</b>: average ' + F(s.mean) + ', lowest ' + F(s.min) + ' (' + esc(rowLabel(argExtreme(target, false))) + '), highest ' + F(s.max) + ' (' + esc(rowLabel(argExtreme(target, true))) + '), total ' + F(s.sum) + '.';
    var f = Q.findings(p).slice(0, 3).map(function (x) { return esc(x.text); });
    return 'Here is what stands out: ' + f.join(' ') + ' <br><br>Try asking about a column by name, for example "What is the average ' + esc(target.name) + '?" or "What affects ' + esc(target.name) + '?"';
  }

  /* ------------------------------------------------------------ AI reply rendering (escaped, then light formatting) */
  function renderAI(text) {
    var h = esc(text);
    h = h.replace(/\*\*([^*]{1,160})\*\*/g, '<b>$1</b>');
    h = h.replace(/(^|\n)\s*[-*•]\s+/g, '$1• ');
    return h.replace(/\n{2,}/g, '<br><br>').replace(/\n/g, '<br>');
  }

  /* ------------------------------------------------------------ UI */
  function add(html, who, note) {
    var m = document.createElement('div'); m.className = 'qa-msg-row ' + who;
    var b = document.createElement('div'); b.className = 'qa-bubble';
    if (who === 'me') b.textContent = html; else b.innerHTML = html;
    m.appendChild(b);
    if (note) { var n = document.createElement('small'); n.textContent = note; m.appendChild(n); }
    log.appendChild(m); log.hidden = false;
    log.scrollTop = log.scrollHeight;
    Array.prototype.forEach.call(b.querySelectorAll('[data-go]'), function (btn) {
      btn.addEventListener('click', function () { app.goTo(btn.getAttribute('data-go')); });
    });
    return m;
  }
  function setBusy(v) { busy = v; form.querySelector('button[type=submit]').disabled = v; }

  function ask(question) {
    question = question.trim(); if (!question || busy || !app || !app.state.prof) return;
    add(question, 'me'); input.value = '';
    var useAI = AI_URL && (!aiToggle || aiToggle.checked);
    if (!useAI) { add(localAnswer(question), 'bot', 'Built-in answer'); return; }
    setBusy(true);
    var typing = add('<span class="qa-dots"><i></i><i></i><i></i></span>', 'bot');
    var ctrl = window.AbortController ? new AbortController() : null;
    var timer = setTimeout(function () { if (ctrl) ctrl.abort(); }, 20000);
    history.push({ role: 'user', content: question.slice(0, 600) });
    fetch(AI_URL, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'data', context: ctx, messages: history.slice(-8) }), signal: ctrl ? ctrl.signal : undefined
    }).then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (d) { if (!d || typeof d.reply !== 'string' || !d.reply) throw 0; return d.reply; })
      .then(function (text) {
        typing.remove(); add(renderAI(text), 'bot', 'AI answer, based on a summary of your data');
        history.push({ role: 'assistant', content: text.slice(0, 600) });
      }, function () {
        typing.remove(); history.pop();
        add(localAnswer(question), 'bot', 'Built-in answer (the AI is not available right now)');
      })
      .then(function () { clearTimeout(timer); setBusy(false); });
  }

  function suggestions() {
    var ns = nums(), out = [];
    if (!ns.length) return ['How many rows are there?', 'Are there any problems in the data?'];
    var t = ns[ns.length - 1].name.replace(/_/g, ' ');
    out.push('What affects ' + t + ' the most?');
    if (dateCol()) { out.push('How has ' + t + ' changed over time?'); out.push('Forecast ' + t + ' for the next 3 periods'); }
    out.push('When was ' + t + ' at its highest?');
    if (cats().length) out.push('Compare ' + t + ' by ' + cats()[0].name.replace(/_/g, ' '));
    out.push('Are there any problems in the data?');
    return out.slice(0, 5);
  }

  form.addEventListener('submit', function (e) { e.preventDefault(); ask(input.value); });

  document.addEventListener('quantai:loaded', function (e) {
    app = e.detail; history = []; log.innerHTML = ''; log.hidden = true;
    try { ctx = buildContext(); } catch (err) { ctx = ''; }
    chipsEl.innerHTML = '';
    suggestions().forEach(function (s) {
      var b = document.createElement('button'); b.type = 'button'; b.textContent = s;
      b.addEventListener('click', function () { ask(s); });
      chipsEl.appendChild(b);
    });
  });

  window.QuantAIAsk = { localAnswer: function (q) { return localAnswer(q); }, context: function () { return ctx; } };
})();

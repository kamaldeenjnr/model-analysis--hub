/* QuantAI interface — Model Analysis Hub. Uses QuantAI (engine), Papa (CSV) and Chart (charts). */
(function () {
  'use strict';
  var Q = window.QuantAI;
  var SHEETJS = 'https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js';
  var MAX_ROWS = 100000;
  var NAVY = '#0C203B', SIGNAL = '#C4412B', STEEL = '#5A6679', GRID = '#E4E9F0', NAVY_SOFT = 'rgba(12,32,59,.14)';

  var $ = function (s, el) { return (el || document).querySelector(s); };
  var $$ = function (s, el) { return Array.prototype.slice.call((el || document).querySelectorAll(s)); };
  var esc = function (s) { return String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };
  var icon = function (n) { return '<svg class="icon" aria-hidden="true"><use href="#i-' + n + '"/></svg>'; };

  var state = { name: '', headers: [], rows: [], prof: null, charts: {} };

  if (window.Chart) {
    Chart.defaults.font.family = '"Plex Sans", system-ui, sans-serif';
    Chart.defaults.color = STEEL;
    Chart.defaults.borderColor = GRID;
    Chart.defaults.animation = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? false : { duration: 500 };
  }

  /* ---------------------------------------------------------------- input */
  var drop = $('#qa-drop'), fileInput = $('#qa-file'), status = $('#qa-status');

  function say(msg, kind) { status.innerHTML = msg ? '<span class="qa-msg ' + (kind || '') + '">' + msg + '</span>' : ''; }

  fileInput.addEventListener('change', function () { if (fileInput.files[0]) readFile(fileInput.files[0]); });
  ['dragenter', 'dragover'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); }); });
  ['dragleave', 'drop'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); }); });
  drop.addEventListener('drop', function (e) { var f = e.dataTransfer.files[0]; if (f) readFile(f); });

  $$('[data-example]').forEach(function (b) {
    b.addEventListener('click', function () {
      var url = b.getAttribute('data-example');
      say('Loading example…');
      fetch(url).then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
        .then(function (t) { parseText(t, b.getAttribute('data-name')); })
        .catch(function () { say('Could not load the example. Check your connection and try again.', 'bad'); });
    });
  });

  $('#qa-paste-go').addEventListener('click', function () {
    var t = $('#qa-paste-text').value.trim();
    if (!t) { say('Paste some rows first, including the header row.', 'bad'); return; }
    parseText(t, 'Pasted data');
  });

  function readFile(file) {
    var ext = (file.name.split('.').pop() || '').toLowerCase();
    if (file.size > 60 * 1024 * 1024) { say('That file is over 60 MB. Try a smaller extract of it.', 'bad'); return; }
    say('Reading ' + esc(file.name) + '…');
    if (['xlsx', 'xls', 'xlsm', 'ods'].indexOf(ext) > -1) {
      loadSheetJS().then(function () {
        var fr = new FileReader();
        fr.onload = function () {
          try {
            var wb = XLSX.read(fr.result, { type: 'array', cellDates: true });
            var name = wb.SheetNames[0];
            var rows = XLSX.utils.sheet_to_json(wb.Sheets[name], { header: 1, raw: true, defval: '' });
            ingest(rows, file.name + (wb.SheetNames.length > 1 ? ' (sheet "' + name + '")' : ''));
            if (wb.SheetNames.length > 1) say('Loaded the first sheet, "' + esc(name) + '". To analyse another sheet, move it to the front in Excel or save it as CSV.', 'info');
          } catch (e) { say('Could not read that spreadsheet. Try saving it as CSV and uploading again.', 'bad'); }
        };
        fr.readAsArrayBuffer(file);
      }).catch(function () { say('Excel support needs an internet connection the first time. You can also save the file as CSV and upload that.', 'bad'); });
    } else if (['csv', 'tsv', 'txt'].indexOf(ext) > -1 || !ext) {
      Papa.parse(file, { skipEmptyLines: 'greedy', complete: function (r) { ingest(r.data, file.name); }, error: function () { say('Could not read that file.', 'bad'); } });
    } else {
      say('QuantAI reads CSV, TSV and Excel files (.xlsx, .xls). "' + esc(file.name) + '" looks like something else.', 'bad');
    }
  }

  function parseText(text, name) {
    var r = Papa.parse(text, { skipEmptyLines: 'greedy' });
    ingest(r.data, name);
  }

  var sheetPromise = null;
  function loadSheetJS() {
    if (window.XLSX) return Promise.resolve();
    if (sheetPromise) return sheetPromise;
    sheetPromise = new Promise(function (res, rej) {
      var s = document.createElement('script'); s.src = SHEETJS; s.onload = res; s.onerror = function () { sheetPromise = null; rej(); };
      document.head.appendChild(s);
    });
    return sheetPromise;
  }

  function ingest(data, name) {
    data = (data || []).filter(function (r) { return r && r.some(function (c) { return String(c).trim() !== ''; }); });
    if (data.length < 3) { say('QuantAI needs a header row and at least two rows of data.', 'bad'); return; }
    var width = Math.max.apply(null, data.slice(0, 50).map(function (r) { return r.length; }));
    var headers = data[0].slice(0, width).map(function (h, i) { return String(h).trim() || ('Column ' + (i + 1)); });
    while (headers.length < width) headers.push('Column ' + (headers.length + 1));
    var seen = {};
    headers = headers.map(function (h) { if (seen[h]) { seen[h]++; return h + ' (' + seen[h] + ')'; } seen[h] = 1; return h; });
    var rows = data.slice(1).map(function (r) { var o = r.slice(0, width); while (o.length < width) o.push(''); return o; });
    var note = '';
    if (rows.length > MAX_ROWS) { rows = rows.slice(0, MAX_ROWS); note = ' Only the first ' + Q.fmtInt(MAX_ROWS) + ' rows were analysed.'; }
    state.name = name; state.headers = headers; state.rows = rows;
    try {
      state.prof = Q.profile(rows, headers);
    } catch (e) { say('Something in this file confused the analysis. Check that the first row holds column names.', 'bad'); return; }
    say(note ? note : '', note ? 'info' : '');
    render();
  }

  /* ---------------------------------------------------------------- results */
  var results = $('#qa-results');
  $('#qa-reset').addEventListener('click', function () {
    results.hidden = true; $('#qa-start').hidden = false; fileInput.value = ''; say('');
    destroyCharts(); $('#qa-start').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  function destroyCharts() { Object.keys(state.charts).forEach(function (k) { state.charts[k].destroy(); }); state.charts = {}; }
  function chart(id, cfg) {
    if (state.charts[id]) state.charts[id].destroy();
    var el = document.getElementById(id); if (!el || !window.Chart) return;
    state.charts[id] = new Chart(el, cfg);
  }

  function render() {
    destroyCharts();
    var p = state.prof;
    $('#qa-start').hidden = true; results.hidden = false;
    $('#qa-file-name').textContent = state.name;
    $('#qa-file-meta').textContent = Q.fmtInt(p.rows) + ' rows · ' + p.cols.length + ' columns';
    renderOverview(); renderColumns(); renderRelations(); setupRegression(); setupForecast(); renderTable();
    selectTab('overview');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* tabs */
  var tabs = $$('#qa-tabs [role="tab"]');
  tabs.forEach(function (t, i) {
    t.addEventListener('click', function () { selectTab(t.getAttribute('data-tab')); });
    t.addEventListener('keydown', function (e) {
      var k = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!k) return; e.preventDefault();
      var n = tabs[(i + k + tabs.length) % tabs.length]; n.focus(); selectTab(n.getAttribute('data-tab'));
    });
  });
  function selectTab(name) {
    tabs.forEach(function (t) { var on = t.getAttribute('data-tab') === name; t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1; });
    $$('.qa-panel').forEach(function (p) { p.hidden = p.id !== 'tab-' + name; });
    Object.keys(state.charts).forEach(function (k) { state.charts[k].resize(); });
  }

  var numeric = function () { return state.prof.cols.filter(function (c) { return c.type === 'number' && !c.isId; }); };
  var cats = function () { return state.prof.cols.filter(function (c) { return c.type === 'category' && c.unique >= 2 && c.unique <= 12; }); };
  var dates = function () { return state.prof.cols.filter(function (c) { return c.type === 'date'; }); };

  /* overview */
  function renderOverview() {
    var p = state.prof;
    var tiles = [
      ['Rows', Q.fmtInt(p.rows)], ['Columns', p.cols.length],
      ['Numeric columns', numeric().length], ['Empty cells', Q.pct(p.missingPct)]
    ];
    $('#qa-tiles').innerHTML = tiles.map(function (t) { return '<div class="qa-tile"><span>' + t[0] + '</span><b>' + t[1] + '</b></div>'; }).join('');
    var icons = { info: 'info', warn: 'triangle-alert', good: 'check', link: 'chart-scatter', trend: 'trending-up' };
    $('#qa-findings').innerHTML = Q.findings(p).map(function (f) {
      return '<li class="f-' + f.kind + '">' + icon(icons[f.kind] || 'info') + '<span>' + esc(f.text) + '</span></li>';
    }).join('');
    var next = [];
    var nums = numeric();
    if (nums.length >= 2) next.push('<button class="btn btn-line" type="button" data-go="regression">' + icon('sigma') + 'Explain "' + esc(nums[nums.length - 1].name) + '" with the other columns</button>');
    if (dates().length && nums.length) next.push('<button class="btn btn-line" type="button" data-go="forecast">' + icon('trending-up') + 'Forecast "' + esc(nums[nums.length - 1].name) + '"</button>');
    if (nums.length >= 2) next.push('<button class="btn btn-line" type="button" data-go="relations">' + icon('grid-3x3') + 'See how the columns relate</button>');
    $('#qa-next').innerHTML = next.join('');
    $$('#qa-next [data-go]').forEach(function (b) { b.addEventListener('click', function () {
      var go = b.getAttribute('data-go'); selectTab(go);
      if (go === 'regression') runRegression(); if (go === 'forecast') runForecast();
      $('#qa-tabs').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }); });
  }

  /* columns */
  function renderColumns() {
    var html = state.prof.cols.map(function (c, i) {
      var badge = '<span class="qa-type t-' + c.type + '">' + c.type + '</span>';
      var body = '';
      if (c.type === 'number') {
        var s = c.stats;
        body = '<dl class="qa-stats">' + [['Average', s.mean], ['Median', s.median], ['Std. dev.', s.sd], ['Min', s.min], ['Max', s.max], ['Total', s.sum]]
          .map(function (r) { return '<div><dt>' + r[0] + '</dt><dd>' + Q.fmt(r[1]) + '</dd></div>'; }).join('') + '</dl>' +
          '<div class="qa-mini"><canvas id="h-' + i + '" role="img" aria-label="Distribution of ' + esc(c.name) + '"></canvas></div>';
      } else if (c.type === 'date') {
        body = '<p class="muted">From <b>' + Q.fmtDate(c.range[0]) + '</b> to <b>' + Q.fmtDate(c.range[1]) + '</b>' + (c.step ? ', roughly one row per ' + c.step.name : '') + '.</p>';
      } else {
        var total = c.count - c.missing;
        body = '<ul class="qa-bars">' + c.freq.top.map(function (t) {
          var w = total ? t.count / total * 100 : 0;
          return '<li><span class="lbl" title="' + esc(t.value) + '">' + esc(t.value) + '</span><span class="bar"><i style="width:' + w.toFixed(1) + '%"></i></span><span class="val">' + Q.fmtInt(t.count) + '</span></li>';
        }).join('') + '</ul>' + (c.unique > c.freq.top.length ? '<p class="muted small">+ ' + (c.unique - c.freq.top.length) + ' more values</p>' : '');
      }
      return '<article class="qa-col"><header><h3>' + esc(c.name) + '</h3>' + badge + '</header>' +
        '<p class="qa-colmeta">' + Q.fmtInt(c.count - c.missing) + ' values · ' + (c.missing ? Q.fmtInt(c.missing) + ' missing · ' : '') + Q.fmtInt(c.unique) + ' unique</p>' + body + '</article>';
    }).join('');
    $('#qa-columns').innerHTML = html;
    state.prof.cols.forEach(function (c, i) {
      if (c.type !== 'number') return;
      chart('h-' + i, { type: 'bar', data: { labels: c.hist.labels, datasets: [{ data: c.hist.counts, backgroundColor: NAVY, borderRadius: 2, barPercentage: 1, categoryPercentage: .92 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { title: function (it) { return it[0].label; }, label: function (it) { return it.raw + ' rows'; } } } },
          scales: { x: { display: false }, y: { display: false, beginAtZero: true } } } });
    });
  }

  /* relationships */
  function renderRelations() {
    var nums = numeric().slice(0, 14), box = $('#qa-heat');
    if (nums.length < 2) { box.innerHTML = '<p class="muted">You need at least two numeric columns to compare.</p>'; $('#qa-scatter-wrap').hidden = true; return; }
    var m = Q.correlationMatrix(nums);
    var head = '<tr><th scope="col"><span class="sr">Column</span></th>' + nums.map(function (c) { return '<th scope="col"><span>' + esc(c.name) + '</span></th>'; }).join('') + '</tr>';
    var body = nums.map(function (a, i) {
      return '<tr><th scope="row">' + esc(a.name) + '</th>' + nums.map(function (b, j) {
        var r = m[i][j].r, bg = heat(r), fg = Math.abs(r) > 0.55 ? '#fff' : 'var(--ink)';
        return '<td><button type="button" ' + (i === j ? 'disabled' : '') + ' data-a="' + i + '" data-b="' + j + '" style="background:' + bg + ';color:' + fg + '" aria-label="' + esc(a.name) + ' and ' + esc(b.name) + ': r = ' + Q.fmt(r, 2) + '">' + Q.fmt(r, 2) + '</button></td>';
      }).join('') + '</tr>';
    }).join('');
    box.innerHTML = '<div class="qa-heat-scroll"><table class="qa-heat"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>' +
      '<p class="qa-legend"><span class="neg"></span>Move in opposite directions <span class="zero"></span>No link <span class="pos"></span>Move together</p>';
    $$('button[data-a]', box).forEach(function (b) { b.addEventListener('click', function () { scatter(nums[+b.dataset.a], nums[+b.dataset.b]); }); });
    // default: strongest pair
    var best = null;
    for (var i = 0; i < nums.length; i++) for (var j = i + 1; j < nums.length; j++) if (!best || Math.abs(m[i][j].r) > Math.abs(best.r)) best = { i: i, j: j, r: m[i][j].r };
    $('#qa-scatter-wrap').hidden = false;
    scatter(nums[best.i], nums[best.j]);
  }
  function heat(r) {
    if (!isFinite(r)) return '#fff';
    var a = Math.min(1, Math.abs(r));
    return r >= 0 ? 'rgba(12,32,59,' + (0.08 + a * 0.85).toFixed(3) + ')' : 'rgba(196,65,43,' + (0.08 + a * 0.85).toFixed(3) + ')';
  }
  function scatter(x, y) {
    var pts = [];
    for (var i = 0; i < x.values.length; i++) if (!isNaN(x.values[i]) && !isNaN(y.values[i])) pts.push({ x: x.values[i], y: y.values[i] });
    if (pts.length > 3000) pts = pts.filter(function (_, i) { return i % Math.ceil(pts.length / 3000) === 0; });
    var c = Q.pearson(x.values, y.values), fit = Q.ols(y.values, [x.values], [x.name]);
    var xs = pts.map(function (p) { return p.x; }), minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var line = fit.coefs ? [{ x: minX, y: fit.coefs[0].estimate + fit.coefs[1].estimate * minX }, { x: maxX, y: fit.coefs[0].estimate + fit.coefs[1].estimate * maxX }] : [];
    $('#qa-scatter-title').textContent = y.name + ' vs ' + x.name;
    $('#qa-scatter-note').innerHTML = isFinite(c.r) ? ('<b>' + Q.strength(c.r) + ' ' + (c.r >= 0 ? 'positive' : 'negative') + ' link</b> (r = ' + Q.fmt(c.r, 2) + ', p ' + (c.p < 0.001 ? '< 0.001' : '= ' + Q.fmt(c.p, 3)) + ', n = ' + c.n + '). ' +
      (fit.coefs ? 'On average, each extra 1 of "' + esc(x.name) + '" goes with ' + Q.fmt(Math.abs(fit.coefs[1].estimate)) + (fit.coefs[1].estimate >= 0 ? ' more' : ' less') + ' "' + esc(y.name) + '". ' : '') +
      'This shows association, not cause.') : 'Not enough paired values.';
    chart('qa-scatter', { type: 'scatter', data: { datasets: [
      { label: 'Rows', data: pts, backgroundColor: 'rgba(12,32,59,.55)', pointRadius: pts.length > 500 ? 2 : 3.5 },
      { label: 'Trend line', type: 'line', data: line, borderColor: SIGNAL, borderWidth: 2, pointRadius: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: x.name } }, y: { title: { display: true, text: y.name } } } } });
  }

  /* regression */
  function setupRegression() {
    var nums = numeric(), cs = cats(), sel = $('#qa-reg-y'), box = $('#qa-reg-x');
    if (nums.length < 2 && !(nums.length && cs.length)) {
      $('#qa-reg-form').hidden = true; $('#qa-reg-out').innerHTML = '<p class="muted">Regression needs a numeric column to explain and at least one other column to explain it with.</p>'; return;
    }
    $('#qa-reg-form').hidden = false; $('#qa-reg-out').innerHTML = '';
    sel.innerHTML = nums.map(function (c) { return '<option value="' + c.index + '">' + esc(c.name) + '</option>'; }).join('');
    sel.value = nums[nums.length - 1].index;
    function fillX() {
      var y = +sel.value;
      box.innerHTML = nums.concat(cs).filter(function (c) { return c.index !== y; }).map(function (c) {
        return '<label class="qa-check"><input type="checkbox" value="' + c.index + '" checked> ' + esc(c.name) + (c.type === 'category' ? ' <small>(category)</small>' : '') + '</label>';
      }).join('');
    }
    sel.onchange = fillX; fillX();
  }
  $('#qa-reg-form').addEventListener('submit', function (e) { e.preventDefault(); runRegression(); });

  function runRegression() {
    var p = state.prof, y = p.cols[+$('#qa-reg-y').value];
    var chosen = $$('#qa-reg-x input:checked').map(function (i) { return p.cols[+i.value]; });
    var out = $('#qa-reg-out');
    if (!chosen.length) { out.innerHTML = '<p class="qa-msg bad">Tick at least one column to explain "' + esc(y.name) + '" with.</p>'; return; }
    var X = [], names = [], parts = [];
    chosen.forEach(function (c) {
      if (c.type === 'number') { X.push(c.values); names.push(c.name); parts.push({ col: c, name: c.name }); }
      else {
        var levels = c.freq.top.map(function (t) { return t.value; }), base = levels[0];
        levels.slice(1).forEach(function (lv) {
          X.push(c.values.map(function (v) { return v === null ? NaN : (v === lv ? 1 : 0); }));
          names.push(c.name + ' = ' + lv); parts.push({ col: c, name: c.name + ' = ' + lv, level: lv, base: base });
        });
      }
    });
    var m = Q.ols(y.values, X, names);
    if (m.error) { out.innerHTML = '<p class="qa-msg bad">' + esc(m.error) + '</p>'; return; }
    var sig = m.coefs.slice(1).filter(function (c) { return c.p < 0.05; });
    var sentences = m.coefs.slice(1).map(function (c, i) {
      var part = parts[i], dir = c.estimate >= 0 ? 'higher' : 'lower', amt = Q.fmt(Math.abs(c.estimate));
      var txt = part.level !== undefined
        ? 'When "' + esc(part.col.name) + '" is "' + esc(part.level) + '" (compared with "' + esc(part.base) + '"), "' + esc(y.name) + '" is ' + amt + ' ' + dir + ' on average'
        : 'Each extra 1 of "' + esc(c.name) + '" goes with "' + esc(y.name) + '" being ' + amt + ' ' + dir;
      return '<li class="' + (c.p < 0.05 ? 'sig' : 'ns') + '">' + txt + ', holding the others fixed. ' +
        (c.p < 0.05 ? '<b>Statistically significant</b> (p ' + (c.p < 0.001 ? '< 0.001' : '= ' + Q.fmt(c.p, 3)) + ').' : 'Not significant (p = ' + Q.fmt(c.p, 2) + '), so this could be noise.') + '</li>';
    }).join('');
    var table = '<div class="qa-table-scroll"><table class="qa-table"><thead><tr><th scope="col">Term</th><th scope="col">Estimate</th><th scope="col">Std. error</th><th scope="col">t</th><th scope="col">p-value</th></tr></thead><tbody>' +
      m.coefs.map(function (c) { return '<tr class="' + (c.p < 0.05 ? 'sig' : '') + '"><th scope="row">' + esc(c.name) + '</th><td>' + Q.fmt(c.estimate, 4) + '</td><td>' + Q.fmt(c.se, 4) + '</td><td>' + Q.fmt(c.t, 2) + '</td><td>' + Q.fmtP(c.p) + '</td></tr>'; }).join('') +
      '</tbody></table></div>';
    var predictInputs = parts.filter(function (pt) { return pt.level === undefined; }).map(function (pt) {
      return '<label class="qa-field"><span>' + esc(pt.name) + '</span><input type="number" step="any" data-name="' + esc(pt.name) + '" value="' + Q.fmt(pt.col.stats.median, 2).replace(/,/g, '') + '"></label>';
    }).join('');
    var catInputs = chosen.filter(function (c) { return c.type === 'category'; }).map(function (c) {
      return '<label class="qa-field"><span>' + esc(c.name) + '</span><select data-cat="' + esc(c.name) + '">' + c.freq.top.map(function (t) { return '<option>' + esc(t.value) + '</option>'; }).join('') + '</select></label>';
    }).join('');
    out.innerHTML =
      '<div class="qa-summary"><div class="qa-tile"><span>R²</span><b>' + Q.fmt(m.r2, 3) + '</b></div><div class="qa-tile"><span>Adjusted R²</span><b>' + Q.fmt(m.adjR2, 3) + '</b></div><div class="qa-tile"><span>Rows used</span><b>' + Q.fmtInt(m.n) + '</b></div><div class="qa-tile"><span>Typical error</span><b>±' + Q.fmt(m.rmse) + '</b></div></div>' +
      '<p class="qa-lead">These columns explain <b>' + Q.pct(Math.max(0, m.r2)) + '</b> of the variation in "' + esc(y.name) + '". ' +
      (m.fp < 0.05 ? 'The model as a whole is statistically significant (F = ' + Q.fmt(m.f, 1) + ', p ' + (m.fp < 0.001 ? '< 0.001' : '= ' + Q.fmt(m.fp, 3)) + ').' : 'The model as a whole is not statistically significant, so treat it with caution.') +
      (sig.length ? '' : ' None of the individual columns are significant on their own.') + '</p>' +
      '<ul class="qa-sentences">' + sentences + '</ul>' + table +
      '<div class="qa-two"><div><h4>Predicted vs actual</h4><div class="qa-chart"><canvas id="qa-avp" role="img" aria-label="Predicted against actual values"></canvas></div></div>' +
      '<div><h4>Try a prediction</h4><form id="qa-predict" class="qa-predict">' + predictInputs + catInputs + '<button class="btn btn-navy" type="submit">' + icon('sparkles') + 'Predict</button><p id="qa-predict-out" class="qa-predict-out" aria-live="polite"></p></form></div></div>' +
      '<p class="muted small">Method: ordinary least squares with an intercept. Rows with a gap in any chosen column are left out. Categories are compared against their most common value.</p>';
    var mn = Math.min.apply(null, m.y.concat(m.fitted)), mx = Math.max.apply(null, m.y.concat(m.fitted));
    chart('qa-avp', { type: 'scatter', data: { datasets: [
      { label: 'Rows', data: m.fitted.map(function (f, i) { return { x: f, y: m.y[i] }; }), backgroundColor: 'rgba(12,32,59,.55)', pointRadius: 3 },
      { label: 'Perfect prediction', type: 'line', data: [{ x: mn, y: mn }, { x: mx, y: mx }], borderColor: SIGNAL, borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Predicted' } }, y: { title: { display: true, text: 'Actual ' + y.name } } } } });
    $('#qa-predict').addEventListener('submit', function (e) {
      e.preventDefault();
      var val = m.coefs[0].estimate;
      m.coefs.slice(1).forEach(function (c, i) {
        var pt = parts[i];
        if (pt.level === undefined) { var inp = $('#qa-predict input[data-name="' + CSS.escape(pt.name) + '"]'); val += c.estimate * (parseFloat(inp.value) || 0); }
        else { var s = $('#qa-predict select[data-cat="' + CSS.escape(pt.col.name) + '"]'); if (s && s.value === pt.level) val += c.estimate; }
      });
      $('#qa-predict-out').innerHTML = 'Predicted "' + esc(y.name) + '": <b>' + Q.fmt(val) + '</b> <span class="muted">(typically within ±' + Q.fmt(1.96 * m.sigma) + ')</span>';
    });
  }

  /* forecast */
  function setupForecast() {
    var nums = numeric(), ds = dates(), f = $('#qa-fc-form');
    if (!nums.length) { f.hidden = true; $('#qa-fc-out').innerHTML = '<p class="muted">Forecasting needs a numeric column.</p>'; return; }
    f.hidden = false; $('#qa-fc-out').innerHTML = '';
    $('#qa-fc-time').innerHTML = ds.map(function (c) { return '<option value="' + c.index + '">' + esc(c.name) + '</option>'; }).join('') + '<option value="-1">Row order (no date column)</option>';
    $('#qa-fc-y').innerHTML = nums.map(function (c) { return '<option value="' + c.index + '">' + esc(c.name) + '</option>'; }).join('');
    $('#qa-fc-y').value = nums[nums.length - 1].index;
    var step = ds.length && ds[0].step ? ds[0].step : null;
    $('#qa-fc-h').value = step && step.period > 1 ? Math.min(12, step.period) : 6;
  }
  $('#qa-fc-form').addEventListener('submit', function (e) { e.preventDefault(); runForecast(); });

  function runForecast() {
    var p = state.prof, ti = +$('#qa-fc-time').value, dc = ti >= 0 ? p.cols[ti] : null, yc = p.cols[+$('#qa-fc-y').value];
    var h = Math.max(1, Math.min(60, parseInt($('#qa-fc-h').value, 10) || 6));
    var ts = Q.series(p, dc, yc), out = $('#qa-fc-out');
    if (ts.y.length < 6) { out.innerHTML = '<p class="qa-msg bad">A forecast needs at least 6 points in time. This column has ' + ts.y.length + '.</p>'; return; }
    var step = dc ? dc.step : null, period = step ? step.period : 1;
    var hw = Q.holtWinters(ts.y, period, h);
    if (!hw) { out.innerHTML = '<p class="qa-msg bad">Could not fit a forecast to this series.</p>'; return; }
    var labels = ts.x.map(function (x) { return dc ? Q.fmtDate(x, step) : String(x + 1); });
    var fLabels = [];
    for (var i = 1; i <= h; i++) fLabels.push(dc ? Q.fmtDate(Q.addSteps(ts.x[ts.x.length - 1], step, i), step) : String(ts.y.length + i));
    var all = labels.concat(fLabels), n = ts.y.length;
    var pad = function (arr, before) { return new Array(before).fill(null).concat(arr); };
    var last = ts.y[n - 1], end = hw.forecast[h - 1], unit = step ? step.name : 'step';
    var change = last ? (end - last) / Math.abs(last) : NaN;
    out.innerHTML =
      '<p class="qa-lead">Over the next <b>' + h + ' ' + unit + (h > 1 ? 's' : '') + '</b>, "' + esc(yc.name) + '" is expected to ' +
      (Math.abs(change) < 0.03 ? 'stay about level' : (change > 0 ? 'rise' : 'fall')) + ', from <b>' + Q.fmt(last) + '</b> now to around <b>' + Q.fmt(end) + '</b> (likely range ' + Q.fmt(hw.lower[h - 1]) + ' to ' + Q.fmt(hw.upper[h - 1]) + '). ' +
      (hw.seasonal ? 'QuantAI found a repeating ' + unit + 'ly pattern and has built it into the forecast. ' : '') +
      'On past data, the model was typically ' + (isFinite(hw.mape) ? 'within ' + Q.pct(hw.mape) + ' of the real value.' : 'close to the real value.') + '</p>' +
      '<div class="qa-chart tall"><canvas id="qa-fc" role="img" aria-label="Forecast chart for ' + esc(yc.name) + '"></canvas></div>' +
      '<div class="qa-table-scroll"><table class="qa-table"><thead><tr><th scope="col">' + (dc ? 'Period' : 'Row') + '</th><th scope="col">Forecast</th><th scope="col">Low (80%)</th><th scope="col">High (80%)</th></tr></thead><tbody>' +
      hw.forecast.map(function (f, i) { return '<tr><th scope="row">' + esc(fLabels[i]) + '</th><td>' + Q.fmt(f) + '</td><td>' + Q.fmt(hw.lower[i]) + '</td><td>' + Q.fmt(hw.upper[i]) + '</td></tr>'; }).join('') +
      '</tbody></table></div>' +
      '<p class="muted small">Method: ' + (hw.seasonal ? 'Holt-Winters exponential smoothing with a ' + hw.period + '-' + unit + ' season' : 'Holt\'s linear trend (exponential smoothing)') +
      ', parameters chosen by best fit (α = ' + hw.alpha + ', β = ' + hw.beta + (hw.seasonal ? ', γ = ' + hw.gamma : '') + '). Rows sharing a date are added together. The shaded band is an approximate 80% range; the further ahead, the less certain.</p>';
    chart('qa-fc', { type: 'line', data: { labels: all, datasets: [
      { label: 'Actual', data: ts.y, borderColor: NAVY, backgroundColor: NAVY, borderWidth: 2, pointRadius: n > 60 ? 0 : 2, tension: .2 },
      { label: 'Model fit', data: hw.fitted, borderColor: 'rgba(90,102,121,.55)', borderDash: [4, 4], borderWidth: 1.2, pointRadius: 0, tension: .2 },
      { label: 'Low', data: pad(hw.lower, n), borderColor: 'transparent', pointRadius: 0, fill: false },
      { label: 'Likely range', data: pad(hw.upper, n), borderColor: 'transparent', backgroundColor: 'rgba(196,65,43,.14)', pointRadius: 0, fill: '-1' },
      { label: 'Forecast', data: pad([last].concat(hw.forecast), n - 1), borderColor: SIGNAL, backgroundColor: SIGNAL, borderWidth: 2.5, pointRadius: 2.5, tension: .2 }] },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { filter: function (it) { return it.text !== 'Low'; }, boxWidth: 14 } } },
        scales: { x: { ticks: { maxTicksLimit: 10, autoSkip: true } }, y: { title: { display: true, text: yc.name } } } } });
  }

  /* data table */
  function renderTable() {
    var p = state.prof, show = state.rows.slice(0, 200);
    $('#qa-data-note').textContent = state.rows.length > 200 ? 'Showing the first 200 of ' + Q.fmtInt(state.rows.length) + ' rows.' : 'Showing all ' + state.rows.length + ' rows.';
    $('#qa-data').innerHTML = '<table class="qa-table data"><thead><tr>' + p.cols.map(function (c) { return '<th scope="col">' + esc(c.name) + '<span class="qa-type t-' + c.type + '">' + c.type + '</span></th>'; }).join('') + '</tr></thead><tbody>' +
      show.map(function (r) { return '<tr>' + r.map(function (v, i) { var s = v instanceof Date ? Q.fmtDate(v) : v; return '<td class="' + (p.cols[i].type === 'number' ? 'num' : '') + '">' + esc(s) + '</td>'; }).join('') + '</tr>'; }).join('') +
      '</tbody></table>';
  }

  /* export */
  $('#qa-download').addEventListener('click', function () {
    var p = state.prof, lines = [['column', 'type', 'values', 'missing', 'unique', 'mean', 'median', 'sd', 'min', 'max']];
    p.cols.forEach(function (c) {
      var s = c.stats || {};
      lines.push([c.name, c.type, c.count - c.missing, c.missing, c.unique, s.mean, s.median, s.sd, s.min, s.max].map(function (v) { return v === undefined || (typeof v === 'number' && !isFinite(v)) ? '' : v; }));
    });
    lines.push([]); lines.push(['Findings']);
    Q.findings(p).forEach(function (f) { lines.push([f.text]); });
    var csv = lines.map(function (r) { return r.map(function (v) { v = String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }).join(','); }).join('\n');
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'quantai-summary-' + (state.name || 'data').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '-') + '.csv';
    document.body.appendChild(a); a.click(); a.remove();
  });
  $('#qa-print').addEventListener('click', function () { window.print(); });
})();

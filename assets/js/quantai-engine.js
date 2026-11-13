/*
 * QuantAI engine — Model Analysis Hub (Amadu Kamal)
 * Pure statistics, no network. Everything here runs in the visitor's browser.
 *   - column type detection (number, date, category, text)
 *   - descriptive statistics, histograms, frequency tables
 *   - Pearson correlation matrix
 *   - ordinary least squares regression with standard errors and p-values
 *   - Holt / Holt-Winters exponential smoothing forecasts with prediction intervals
 *   - plain-language findings
 */
(function (root) {
  'use strict';

  /* ------------------------------------------------------------------ parsing helpers */
  var NUM_RE = /^[\s$€£₵¢]*[-+(]?[\d.,\s]+%?\)?[\s$€£₵¢]*$/;
  var MISSING = { '': 1, 'na': 1, 'n/a': 1, 'nan': 1, 'null': 1, 'none': 1, '-': 1, '--': 1, '?': 1, '#n/a': 1 };

  function isMissing(v) {
    return v === null || v === undefined || (typeof v === 'number' && isNaN(v)) || MISSING[String(v).trim().toLowerCase()] === 1;
  }

  function toNumber(v) {
    if (typeof v === 'number') return isFinite(v) ? v : NaN;
    var s = String(v).trim().replace(/^(GH₵|GH¢|GHS|GHC|USD|US\$|EUR|GBP|NGN|KES|ZAR|CFA)\s*/i, '').replace(/\s*(GH₵|GHS|USD|EUR|GBP|NGN|KES|ZAR|CFA)$/i, '');
    if (!s || !NUM_RE.test(s)) return NaN;
    var neg = /^\(.*\)$/.test(s) || /^-/.test(s.replace(/[\s$€£₵¢]/g, ''));
    s = s.replace(/[\s$€£₵¢()%+-]/g, '');
    // 1,234.56 or 1.234,56 or 1234,56
    var lastComma = s.lastIndexOf(','), lastDot = s.lastIndexOf('.');
    if (lastComma > -1 && lastDot > -1) {
      if (lastComma > lastDot) s = s.replace(/\./g, '').replace(',', '.');
      else s = s.replace(/,/g, '');
    } else if (lastComma > -1) {
      var parts = s.split(',');
      s = (parts.length === 2 && parts[1].length !== 3) ? s.replace(',', '.') : s.replace(/,/g, '');
    }
    if (!/^\d*\.?\d+$|^\d+\.$/.test(s)) return NaN;
    var n = parseFloat(s);
    return neg ? -n : n;
  }

  var MONTHS = { jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5, jul: 6, aug: 7, sep: 8, sept: 8, oct: 9, nov: 10, dec: 11 };
  function toDate(v) {
    if (v instanceof Date) return isNaN(v) ? null : v;
    var s = String(v).trim();
    if (!s || s.length < 4 || s.length > 30) return null;
    var m;
    if ((m = s.match(/^(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?(?:[ T](\d{1,2}):(\d{2}))?/))) {
      var d = new Date(Date.UTC(+m[1], +m[2] - 1, m[3] ? +m[3] : 1));
      return (+m[2] >= 1 && +m[2] <= 12) ? d : null;
    }
    if ((m = s.match(/^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$/))) {
      var y = +m[3] < 100 ? 2000 + +m[3] : +m[3], a = +m[1], b = +m[2];
      var day = a > 12 ? a : (b > 12 ? b : a), mon = a > 12 ? b : (b > 12 ? a : b); // prefer day-first (Ghana/UK)
      if (mon < 1 || mon > 12 || day < 1 || day > 31) return null;
      return new Date(Date.UTC(y, mon - 1, day));
    }
    if ((m = s.match(/^([A-Za-z]{3,9})[\s-]+(\d{4})$/)) && MONTHS[m[1].slice(0, 3).toLowerCase()] !== undefined) {
      return new Date(Date.UTC(+m[2], MONTHS[m[1].slice(0, 3).toLowerCase()], 1));
    }
    if ((m = s.match(/^(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})$/)) && MONTHS[m[2].slice(0, 3).toLowerCase()] !== undefined) {
      return new Date(Date.UTC(+m[3], MONTHS[m[2].slice(0, 3).toLowerCase()], +m[1]));
    }
    return null;
  }

  /* ------------------------------------------------------------------ basic statistics */
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  function mean(a) { return a.length ? sum(a) / a.length : NaN; }
  function variance(a) {
    if (a.length < 2) return NaN;
    var m = mean(a), s = 0;
    for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return s / (a.length - 1);
  }
  function sd(a) { return Math.sqrt(variance(a)); }
  function sorted(a) { return a.slice().sort(function (x, y) { return x - y; }); }
  function quantile(sortedArr, q) {
    if (!sortedArr.length) return NaN;
    var pos = (sortedArr.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (pos - lo);
  }
  function skewness(a) {
    var n = a.length; if (n < 3) return NaN;
    var m = mean(a), s = sd(a); if (!s) return 0;
    var t = 0; for (var i = 0; i < n; i++) t += Math.pow((a[i] - m) / s, 3);
    return n / ((n - 1) * (n - 2)) * t;
  }

  function describe(values) {
    var s = sorted(values), q1 = quantile(s, 0.25), q3 = quantile(s, 0.75), iqr = q3 - q1;
    var lo = q1 - 1.5 * iqr, hi = q3 + 1.5 * iqr, out = 0;
    for (var i = 0; i < s.length; i++) if (s[i] < lo || s[i] > hi) out++;
    return {
      n: s.length, mean: mean(s), sd: sd(s), min: s[0], q1: q1, median: quantile(s, 0.5), q3: q3, max: s[s.length - 1],
      sum: sum(s), skew: skewness(s), outliers: out, cv: mean(s) ? sd(s) / Math.abs(mean(s)) : NaN
    };
  }

  function histogram(values, bins) {
    if (!values.length) return { edges: [], counts: [] };
    var s = sorted(values), min = s[0], max = s[s.length - 1];
    if (!bins) bins = Math.max(5, Math.min(24, Math.ceil(Math.log2(s.length) + 1)));
    if (min === max) return { edges: [min, max], counts: [s.length], labels: [fmt(min)] };
    var w = (max - min) / bins, counts = new Array(bins).fill(0), edges = [];
    for (var i = 0; i <= bins; i++) edges.push(min + i * w);
    for (var j = 0; j < s.length; j++) counts[Math.min(bins - 1, Math.floor((s[j] - min) / w))]++;
    var labels = counts.map(function (_, k) { return fmt(edges[k]) + '–' + fmt(edges[k + 1]); });
    return { edges: edges, counts: counts, labels: labels };
  }

  function frequencies(values, top) {
    var map = new Map();
    values.forEach(function (v) { v = String(v).trim(); map.set(v, (map.get(v) || 0) + 1); });
    var arr = Array.from(map, function (e) { return { value: e[0], count: e[1] }; }).sort(function (a, b) { return b.count - a.count; });
    return { unique: arr.length, top: arr.slice(0, top || 8) };
  }

  /* ------------------------------------------------------------------ dataset profiling */
  function profile(rows, headers) {
    var cols = headers.map(function (name, ci) {
      var raw = rows.map(function (r) { return r[ci]; });
      var present = raw.filter(function (v) { return !isMissing(v); });
      var nums = [], dates = [];
      present.forEach(function (v) {
        var n = toNumber(v); if (!isNaN(n)) nums.push(n);
        var d = (typeof v === 'number') ? null : toDate(v); if (d) dates.push(d);
      });
      var type = 'text';
      var p = present.length || 1;
      var uniq = new Set(present.map(function (v) { return String(v).trim().toLowerCase(); })).size;
      if (dates.length / p >= 0.9 && !/^\d+(\.\d+)?$/.test(String(present[0]).trim())) type = 'date';
      else if (nums.length / p >= 0.9) type = 'number';
      else if (uniq <= Math.max(12, present.length * 0.5) || uniq <= 30) type = 'category';
      var col = { name: String(name || ('Column ' + (ci + 1))), index: ci, type: type, count: raw.length, missing: raw.length - present.length, unique: uniq };
      if (type === 'number') {
        col.values = raw.map(function (v) { return isMissing(v) ? NaN : toNumber(v); });
        var clean = col.values.filter(function (x) { return !isNaN(x); });
        col.stats = describe(clean);
        col.hist = histogram(clean);
        col.isId = uniq === present.length && present.length > 10 && clean.every(function (x, i, a) { return i === 0 || x === a[i - 1] + 1; });
      } else if (type === 'date') {
        col.values = raw.map(function (v) { return isMissing(v) ? null : toDate(v); });
        var ds = col.values.filter(Boolean).map(function (d) { return +d; }).sort(function (a, b) { return a - b; });
        col.range = [new Date(ds[0]), new Date(ds[ds.length - 1])];
        col.step = inferStep(ds);
      } else {
        col.values = raw.map(function (v) { return isMissing(v) ? null : String(v).trim(); });
        col.freq = frequencies(present);
      }
      return col;
    });
    var totalCells = rows.length * headers.length, missing = sum(cols.map(function (c) { return c.missing; }));
    return { rows: rows.length, cols: cols, missingCells: missing, missingPct: totalCells ? missing / totalCells : 0,
             duplicates: countDuplicates(rows) };
  }

  function countDuplicates(rows) {
    var seen = new Set(), d = 0;
    rows.forEach(function (r) { var k = JSON.stringify(r); if (seen.has(k)) d++; else seen.add(k); });
    return d;
  }

  function inferStep(ms) {
    if (ms.length < 3) return null;
    var diffs = [];
    for (var i = 1; i < ms.length; i++) diffs.push((ms[i] - ms[i - 1]) / 86400000);
    var med = quantile(sorted(diffs), 0.5);
    if (med >= 27 && med <= 32) return { name: 'month', period: 12 };
    if (med >= 6 && med <= 8) return { name: 'week', period: 52 };
    if (med >= 0.9 && med <= 1.1) return { name: 'day', period: 7 };
    if (med >= 88 && med <= 93) return { name: 'quarter', period: 4 };
    if (med >= 360 && med <= 370) return { name: 'year', period: 1 };
    return { name: 'step', period: 1 };
  }

  /* ------------------------------------------------------------------ correlation */
  function pairwise(x, y) {
    var a = [], b = [];
    for (var i = 0; i < x.length; i++) if (!isNaN(x[i]) && !isNaN(y[i])) { a.push(x[i]); b.push(y[i]); }
    return [a, b];
  }
  function pearson(x, y) {
    var p = pairwise(x, y), a = p[0], b = p[1], n = a.length;
    if (n < 3) return { r: NaN, n: n, p: NaN };
    var ma = mean(a), mb = mean(b), sxy = 0, sxx = 0, syy = 0;
    for (var i = 0; i < n; i++) { var dx = a[i] - ma, dy = b[i] - mb; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
    var r = (sxx && syy) ? sxy / Math.sqrt(sxx * syy) : NaN;
    var t = r * Math.sqrt((n - 2) / Math.max(1e-12, 1 - r * r));
    return { r: r, n: n, p: tPValue(t, n - 2) };
  }
  function correlationMatrix(cols) {
    return cols.map(function (a) { return cols.map(function (b) { return a === b ? { r: 1, n: a.stats.n, p: 0 } : pearson(a.values, b.values); }); });
  }

  /* ------------------------------------------------------------------ distributions (t, F) */
  function logGamma(z) {
    var g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    var x = z, y = z, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp);
    var ser = 1.000000000190015; for (var j = 0; j < 6; j++) ser += g[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  function betacf(a, b, x) {
    var MAXIT = 200, EPS = 3e-14, FPMIN = 1e-300, qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; var h = d;
    for (var m = 1; m <= MAXIT; m++) {
      var m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d;
      var del = d * c; h *= del; if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function incBeta(x, a, b) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    var bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  function tPValue(t, df) { // two-sided
    if (!isFinite(t) || df <= 0) return NaN;
    return incBeta(df / (df + t * t), df / 2, 0.5);
  }
  function fPValue(f, d1, d2) {
    if (!isFinite(f) || f < 0) return NaN;
    return incBeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
  }

  /* ------------------------------------------------------------------ linear algebra + OLS */
  function invert(M) {
    var n = M.length, A = M.map(function (r, i) { return r.concat(Array.from({ length: n }, function (_, j) { return i === j ? 1 : 0; })); });
    for (var c = 0; c < n; c++) {
      var piv = c;
      for (var r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      if (Math.abs(A[piv][c]) < 1e-12) return null;
      var tmp = A[c]; A[c] = A[piv]; A[piv] = tmp;
      var pv = A[c][c];
      for (var k = 0; k < 2 * n; k++) A[c][k] /= pv;
      for (var r2 = 0; r2 < n; r2++) if (r2 !== c) {
        var f = A[r2][c]; if (!f) continue;
        for (var k2 = 0; k2 < 2 * n; k2++) A[r2][k2] -= f * A[c][k2];
      }
    }
    return A.map(function (r) { return r.slice(n); });
  }

  function ols(y, Xcols, names) {
    var rows = [];
    for (var i = 0; i < y.length; i++) {
      if (isNaN(y[i])) continue;
      var r = [1], ok = true;
      for (var j = 0; j < Xcols.length; j++) { var v = Xcols[j][i]; if (isNaN(v)) { ok = false; break; } r.push(v); }
      if (ok) rows.push({ x: r, y: y[i] });
    }
    var n = rows.length, k = Xcols.length + 1;
    if (n <= k + 1) return { error: 'Not enough complete rows for this model. You need more rows than predictors.' };
    var XtX = Array.from({ length: k }, function () { return new Array(k).fill(0); }), Xty = new Array(k).fill(0);
    rows.forEach(function (r) {
      for (var a = 0; a < k; a++) { Xty[a] += r.x[a] * r.y; for (var b = 0; b < k; b++) XtX[a][b] += r.x[a] * r.x[b]; }
    });
    var inv = invert(XtX);
    if (!inv) return { error: 'These predictors are too closely related to each other (perfect multicollinearity). Remove one and try again.' };
    var beta = inv.map(function (row) { return sum(row.map(function (v, j) { return v * Xty[j]; })); });
    var ys = rows.map(function (r) { return r.y; }), my = mean(ys), sse = 0, sst = 0;
    var fitted = [], resid = [];
    rows.forEach(function (r) {
      var f = sum(r.x.map(function (v, j) { return v * beta[j]; }));
      fitted.push(f); resid.push(r.y - f);
      sse += (r.y - f) * (r.y - f); sst += (r.y - my) * (r.y - my);
    });
    var df = n - k, s2 = sse / df, r2 = sst ? 1 - sse / sst : NaN, adj = 1 - (1 - r2) * (n - 1) / df;
    var coefs = beta.map(function (b, j) {
      var se = Math.sqrt(Math.max(0, s2 * inv[j][j])), t = se ? b / se : NaN;
      return { name: j === 0 ? '(Intercept)' : names[j - 1], estimate: b, se: se, t: t, p: tPValue(t, df) };
    });
    var fstat = (k > 1 && sse) ? ((sst - sse) / (k - 1)) / s2 : NaN;
    return { n: n, k: k, df: df, coefs: coefs, r2: r2, adjR2: adj, rmse: Math.sqrt(sse / n), sigma: Math.sqrt(s2),
             f: fstat, fp: fPValue(fstat, k - 1, df), fitted: fitted, resid: resid, y: ys };
  }

  /* ------------------------------------------------------------------ forecasting */
  function holtWinters(series, period, h) {
    var n = series.length;
    var seasonal = period > 1 && n >= 2 * period + 2;
    var best = null, grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    var gGrid = seasonal ? [0.05, 0.1, 0.2, 0.3, 0.5] : [0];
    var bGrid = [0.01, 0.05, 0.1, 0.2, 0.3];
    grid.forEach(function (a) { bGrid.forEach(function (b) { gGrid.forEach(function (g) {
      var r = runHW(series, period, a, b, g, seasonal);
      if (r && (!best || r.sse < best.sse)) best = { a: a, b: b, g: g, sse: r.sse, state: r };
    }); }); });
    if (!best) return null;
    var st = best.state, fc = [], lo = [], hi = [];
    var resSd = Math.sqrt(best.sse / Math.max(1, st.nfit - (seasonal ? 3 : 2)));
    for (var i = 1; i <= h; i++) {
      var s = seasonal ? st.season[(n + i - 1) % period] : 0;
      var f = st.level + i * st.trend + s;
      var width = 1.2816 * resSd * Math.sqrt(1 + (i - 1) * best.a * best.a * (1 + i * best.b) * 0.5); // ~80% band
      fc.push(f); lo.push(f - width); hi.push(f + width);
    }
    return { forecast: fc, lower: lo, upper: hi, fitted: st.fitted, seasonal: seasonal, period: period,
             alpha: best.a, beta: best.b, gamma: best.g, rmse: resSd, trendPerStep: st.trend,
             mape: mape(series, st.fitted) };
  }
  function runHW(y, p, a, b, g, seasonal) {
    var n = y.length, level, trend, season = [], fitted = new Array(n).fill(NaN), sse = 0, nfit = 0, start;
    if (seasonal) {
      var m1 = mean(y.slice(0, p)), m2 = mean(y.slice(p, 2 * p));
      level = m1; trend = (m2 - m1) / p;
      for (var i = 0; i < p; i++) season.push(y[i] - m1);
      start = p;
    } else {
      level = y[0]; trend = y.length > 1 ? y[1] - y[0] : 0; start = 1;
    }
    for (var t = start; t < n; t++) {
      var s = seasonal ? season[t % p] : 0, f = level + trend + s;
      fitted[t] = f; sse += (y[t] - f) * (y[t] - f); nfit++;
      var prevLevel = level;
      level = a * (y[t] - s) + (1 - a) * (level + trend);
      trend = b * (level - prevLevel) + (1 - b) * trend;
      if (seasonal) season[t % p] = g * (y[t] - level) + (1 - g) * s;
    }
    return { level: level, trend: trend, season: season, fitted: fitted, sse: sse, nfit: nfit };
  }
  function mape(y, f) {
    var s = 0, c = 0;
    for (var i = 0; i < y.length; i++) if (!isNaN(f[i]) && y[i]) { s += Math.abs((y[i] - f[i]) / y[i]); c++; }
    return c ? s / c : NaN;
  }

  /* ------------------------------------------------------------------ plain-language findings */
  function strength(r) {
    var a = Math.abs(r);
    return a >= 0.8 ? 'very strong' : a >= 0.6 ? 'strong' : a >= 0.4 ? 'moderate' : a >= 0.2 ? 'weak' : 'very weak';
  }
  function findings(prof) {
    var out = [], nums = prof.cols.filter(function (c) { return c.type === 'number' && !c.isId && c.stats.n > 2; });
    out.push({ kind: 'info', text: 'Your data has ' + fmtInt(prof.rows) + ' rows and ' + prof.cols.length + ' columns: ' +
      count(prof.cols, 'number') + ' numeric, ' + count(prof.cols, 'category') + ' categorical' +
      (count(prof.cols, 'date') ? ', ' + count(prof.cols, 'date') + ' date' : '') +
      (count(prof.cols, 'text') ? ', ' + count(prof.cols, 'text') + ' free text' : '') + '.' });
    if (prof.missingCells) {
      var worst = prof.cols.slice().sort(function (a, b) { return b.missing - a.missing; })[0];
      out.push({ kind: prof.missingPct > 0.05 ? 'warn' : 'info', text: pct(prof.missingPct) + ' of cells are empty. The most gaps are in "' + worst.name + '" (' + fmtInt(worst.missing) + ' missing).' });
    } else out.push({ kind: 'good', text: 'No missing values. Every cell is filled in.' });
    if (prof.duplicates) out.push({ kind: 'warn', text: fmtInt(prof.duplicates) + ' rows are exact duplicates of another row. Check whether they were entered twice.' });
    if (nums.length >= 2) {
      var pairs = [];
      for (var i = 0; i < nums.length; i++) for (var j = i + 1; j < nums.length; j++) {
        var c = pearson(nums[i].values, nums[j].values);
        if (!isNaN(c.r)) pairs.push({ a: nums[i].name, b: nums[j].name, r: c.r, p: c.p, n: c.n });
      }
      pairs.sort(function (x, y) { return Math.abs(y.r) - Math.abs(x.r); });
      pairs.slice(0, 3).forEach(function (pr) {
        if (Math.abs(pr.r) < 0.3) return;
        out.push({ kind: 'link', text: '"' + pr.a + '" and "' + pr.b + '" move ' + (pr.r > 0 ? 'together' : 'in opposite directions') +
          ' (' + strength(pr.r) + ' ' + (pr.r > 0 ? 'positive' : 'negative') + ' correlation, r = ' + fmt(pr.r, 2) + ')' +
          (pr.p < 0.05 ? '. This is unlikely to be chance.' : ', but with this many rows it could still be chance.'), pair: pr });
      });
      if (!pairs.length || Math.abs(pairs[0].r) < 0.3) out.push({ kind: 'info', text: 'None of the numeric columns are strongly related to each other.' });
    }
    nums.forEach(function (c) {
      if (Math.abs(c.stats.skew) > 1.2 && c.stats.n > 15)
        out.push({ kind: 'info', text: '"' + c.name + '" is ' + (c.stats.skew > 0 ? 'right' : 'left') + '-skewed: most values are ' + (c.stats.skew > 0 ? 'low' : 'high') + ', with a few ' + (c.stats.skew > 0 ? 'very large' : 'very small') + ' ones. The median (' + fmt(c.stats.median) + ') describes a typical value better than the average (' + fmt(c.stats.mean) + ').' });
      else if (c.stats.outliers && c.stats.outliers / c.stats.n > 0.02 && c.stats.n > 15)
        out.push({ kind: 'warn', text: '"' + c.name + '" has ' + c.stats.outliers + ' unusual value' + (c.stats.outliers > 1 ? 's' : '') + ' far from the rest. Worth checking for typing errors.' });
    });
    var date = prof.cols.find(function (c) { return c.type === 'date'; });
    if (date && nums.length) {
      var target = nums[nums.length - 1], ts = series(prof, date, target);
      if (ts.y.length >= 6) {
        var half = Math.floor(ts.y.length / 2), a = mean(ts.y.slice(0, half)), b = mean(ts.y.slice(half));
        var ch = a ? (b - a) / Math.abs(a) : NaN;
        if (isFinite(ch) && Math.abs(ch) > 0.05)
          out.push({ kind: 'trend', text: 'Over time, "' + target.name + '" is ' + (ch > 0 ? 'rising' : 'falling') + ': the second half of the period averages ' + pct(Math.abs(ch)) + (ch > 0 ? ' higher' : ' lower') + ' than the first half.' });
      }
    }
    prof.cols.filter(function (c) { return c.type === 'category'; }).slice(0, 2).forEach(function (c) {
      var t = c.freq.top[0]; if (!t) return;
      out.push({ kind: 'info', text: 'In "' + c.name + '", the most common value is "' + t.value + '" (' + pct(t.count / (c.count - c.missing)) + ' of rows, ' + c.unique + ' different values in total).' });
    });
    return out;
  }
  function count(cols, t) { return cols.filter(function (c) { return c.type === t; }).length; }

  function series(prof, dateCol, valueCol) {
    var pts = [];
    for (var i = 0; i < prof.rows; i++) {
      var d = dateCol ? dateCol.values[i] : i, v = valueCol.values[i];
      if ((dateCol ? d : true) && !isNaN(v)) pts.push({ d: d, v: v });
    }
    if (dateCol) {
      pts.sort(function (a, b) { return a.d - b.d; });
      // combine rows that share the same date (sum)
      var merged = [];
      pts.forEach(function (p) {
        var last = merged[merged.length - 1];
        if (last && +last.d === +p.d) last.v += p.v; else merged.push({ d: p.d, v: p.v });
      });
      pts = merged;
    }
    return { x: pts.map(function (p) { return p.d; }), y: pts.map(function (p) { return p.v; }) };
  }

  function addSteps(date, step, k) {
    var d = new Date(+date);
    if (!step) return d;
    if (step.name === 'month') d.setUTCMonth(d.getUTCMonth() + k);
    else if (step.name === 'quarter') d.setUTCMonth(d.getUTCMonth() + 3 * k);
    else if (step.name === 'year') d.setUTCFullYear(d.getUTCFullYear() + k);
    else if (step.name === 'week') d.setUTCDate(d.getUTCDate() + 7 * k);
    else d.setUTCDate(d.getUTCDate() + k);
    return d;
  }

  /* ------------------------------------------------------------------ formatting */
  function fmt(v, dp) {
    if (v === null || v === undefined || (typeof v === 'number' && !isFinite(v))) return '–';
    var a = Math.abs(v);
    if (dp === undefined) dp = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 1 ? 2 : a === 0 ? 0 : 3;
    return v.toLocaleString('en-GB', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  }
  function fmtInt(v) { return Math.round(v).toLocaleString('en-GB'); }
  function pct(v) { return (v * 100).toLocaleString('en-GB', { maximumFractionDigits: v < 0.1 ? 1 : 0 }) + '%'; }
  function fmtP(p) { return !isFinite(p) ? '–' : p < 0.001 ? '< 0.001' : fmt(p, 3); }
  function fmtDate(d, step) {
    if (!(d instanceof Date)) return String(d);
    var opts = step && (step.name === 'month' || step.name === 'quarter') ? { month: 'short', year: 'numeric', timeZone: 'UTC' }
      : step && step.name === 'year' ? { year: 'numeric', timeZone: 'UTC' } : { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' };
    return d.toLocaleDateString('en-GB', opts);
  }

  root.QuantAI = {
    isMissing: isMissing, toNumber: toNumber, toDate: toDate, profile: profile, describe: describe, histogram: histogram,
    pearson: pearson, correlationMatrix: correlationMatrix, ols: ols, holtWinters: holtWinters, findings: findings,
    series: series, addSteps: addSteps, strength: strength, tPValue: tPValue,
    fmt: fmt, fmtInt: fmtInt, pct: pct, fmtP: fmtP, fmtDate: fmtDate, mean: mean, sd: sd
  };
})(typeof window !== 'undefined' ? window : globalThis);

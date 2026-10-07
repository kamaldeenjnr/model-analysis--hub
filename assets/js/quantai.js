/*! QuantAI — Model Analysis Hub (modelanalysishub.com). Methodology and analysis by fixed statistical rules.
    Every number is computed in the browser by deterministic code; no data leave the device and no AI service is used. */
(function () {
"use strict";
/* =====================================================================
   Stats — deterministic statistics engine. No AI computes any number.
   Every function is a standard published algorithm; each is tested
   against scipy / statsmodels in test/compare.py.
   ===================================================================== */
const Stats = (function () {
  "use strict";

  /* ---------- special functions ---------- */
  function logGamma(x) {
    const g = 7;
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    x -= 1; let a = c[0]; const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  const lfCache = [0];
  function logFactorial(n) {
    if (n < 0) return NaN;
    if (n < 1000) {
      for (let i = lfCache.length; i <= n; i++) lfCache[i] = lfCache[i - 1] + Math.log(i);
      return lfCache[n];
    }
    return logGamma(n + 1);
  }
  function gammpSeries(a, x) {
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 1; n <= 1000; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
    return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
  }
  function gammqCF(a, x) {
    let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;
    for (let i = 1; i <= 1000; i++) {
      const an = -i * (i - a);
      b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
  }
  /** Upper regularized incomplete gamma Q(a,x). */
  function gammq(a, x) {
    if (x < 0 || a <= 0) return NaN;
    if (x === 0) return 1;
    if (x < a + 1) return 1 - gammpSeries(a, x);
    return gammqCF(a, x);
  }
  function betacf(a, b, x) {
    const FPMIN = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d;
    for (let m = 1; m <= 1000; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return h;
  }
  /** Regularized incomplete beta I_x(a,b). */
  function betai(a, b, x) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return bt * betacf(a, b, x) / a;
    return 1 - bt * betacf(b, a, 1 - x) / b;
  }

  /* ---------- distributions ---------- */
  function normCdf(z) {
    if (!isFinite(z)) return z > 0 ? 1 : 0;
    if (z === 0) return 0.5;
    const q = 0.5 * gammq(0.5, z * z / 2); // P(Z > |z|)
    return z > 0 ? 1 - q : q;
  }
  function normSf(z) { return normCdf(-z); }
  /** Inverse standard normal (Acklam) with one Halley refinement. */
  function normQuantile(p) {
    if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425; let x;
    if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    else if (p <= 1 - pl) { const q = p - 0.5, r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
    else { const q = Math.sqrt(-2 * Math.log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    const e = normCdf(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
    return x - u / (1 + x * u / 2);
  }
  /** Two-tailed p for t. */
  function tP2(t, df) { if (!isFinite(t)) return 0; return clamp01(betai(df / 2, 0.5, df / (df + t * t))); }
  function tCdf(t, df) { const half = 0.5 * betai(df / 2, 0.5, df / (df + t * t)); return t > 0 ? 1 - half : half; }
  function tQuantile(p, df) {
    if (df > 1e7) return normQuantile(p);
    let lo = -1e3, hi = 1e3;
    for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (tCdf(mid, df) < p) lo = mid; else hi = mid; if (hi - lo < 1e-12) break; }
    return (lo + hi) / 2;
  }
  function fSf(F, d1, d2) { if (!(F > 0)) return 1; return clamp01(betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * F))); }
  function chi2Sf(x, df) { if (!(x > 0)) return 1; return clamp01(gammq(df / 2, x / 2)); }
  function binomCdf(k, n, p) { let s = 0; for (let i = 0; i <= k; i++) s += Math.exp(logFactorial(n) - logFactorial(i) - logFactorial(n - i) + i * Math.log(p) + (n - i) * Math.log(1 - p)); return Math.min(1, s); }
  function clamp01(v) { return Math.max(0, Math.min(1, v)); }

  /* ---------- descriptives ---------- */
  const sum = a => { let s = 0; for (const v of a) s += v; return s; };
  const mean = a => sum(a) / a.length;
  function variance(a) { const m = mean(a); let s = 0; for (const v of a) s += (v - m) * (v - m); return s / (a.length - 1); }
  const sd = a => Math.sqrt(variance(a));
  /** Quantile, type 7 (default in R, numpy and Excel PERCENTILE.INC). */
  function quantile(a, q) {
    const s = [...a].sort((x, y) => x - y), n = s.length;
    if (!n) return NaN;
    const h = (n - 1) * q, lo = Math.floor(h), hi = Math.ceil(h);
    return s[lo] + (h - lo) * (s[hi] - s[lo]);
  }
  const median = a => quantile(a, 0.5);
  /** Skewness g1 (moment-based, as Stata `summarize, detail` and scipy default). */
  function skewness(a) { const n = a.length, m = mean(a); let m2 = 0, m3 = 0; for (const v of a) { const d = v - m; m2 += d * d; m3 += d * d * d; } m2 /= n; m3 /= n; return m2 > 0 ? m3 / Math.pow(m2, 1.5) : 0; }
  /** Kurtosis (Pearson, normal = 3, as Stata reports). */
  function kurtosis(a) { const n = a.length, m = mean(a); let m2 = 0, m4 = 0; for (const v of a) { const d = v - m; m2 += d * d; m4 += d * d * d * d; } m2 /= n; m4 /= n; return m2 > 0 ? m4 / (m2 * m2) : 0; }
  function describe(a) {
    const n = a.length;
    if (!n) return { n: 0 };
    const m = mean(a), s = n > 1 ? sd(a) : NaN, tq = n > 1 ? tQuantile(0.975, n - 1) : NaN;
    return {
      n, mean: m, sd: s, se: s / Math.sqrt(n), ciLow: m - tq * s / Math.sqrt(n), ciHigh: m + tq * s / Math.sqrt(n),
      median: median(a), q1: quantile(a, 0.25), q3: quantile(a, 0.75), min: Math.min(...a), max: Math.max(...a),
      skew: n > 2 ? skewness(a) : NaN, kurt: n > 3 ? kurtosis(a) : NaN,
    };
  }
  /** Average ranks (ties share the mean rank) + tie group sizes. */
  function rank(values) {
    const idx = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const r = new Array(values.length), ties = [];
    for (let i = 0; i < idx.length;) {
      let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const avg = (i + j + 2) / 2;
      for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
      if (j > i) ties.push(j - i + 1);
      i = j + 1;
    }
    return { ranks: r, ties };
  }
  const tieSum = ties => ties.reduce((s, t) => s + (t * t * t - t), 0);

  /* ---------- assumption tests ---------- */
  function poly(cc, x) { let r = cc[cc.length - 1]; for (let i = cc.length - 2; i >= 0; i--) r = r * x + cc[i]; return r; }
  /** Shapiro–Wilk W and p-value (Royston 1995, algorithm AS R94 — same as R and scipy). */
  function shapiroWilk(xIn) {
    const x = [...xIn].sort((a, b) => a - b), n = x.length;
    if (n < 3) return { W: NaN, p: NaN, n, error: "Needs at least 3 values" };
    if (n > 5000) return { W: NaN, p: NaN, n, error: "Shapiro–Wilk is only defined up to 5,000 values" };
    if (x[n - 1] - x[0] < 1e-12) return { W: NaN, p: NaN, n, error: "All values are identical" };
    const nn2 = Math.floor(n / 2), a = new Array(nn2);
    const c1 = [0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
    const c2 = [0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
    if (n === 3) { a[0] = Math.SQRT1_2; }
    else {
      const an25 = n + 0.25, m = new Array(nn2);
      let summ2 = 0;
      for (let i = 0; i < nn2; i++) { m[i] = normQuantile((i + 1 - 0.375) / an25); summ2 += m[i] * m[i]; }
      summ2 *= 2;
      const ssumm2 = Math.sqrt(summ2), rsn = 1 / Math.sqrt(n);
      const a1 = poly(c1, rsn) - m[0] / ssumm2;
      let i1, fac;
      if (n > 5) {
        i1 = 2;
        const a2 = -m[1] / ssumm2 + poly(c2, rsn);
        fac = Math.sqrt((summ2 - 2 * m[0] * m[0] - 2 * m[1] * m[1]) / (1 - 2 * a1 * a1 - 2 * a2 * a2));
        a[1] = a2;
      } else {
        i1 = 1;
        fac = Math.sqrt((summ2 - 2 * m[0] * m[0]) / (1 - 2 * a1 * a1));
      }
      a[0] = a1;
      for (let i = i1; i < nn2; i++) a[i] = -m[i] / fac;
    }
    let num = 0; for (let i = 0; i < nn2; i++) num += a[i] * (x[n - 1 - i] - x[i]);
    const mu = mean(x); let ss = 0; for (const v of x) ss += (v - mu) * (v - mu);
    let W = Math.min(1, num * num / ss);
    let p;
    if (n === 3) {
      p = Math.max(0, (6 / Math.PI) * (Math.asin(Math.sqrt(W)) - Math.asin(Math.sqrt(0.75))));
    } else {
      let y = Math.log(1 - W), mm, s;
      if (n <= 11) {
        const gamma = poly([-2.273, 0.459], n);
        if (y >= gamma) return { W, p: 1e-99, n };
        y = -Math.log(gamma - y);
        mm = poly([0.544, -0.39978, 0.025054, -6.714e-4], n);
        s = Math.exp(poly([1.3822, -0.77857, 0.062767, -0.0020322], n));
      } else {
        const xx = Math.log(n);
        mm = poly([-1.5861, -0.31082, -0.083751, 0.0038915], xx);
        s = Math.exp(poly([-0.4803, -0.082676, 0.0030302], xx));
      }
      p = normSf((y - mm) / s);
    }
    return { W, p: clamp01(p), n };
  }
  /** Levene's test, Brown–Forsythe (median-centred) version — Stata robvar W50, R car::leveneTest default, scipy default. */
  function leveneBF(groups) {
    const z = groups.map(g => { const md = median(g); return g.map(v => Math.abs(v - md)); });
    const r = anova(z);
    return { F: r.F, df1: r.df1, df2: r.df2, p: r.p };
  }

  /* ---------- two-group and paired tests ---------- */
  function tTestStudent(g1, g2) {
    const n1 = g1.length, n2 = g2.length, m1 = mean(g1), m2 = mean(g2), v1 = variance(g1), v2 = variance(g2);
    const df = n1 + n2 - 2, sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df, se = Math.sqrt(sp2 * (1 / n1 + 1 / n2));
    const t = (m1 - m2) / se, tq = tQuantile(0.975, df);
    return { t, df, p: tP2(t, df), diff: m1 - m2, ciLow: m1 - m2 - tq * se, ciHigh: m1 - m2 + tq * se, d: (m1 - m2) / Math.sqrt(sp2) };
  }
  function tTestWelch(g1, g2) {
    const n1 = g1.length, n2 = g2.length, m1 = mean(g1), m2 = mean(g2), v1 = variance(g1), v2 = variance(g2);
    const se2 = v1 / n1 + v2 / n2, se = Math.sqrt(se2);
    const df = se2 * se2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1));
    const t = (m1 - m2) / se, tq = tQuantile(0.975, df);
    const sp = Math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2));
    return { t, df, p: tP2(t, df), diff: m1 - m2, ciLow: m1 - m2 - tq * se, ciHigh: m1 - m2 + tq * se, d: (m1 - m2) / sp };
  }
  function tTestPaired(x, y) {
    const d = x.map((v, i) => v - y[i]), n = d.length, md = mean(d), s = sd(d), se = s / Math.sqrt(n), df = n - 1;
    const t = md / se, tq = tQuantile(0.975, df);
    return { t, df, p: tP2(t, df), diff: md, ciLow: md - tq * se, ciHigh: md + tq * se, d: md / s, n };
  }
  /** Mann–Whitney U / Wilcoxon rank-sum: normal approximation with tie and continuity correction
      (R wilcox.test(exact=FALSE, correct=TRUE); scipy mannwhitneyu(method="asymptotic")). */
  function mannWhitney(g1, g2) {
    const n1 = g1.length, n2 = g2.length, N = n1 + n2;
    const { ranks, ties } = rank([...g1, ...g2]);
    const R1 = sum(ranks.slice(0, n1));
    const U = R1 - n1 * (n1 + 1) / 2;
    const mu = n1 * n2 / 2;
    const sigma = Math.sqrt(n1 * n2 / 12 * ((N + 1) - tieSum(ties) / (N * (N - 1))));
    let zr = U - mu; const corr = zr === 0 ? 0 : 0.5 * Math.sign(zr);
    const z = sigma > 0 ? (zr - corr) / sigma : 0;
    const zNoCorr = sigma > 0 ? zr / sigma : 0;
    return { U, R1, z, p: clamp01(2 * normSf(Math.abs(z))), r: Math.abs(zNoCorr) / Math.sqrt(N), zNoCorr, pNoCorr: clamp01(2 * normSf(Math.abs(zNoCorr))) };
  }
  /** Wilcoxon signed-rank, zeros dropped, tie + continuity correction (R default with exact=FALSE). */
  function wilcoxonSignedRank(x, y) {
    const d = x.map((v, i) => v - y[i]).filter(v => v !== 0), n = d.length;
    if (n === 0) return { V: 0, z: 0, p: 1, n: 0, nZero: x.length };
    const { ranks, ties } = rank(d.map(Math.abs));
    const V = sum(ranks.filter((_, i) => d[i] > 0));
    const mu = n * (n + 1) / 4, sigma = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tieSum(ties) / 48);
    const zr = V - mu, corr = zr === 0 ? 0 : 0.5 * Math.sign(zr), z = (zr - corr) / sigma;
    return { V, z, p: clamp01(2 * normSf(Math.abs(z))), n, nZero: x.length - n, r: Math.abs(zr / sigma) / Math.sqrt(n) };
  }

  /* ---------- k-group tests ---------- */
  function anova(groups) {
    const all = groups.flat(), gm = mean(all), N = all.length, k = groups.length;
    let ssb = 0, ssw = 0;
    groups.forEach(g => { const m = mean(g); ssb += g.length * (m - gm) ** 2; for (const v of g) ssw += (v - m) ** 2; });
    const df1 = k - 1, df2 = N - k, F = (ssb / df1) / (ssw / df2);
    return { F, df1, df2, p: fSf(F, df1, df2), etaSq: ssb / (ssb + ssw), mse: ssw / df2 };
  }
  /** Welch's ANOVA (R oneway.test(var.equal=FALSE)). */
  function welchAnova(groups) {
    const k = groups.length, n = groups.map(g => g.length), m = groups.map(mean), v = groups.map(variance);
    const w = n.map((ni, i) => ni / v[i]), sw = sum(w), mw = sum(w.map((wi, i) => wi * m[i])) / sw;
    const A = sum(w.map((wi, i) => wi * (m[i] - mw) ** 2)) / (k - 1);
    const tmp = sum(w.map((wi, i) => (1 - wi / sw) ** 2 / (n[i] - 1)));
    const B = 1 + 2 * (k - 2) / (k * k - 1) * tmp;
    const F = A / B, df1 = k - 1, df2 = (k * k - 1) / (3 * tmp);
    return { F, df1, df2, p: fSf(F, df1, df2) };
  }
  /** Kruskal–Wallis H with tie correction. */
  function kruskal(groups) {
    const all = groups.flat(), N = all.length, k = groups.length;
    const { ranks, ties } = rank(all);
    let off = 0, s = 0;
    groups.forEach(g => { const R = sum(ranks.slice(off, off + g.length)); s += R * R / g.length; off += g.length; });
    const Hraw = 12 / (N * (N + 1)) * s - 3 * (N + 1);
    const C = 1 - tieSum(ties) / (N * N * N - N);
    const H = Hraw / C, df = k - 1;
    return { H, df, p: chi2Sf(H, df), epsSq: H / ((N * N - 1) / (N + 1)) };
  }
  function adjustP(ps, method) {
    const m = ps.length;
    if (method === "bonferroni") return ps.map(p => Math.min(1, p * m));
    if (method === "holm") {
      const o = ps.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]), out = new Array(m); let run = 0;
      o.forEach(([p, i], r) => { run = Math.max(run, Math.min(1, (m - r) * p)); out[i] = run; });
      return out;
    }
    return ps;
  }
  /** Pairwise post-hoc comparisons, Bonferroni-adjusted.
      kind: "pooled-t" (R pairwise.t.test pool.sd=TRUE; Stata oneway, bonferroni; SPSS BONFERRONI),
            "welch-t" (pairwise.t.test pool.sd=FALSE), "mwu" (pairwise.wilcox.test exact=FALSE). */
  function pairwise(groups, labels, kind) {
    const out = [];
    let mse = 0, dfw = 0;
    if (kind === "pooled-t") { const a = anova(groups); mse = a.mse; dfw = a.df2; }
    for (let i = 0; i < groups.length; i++) for (let j = i + 1; j < groups.length; j++) {
      let p, stat;
      if (kind === "pooled-t") {
        const se = Math.sqrt(mse * (1 / groups[i].length + 1 / groups[j].length));
        stat = (mean(groups[i]) - mean(groups[j])) / se; p = tP2(stat, dfw);
      } else if (kind === "welch-t") { const r = tTestWelch(groups[i], groups[j]); stat = r.t; p = r.p; }
      else { const r = mannWhitney(groups[i], groups[j]); stat = r.z; p = r.p; }
      out.push({ a: labels[i], b: labels[j], stat, p });
    }
    const adj = adjustP(out.map(o => o.p), "bonferroni");
    out.forEach((o, i) => o.pAdj = adj[i]);
    return out;
  }

  /* ---------- correlation ---------- */
  function pearson(x, y) {
    const n = x.length, mx = mean(x), my = mean(y);
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) { num += (x[i] - mx) * (y[i] - my); dx += (x[i] - mx) ** 2; dy += (y[i] - my) ** 2; }
    const r = Math.max(-1, Math.min(1, num / Math.sqrt(dx * dy))), df = n - 2;
    const t = Math.abs(r) >= 1 ? Infinity * Math.sign(r) : r * Math.sqrt(df / (1 - r * r));
    const z = Math.atanh(Math.max(-0.9999999, Math.min(0.9999999, r))), se = 1 / Math.sqrt(n - 3), q = normQuantile(0.975);
    return { r, t, df, n, p: tP2(t, df), ciLow: n > 3 ? Math.tanh(z - q * se) : NaN, ciHigh: n > 3 ? Math.tanh(z + q * se) : NaN };
  }
  /** Spearman rho with t-approximation p (Stata spearman; scipy; R cor.test(exact=FALSE)). */
  function spearman(x, y) { const r = pearson(rank(x).ranks, rank(y).ranks); return { rho: r.r, t: r.t, df: r.df, n: r.n, p: r.p }; }

  /* ---------- categorical ---------- */
  function chiSquare(table) {
    const R = table.length, C = table[0].length;
    const rs = table.map(r => sum(r)), cs = table[0].map((_, j) => sum(table.map(r => r[j]))), n = sum(rs);
    let stat = 0, minE = Infinity, low = 0; const expected = [];
    for (let i = 0; i < R; i++) { expected.push([]); for (let j = 0; j < C; j++) {
      const e = rs[i] * cs[j] / n; expected[i].push(e); minE = Math.min(minE, e); if (e < 5) low++;
      stat += (table[i][j] - e) ** 2 / e;
    } }
    const df = (R - 1) * (C - 1);
    return { stat, df, p: chi2Sf(stat, df), n, expected, minExpected: minE, pctLow: 100 * low / (R * C), cramersV: Math.sqrt(stat / (n * (Math.min(R, C) - 1))) };
  }
  function tableLogP(t, rs, cs, n) {
    let s = -logFactorial(n);
    rs.forEach(v => s += logFactorial(v)); cs.forEach(v => s += logFactorial(v));
    t.forEach(r => r.forEach(v => s -= logFactorial(v)));
    return s;
  }
  /** Fisher's exact test (two-sided, sum of tables no more probable than observed).
      Exact enumeration for any r×c table up to a work limit; returns {p, exact:false} with a
      seeded Monte-Carlo estimate beyond it. Matches R fisher.test and Stata tab, exact. */
  function fisherExact(table) {
    const R = table.length, C = table[0].length;
    const rs = table.map(r => sum(r)), cs = table[0].map((_, j) => sum(table.map(r => r[j]))), n = sum(rs);
    const lpObs = tableLogP(table, rs, cs, n), tol = 1e-7;
    let p = 0, count = 0, aborted = false;
    const LIMIT = 3e6;
    const cur = Array.from({ length: R }, () => new Array(C).fill(0));
    const colLeft = [...cs];
    // fill row by row, cell by cell
    function fill(i, j, rowLeft) {
      if (aborted) return;
      if (i === R - 1) { // last row is determined by column remainders
        for (let jj = 0; jj < C; jj++) cur[i][jj] = colLeft[jj];
        if (++count > LIMIT) { aborted = true; return; }
        const lp = tableLogP(cur, rs, cs, n);
        if (lp <= lpObs + tol) p += Math.exp(lp);
        return;
      }
      if (j === C - 1) {
        if (rowLeft > colLeft[j]) return;
        cur[i][j] = rowLeft; colLeft[j] -= rowLeft;
        fill(i + 1, 0, rs[i + 1]);
        colLeft[j] += rowLeft;
        return;
      }
      // remaining capacity of later columns in this row
      let capLater = 0; for (let jj = j + 1; jj < C; jj++) capLater += colLeft[jj];
      const lo = Math.max(0, rowLeft - capLater), hi = Math.min(rowLeft, colLeft[j]);
      for (let v = lo; v <= hi; v++) {
        cur[i][j] = v; colLeft[j] -= v;
        fill(i, j + 1, rowLeft - v);
        colLeft[j] += v;
        if (aborted) return;
      }
    }
    if (R === 1 || C === 1) return { p: 1, exact: true };
    fill(0, 0, rs[0]);
    if (!aborted) return { p: clamp01(p), exact: true };
    // Monte Carlo fallback: permute column labels (fixed margins), seeded for reproducibility
    let seed = 20240607; const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
    const rowsLab = [], colsLab = [];
    table.forEach((r, i) => r.forEach((v, j) => { for (let k = 0; k < v; k++) { rowsLab.push(i); colsLab.push(j); } }));
    const B = 20000; let hit = 0;
    for (let b = 0; b < B; b++) {
      for (let k = colsLab.length - 1; k > 0; k--) { const m = Math.floor(rnd() * (k + 1)); [colsLab[k], colsLab[m]] = [colsLab[m], colsLab[k]]; }
      const t = Array.from({ length: R }, () => new Array(C).fill(0));
      for (let k = 0; k < rowsLab.length; k++) t[rowsLab[k]][colsLab[k]]++;
      if (tableLogP(t, rs, cs, n) <= lpObs + tol) hit++;
    }
    return { p: (hit + 1) / (B + 1), exact: false, sims: B };
  }
  /** 2×2 effect measures. a = exposed & outcome, b = exposed & no outcome, c = unexposed & outcome, d = unexposed & no outcome. */
  function twoByTwo(a, b, c, d) {
    let corrected = false;
    let A = a, B = b, Cc = c, D = d;
    if ([a, b, c, d].some(v => v === 0)) { A += 0.5; B += 0.5; Cc += 0.5; D += 0.5; corrected = true; }
    const q = normQuantile(0.975);
    const or = A * D / (B * Cc), seOr = Math.sqrt(1 / A + 1 / B + 1 / Cc + 1 / D);
    const r1 = A / (A + B), r0 = Cc / (Cc + D), rr = r1 / r0, seRr = Math.sqrt(1 / A - 1 / (A + B) + 1 / Cc - 1 / (Cc + D));
    return { or, orLow: Math.exp(Math.log(or) - q * seOr), orHigh: Math.exp(Math.log(or) + q * seOr),
      rr, rrLow: Math.exp(Math.log(rr) - q * seRr), rrHigh: Math.exp(Math.log(rr) + q * seRr), corrected };
  }
  /** McNemar for paired binary data: chi-square with continuity correction (R default) + exact binomial p. */
  function mcnemar(b, c) {
    const nd = b + c;
    if (nd === 0) return { chi2: 0, p: 1, pExact: 1, b, c };
    const chi2 = (Math.abs(b - c) - 1) ** 2 / nd;
    return { chi2, p: chi2Sf(chi2, 1), pExact: Math.min(1, 2 * binomCdf(Math.min(b, c), nd, 0.5)), b, c };
  }

  /* ---------- matrix utilities ---------- */
  function matInverse(A) {
    const n = A.length;
    const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => i === j ? 1 : 0)]);
    const scale = Math.max(1e-300, ...A.map((r, i) => Math.abs(r[i])));
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      if (Math.abs(M[piv][col]) < 1e-11 * scale) { const e = new Error("singular"); e.code = "singular"; throw e; }
      [M[col], M[piv]] = [M[piv], M[col]];
      const pv = M[col][col];
      for (let j = 0; j < 2 * n; j++) M[col][j] /= pv;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col]; if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
      }
    }
    return M.map(row => row.slice(n));
  }
  function xtwx(Xd, w) {
    const p = Xd[0].length, out = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let k = 0; k < Xd.length; k++) { const x = Xd[k], wk = w ? w[k] : 1;
      for (let i = 0; i < p; i++) { const xi = x[i] * wk; if (xi === 0) continue; for (let j = i; j < p; j++) out[i][j] += xi * x[j]; } }
    for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) out[i][j] = out[j][i];
    return out;
  }
  const mv = (A, v) => A.map(r => { let s = 0; for (let i = 0; i < v.length; i++) s += r[i] * v[i]; return s; });
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  /* ---------- linear regression with diagnostics ---------- */
  function ols(X, y, names, opts) {
    opts = opts || {};
    const n = X.length, Xd = X.map(r => [1, ...r]), p = Xd[0].length, k = p - 1;
    const XtXinv = matInverse(xtwx(Xd));
    const Xty = new Array(p).fill(0); Xd.forEach((r, i) => r.forEach((v, j) => Xty[j] += v * y[i]));
    const beta = mv(XtXinv, Xty);
    const fitted = Xd.map(r => dot(r, beta)), resid = y.map((v, i) => v - fitted[i]);
    const rss = sum(resid.map(e => e * e)), ym = mean(y), tss = sum(y.map(v => (v - ym) ** 2));
    const df = n - p, mse = rss / df, tq = tQuantile(0.975, df);
    let se = XtXinv.map((r, i) => Math.sqrt(Math.max(mse * r[i], 0)));
    if (opts.robust) { // HC1 heteroskedasticity-robust SEs (Stata vce(robust); R sandwich HC1; statsmodels HC1)
      const meat = Array.from({ length: p }, () => new Array(p).fill(0));
      for (let i = 0; i < n; i++) { const e2 = resid[i] * resid[i]; for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) meat[a][b] += Xd[i][a] * Xd[i][b] * e2; }
      const t1 = XtXinv.map(row => meat[0].map((_, j) => dot(row, meat.map(r => r[j]))));
      const cov = t1.map(row => XtXinv[0].map((_, j) => dot(row, XtXinv.map(r => r[j]))));
      se = cov.map((r, i) => Math.sqrt(Math.max(r[i] * n / df, 0)));
    }
    const coefs = beta.map((b, i) => ({ name: i === 0 ? "Intercept" : names[i - 1], coef: b, se: se[i], t: b / se[i], p: tP2(b / se[i], df), ciLow: b - tq * se[i], ciHigh: b + tq * se[i] }));
    let F = k > 0 ? ((tss - rss) / k) / mse : NaN;
    if (opts.robust && k > 0) { // robust Wald F for all slopes = 0 (as Stata after vce(robust))
      const meat = Array.from({ length: p }, () => new Array(p).fill(0));
      for (let i = 0; i < n; i++) { const e2 = resid[i] * resid[i]; for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) meat[a][b] += Xd[i][a] * Xd[i][b] * e2; }
      const t1 = XtXinv.map(row => meat[0].map((_, j) => dot(row, meat.map(r => r[j]))));
      const cov = t1.map(row => XtXinv[0].map((_, j) => dot(row, XtXinv.map(r => r[j])) * n / df));
      const idx = Array.from({ length: k }, (_, i) => i + 1), V = idx.map(i => idx.map(j => cov[i][j])), b = idx.map(i => beta[i]);
      F = dot(b, mv(matInverse(V), b)) / k;
    }
    // leverage and Cook's distance
    const lev = Xd.map(r => dot(r, mv(XtXinv, r)));
    const cooks = resid.map((e, i) => (e * e / (p * mse)) * lev[i] / ((1 - lev[i]) ** 2));
    // Breusch–Pagan (Koenker studentized): regress e^2 on X, LM = n R^2, df = k
    let bp = null;
    if (k > 0) {
      const e2 = resid.map(e => e * e), aux = olsCore(Xd, e2);
      const lm = n * aux.r2; bp = { lm, df: k, p: chi2Sf(lm, k) };
    }
    const sw = n >= 3 && n <= 5000 ? shapiroWilk(resid) : null;
    return { coefs, n, df, k, r2: 1 - rss / tss, adjR2: 1 - (rss / df) / (tss / (n - 1)), F, fP: k > 0 ? fSf(F, k, df) : NaN, rmse: Math.sqrt(mse),
      resid, fitted, lev, cooks, bp, residNormality: sw, nCooksHigh: cooks.filter(c => c > 4 / n).length, maxCook: Math.max(...cooks) };
  }
  function olsCore(Xd, y) {
    const inv = matInverse(xtwx(Xd)); const Xty = new Array(Xd[0].length).fill(0);
    Xd.forEach((r, i) => r.forEach((v, j) => Xty[j] += v * y[i]));
    const b = mv(inv, Xty), f = Xd.map(r => dot(r, b)), ym = mean(y);
    const rss = sum(y.map((v, i) => (v - f[i]) ** 2)), tss = sum(y.map(v => (v - ym) ** 2));
    return { r2: tss > 0 ? 1 - rss / tss : 0 };
  }
  /** Variance inflation factors: regress each column on the others. */
  function vif(X, names) {
    const k = X[0].length; if (k < 2) return null;
    return names.map((nm, j) => {
      try { const r2 = olsCore(X.map(r => [1, ...r.filter((_, m) => m !== j)]), X.map(r => r[j])).r2; return { name: nm, vif: r2 >= 1 ? Infinity : 1 / (1 - r2) }; }
      catch (e) { return { name: nm, vif: Infinity }; }
    });
  }

  /* ---------- generalized linear models (IRLS) ---------- */
  const FAMILIES = {
    binomial: { inv: e => 1 / (1 + Math.exp(-e)), varf: m => m * (1 - m), ll: (y, m) => y * Math.log(Math.max(m, 1e-300)) + (1 - y) * Math.log(Math.max(1 - m, 1e-300)) },
    poisson: { inv: e => Math.exp(Math.min(e, 50)), varf: m => m, ll: (y, m) => y * Math.log(Math.max(m, 1e-300)) - m - logFactorial(y) },
  };
  /** Logistic or Poisson regression by IRLS (canonical links). opts.robust → HC0 sandwich SEs. */
  function glm(X, y, names, family, opts) {
    opts = opts || {};
    const F = FAMILIES[family], n = X.length, Xd = X.map(r => [1, ...r]), p = Xd[0].length;
    let beta = new Array(p).fill(0);
    if (family === "poisson") beta[0] = Math.log(Math.max(mean(y), 1e-8));
    else { const m = Math.min(Math.max(mean(y), 1e-6), 1 - 1e-6); beta[0] = Math.log(m / (1 - m)); }
    let converged = false, inv = null, mu = null, llOld = -Infinity, iter = 0;
    const llOf = b => { let s = 0; for (let i = 0; i < n; i++) s += F.ll(y[i], F.inv(dot(Xd[i], b))); return s; };
    llOld = llOf(beta);
    for (iter = 0; iter < 100; iter++) {
      mu = Xd.map(r => F.inv(dot(r, beta)));
      const W = mu.map(m => Math.max(F.varf(m), 1e-10));
      inv = matInverse(xtwx(Xd, W));
      const grad = new Array(p).fill(0);
      for (let i = 0; i < n; i++) { const r = y[i] - mu[i]; for (let j = 0; j < p; j++) grad[j] += Xd[i][j] * r; }
      let step = mv(inv, grad), nb = beta.map((b, j) => b + step[j]), ll = llOf(nb), half = 0;
      while (ll < llOld - 1e-10 && half < 30) { step = step.map(s => s / 2); nb = beta.map((b, j) => b + step[j]); ll = llOf(nb); half++; }
      beta = nb;
      const change = Math.abs(ll - llOld);
      llOld = ll;
      if (change < 1e-10 * (Math.abs(ll) + 1e-8) || Math.max(...step.map(Math.abs)) < 1e-9) { converged = true; break; }
    }
    mu = Xd.map(r => F.inv(dot(r, beta)));
    inv = matInverse(xtwx(Xd, mu.map(m => Math.max(F.varf(m), 1e-10))));
    let cov = inv;
    if (opts.robust) {
      const meat = Array.from({ length: p }, () => new Array(p).fill(0));
      for (let i = 0; i < n; i++) { const r2 = (y[i] - mu[i]) ** 2; for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) meat[a][b] += Xd[i][a] * Xd[i][b] * r2; }
      const t1 = inv.map(row => meat[0].map((_, j) => dot(row, meat.map(r => r[j]))));
      cov = t1.map(row => inv[0].map((_, j) => dot(row, inv.map(r => r[j]))));
    }
    const q = normQuantile(0.975);
    const coefs = beta.map((b, i) => { const se = Math.sqrt(Math.max(cov[i][i], 0)), z = b / se;
      return { name: i === 0 ? "Intercept" : names[i - 1], coef: b, se, z, p: clamp01(2 * normSf(Math.abs(z))), exp: Math.exp(b), expLow: Math.exp(b - q * se), expHigh: Math.exp(b + q * se), ciLow: b - q * se, ciHigh: b + q * se }; });
    const ll = llOf(beta);
    // null model log-likelihood
    const ym = mean(y); let ll0 = 0; for (let i = 0; i < n; i++) ll0 += F.ll(y[i], ym);
    const lr = 2 * (ll - ll0), k = p - 1;
    let pearsonChi2 = 0; for (let i = 0; i < n; i++) pearsonChi2 += (y[i] - mu[i]) ** 2 / Math.max(F.varf(mu[i]), 1e-12);
    const sep = family === "binomial" && (beta.some((b, i) => i > 0 && Math.abs(b) > 15) || mu.filter(m => m < 1e-8 || m > 1 - 1e-8).length > 0.05 * n || !converged);
    return { coefs, n, k, converged, iterations: iter + 1, ll, ll0, lr, lrDf: k, lrP: chi2Sf(lr, k), mcFadden: 1 - ll / ll0,
      aic: -2 * ll + 2 * p, fitted: mu, dispersion: pearsonChi2 / (n - p), pearsonChi2, residDf: n - p, separation: sep };
  }
  /** Hosmer–Lemeshow, g groups at quantile breaks of fitted probabilities (as R ResourceSelection::hoslem.test). */
  function hosmerLemeshow(y, p, g) {
    g = g || 10;
    const qs = []; for (let i = 0; i <= g; i++) qs.push(quantile(p, i / g));
    const breaks = [...new Set(qs)];
    const G = breaks.length - 1; if (G < 3) return null;
    const obs1 = new Array(G).fill(0), exp1 = new Array(G).fill(0), cnt = new Array(G).fill(0);
    p.forEach((pi, i) => {
      let gi = 0; while (gi < G - 1 && pi > breaks[gi + 1]) gi++; // intervals (b_i, b_{i+1}], first includes lowest
      obs1[gi] += y[i]; exp1[gi] += pi; cnt[gi]++;
    });
    let chi = 0;
    for (let i = 0; i < G; i++) { if (!cnt[i]) continue; const e0 = cnt[i] - exp1[i], o0 = cnt[i] - obs1[i];
      chi += (obs1[i] - exp1[i]) ** 2 / exp1[i] + (o0 - e0) ** 2 / e0; }
    const df = G - 2;
    return { chi2: chi, df, p: chi2Sf(chi, df), groups: G };
  }
  /** ROC curve and AUC; tied predictions are handled as one block. */
  function roc(yTrue, probs) {
    const pr = yTrue.map((y, i) => ({ y, p: probs[i] })).sort((a, b) => b.p - a.p);
    const P = sum(yTrue), N = yTrue.length - P;
    let tp = 0, fp = 0; const pts = [{ fpr: 0, tpr: 0 }];
    for (let i = 0; i < pr.length;) { let j = i;
      while (j < pr.length && Math.abs(pr[j].p - pr[i].p) < 1e-12) { if (pr[j].y === 1) tp++; else fp++; j++; }
      pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0 }); i = j; }
    let auc = 0; for (let i = 1; i < pts.length; i++) auc += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2;
    return { points: pts, auc, P, N };
  }

  /* ---------- sample size ---------- */
  const z = p => normQuantile(p);
  const SampleSize = {
    /** Single proportion (prevalence): n = z² p(1-p) / d² */
    oneProportion({ p, d, conf = 0.95 }) { const za = z(1 - (1 - conf) / 2); return { n: za * za * p * (1 - p) / (d * d), za }; },
    oneMean({ sd: s, d, conf = 0.95 }) { const za = z(1 - (1 - conf) / 2); return { n: (za * s / d) ** 2, za }; },
    /** Two independent proportions (Fleiss, pooled-variance formula without continuity correction; Stata `power twoproportions`). */
    twoProportions({ p1, p2, alpha = 0.05, power = 0.8, ratio = 1 }) {
      const za = z(1 - alpha / 2), zb = z(power), pbar = (p1 + ratio * p2) / (1 + ratio);
      const n1 = (za * Math.sqrt((1 + 1 / ratio) * pbar * (1 - pbar)) + zb * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2 / (p1 - p2) ** 2;
      return { n1, n2: n1 * ratio, za, zb };
    },
    /** Two means, normal approximation: n per group = (1+1/k)(zα+zβ)² σ² / Δ² */
    twoMeans({ sd: s, diff, alpha = 0.05, power = 0.8, ratio = 1 }) {
      const za = z(1 - alpha / 2), zb = z(power), nz = (1 + 1 / ratio) * (za + zb) ** 2 * s * s / (diff * diff);
      // iterate with t quantiles (as G*Power / Stata `power twomeans` do) — usually 1–2 more per group
      let n1 = nz;
      for (let i = 0; i < 50; i++) { const df = Math.max(2, n1 * (1 + ratio) - 2); const nn = (1 + 1 / ratio) * (tQuantile(1 - alpha / 2, df) + tQuantile(power, df)) ** 2 * s * s / (diff * diff); if (Math.abs(nn - n1) < 1e-6) { n1 = nn; break; } n1 = nn; }
      return { n1, n2: n1 * ratio, za, zb, nNormal: nz };
    },
    /** Unmatched case–control from expected OR and exposure prevalence among controls (Kelsey/Fleiss). */
    caseControl({ p0, or, alpha = 0.05, power = 0.8, ratio = 1 }) {
      const p1 = or * p0 / (1 + p0 * (or - 1));
      const r = SampleSize.twoProportions({ p1, p2: p0, alpha, power, ratio });
      return { cases: r.n1, controls: r.n2, p1, za: r.za, zb: r.zb };
    },
    correlation({ r, alpha = 0.05, power = 0.8 }) { const za = z(1 - alpha / 2), zb = z(power); return { n: ((za + zb) / Math.atanh(r)) ** 2 + 3, za, zb }; },
    /** Finite population correction, design effect and non-response. */
    adjust(n, { N, deff = 1, nonResponse = 0 }) {
      let a = n * deff; const steps = [];
      if (deff !== 1) steps.push({ label: `× design effect ${deff}`, value: a });
      if (N && N > 0) { a = a / (1 + (a - 1) / N); steps.push({ label: `finite population correction (N = ${N})`, value: a }); }
      if (nonResponse > 0) { a = a / (1 - nonResponse); steps.push({ label: `÷ (1 − ${nonResponse}) for non-response`, value: a }); }
      return { n: a, steps };
    },
  };

  return {
    logGamma, gammq, betai, normCdf, normSf, normQuantile, tP2, tCdf, tQuantile, fSf, chi2Sf, binomCdf,
    sum, mean, variance, sd, quantile, median, skewness, kurtosis, describe, rank,
    shapiroWilk, leveneBF, tTestStudent, tTestWelch, tTestPaired, mannWhitney, wilcoxonSignedRank,
    anova, welchAnova, kruskal, pairwise, adjustP, pearson, spearman,
    chiSquare, fisherExact, twoByTwo, mcnemar,
    matInverse, ols, vif, glm, hosmerLemeshow, roc, SampleSize,
  };
})();
if (false) module.exports = Stats;

/* =====================================================================
   Stats extensions: survival analysis, ordinal and multinomial logistic
   regression, linear mixed model (random intercept, REML).
   Verified against statsmodels in test/compare_ext.py.
   ===================================================================== */
(function (S) {
  "use strict";
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const mv = (A, v) => A.map(r => dot(r, v));
  const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  const q975 = () => S.normQuantile(0.975);
  const pz = z => Math.max(0, Math.min(1, 2 * S.normSf(Math.abs(z))));

  /* ---------------- Kaplan–Meier with Greenwood SE and log(-log) CI ---------------- */
  function kaplanMeier(time, event) {
    const idx = time.map((t, i) => i).sort((a, b) => time[a] - time[b]);
    const out = [{ t: 0, n: time.length, d: 0, c: 0, s: 1, lo: 1, hi: 1 }];
    let s = 1, gw = 0, atRisk = time.length, k = 0;
    while (k < idx.length) {
      const t = time[idx[k]]; let d = 0, c = 0;
      while (k < idx.length && time[idx[k]] === t) { if (event[idx[k]]) d++; else c++; k++; }
      if (d > 0) {
        s *= 1 - d / atRisk;
        if (atRisk > d) gw += d / (atRisk * (atRisk - d));
        let lo = s, hi = s;
        if (s > 0 && s < 1) { const se = Math.sqrt(gw) / Math.abs(Math.log(s)), z = q975(); lo = Math.pow(s, Math.exp(z * se)); hi = Math.pow(s, Math.exp(-z * se)); }
        out.push({ t, n: atRisk, d, c, s, lo, hi });
      }
      atRisk -= d + c;
    }
    const med = out.find(r => r.s <= 0.5);
    const medLo = out.find(r => r.hi <= 0.5), medHi = out.find(r => r.lo <= 0.5);
    return { table: out, median: med ? med.t : null, medianLo: medHi ? medHi.t : null, medianHi: medLo ? medLo.t : null, n: time.length, events: event.filter(Boolean).length };
  }
  /** Log-rank test for k groups (Mantel–Haenszel), as R survdiff / statsmodels survdiff. */
  function logRank(time, event, group) {
    const lv = [...new Set(group)], k = lv.length, gi = group.map(g => lv.indexOf(g));
    const ts = [...new Set(time.filter((t, i) => event[i]))].sort((a, b) => a - b);
    const O = new Array(k).fill(0), E = new Array(k).fill(0), V = zeros(k, k);
    ts.forEach(t => {
      const n = new Array(k).fill(0), d = new Array(k).fill(0);
      for (let i = 0; i < time.length; i++) { if (time[i] >= t) n[gi[i]]++; if (time[i] === t && event[i]) d[gi[i]]++; }
      const N = S.sum(n), D = S.sum(d); if (N < 2) { for (let j = 0; j < k; j++) { O[j] += d[j]; E[j] += D * n[j] / N; } return; }
      for (let j = 0; j < k; j++) {
        O[j] += d[j]; E[j] += D * n[j] / N;
        for (let m = 0; m < k; m++) V[j][m] += (j === m ? D * (n[j] / N) * (1 - n[j] / N) : -D * n[j] * n[m] / (N * N)) * (N - D) / (N - 1);
      }
    });
    const U = O.map((o, j) => o - E[j]).slice(0, k - 1), Vr = V.slice(0, k - 1).map(r => r.slice(0, k - 1));
    const chi2 = dot(U, mv(S.matInverse(Vr), U)), df = k - 1;
    return { chi2, df, p: S.chi2Sf(chi2, df), levels: lv, observed: O, expected: E };
  }
  /** Cox proportional hazards, Efron ties, Newton–Raphson with step halving. */
  function coxPH(X, time, event, names) {
    const n = X.length, p = X[0].length;
    const order = time.map((t, i) => i).sort((a, b) => time[b] - time[a]); // descending
    function evalAt(beta, needH) {
      const eta = X.map(r => dot(r, beta)), w = eta.map(Math.exp);
      let ll = 0; const g = new Array(p).fill(0), H = needH ? zeros(p, p) : null;
      let S0 = 0; const S1 = new Array(p).fill(0), S2 = needH ? zeros(p, p) : null;
      let k = 0;
      while (k < n) {
        const t = time[order[k]], grp = [];
        while (k < n && time[order[k]] === t) { grp.push(order[k]); k++; }
        grp.forEach(i => { S0 += w[i]; for (let a = 0; a < p; a++) { S1[a] += w[i] * X[i][a]; if (needH) for (let b = 0; b < p; b++) S2[a][b] += w[i] * X[i][a] * X[i][b]; } });
        const D = grp.filter(i => event[i]); const d = D.length; if (!d) continue;
        let D0 = 0; const D1 = new Array(p).fill(0), D2 = needH ? zeros(p, p) : null;
        D.forEach(i => { ll += eta[i]; D0 += w[i]; for (let a = 0; a < p; a++) { g[a] += X[i][a]; D1[a] += w[i] * X[i][a]; if (needH) for (let b = 0; b < p; b++) D2[a][b] += w[i] * X[i][a] * X[i][b]; } });
        for (let l = 0; l < d; l++) {
          const f = l / d, phi = S0 - f * D0; ll -= Math.log(phi);
          const a1 = S1.map((v, a) => v - f * D1[a]);
          for (let a = 0; a < p; a++) { g[a] -= a1[a] / phi; if (needH) for (let b = 0; b < p; b++) H[a][b] -= (S2[a][b] - f * D2[a][b]) / phi - a1[a] * a1[b] / (phi * phi); }
        }
      }
      return { ll, g, H };
    }
    let beta = new Array(p).fill(0), cur = evalAt(beta, true), converged = false, it = 0;
    const ll0 = cur.ll;
    for (it = 0; it < 60; it++) {
      const inv = S.matInverse(cur.H.map(r => r.map(v => -v)));
      let step = mv(inv, cur.g), nb = beta.map((b, i) => b + step[i]), nx = evalAt(nb, true), h = 0;
      while (nx.ll < cur.ll - 1e-10 && h < 30) { step = step.map(s => s / 2); nb = beta.map((b, i) => b + step[i]); nx = evalAt(nb, true); h++; }
      const ch = Math.abs(nx.ll - cur.ll); beta = nb; cur = nx;
      if (ch < 1e-10 || Math.max(...step.map(Math.abs)) < 1e-9) { converged = true; break; }
    }
    const cov = S.matInverse(cur.H.map(r => r.map(v => -v))), z = q975();
    const coefs = beta.map((b, i) => { const se = Math.sqrt(Math.max(cov[i][i], 0)); return { name: names[i], coef: b, se, z: b / se, p: pz(b / se), exp: Math.exp(b), expLow: Math.exp(b - z * se), expHigh: Math.exp(b + z * se) }; });
    const lr = 2 * (cur.ll - ll0);
    return { coefs, ll: cur.ll, ll0, lr, lrDf: p, lrP: S.chi2Sf(lr, p), converged, n, events: event.filter(Boolean).length, separation: !converged || beta.some(b => Math.abs(b) > 15) };
  }

  /* ---------------- generic Newton maximiser with numerical Hessian ---------------- */
  function maximize(f, grad, x0, opts) {
    opts = opts || {};
    let x = [...x0], fx = f(x), converged = false, it = 0;
    const hess = xx => { const k = xx.length, H = zeros(k, k), g0 = grad(xx); for (let j = 0; j < k; j++) { const h = 1e-5 * Math.max(1, Math.abs(xx[j])); const xp = [...xx]; xp[j] += h; const g1 = grad(xp); for (let i = 0; i < k; i++) H[i][j] = (g1[i] - g0[i]) / h; } for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) { const a = (H[i][j] + H[j][i]) / 2; H[i][j] = a; H[j][i] = a; } return H; };
    for (it = 0; it < (opts.maxIter || 100); it++) {
      const g = grad(x), H = hess(x);
      let step; try { step = mv(S.matInverse(H.map(r => r.map(v => -v))), g); } catch (e) { step = g.map(v => v * 1e-3); }
      let nx = x.map((v, i) => v + step[i]), nf = f(nx), h = 0;
      while ((!isFinite(nf) || nf < fx - 1e-12) && h < 40) { step = step.map(s => s / 2); nx = x.map((v, i) => v + step[i]); nf = f(nx); h++; }
      const ch = Math.abs(nf - fx); x = nx; fx = nf;
      if (ch < 1e-11 && Math.max(...step.map(Math.abs)) < 1e-7) { converged = true; break; }
    }
    return { x, fx, converged, H: hess(x), iterations: it + 1 };
  }
  const logistic = v => v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v));

  /* ---------------- ordinal logistic (proportional odds) ---------------- */
  /** P(Y ≤ j) = logistic(θ_j − xβ). y in 0..K−1. Same sign convention as Stata ologit, R polr, statsmodels OrderedModel. */
  function ordinalLogit(X, y, names, levelNames) {
    const n = X.length, p = X[0].length, K = Math.max(...y) + 1, nt = K - 1;
    const cum = []; let acc = 0; for (let j = 0; j < nt; j++) { acc += y.filter(v => v === j).length / n; cum.push(Math.log(Math.min(.999, Math.max(.001, acc)) / (1 - Math.min(.999, Math.max(.001, acc))))); }
    const f = par => {
      const th = par.slice(0, nt), b = par.slice(nt); for (let j = 1; j < nt; j++) if (th[j] <= th[j - 1]) return -Infinity;
      let ll = 0;
      for (let i = 0; i < n; i++) { const e = dot(X[i], b), yi = y[i]; const up = yi < nt ? logistic(th[yi] - e) : 1, lo = yi > 0 ? logistic(th[yi - 1] - e) : 0; ll += Math.log(Math.max(up - lo, 1e-300)); }
      return ll;
    };
    const grad = par => {
      const th = par.slice(0, nt), b = par.slice(nt), g = new Array(nt + p).fill(0);
      for (let i = 0; i < n; i++) {
        const e = dot(X[i], b), yi = y[i];
        const Fu = yi < nt ? logistic(th[yi] - e) : 1, Fl = yi > 0 ? logistic(th[yi - 1] - e) : 0, pr = Math.max(Fu - Fl, 1e-300);
        const fu = yi < nt ? Fu * (1 - Fu) : 0, fl = yi > 0 ? Fl * (1 - Fl) : 0;
        if (yi < nt) g[yi] += fu / pr; if (yi > 0) g[yi - 1] -= fl / pr;
        const db = -(fu - fl) / pr; for (let a = 0; a < p; a++) g[nt + a] += db * X[i][a];
      }
      return g;
    };
    const r = maximize(f, grad, [...cum, ...new Array(p).fill(0)]);
    const cov = S.matInverse(r.H.map(row => row.map(v => -v))), z = q975();
    const se = cov.map((row, i) => Math.sqrt(Math.max(row[i], 0)));
    const coefs = names.map((nm, a) => { const b = r.x[nt + a], s = se[nt + a]; return { name: nm, coef: b, se: s, z: b / s, p: pz(b / s), exp: Math.exp(b), expLow: Math.exp(b - z * s), expHigh: Math.exp(b + z * s) }; });
    const thresholds = r.x.slice(0, nt).map((t, j) => ({ name: `${levelNames[j]} | ${levelNames[j + 1]}`, coef: t, se: se[j] }));
    let ll0 = 0; for (let j = 0; j < K; j++) { const c = y.filter(v => v === j).length; if (c) ll0 += c * Math.log(c / n); }
    const lr = 2 * (r.fx - ll0);
    return { coefs, thresholds, ll: r.fx, ll0, lr, lrDf: p, lrP: S.chi2Sf(lr, p), mcFadden: 1 - r.fx / ll0, converged: r.converged, n, K, aic: -2 * r.fx + 2 * (nt + p) };
  }

  /* ---------------- multinomial logistic (reference = category 0) ---------------- */
  function multinomialLogit(X, y, names, levelNames) {
    const n = X.length, Xd = X.map(r => [1, ...r]), p = Xd[0].length, K = Math.max(...y) + 1, m = K - 1, P = m * p;
    let B = zeros(m, p);
    const probs = (Bm, x) => { const e = Bm.map(b => dot(b, x)), mx = Math.max(0, ...e), ex = e.map(v => Math.exp(v - mx)), den = Math.exp(-mx) + S.sum(ex); return ex.map(v => v / den); };
    const ll = Bm => { let s = 0; for (let i = 0; i < n; i++) { const pr = probs(Bm, Xd[i]); const p0 = 1 - S.sum(pr); s += Math.log(Math.max(y[i] === 0 ? p0 : pr[y[i] - 1], 1e-300)); } return s; };
    let cur = ll(B), converged = false, H = null, it;
    for (it = 0; it < 100; it++) {
      const g = new Array(P).fill(0); H = zeros(P, P);
      for (let i = 0; i < n; i++) {
        const pr = probs(B, Xd[i]), x = Xd[i];
        for (let a = 0; a < m; a++) {
          const r = (y[i] === a + 1 ? 1 : 0) - pr[a];
          for (let u = 0; u < p; u++) g[a * p + u] += r * x[u];
          for (let b = 0; b < m; b++) { const w = pr[a] * ((a === b ? 1 : 0) - pr[b]); for (let u = 0; u < p; u++) for (let v = 0; v < p; v++) H[a * p + u][b * p + v] -= w * x[u] * x[v]; }
        }
      }
      let step = mv(S.matInverse(H.map(r => r.map(v => -v))), g), h = 0;
      let NB = B.map((row, a) => row.map((v, u) => v + step[a * p + u])), nl = ll(NB);
      while (nl < cur - 1e-10 && h < 30) { step = step.map(s => s / 2); NB = B.map((row, a) => row.map((v, u) => v + step[a * p + u])); nl = ll(NB); h++; }
      const ch = Math.abs(nl - cur); B = NB; cur = nl;
      if (ch < 1e-10 || Math.max(...step.map(Math.abs)) < 1e-9) { converged = true; break; }
    }
    // final Hessian at the optimum
    H = zeros(P, P);
    for (let i = 0; i < n; i++) { const pr = probs(B, Xd[i]), x = Xd[i]; for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) { const w = pr[a] * ((a === b ? 1 : 0) - pr[b]); for (let u = 0; u < p; u++) for (let v = 0; v < p; v++) H[a * p + u][b * p + v] -= w * x[u] * x[v]; } }
    const cov = S.matInverse(H.map(r => r.map(v => -v))), z = q975();
    const equations = B.map((row, a) => ({ level: levelNames[a + 1], coefs: row.map((b, u) => { const se = Math.sqrt(Math.max(cov[a * p + u][a * p + u], 0)); return { name: u === 0 ? "Intercept" : names[u - 1], coef: b, se, z: b / se, p: pz(b / se), exp: Math.exp(b), expLow: Math.exp(b - z * se), expHigh: Math.exp(b + z * se) }; }) }));
    let ll0 = 0; for (let j = 0; j < K; j++) { const c = y.filter(v => v === j).length; if (c) ll0 += c * Math.log(c / n); }
    const lr = 2 * (cur - ll0), df = m * (p - 1);
    return { equations, reference: levelNames[0], ll: cur, ll0, lr, lrDf: df, lrP: S.chi2Sf(lr, df), mcFadden: 1 - cur / ll0, converged, n, K, separation: !converged || B.some(r => r.some(v => Math.abs(v) > 15)) };
  }

  /* ---------------- linear mixed model, random intercept, REML ---------------- */
  function mixedRandomIntercept(X, y, groups, names) {
    const n = X.length, Xd = X.map(r => [1, ...r]), p = Xd[0].length;
    const glev = [...new Set(groups)], G = glev.map(g => []); groups.forEach((g, i) => G[glev.indexOf(g)].push(i));
    function fitAt(lam) {
      // V_g = I + lam J ; V_g^-1 = I - c_g J, c_g = lam/(1+n_g lam)
      const XtVX = zeros(p, p), XtVy = new Array(p).fill(0); let logdet = 0;
      G.forEach(ix => {
        const ng = ix.length, c = lam / (1 + ng * lam); logdet += Math.log(1 + ng * lam);
        const sx = new Array(p).fill(0); let sy = 0; ix.forEach(i => { for (let a = 0; a < p; a++) sx[a] += Xd[i][a]; sy += y[i]; });
        ix.forEach(i => { for (let a = 0; a < p; a++) { XtVy[a] += Xd[i][a] * y[i]; for (let b = 0; b < p; b++) XtVX[a][b] += Xd[i][a] * Xd[i][b]; } });
        for (let a = 0; a < p; a++) { XtVy[a] -= c * sx[a] * sy; for (let b = 0; b < p; b++) XtVX[a][b] -= c * sx[a] * sx[b]; }
      });
      const inv = S.matInverse(XtVX), beta = mv(inv, XtVy);
      let q = 0; G.forEach(ix => { const ng = ix.length, c = lam / (1 + ng * lam); let sr = 0, ss = 0; ix.forEach(i => { const r = y[i] - dot(Xd[i], beta); sr += r; ss += r * r; }); q += ss - c * sr * sr; });
      const s2 = q / (n - p);
      // log|X'V^-1X| via LU-free determinant from Cholesky-like elimination
      let ld = 0; { const M = XtVX.map(r => [...r]); for (let k = 0; k < p; k++) { let piv = k; for (let r = k + 1; r < p; r++) if (Math.abs(M[r][k]) > Math.abs(M[piv][k])) piv = r; [M[k], M[piv]] = [M[piv], M[k]]; ld += Math.log(Math.abs(M[k][k])); for (let r = k + 1; r < p; r++) { const f = M[r][k] / M[k][k]; for (let c2 = k; c2 < p; c2++) M[r][c2] -= f * M[k][c2]; } } }
      const reml = -0.5 * ((n - p) * Math.log(s2) + logdet + ld + (n - p) * (1 + Math.log(2 * Math.PI)));
      return { beta, inv, s2, reml };
    }
    // golden-section on t = log(lam), plus the boundary lam = 0
    let a = Math.log(1e-8), b = Math.log(1e4); const gr = (Math.sqrt(5) - 1) / 2;
    let c1 = b - gr * (b - a), c2 = a + gr * (b - a), f1 = fitAt(Math.exp(c1)).reml, f2 = fitAt(Math.exp(c2)).reml;
    for (let i = 0; i < 200 && b - a > 1e-10; i++) { if (f1 > f2) { b = c2; c2 = c1; f2 = f1; c1 = b - gr * (b - a); f1 = fitAt(Math.exp(c1)).reml; } else { a = c1; c1 = c2; f1 = f2; c2 = a + gr * (b - a); f2 = fitAt(Math.exp(c2)).reml; } }
    let lam = Math.exp((a + b) / 2), best = fitAt(lam); const at0 = fitAt(0);
    if (at0.reml >= best.reml) { lam = 0; best = at0; }
    const z = q975();
    const coefs = best.beta.map((bv, i) => { const se = Math.sqrt(Math.max(best.s2 * best.inv[i][i], 0)); return { name: i === 0 ? "Intercept" : names[i - 1], coef: bv, se, z: bv / se, p: pz(bv / se), ciLow: bv - z * se, ciHigh: bv + z * se }; });
    const tau2 = lam * best.s2;
    return { coefs, sigma2: best.s2, tau2, icc: tau2 / (tau2 + best.s2), reml: best.reml, nGroups: glev.length, n, groupSizes: G.map(g => g.length), boundary: lam === 0 };
  }

  Object.assign(S, { kaplanMeier, logRank, coxPH, ordinalLogit, multinomialLogit, mixedRandomIntercept });
})(typeof Stats !== "undefined" ? Stats : require("./stats.js"));
if (false) module.exports = require("./stats.js");

/* =====================================================================
   Data layer: parsing, clean names, missing values, type inference.
   ===================================================================== */
const Data = (function () {
  "use strict";
  const DEFAULT_MISSING = ["", "na", "n/a", "nan", "null", ".", "#n/a"];
  const RESERVED = new Set(["_n", "_N", "_all", "_b", "_cons", "_se", "in", "if", "using", "with", "byte", "int", "long", "float", "double", "str",
    "all", "and", "by", "eq", "ge", "gt", "le", "lt", "ne", "not", "or", "to", "function", "true", "false", "null", "na", "nan", "inf", "else", "for", "while", "repeat", "break", "next", "class", "def", "lambda", "import", "from", "pass", "return", "c", "t", "f"]);

  /** RFC-4180 CSV parser with quoted fields, embedded newlines, and delimiter detection. */
  function parseDelimited(text) {
    text = text.replace(/^\uFEFF/, "");
    // choose the delimiter that gives the most consistent, widest rows over the first lines (title lines may sit above the table)
    const lines = text.split(/\r?\n/).filter(l => l.trim()).slice(0, 30);
    const cands = [",", ";", "\t", "|"];
    const score = d => { const w = lines.map(l => l.split(d).length); const mode = modeOf(w); return mode > 1 ? mode * w.filter(x => x === mode).length : 0; };
    const delim = cands.map(d => [d, score(d)]).sort((a, b) => b[1] - a[1])[0][0];
    const rows = []; let row = [], cur = "", inQ = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQ) {
        if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else inQ = false; }
        else cur += c;
      } else if (c === '"') inQ = true;
      else if (c === delim) { row.push(cur); cur = ""; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(cur); cur = "";
        if (row.length > 1 || row[0] !== "") rows.push(row);
        row = [];
      } else cur += c;
    }
    if (cur !== "" || row.length) { row.push(cur); if (row.length > 1 || row[0] !== "") rows.push(row); }
    const t = tidyTable(rows);
    return { headers: t.headers, rows: t.rows, delimiter: delim, notes: t.notes };
  }
  function modeOf(arr) { const c = new Map(); let best = arr[0] || 0, bc = 0; arr.forEach(x => { const k = (c.get(x) || 0) + 1; c.set(x, k); if (k > bc || (k === bc && x > best)) { bc = k; best = x; } }); return best; }

  /** Find the real header row in a raw grid (rows of strings): skips title/notes lines above the table,
      drops empty columns, and removes blank and "Total" rows below it. Returns { headers, rows, notes }. */
  function tidyTable(grid) {
    const notes = [];
    grid = grid.map(r => r.map(c => (c == null ? "" : String(c))));
    const filled = r => r.filter(c => c.trim() !== "").length;
    const widths = grid.slice(0, 200).map(filled).filter(w => w > 0);
    const W = widths.length ? modeOf(widths) : 0;
    let h = 0;
    const looksHeader = r => { const f = r.filter(c => c.trim() !== ""); return f.length >= Math.max(2, Math.ceil(0.6 * W)) && f.filter(c => !isNum(c.trim())).length >= 0.6 * f.length; };
    for (let i = 0; i < Math.min(25, grid.length - 1); i++) { if (looksHeader(grid[i]) && filled(grid[i + 1]) >= Math.max(2, Math.ceil(0.5 * W))) { h = i; break; } }
    if (h > 0) notes.push({ kind: "header", msg: `Skipped ${h} line${h > 1 ? "s" : ""} above the table (titles or notes); variable names were found on line ${h + 1}.` });
    let headers = grid[h] || [], rows = grid.slice(h + 1);
    // drop columns that are empty in the header and every row
    const ncol = Math.max(headers.length, ...rows.slice(0, 500).map(r => r.length));
    const keep = [];
    for (let j = 0; j < ncol; j++) if ((headers[j] || "").trim() !== "" || rows.some(r => (r[j] || "").trim() !== "")) keep.push(j);
    if (keep.length < ncol && ncol - keep.length > 0 && keep.length) { const nd = ncol - keep.length; if (nd > 0 && keep.length < ncol) notes.push({ kind: "cols", msg: `Removed ${nd} empty column${nd > 1 ? "s" : ""}.` }); }
    headers = keep.map(j => (headers[j] || "").trim());
    rows = rows.map(r => keep.map(j => r[j] == null ? "" : r[j]));
    const before = rows.length;
    rows = rows.filter(r => r.some(c => c.trim() !== ""));
    // a summary row at the bottom ("Total", "Mean") is not an observation
    const tot = /^(total|totals|grand total|sum|mean|average|overall)$/i;
    let dropped = 0;
    while (rows.length > 1 && rows[rows.length - 1].some(c => tot.test(c.trim()))) { rows.pop(); dropped++; }
    if (dropped) notes.push({ kind: "total", msg: `Removed ${dropped} summary row${dropped > 1 ? "s" : ""} ("Total"/"Mean") at the bottom of the table.` });
    if (before - rows.length - dropped > 0) notes.push({ kind: "blank", msg: `Removed ${before - rows.length - dropped} blank row${before - rows.length - dropped > 1 ? "s" : ""}.` });
    return { headers, rows, notes };
  }

  /** Make names valid and identical in Stata, R, Python and SPSS: lowercase, a–z0–9_, ≤ 30 chars, unique. */
  function cleanNames(headers) {
    const used = new Set();
    return headers.map((h, i) => {
      let s = String(h || "").normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase()
        .replace(/[^a-z0-9_]+/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "");
      if (!s) s = "var" + (i + 1);
      if (/^[0-9]/.test(s)) s = "v" + s;
      s = s.slice(0, 28).replace(/_+$/, "");
      if (RESERVED.has(s)) s = s + "_v";
      let base = s, k = 2;
      while (used.has(s)) { s = (base.slice(0, 26) + "_" + k++); }
      used.add(s);
      return s;
    });
  }

  const isNum = s => s !== "" && /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s);
  function sortLevels(levels, numeric) {
    return [...levels].sort((a, b) => numeric ? Number(a) - Number(b) : (a < b ? -1 : a > b ? 1 : 0));
  }

  /** Build a dataset: columns of cleaned string values (null = missing), plus variable metadata. */
  function build(headers, rows, opts) {
    opts = opts || {};
    const extraMissing = (opts.missingCodes || []).map(s => String(s).trim().toLowerCase()).filter(Boolean);
    const missSet = new Set([...DEFAULT_MISSING, ...extraMissing]);
    const names = cleanNames(headers);
    const nRows = rows.length;
    const varMiss = opts.varMissing || {};
    const vars = headers.map((h, j) => {
      const raw = rows.map(r => (r[j] == null ? "" : String(r[j])).trim());
      const own = new Set((varMiss[names[j]] || []).map(x => String(x).trim().toLowerCase()));
      let vals = raw.map(v => missSet.has(v.toLowerCase()) || own.has(v.toLowerCase()) ? null : v);
      const fixes = [];
      const cn = cleanNumbers(vals); if (cn) { vals = cn.vals; fixes.push(cn.note); }
      const pres = vals.filter(v => v !== null), nonNum = pres.filter(x => !isNum(x)), nonNumU = [...new Set(nonNum)];
      if (pres.length - nonNum.length >= 10 && nonNum.length && nonNum.length <= 0.1 * pres.length && nonNumU.length <= 5) {
        vals = vals.map(v => v !== null && !isNum(v) ? null : v);
        fixes.push(`Set ${nonNum.length} text entr${nonNum.length > 1 ? "ies" : "y"} in a numeric column to missing: ${nonNumU.map(x => `"${x}"`).join(", ")}.`);
      }
      const cm = mergeCase(vals); if (cm) { vals = cm.vals; fixes.push(cm.note); }
      const sy = mergeSynonyms(vals); if (sy) { vals = sy.vals; fixes.push(sy.note); }
      const v = makeVar(names[j], h, vals, opts);
      v.fixes = fixes;
      return v;
    });
    likertBatch(vars);
    return { vars, nRows, names, labels: headers };
  }

  /** "1,250", "GH₵ 40", "45%", "12 kg" → plain numbers, when every value in the column has the same form. */
  function cleanNumbers(vals) {
    const present = vals.filter(v => v !== null);
    if (!present.length || present.every(isNum)) return null;
    const re = /^(gh₵|ghs|gh¢|ghc|us\$|usd|\$|€|eur|£|gbp|¥|₦|ngn|ksh|kes|rs|inr|cfa|xof|₵|¢|)\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?)\s*(%|kg|g|mg|mcg|lb|lbs|cm|mm|m|km|ml|l|yrs?|years?|months?|mos?|wks?|weeks?|days?|hrs?|hours?|mins?|minutes?|secs?|mmhg|bpm|ha|acres?|°c|c)?\.?$/i;
    const pre = new Set(), suf = new Set(), shown = []; let ok = 0, changed = 0;
    const out = vals.map(v => {
      if (v === null) return null;
      const m = v.match(re); if (!m || !m[2] || !/\d/.test(m[2])) return v;
      if (!pre.has(m[1].toLowerCase()) && m[1]) shown.push(m[1]); if (!suf.has((m[3] || "").toLowerCase()) && m[3]) shown.push(m[3]); pre.add(m[1].toLowerCase()); suf.add((m[3] || "").toLowerCase()); ok++;
      const x = String(Number(m[2].replace(/,/g, ""))); if (x !== v) changed++;
      return x;
    });
    // all values (or all but a few words such as "refused", handled next) must share one number format
    if (!changed || ok < 0.9 * present.length || ok < present.length - 5 || pre.size > 2 || suf.size > 2) return null;
    const unit = shown.join(" ");
    return { vals: out, note: `Read as numbers${unit ? ` after removing "${unit}"` : ""} and thousands separators.` };
  }
  /** Short forms and punctuation variants that mean the same answer:
      Y/N → Yes/No, M/F → Male/Female, "S.H.S" = "SHS", "Don't know" = "Dont know". */
  const SYN_SETS = [
    { canon: { yes: "Yes", no: "No" }, map: { y: "yes", yes: "yes", yeah: "yes", yea: "yes", true: "yes", n: "no", no: "no", nope: "no", false: "no" } },
    { canon: { male: "Male", female: "Female" }, map: { m: "male", male: "male", man: "male", men: "male", boy: "male", f: "female", female: "female", woman: "female", women: "female", girl: "female", fem: "female" } },
  ];
  function mergeSynonyms(vals) {
    const present = vals.filter(v => v !== null);
    if (!present.length || present.every(isNum)) return null;
    const low = v => v.toLowerCase().trim();
    const notes = [];
    let out = vals;
    // 1) whole-column answer sets (only when every value belongs to the set, so other meanings of "N" or "F" are left alone)
    for (const set of SYN_SETS) {
      const keys = new Set(present.map(low));
      if (![...keys].every(k => k in set.map)) continue;
      const targets = new Set([...keys].map(k => set.map[k]));
      if (targets.size < 2 && keys.size < 2) continue;
      const before = new Set(present);
      out = vals.map(v => v === null ? null : set.canon[set.map[low(v)]]);
      const changed = [...before].filter(x => x !== set.canon[set.map[low(x)]]);
      if (changed.length) notes.push(`Read ${changed.map(x => `"${x}"`).join(", ")} as ${[...targets].map(t => `"${set.canon[t]}"`).join(" / ")}`);
      break;
    }
    // 2) spellings that differ only in dots, spaces, hyphens or apostrophes ("S.H.S" = "SHS", "Don't know" = "Dont know")
    const pres2 = out.filter(v => v !== null), key = v => v.toLowerCase().replace(/[.\s\-_'’]+/g, "");
    const groups = new Map();
    pres2.forEach(v => { const k = key(v); if (!k) return; if (!groups.has(k)) groups.set(k, new Map()); const g = groups.get(k); g.set(v, (g.get(v) || 0) + 1); });
    const multi = [...groups.values()].filter(g => g.size > 1);
    if (multi.length && groups.size <= 50) {
      const canon = new Map(); groups.forEach((g, k) => canon.set(k, [...g.entries()].sort((a, b) => b[1] - a[1])[0][0]));
      out = out.map(v => v === null ? null : (canon.get(key(v)) || v));
      notes.push(`Merged ${multi.slice(0, 3).map(g => [...g.keys()].map(x => `"${x}"`).join(" = ")).join("; ")}`);
    }
    return notes.length ? { vals: out, note: notes.join(". ") + "." } : null;
  }
  /** "Yes", "yes", "YES " → one spelling (the most common). */
  function mergeCase(vals) {
    const present = vals.filter(v => v !== null);
    if (!present.length || present.every(isNum)) return null;
    const groups = new Map();
    present.forEach(v => { const k = v.toLowerCase().replace(/\s+/g, " ").trim(); if (!groups.has(k)) groups.set(k, new Map()); const g = groups.get(k); g.set(v, (g.get(v) || 0) + 1); });
    if (groups.size === new Set(present).size) return null;
    if (groups.size > 50) return null;
    const canon = new Map(); groups.forEach((g, k) => canon.set(k, [...g.entries()].sort((a, b) => b[1] - a[1] || (/^[A-Z]/.test(b[0]) - /^[A-Z]/.test(a[0])))[0][0]));
    const merged = [...groups.values()].filter(g => g.size > 1).map(g => [...g.keys()].map(x => `"${x}"`).join(" = "));
    return { vals: vals.map(v => v === null ? null : canon.get(v.toLowerCase().replace(/\s+/g, " ").trim())), note: `Merged spellings that differ only in capitals or spaces: ${merged.slice(0, 3).join("; ")}${merged.length > 3 ? " …" : ""}.` };
  }
  /** Three or more columns sharing the same small 1..k (or 0..k) integer scale are questionnaire items: treat as ordinal. */
  function likertBatch(vars) {
    const key = v => v.numeric && v.type === "categorical" && !v.hint ? v.allLevels.join(",") : null;
    const groups = new Map();
    vars.forEach(v => { const k = key(v); if (k && /^(0,)?1,2,3(,4(,5(,6(,7)?)?)?)?$/.test(k)) { if (!groups.has(k)) groups.set(k, []); groups.get(k).push(v); } });
    groups.forEach(g => { if (g.length >= 3) g.forEach(v => { v.type = v.inferredType = "ordinal"; v.why = `one of ${g.length} items on the same ${v.allLevels[0]}–${v.allLevels[v.allLevels.length - 1]} scale (questionnaire items)`; setLevels(v); }); });
  }
  function makeVar(name, label, vals) {
    const present = vals.filter(v => v !== null);
    const numeric = present.length > 0 && present.every(isNum);
    const uniq = [...new Set(present)];
    let levels = sortLevels(uniq, numeric);
    const v = { name, label, values: vals, numeric, nMissing: vals.length - present.length, nUnique: uniq.length, allLevels: levels };
    v.type = inferType(v, present);
    v.inferredType = v.type;
    setLevels(v);
    if (v.type === "ordinal" && v.scaleOrder) v.levels = v.scaleOrder;
    return v;
  }

  /* ---- what a variable's name says about it ---- */
  const toks = name => name.toLowerCase().split(/[_\s]+|(?<=[a-z])(?=\d)/).filter(Boolean);
  const HINT = {
    id: /^(id|uid|code|serial|sn|no|num|number|record|respondent|participant|subject|patient|case|caseid|pid|hhid|key|index)$/,
    group: /^(group|grp|cat|category|band|class|bracket|range|level)$/,
    cat: /^(sex|gender|region|district|state|province|county|community|village|town|city|zone|area|site|clinic|facility|hospital|school|marital|religion|ethnicity|ethnic|tribe|occupation|employment|employed|job|profession|residence|location|urban|rural|status|type|arm|treatment|intervention|race|nationality|language|brand|method|mode|source|department|ward|team|country|colour|color|blood|species|variety|breed|crop|product|channel|sector|industry)$/,
    ord: /^(education|educ|edu|grade|stage|severity|satisfaction|satisfied|agree|agreement|likert|rating|rank|frequency|often|quintile|quartile|tertile|decile|wealth|ses|class|scale|level|importance|priority|stars|pain)$/,
    cont: /^(age|weight|wt|height|ht|bmi|bp|sbp|dbp|pressure|systolic|diastolic|glucose|sugar|cholesterol|hb|hgb|haemoglobin|hemoglobin|income|salary|wage|price|cost|amount|expenditure|spend|revenue|sales|profit|time|duration|days|months|years|weeks|hours|minutes|seconds|distance|temperature|temp|score|marks|mark|percent|percentage|pct|rate|ratio|length|width|size|volume|dose|yield|area_ha|kg|cm|mm|count|visits|children|births|members|household_size|hhsize|number_of|steps|calories|gpa|cgpa|iq|bmi_z|waist|hip|muac|viral|cd4|creatinine|potassium|sodium|pulse|heart_rate|rr|spo2|tenure|experience|quantity|qty|units|population|density|rainfall|humidity)$/,
  };
  function nameHint(name) {
    const t = toks(name), has = re => t.some(x => re.test(x));
    if (has(HINT.group) && (has(HINT.cont) || t.includes("age"))) return "ord";
    if (/^(no|num|number|n)_of/.test(name)) return "cont";
    // the last meaningful word decides ("clinic_visits" is a count, "visit_clinic" a category)
    for (let i = t.length - 1; i >= 0; i--) {
      const x = t[i];
      if (HINT.id.test(x)) { if (!has(HINT.cont) && !has(HINT.cat)) return "id"; continue; }
      if (HINT.ord.test(x)) return "ord";
      if (HINT.cat.test(x) || HINT.group.test(x)) return "cat";
      if (HINT.cont.test(x)) return "cont";
    }
    if (/^q\d+[a-z]?$|^item\d+$|^[a-z]{1,4}\d{1,2}$/.test(name.toLowerCase())) return "item";
    return null;
  }
  /* ---- ordered answer scales recognised from their words ---- */
  const SCALES = [
    ["strongly disagree", "disagree", "somewhat disagree", "neither agree nor disagree", "neutral", "undecided", "not sure", "somewhat agree", "agree", "strongly agree"],
    ["very dissatisfied", "dissatisfied", "somewhat dissatisfied", "neutral", "neither", "somewhat satisfied", "satisfied", "very satisfied"],
    ["never", "rarely", "seldom", "occasionally", "sometimes", "often", "frequently", "usually", "very often", "always"],
    ["very poor", "poor", "fair", "average", "good", "very good", "excellent"],
    ["very bad", "bad", "average", "good", "very good"],
    ["none", "no education", "no formal education", "no schooling", "nursery", "kindergarten", "primary", "basic", "middle", "jhs", "junior high", "junior high school", "junior secondary", "secondary", "shs", "senior high", "senior high school", "senior secondary", "high school", "o level", "a level", "vocational", "technical", "vocational/technical", "diploma", "certificate", "hnd", "tertiary", "college", "university", "bachelor", "bachelors", "degree", "first degree", "undergraduate", "graduate", "masters", "master", "postgraduate", "phd", "doctorate"],
    ["very low", "low", "lower", "below average", "medium", "moderate", "average", "middle", "above average", "high", "higher", "very high"],
    ["mild", "moderate", "severe", "very severe"],
    ["none", "minimal", "mild", "moderate", "moderately severe", "severe"],
    ["very small", "small", "medium", "large", "very large"],
    ["poorest", "poorer", "poor", "middle", "richer", "rich", "richest"],
    ["lowest", "second", "middle", "fourth", "highest"],
    ["not at all", "a little", "slightly", "somewhat", "moderately", "quite a bit", "very", "very much", "extremely"],
    ["very unlikely", "unlikely", "neutral", "likely", "very likely"],
    ["not important", "slightly important", "moderately important", "important", "very important"],
    ["stage i", "stage ii", "stage iii", "stage iv"],
    ["first", "second", "third", "fourth", "fifth"],
  ];
  function scaleOrder(levels) {
    if (levels.length < 3) return null;
    const norm = levels.map(l => l.toLowerCase().replace(/^\s*\d+\s*[.)=:-]\s*/, "").replace(/\s+/g, " ").trim());
    for (const sc of SCALES) { const idx = norm.map(l => sc.indexOf(l)); if (idx.every(i => i >= 0)) return levels.map((l, i) => [l, idx[i]]).sort((a, b) => a[1] - b[1]).map(x => x[0]); }
    // "1 = Poor", "2 = Fair" ... : ordered by their leading number
    if (levels.every(l => /^\s*\d+\s*[.)=:-]/.test(l))) return [...levels].sort((a, b) => parseInt(a) - parseInt(b));
    // ranges such as "18-24", "25-34", "<18", "65+"
    if (levels.every(l => /^\s*(<|>|≤|≥|under|over|below|above)?\s*\d+(\.\d+)?\s*(-|–|to|\+|and above|and over|or more|plus)?\s*(\d+(\.\d+)?)?\s*[a-z]*\s*$/i.test(l))) {
      const key = l => { const m = l.match(/\d+(\.\d+)?/); const x = m ? +m[0] : 0; return /^\s*(<|≤|under|below)/i.test(l) ? x - 0.5 : x; };
      return [...levels].sort((a, b) => key(a) - key(b));
    }
    return null;
  }
  function inferType(v, present) {
    if (!present.length) { v.why = "no values"; return "id"; }
    const hint = nameHint(v.name); v.hint = hint;
    if (v.numeric) {
      const nums = present.map(Number), allInt = nums.every(Number.isInteger);
      const mn = Math.min(...nums), mx = Math.max(...nums);
      const consecutive = allInt && v.nUnique === mx - mn + 1;
      if (v.nUnique <= 2) { v.why = "two values"; return "binary"; }
      // a running number, one per row: an identifier, not a measurement
      if (allInt && v.nUnique === present.length && present.length >= 20 && (hint === "id" || (consecutive && hint !== "cont"))) { v.why = "a different whole number on every row (an identifier)"; return "id"; }
      if (hint === "id" && v.nUnique >= 0.9 * present.length) { v.why = "named like an identifier"; return "id"; }
      if (allInt && hint === "cat" && v.nUnique <= 40) { v.why = "whole-number codes in a variable named like a category"; return "categorical"; }
      if (allInt && hint === "ord" && v.nUnique <= 12) { v.why = "whole-number codes in a variable named like an ordered scale"; return "ordinal"; }
      if (allInt && hint === "item" && consecutive && v.nUnique <= 11 && mn >= 0 && mn <= 1) { v.why = "a questionnaire item on a " + mn + "–" + mx + " scale"; return "ordinal"; }
      if (hint === "cont") {
        if (allInt && mn >= 0 && mn <= 1 && mx <= 1000 && !/(^|_)(age|bp|sbp|dbp|systolic|diastolic|weight|height|income|salary|price|cost|score|marks?)($|_)/i.test(v.name)) { v.why = "whole numbers from " + mn + " in a variable named like a count"; return "count"; }
        v.why = "numbers in a variable named like a measurement"; return "continuous";
      }
      if (allInt && v.nUnique <= 7) { v.why = `${v.nUnique} whole-number codes`; return "categorical"; }
      // counts start at 0 or 1 (visits, children, episodes); whole-number measurements (age, blood pressure) do not
      if (allInt && mn >= 0 && mn <= 1) { v.why = "whole numbers starting at " + mn; return "count"; }
      v.why = "numbers with many values"; return "continuous";
    }
    if (v.nUnique <= 2) { v.why = "two categories"; return "binary"; }
    const ord = scaleOrder(v.allLevels);
    if (ord && v.nUnique <= 15) { v.scaleOrder = ord; v.why = "categories that form an ordered scale"; return "ordinal"; }
    if (v.nUnique <= Math.max(15, 0.05 * present.length) || (hint === "cat" && v.nUnique <= 60 && v.nUnique < 0.5 * present.length)) { v.why = `${v.nUnique} categories`; return "categorical"; }
    v.why = v.nUnique === present.length ? "different text on every row (names or IDs)" : "free text with many different values";
    return "id";
  }
  /** Levels in analysis order: reference/unexposed first. For binary, levels[1] is the event. */
  function setLevels(v) {
    if (["binary", "categorical", "ordinal"].includes(v.type)) {
      let lv = [...v.allLevels];
      if (v.type === "binary" && !v.numeric) {
        // put the "absence" level first when it is recognisable
        const neg = /^(no|n|false|negative|neg|absent|none|never|normal|alive|0|female|f|control|unexposed|non.*|not .*)$/i;
        const iNeg = lv.findIndex(l => neg.test(l));
        if (iNeg > 0) lv = [lv[iNeg], ...lv.filter((_, i) => i !== iNeg)];
      }
      if (v.refLevel && lv.includes(v.refLevel)) lv = [v.refLevel, ...lv.filter(l => l !== v.refLevel)];
      v.levels = lv;
    } else v.levels = null;
  }
  function setType(v, type) { v.type = type; setLevels(v); }
  function setReference(v, level) { v.refLevel = level; setLevels(v); }

  /** Numeric vector of a variable (NaN where missing). */
  const num = (v, i) => v.values[i] === null ? NaN : Number(v.values[i]);

  /** Rows (indices) complete for all given variables. */
  function completeRows(ds, vars) {
    const out = [];
    for (let i = 0; i < ds.nRows; i++) {
      let ok = true;
      for (const v of vars) {
        const x = v.values[i];
        if (x === null) { ok = false; break; }
        if ((v.type === "continuous" || v.type === "count") && !isNum(x)) { ok = false; break; }
        if (v.levels && !v.levels.includes(x)) { ok = false; break; }
      }
      if (ok) out.push(i);
    }
    return out;
  }


  /* ---------- data check: problems a careful analyst looks for before any test ---------- */
  const MISS_CODES = [9, 99, 999, 9999, 99999, -9, -99, -999, -1, 88, 888, 98, 998, 97, 997, 77, 777, 66, 666];
  const LIMITS = [
    [/(^|_)age($|_|_years|_yrs)/, 0, 120, "an age"], [/(^|_)(bmi)($|_)/, 10, 80, "a BMI"], [/(^|_)(sbp|systolic)/, 50, 300, "a systolic blood pressure"],
    [/(^|_)(dbp|diastolic)/, 20, 200, "a diastolic blood pressure"], [/(^|_)(percent|percentage|pct)($|_)/, 0, 100, "a percentage"],
    [/(^|_)(height|ht)(_cm)?($|_)/, 0, 260, "a height"], [/(^|_)(weight|wt)(_kg)?($|_)/, 0, 400, "a weight"],
    [/(^|_)(temp|temperature)($|_)/, 25, 45, "a body temperature (°C)"], [/(^|_)(hb|hgb|haemoglobin|hemoglobin)($|_)/, 2, 25, "a haemoglobin (g/dL)"],
    [/(^|_)(gpa|cgpa)($|_)/, 0, 5, "a grade point average"],
  ];
  const NONNEG = /(^|_)(income|salary|wage|price|cost|amount|expenditure|spend|revenue|sales|count|visits|children|births|members|size|duration|days|months|years|weeks|hours|minutes|distance|weight|height|age|dose|quantity|qty|units|time|length|volume|yield|population)($|_)/;
  function quantile(sorted, q) { const pos = (sorted.length - 1) * q, lo = Math.floor(pos); return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (pos - lo); }
  function quality(ds, opts) {
    opts = opts || {};
    const out = [], n = ds.nRows, add = (sev, msg, extra) => out.push(Object.assign({ sev, msg }, extra || {}));
    (opts.notes || []).forEach(x => add("info", x.msg));
    ds.vars.forEach(v => (v.fixes || []).forEach(f => add("info", `${v.label}: ${f}`, { var: v.name })));
    if (n < 30) add("warn", `Only ${n} rows. Tests have little power and normality checks are unreliable with so few observations; report results with caution.`);
    // duplicate rows
    const seen = new Map(); let dup = 0;
    for (let i = 0; i < n; i++) { const k = ds.vars.map(v => v.values[i]).join("\u0001"); if (seen.has(k)) dup++; else seen.set(k, i); }
    if (dup) add("warn", `${dup} row${dup > 1 ? "s are exact duplicates" : " is an exact duplicate"} of an earlier row. Check whether the same person was entered twice.`, { fix: { kind: "dedupe" }, fixLabel: "Remove duplicate rows" });
    ds.vars.forEach(v => {
      const pres = v.values.filter(x => x !== null);
      if (v.type === "id" && v.hint === "id" && pres.length) { const u = new Set(pres).size; if (u < pres.length) add("warn", `${v.label} looks like an ID but ${pres.length - u} value${pres.length - u > 1 ? "s are" : " is"} repeated. If each row should be a different person, some were entered twice; if people were measured more than once, the data are in long format (use a mixed model with this as the cluster).`, { var: v.name }); }
      if (!pres.length) { add("warn", `${v.label} is empty in every row.`, { var: v.name }); return; }
      if (v.nUnique === 1) add("info", `${v.label} has the same value ("${pres[0]}") in every row, so it can't explain any differences.`, { var: v.name });
      const pm = v.nMissing / n;
      if (pm >= 0.2 && v.type !== "id") add(pm >= 0.5 ? "warn" : "info", `${v.label} is missing for ${(100 * pm).toFixed(0)}% of rows. Analyses with it use only complete cases, which can bias results if the missingness isn't random.`, { var: v.name });
      if (v.numeric && v.type !== "id") {
        const nums = pres.map(Number), srt = [...nums].sort((a, b) => a - b), q1 = quantile(srt, 0.25), q3 = quantile(srt, 0.75), iqr = q3 - q1;
        // missing-value codes such as 99 or 999 hiding as real numbers
        const cnt = new Map(); nums.forEach(x => cnt.set(x, (cnt.get(x) || 0) + 1));
        const nonCode = srt.filter(x => !MISS_CODES.includes(x) && (iqr <= 0 || x <= q3 + 3 * iqr)), topReal = nonCode.length ? nonCode[nonCode.length - 1] : Infinity;
        const codes = MISS_CODES.filter(c => cnt.has(c) && (v.type === "continuous" || v.type === "count" ? (iqr > 0 ? (c > q3 + 3 * iqr || c < q1 - 3 * iqr || (c >= 98 && cnt.get(c) >= 2 && c > topReal + 5)) : c !== srt[Math.floor(srt.length / 2)]) : (v.levels && Math.abs(c) >= 9 && (() => { const others = srt.filter(x => !MISS_CODES.includes(x)); return others.length && (c > Math.max(...others) + 1 || c < Math.min(...others) - 1); })())));
        if (codes.length) add("warn", `${v.label} contains ${codes.map(c => `${c} (${cnt.get(c)}×)`).join(", ")}, far outside its other values. This is usually a code for "missing" or "don't know"; left as a number it distorts means and tests.`, { var: v.name, fix: { kind: "missing", var: v.name, codes: codes.map(String) }, fixLabel: `Treat ${codes.join(", ")} as missing` });
        const real = nums.filter(x => !codes.includes(x)); let impossible = new Set();
        // impossible values for well-known measurements
        const lim = LIMITS.find(([re]) => re.test(v.name));
        if (lim && v.type !== "categorical" && v.type !== "binary") { const bad = real.filter(x => x < lim[1] || x > lim[2]); if (bad.length) { impossible = new Set(bad); add("warn", `${v.label} has ${bad.length} value${bad.length > 1 ? "s" : ""} that can't be ${lim[3]} (${[...impossible].slice(0, 5).join(", ")}). Check them against the source records, or set them to missing.`, { var: v.name, fix: { kind: "missing", var: v.name, codes: [...impossible].map(String) }, fixLabel: "Set to missing" }); } }
        else if (NONNEG.test(v.name) && v.type !== "categorical") { const neg = real.filter(x => x < 0); if (neg.length) { impossible = new Set(neg); add("warn", `${v.label} has ${neg.length} negative value${neg.length > 1 ? "s" : ""}, which isn't possible for this kind of variable.`, { var: v.name, fix: { kind: "missing", var: v.name, codes: [...impossible].map(String) }, fixLabel: "Set to missing" }); } }
        // extreme values
        if (v.type === "continuous" && iqr > 0 && real.length >= 20) {
          const lo = q1 - 3 * iqr, hi = q3 + 3 * iqr, ext = real.filter(x => (x < lo || x > hi) && !impossible.has(x));
          if (ext.length) add("info", `${v.label} has ${ext.length} extreme value${ext.length > 1 ? "s" : ""} (beyond 3 IQR from the quartiles: ${[...new Set(ext)].sort((a, b) => a - b).slice(0, 5).join(", ")}). Confirm they are real; QuantAI's assumption checks will then pick rank-based tests if they distort the distribution.`, { var: v.name });
        }
      }
      if (v.levels && v.type !== "binary") { const small = v.levels.filter(l => pres.filter(x => x === l).length < 5); if (small.length && small.length <= 6) add("info", `${v.label}: ${small.length === 1 ? "category" : "categories"} ${small.map(x => `"${x}"`).join(", ")} ${small.length === 1 ? "has" : "have"} fewer than 5 observations. Consider merging small categories before modelling.`, { var: v.name }); }
    });
    // wide layout: the same measurement repeated across columns (bp1, bp2, bp3 / score_t0, score_t1)
    const stems = new Map();
    ds.vars.forEach(v => { const m = v.name.match(/^(.*?)(_?(t|v|visit|wave|time|round|month|m|week|w|day|d|y|year)?_?)(\d{1,2})$/); if (m && m[1].length >= 2 && v.numeric) { const k = m[1].replace(/_$/, ""); if (!stems.has(k)) stems.set(k, []); stems.get(k).push(v.name); } });
    ds.vars.forEach(v => { const m = v.name.match(/^(.+?)_(baseline|\d+_?(months?|weeks?|days?|years?|m|w|d|y))$/); if (m && v.numeric) { if (!stems.has(m[1])) stems.set(m[1], []); if (!stems.get(m[1]).includes(v.name)) stems.get(m[1]).push(v.name); } });
    stems.forEach((cols, k) => {
      const base = ds.vars.find(v => v.name === k); if (base && base.numeric && !cols.includes(k)) cols.unshift(k);
      if (cols.length >= 3 && !cols.every(c => /^q\d|^item\d/.test(c))) add("info", `${cols.slice(0, 4).join(", ")}${cols.length > 4 ? " …" : ""} look like the same measurement taken ${cols.length} times (wide format). Compare two time points with a paired test; for all of them, reshape to one row per person per visit and use a mixed model.`);
    });
    const complete = completeRows(ds, ds.vars.filter(v => v.type !== "id")).length;
    if (complete < 0.7 * n && ds.vars.length > 2) add("info", `Only ${complete} of ${n} rows (${(100 * complete / n).toFixed(0)}%) are complete for every variable. Each analysis uses the rows complete for its own variables, so sample sizes will differ between tables.`);
    const order = { warn: 0, info: 1 };
    return out.sort((a, b) => order[a.sev] - order[b.sev]);
  }

  /** CSV text of the cleaned dataset (clean names, missing as empty) — the file every exported script reads. */
  function toCSV(ds) {
    const q = s => /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    const used = ds.vars.filter(v => v.type !== "id" || true);
    const lines = [used.map(v => v.name).join(",")];
    for (let i = 0; i < ds.nRows; i++) lines.push(used.map(v => v.values[i] === null ? "" : q(v.values[i])).join(","));
    return lines.join("\n") + "\n";
  }

  return { parseDelimited, tidyTable, scaleOrder, nameHint, quality, cleanNames, build, makeVar, setType, setReference, setLevels, num, completeRows, toCSV, isNum, sortLevels, DEFAULT_MISSING };
})();
if (false) module.exports = Data;

/* =====================================================================
   Analysis engine: applies the decision rules, runs the right test,
   records every rule it applied, writes an APA-style summary, and
   emits a spec the code generator turns into Stata/Python/R/SPSS.
   ===================================================================== */
const Analysis = (function () {
  "use strict";
  const S = (typeof Stats !== "undefined") ? Stats : require("./stats.js");
  const D = (typeof Data !== "undefined") ? Data : require("./data.js");

  /* ---------- formatting (APA 7) ---------- */
  const f = (x, d = 2) => (x === null || x === undefined || !isFinite(x)) ? (x === Infinity ? "∞" : "–") : Number(x).toFixed(d);
  const noLead = s => s.replace(/^(-?)0\./, "$1.");
  /** APA p: "p < .001" or "p = .034" */
  const pAPA = p => !isFinite(p) ? "p = –" : p < 0.001 ? "p < .001" : "p = " + noLead(p.toFixed(3));
  const pCell = p => !isFinite(p) ? "–" : p < 0.001 ? "<.001" : noLead(p.toFixed(3));
  const sig = (p, a) => p < a;
  const CONT = ["continuous", "count"], ORDERED = ["continuous", "count", "ordinal"], CAT = ["binary", "categorical", "ordinal"];

  const RULES = {
    alpha: 0.05,
    normality: "Shapiro–Wilk p ≥ .05 in every group → treated as normal. If p < .05 but the group has ≥ 50 values and |skewness| < 1, the mean is still reliable (central limit theorem) and the group is treated as approximately normal.",
    variance: "Brown–Forsythe (median-centred Levene) p ≥ .05 → equal variances assumed; otherwise the Welch version of the test is used.",
    expected: "Chi-square needs expected counts ≥ 5 in at least 80% of cells and none below 1 (Cochran's rule). Otherwise Fisher's exact test is used.",
    epv: "At least 10 events (the rarer outcome category) per predictor parameter for logistic regression; at least 10 observations per parameter for linear and Poisson models.",
    vif: "Variance inflation factor above 5 is a warning and above 10 means predictors are redundant with each other.",
    dispersion: "For Poisson models, Pearson χ²/df above 1.5 indicates overdispersion; robust (sandwich) standard errors are then used automatically.",
    hetero: "Breusch–Pagan (Koenker) p < .05 indicates heteroskedastic residuals; robust (HC1) standard errors are then used automatically.",
    commonOutcome: "When a binary outcome is common (more than 10% have it), an odds ratio overstates the risk/prevalence ratio; a modified Poisson model (robust SEs) reports prevalence ratios instead.",
    posthoc: "Post-hoc pairwise comparisons run only when the overall test is significant, with Bonferroni-adjusted p-values.",
  };

  function groupValues(ds, y, g, rows) {
    return g.levels.map(l => rows.filter(i => g.values[i] === l).map(i => Number(y.values[i])));
  }
  function ordinalValue(v, i) { return v.type === "ordinal" ? v.levels.indexOf(v.values[i]) + 1 : Number(v.values[i]); }

  /** Normality judgement per rule; returns {ok, detail} */
  function judgeNormal(vals, label, alpha) {
    const n = vals.length, sk = n > 2 ? S.skewness(vals) : NaN;
    if (n < 3) return { ok: false, detail: `${label}: n = ${n}, too few values to check normality — treated as not normal`, n };
    if (n > 5000) { const ok = Math.abs(sk) < 1; return { ok, detail: `${label}: n = ${n} (> 5,000), skewness ${f(sk)} → ${ok ? "approximately normal" : "skewed"}`, n, skew: sk }; }
    const sw = S.shapiroWilk(vals);
    if (!isFinite(sw.p)) return { ok: false, detail: `${label}: ${sw.error || "normality could not be assessed"}`, n };
    if (sw.p >= alpha) return { ok: true, detail: `${label}: Shapiro–Wilk W = ${f(sw.W, 3)}, ${pAPA(sw.p)} → normal`, n, W: sw.W, p: sw.p, skew: sk };
    if (n >= 50 && Math.abs(sk) < 1) return { ok: true, detail: `${label}: Shapiro–Wilk ${pAPA(sw.p)}, but n = ${n} and skewness ${f(sk)} → approximately normal (large sample)`, n, W: sw.W, p: sw.p, skew: sk };
    return { ok: false, detail: `${label}: Shapiro–Wilk W = ${f(sw.W, 3)}, ${pAPA(sw.p)}, skewness ${f(sk)} → not normal`, n, W: sw.W, p: sw.p, skew: sk };
  }
  const effectWord = (v, cuts, words) => { const a = Math.abs(v); return a < cuts[0] ? words[0] : a < cuts[1] ? words[1] : a < cuts[2] ? words[2] : words[3]; };
  function boxStats(vals) {
    const q1 = S.quantile(vals, .25), q3 = S.quantile(vals, .75), md = S.median(vals), iqr = q3 - q1;
    const inside = vals.filter(v => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr);
    return { q1, q3, median: md, lo: Math.min(...inside), hi: Math.max(...inside), outliers: vals.filter(v => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr).slice(0, 200), mean: S.mean(vals) };
  }
  const lab = v => v.label && v.label !== v.name ? v.label : v.name;

  /* =================================================================
     1. Continuous / ordinal outcome across groups
     ================================================================= */
  function compareGroups(ds, y, g, opts) {
    opts = opts || {}; const alpha = opts.alpha || RULES.alpha;
    const rows = D.completeRows(ds, [y, g]);
    const levelsAll = g.levels;
    let groups = levelsAll.map(l => rows.filter(i => g.values[i] === l).map(i => ordinalValue(y, i)));
    const keep = groups.map(a => a.length > 0);
    const levels = levelsAll.filter((_, i) => keep[i]); groups = groups.filter((_, i) => keep[i]);
    const k = groups.length, decision = [], warnings = [];
    if (k < 2) throw userError(`"${lab(g)}" has fewer than two groups with data for "${lab(y)}".`);
    if (groups.some(a => a.length < 2)) throw userError(`Every group needs at least 2 values. Check the groups of "${lab(g)}".`);
    decision.push({ rule: "Variable types", detail: `Outcome "${lab(y)}" is ${y.type}; grouping variable "${lab(g)}" has ${k} groups → compare ${k === 2 ? "two independent groups" : k + " independent groups"}.` });
    let parametric, normalAll = true, equalVar = true, lev = null;
    if (y.type === "ordinal") { parametric = false; decision.push({ rule: "Measurement level", detail: "Outcome is ordinal, so means are not meaningful → rank-based (non-parametric) test." }); }
    else {
      const checks = groups.map((a, i) => judgeNormal(a, levels[i], alpha));
      normalAll = checks.every(c => c.ok);
      decision.push({ rule: "Normality (Shapiro–Wilk, per group)", detail: checks.map(c => c.detail).join("; "), ok: normalAll });
      lev = S.leveneBF(groups); equalVar = lev.p >= alpha;
      decision.push({ rule: "Equal variances (Brown–Forsythe)", detail: `F(${lev.df1}, ${lev.df2}) = ${f(lev.F)}, ${pAPA(lev.p)} → ${equalVar ? "equal variances" : "unequal variances"}`, ok: equalVar });
      parametric = normalAll;
    }
    if (opts.force === "parametric" && y.type !== "ordinal") { parametric = true; decision.push({ rule: "Override", detail: "You chose to force the parametric test." , ok: false}); }
    if (opts.force === "nonparametric") { parametric = false; decision.push({ rule: "Override", detail: "You chose to force the non-parametric test.", ok: false }); }

    const desc = groups.map((a, i) => ({ level: levels[i], ...S.describe(a) }));
    let method, test, chosen, posthoc = null, writeup;
    const yl = lab(y), gl = lab(g);
    if (k === 2) {
      if (parametric && equalVar) { chosen = "student"; method = "Independent-samples t-test (Student)"; test = S.tTestStudent(groups[0], groups[1]); }
      else if (parametric) { chosen = "welch"; method = "Welch's t-test (unequal variances)"; test = S.tTestWelch(groups[0], groups[1]); }
      else { chosen = "mwu"; method = "Mann–Whitney U test (Wilcoxon rank-sum)"; test = S.mannWhitney(groups[0], groups[1]); }
    } else {
      if (parametric && equalVar) { chosen = "anova"; method = "One-way ANOVA"; test = S.anova(groups); }
      else if (parametric) { chosen = "welch_anova"; method = "Welch's ANOVA (unequal variances)"; test = S.welchAnova(groups); }
      else { chosen = "kruskal"; method = "Kruskal–Wallis H test"; test = S.kruskal(groups); }
      const pk = chosen === "anova" ? "pooled-t" : chosen === "welch_anova" ? "welch-t" : "mwu";
      if (test.p < alpha) {
        posthoc = { kind: pk, rows: S.pairwise(groups, levels, pk) };
        decision.push({ rule: "Post-hoc", detail: `Overall test significant → pairwise ${pk === "pooled-t" ? "t-tests (pooled variance)" : pk === "welch-t" ? "Welch t-tests" : "Mann–Whitney tests"} with Bonferroni adjustment.` });
      } else decision.push({ rule: "Post-hoc", detail: "Overall test not significant → no pairwise comparisons (avoids false positives)." });
    }
    decision.push({ rule: "Test chosen", detail: method, final: true });

    // tables
    const descTable = { title: `${yl} by ${gl}`, columns: ["Group", "n", "Mean", "SD", "Median", "IQR (Q1–Q3)", "Min–Max"],
      rows: desc.map(d => [d.level, d.n, f(d.mean), f(d.sd), f(d.median), `${f(d.q1)}–${f(d.q3)}`, `${f(d.min)}–${f(d.max)}`]) };
    let testTable, writeupParts = [];
    const [d0, d1] = desc;
    if (chosen === "student" || chosen === "welch") {
      testTable = { title: method, columns: ["t", "df", "p", "Mean difference", "95% CI", "Cohen's d"],
        rows: [[f(test.t), chosen === "welch" ? f(test.df, 1) : test.df, pCell(test.p), f(test.diff), `${f(test.ciLow)} to ${f(test.ciHigh)}`, f(test.d)]] };
      const dir = test.diff > 0 ? "higher" : "lower";
      writeup = `${chosen === "welch" ? "Welch's t-test" : "An independent-samples t-test"} showed that ${yl} was ${sig(test.p, alpha) ? "significantly " : ""}${sig(test.p, alpha) ? dir : "not significantly different"} in ${d0.level} (M = ${f(d0.mean)}, SD = ${f(d0.sd)})${sig(test.p, alpha) ? " than" : " compared with"} ${d1.level} (M = ${f(d1.mean)}, SD = ${f(d1.sd)}), t(${chosen === "welch" ? f(test.df, 2) : test.df}) = ${f(test.t)}, ${pAPA(test.p)}, mean difference = ${f(test.diff)}, 95% CI [${f(test.ciLow)}, ${f(test.ciHigh)}], d = ${f(test.d)} (${effectWord(test.d, [.2, .5, .8], ["negligible", "small", "medium", "large"])} effect).`;
    } else if (chosen === "mwu") {
      testTable = { title: method, columns: ["U", "z", "p", "Effect size r"], rows: [[f(test.U, 1), f(test.z), pCell(test.p), f(test.r)]] };
      writeup = `A Mann–Whitney U test indicated that ${yl} ${sig(test.p, alpha) ? "differed significantly" : "did not differ significantly"} between ${d0.level} (Mdn = ${f(d0.median)}, IQR ${f(d0.q1)}–${f(d0.q3)}) and ${d1.level} (Mdn = ${f(d1.median)}, IQR ${f(d1.q1)}–${f(d1.q3)}), U = ${f(test.U, 1)}, z = ${f(test.z)}, ${pAPA(test.p)}, r = ${f(test.r)} (${effectWord(test.r, [.1, .3, .5], ["negligible", "small", "medium", "large"])} effect).`;
    } else if (chosen === "anova" || chosen === "welch_anova") {
      testTable = { title: method, columns: ["F", "df1", "df2", "p", chosen === "anova" ? "η²" : ""].filter(Boolean),
        rows: [[f(test.F), test.df1, chosen === "welch_anova" ? f(test.df2, 2) : test.df2, pCell(test.p), ...(chosen === "anova" ? [f(test.etaSq, 3)] : [])]] };
      writeup = `${chosen === "anova" ? "A one-way ANOVA" : "Welch's ANOVA"} showed ${sig(test.p, alpha) ? "a significant" : "no significant"} difference in ${yl} across ${gl} groups, F(${test.df1}, ${chosen === "welch_anova" ? f(test.df2, 2) : test.df2}) = ${f(test.F)}, ${pAPA(test.p)}${chosen === "anova" ? `, η² = ${noLead(f(test.etaSq, 3))}` : ""}.`;
    } else {
      testTable = { title: method, columns: ["H", "df", "p", "ε²"], rows: [[f(test.H), test.df, pCell(test.p), noLead(f(test.epsSq, 3))]] };
      writeup = `A Kruskal–Wallis test showed ${sig(test.p, alpha) ? "a significant" : "no significant"} difference in ${yl} across ${gl} groups, H(${test.df}) = ${f(test.H)}, ${pAPA(test.p)}, ε² = ${noLead(f(test.epsSq, 3))}.`;
    }
    const tables = [descTable, testTable];
    if (posthoc) {
      tables.push({ title: "Pairwise comparisons (Bonferroni-adjusted)", columns: ["Comparison", posthoc.kind === "mwu" ? "z" : "t", "p (unadjusted)", "p (Bonferroni)"],
        rows: posthoc.rows.map(r => [`${r.a} vs ${r.b}`, f(r.stat), pCell(r.p), pCell(r.pAdj)]) });
      const sigPairs = posthoc.rows.filter(r => r.pAdj < alpha).map(r => `${r.a} vs ${r.b} (adjusted ${pAPA(r.pAdj)})`);
      writeup += sigPairs.length ? ` Bonferroni-adjusted pairwise comparisons showed significant differences for ${sigPairs.join(", ")}.` : " No pairwise comparison remained significant after Bonferroni adjustment.";
    }
    if (rows.length < ds.nRows) warnings.push(`${ds.nRows - rows.length} row(s) with missing values in "${yl}" or "${gl}" were excluded (complete-case analysis).`);
    return {
      kind: "compare", title: `${yl} by ${gl}`, method, decision, tables, writeup, warnings, n: rows.length,
      chart: { type: "box", ylabel: yl, groups: groups.map((a, i) => ({ label: levels[i], ...boxStats(a) })) },
      spec: { kind: "compare", y: y.name, g: g.name, chosen, posthoc: posthoc ? posthoc.kind : null, alpha, levels, yOrdinal: y.type === "ordinal" },
    };
  }

  /* =================================================================
     2. Correlation
     ================================================================= */
  function correlate(ds, x, y, opts) {
    opts = opts || {}; const alpha = opts.alpha || RULES.alpha;
    const rows = D.completeRows(ds, [x, y]);
    if (rows.length < 4) throw userError("At least 4 complete pairs are needed for a correlation.");
    const xv = rows.map(i => ordinalValue(x, i)), yv = rows.map(i => ordinalValue(y, i));
    const decision = [{ rule: "Variable types", detail: `"${lab(x)}" (${x.type}) and "${lab(y)}" (${y.type}) are both numeric/ordered → correlation.` }];
    let pearsonOk;
    if (x.type === "ordinal" || y.type === "ordinal") { pearsonOk = false; decision.push({ rule: "Measurement level", detail: "At least one variable is ordinal → Spearman's rank correlation." }); }
    else {
      const a = judgeNormal(xv, lab(x), alpha), b = judgeNormal(yv, lab(y), alpha);
      pearsonOk = a.ok && b.ok;
      decision.push({ rule: "Normality (Shapiro–Wilk)", detail: a.detail + "; " + b.detail, ok: pearsonOk });
    }
    if (opts.force === "parametric" && x.type !== "ordinal" && y.type !== "ordinal") { pearsonOk = true; decision.push({ rule: "Override", detail: "You chose to force Pearson's r.", ok: false }); }
    if (opts.force === "nonparametric") { pearsonOk = false; decision.push({ rule: "Override", detail: "You chose to force Spearman's rho.", ok: false }); }
    const chosen = pearsonOk ? "pearson" : "spearman";
    const r = pearsonOk ? S.pearson(xv, yv) : S.spearman(xv, yv);
    const coef = pearsonOk ? r.r : r.rho;
    const method = pearsonOk ? "Pearson correlation" : "Spearman's rank correlation";
    decision.push({ rule: "Test chosen", detail: method, final: true });
    const strength = effectWord(coef, [.1, .3, .5], ["negligible", "weak", "moderate", "strong"]);
    const writeup = `${pearsonOk ? "A Pearson correlation" : "Spearman's rank correlation"} showed ${sig(r.p, alpha) ? "a significant" : "no significant"} ${sig(r.p, alpha) ? strength + " " + (coef > 0 ? "positive" : "negative") + " " : ""}association between ${lab(x)} and ${lab(y)}, ${pearsonOk ? "r" : "rₛ"}(${r.df}) = ${noLead(f(coef))}, ${pAPA(r.p)}${pearsonOk ? `, 95% CI [${noLead(f(r.ciLow))}, ${noLead(f(r.ciHigh))}]` : ""}, n = ${r.n}.`;
    const pts = rows.slice(0, 3000).map((i, k) => [xv[k], yv[k]]);
    return {
      kind: "correlate", title: `${lab(x)} and ${lab(y)}`, method, decision, writeup, warnings: missingWarn(ds, rows, [x, y]), n: rows.length,
      tables: [{ title: method, columns: [pearsonOk ? "r" : "rho", "df", "p", ...(pearsonOk ? ["95% CI"] : []), "n"],
        rows: [[f(coef, 3), r.df, pCell(r.p), ...(pearsonOk ? [`${f(r.ciLow, 3)} to ${f(r.ciHigh, 3)}`] : []), r.n]] }],
      chart: { type: "scatter", xlabel: lab(x), ylabel: lab(y), points: pts },
      spec: { kind: "correlate", x: x.name, y: y.name, chosen },
    };
  }

  /* =================================================================
     3. Cross-tabulation: chi-square / Fisher, OR and RR for 2×2
     ================================================================= */
  function crosstab(ds, rowV, colV, opts) {
    opts = opts || {}; const alpha = opts.alpha || RULES.alpha;
    const rows = D.completeRows(ds, [rowV, colV]);
    const rl = rowV.levels.filter(l => rows.some(i => rowV.values[i] === l)), cl = colV.levels.filter(l => rows.some(i => colV.values[i] === l));
    if (rl.length < 2 || cl.length < 2) throw userError("Both variables need at least two categories with data.");
    const table = rl.map(a => cl.map(b => rows.filter(i => rowV.values[i] === a && colV.values[i] === b).length));
    const chi = S.chiSquare(table), decision = [];
    decision.push({ rule: "Variable types", detail: `"${lab(rowV)}" (${rl.length} categories) × "${lab(colV)}" (${cl.length} categories) → test of association for categorical data.` });
    const cochranOk = chi.pctLow <= 20 && chi.minExpected >= 1;
    decision.push({ rule: "Expected counts (Cochran)", detail: `Smallest expected count ${f(chi.minExpected)}; ${f(chi.pctLow, 0)}% of cells below 5 → ${cochranOk ? "chi-square is valid" : "chi-square unreliable, use Fisher's exact test"}`, ok: cochranOk });
    let useFisher = !cochranOk;
    if (opts.force === "fisher") { useFisher = true; decision.push({ rule: "Override", detail: "You chose Fisher's exact test.", ok: false }); }
    if (opts.force === "chi2") { useFisher = false; decision.push({ rule: "Override", detail: "You chose the chi-square test.", ok: false }); }
    const fis = useFisher ? S.fisherExact(table) : null;
    if (fis && !fis.exact) decision.push({ rule: "Fisher computation", detail: `The table is too large for full enumeration; p is a Monte Carlo estimate from ${fis.sims.toLocaleString()} simulated tables (as R fisher.test(simulate.p.value = TRUE)).` });
    const method = useFisher ? "Fisher's exact test" : "Pearson chi-square test of independence";
    decision.push({ rule: "Test chosen", detail: method, final: true });
    const p = useFisher ? fis.p : chi.p;
    const cs = cl.map((_, j) => S.sum(table.map(r => r[j]))), rs = table.map(r => S.sum(r)), n = S.sum(rs);
    const cnt = { title: `${lab(rowV)} × ${lab(colV)}: n (row %)`, columns: [lab(rowV), ...cl, "Total"],
      rows: [...table.map((r, i) => [rl[i], ...r.map(v => `${v} (${f(100 * v / rs[i], 1)}%)`), rs[i]]), ["Total", ...cs.map(v => `${v} (${f(100 * v / n, 1)}%)`), n]] };
    const testT = { title: method, columns: useFisher ? ["p", "Cramér's V"] : ["χ²", "df", "p", "Cramér's V"],
      rows: [useFisher ? [pCell(p), f(chi.cramersV, 3)] : [f(chi.stat), chi.df, pCell(p), f(chi.cramersV, 3)]] };
    const tables = [cnt, testT];
    let writeup = useFisher
      ? `Fisher's exact test showed ${sig(p, alpha) ? "a significant" : "no significant"} association between ${lab(rowV)} and ${lab(colV)}, ${pAPA(p)}, Cramér's V = ${noLead(f(chi.cramersV))}.`
      : `A chi-square test of independence showed ${sig(p, alpha) ? "a significant" : "no significant"} association between ${lab(rowV)} and ${lab(colV)}, χ²(${chi.df}, N = ${n}) = ${f(chi.stat)}, ${pAPA(p)}, Cramér's V = ${noLead(f(chi.cramersV))}.`;
    let or = null;
    if (rl.length === 2 && cl.length === 2) {
      // rows: [reference, exposed]; cols: [no outcome, outcome]
      const a = table[1][1], b = table[1][0], c = table[0][1], d = table[0][0];
      or = S.twoByTwo(a, b, c, d);
      tables.push({ title: `Effect size: ${lab(rowV)} = ${rl[1]} vs ${rl[0]} for ${lab(colV)} = ${cl[1]}`, columns: ["Measure", "Estimate", "95% CI"],
        rows: [["Odds ratio", f(or.or), `${f(or.orLow)} to ${f(or.orHigh)}`], ["Risk / prevalence ratio", f(or.rr), `${f(or.rrLow)} to ${f(or.rrHigh)}`]],
        note: or.corrected ? "A cell was zero, so 0.5 was added to every cell (Haldane correction) before computing the ratios." : "Woolf (log) confidence intervals." });
      writeup += ` The odds of ${lab(colV)} = ${cl[1]} were ${f(or.or)} times as high for ${rl[1]} as for ${rl[0]} (OR = ${f(or.or)}, 95% CI [${f(or.orLow)}, ${f(or.orHigh)}]).`;
    }
    return {
      kind: "crosstab", title: `${lab(rowV)} × ${lab(colV)}`, method, decision, tables, writeup, warnings: missingWarn(ds, rows, [rowV, colV]), n,
      chart: { type: "stack", rowLabels: rl, colLabels: cl, table },
      spec: { kind: "crosstab", r: rowV.name, c: colV.name, chosen: useFisher ? "fisher" : "chi2", twoByTwo: !!or, exact: fis ? fis.exact : true },
    };
  }

  /* =================================================================
     4. Paired data (two measurements on the same people)
     ================================================================= */
  function paired(ds, a, b, opts) {
    opts = opts || {}; const alpha = opts.alpha || RULES.alpha;
    const rows = D.completeRows(ds, [a, b]), decision = [];
    if (CAT.includes(a.type) && CAT.includes(b.type) && a.type !== "ordinal") {
      const lv = a.levels;
      if (lv.length !== 2 || b.levels.length !== 2 || lv.some(l => !b.levels.includes(l))) throw userError("McNemar's test needs two binary variables with the same two categories (e.g. Yes/No before and after).");
      const [n0, n1] = lv;
      const cnt = (x, y) => rows.filter(i => a.values[i] === x && b.values[i] === y).length;
      const t = [[cnt(n0, n0), cnt(n0, n1)], [cnt(n1, n0), cnt(n1, n1)]];
      const r = S.mcnemar(t[0][1], t[1][0]), nd = t[0][1] + t[1][0];
      const useExact = nd < 25;
      decision.push({ rule: "Variable types", detail: "Two binary measurements on the same individuals → McNemar's test." });
      decision.push({ rule: "Discordant pairs", detail: `${nd} discordant pairs → ${useExact ? "fewer than 25, so the exact binomial p-value is reported" : "25 or more, so the chi-square (continuity-corrected) p-value is reported"}.`, ok: true });
      const p = useExact ? r.pExact : r.p, method = useExact ? "McNemar's test (exact binomial)" : "McNemar's test (χ² with continuity correction)";
      decision.push({ rule: "Test chosen", detail: method, final: true });
      return {
        kind: "paired", title: `${lab(a)} vs ${lab(b)} (paired)`, method, decision, warnings: missingWarn(ds, rows, [a, b]), n: rows.length,
        tables: [{ title: "Paired table", columns: [`${lab(a)} \\ ${lab(b)}`, n0, n1], rows: [[n0, t[0][0], t[0][1]], [n1, t[1][0], t[1][1]]] },
          { title: method, columns: ["χ² (corrected)", "p (χ²)", "p (exact)", "Discordant pairs"], rows: [[f(r.chi2), pCell(r.p), pCell(r.pExact), nd]] }],
        writeup: `McNemar's test showed ${sig(p, alpha) ? "a significant" : "no significant"} change in the proportion with ${n1} from ${lab(a)} (${f(100 * (t[1][0] + t[1][1]) / rows.length, 1)}%) to ${lab(b)} (${f(100 * (t[0][1] + t[1][1]) / rows.length, 1)}%), ${useExact ? "exact " : `χ²(1) = ${f(r.chi2)}, `}${pAPA(p)}, n = ${rows.length}.`,
        spec: { kind: "mcnemar", a: a.name, b: b.name, exact: useExact },
      };
    }
    if (!ORDERED.includes(a.type) || !ORDERED.includes(b.type)) throw userError("Paired comparison needs two numeric (or two binary) measurements.");
    const xv = rows.map(i => ordinalValue(a, i)), yv = rows.map(i => ordinalValue(b, i)), diff = xv.map((v, i) => v - yv[i]);
    decision.push({ rule: "Variable types", detail: "Two numeric measurements on the same individuals → paired comparison of the differences." });
    let normal;
    if (a.type === "ordinal" || b.type === "ordinal") { normal = false; decision.push({ rule: "Measurement level", detail: "Ordinal data → Wilcoxon signed-rank test." }); }
    else { const j = judgeNormal(diff, "Differences", alpha); normal = j.ok; decision.push({ rule: "Normality of the differences", detail: j.detail, ok: normal }); }
    if (opts.force === "parametric") { normal = true; decision.push({ rule: "Override", detail: "You chose the paired t-test.", ok: false }); }
    if (opts.force === "nonparametric") { normal = false; decision.push({ rule: "Override", detail: "You chose the Wilcoxon signed-rank test.", ok: false }); }
    const da = S.describe(xv), db = S.describe(yv), dd = S.describe(diff);
    const descT = { title: "Paired descriptives", columns: ["Measurement", "n", "Mean", "SD", "Median", "IQR"],
      rows: [[lab(a), da.n, f(da.mean), f(da.sd), f(da.median), `${f(da.q1)}–${f(da.q3)}`], [lab(b), db.n, f(db.mean), f(db.sd), f(db.median), `${f(db.q1)}–${f(db.q3)}`], [`Difference (${lab(a)} − ${lab(b)})`, dd.n, f(dd.mean), f(dd.sd), f(dd.median), `${f(dd.q1)}–${f(dd.q3)}`]] };
    if (normal) {
      const t = S.tTestPaired(xv, yv);
      decision.push({ rule: "Test chosen", detail: "Paired-samples t-test", final: true });
      return { kind: "paired", title: `${lab(a)} vs ${lab(b)} (paired)`, method: "Paired-samples t-test", decision, warnings: missingWarn(ds, rows, [a, b]), n: rows.length,
        tables: [descT, { title: "Paired-samples t-test", columns: ["t", "df", "p", "Mean difference", "95% CI", "Cohen's d (dz)"], rows: [[f(t.t), t.df, pCell(t.p), f(t.diff), `${f(t.ciLow)} to ${f(t.ciHigh)}`, f(t.d)]] }],
        writeup: `A paired-samples t-test showed that ${lab(a)} (M = ${f(da.mean)}, SD = ${f(da.sd)}) was ${sig(t.p, alpha) ? "significantly " + (t.diff > 0 ? "higher" : "lower") + " than" : "not significantly different from"} ${lab(b)} (M = ${f(db.mean)}, SD = ${f(db.sd)}), t(${t.df}) = ${f(t.t)}, ${pAPA(t.p)}, mean difference = ${f(t.diff)}, 95% CI [${f(t.ciLow)}, ${f(t.ciHigh)}], dz = ${f(t.d)}.`,
        chart: { type: "box", ylabel: "Value", groups: [{ label: lab(a), ...boxStats(xv) }, { label: lab(b), ...boxStats(yv) }] },
        spec: { kind: "paired", a: a.name, b: b.name, chosen: "paired_t" } };
    }
    const w = S.wilcoxonSignedRank(xv, yv);
    decision.push({ rule: "Test chosen", detail: "Wilcoxon signed-rank test", final: true });
    return { kind: "paired", title: `${lab(a)} vs ${lab(b)} (paired)`, method: "Wilcoxon signed-rank test", decision, warnings: missingWarn(ds, rows, [a, b]).concat(w.nZero ? [`${w.nZero} pair(s) with zero difference were dropped, as is standard for this test.`] : []), n: rows.length,
      tables: [descT, { title: "Wilcoxon signed-rank test", columns: ["V (sum of positive ranks)", "z", "p", "Effect size r", "Non-zero pairs"], rows: [[f(w.V, 1), f(w.z), pCell(w.p), f(w.r), w.n]] }],
      writeup: `A Wilcoxon signed-rank test showed ${sig(w.p, alpha) ? "a significant" : "no significant"} difference between ${lab(a)} (Mdn = ${f(da.median)}) and ${lab(b)} (Mdn = ${f(db.median)}), V = ${f(w.V, 1)}, z = ${f(w.z)}, ${pAPA(w.p)}, r = ${f(w.r)}.`,
      chart: { type: "box", ylabel: "Value", groups: [{ label: lab(a), ...boxStats(xv) }, { label: lab(b), ...boxStats(yv) }] },
      spec: { kind: "paired", a: a.name, b: b.name, chosen: "wilcoxon" } };
  }

  /* =================================================================
     5. Regression models
     ================================================================= */
  function designFor(ds, preds, rows) {
    const terms = preds.map(v => {
      if (CONT.includes(v.type)) return { v, cols: [{ name: lab(v), level: null }] };
      const lv = v.levels.filter(l => rows.some(i => v.values[i] === l));
      return { v, ref: lv[0], cols: lv.slice(1).map(l => ({ name: `${lab(v)}: ${l} (vs ${lv[0]})`, level: l })) };
    });
    const names = terms.flatMap(t => t.cols.map(c => c.name));
    const X = rows.map(i => terms.flatMap(t => t.cols.map(c => c.level === null ? Number(t.v.values[i]) : (t.v.values[i] === c.level ? 1 : 0))));
    return { terms, names, X };
  }
  function regression(ds, y, preds, opts) {
    opts = opts || {}; const alpha = opts.alpha || RULES.alpha;
    if (!preds.length) throw userError("Choose at least one predictor.");
    if (preds.some(p => p.name === y.name)) throw userError("The outcome cannot also be a predictor.");
    const rows = D.completeRows(ds, [y, ...preds]);
    const decision = [], warnings = missingWarn(ds, rows, [y, ...preds]);
    let model = opts.model || "auto";
    const yIsBin = y.type === "binary";
    let yv;
    if (yIsBin) yv = rows.map(i => y.values[i] === y.levels[1] ? 1 : 0);
    else yv = rows.map(i => Number(y.values[i]));
    if (model === "auto") {
      model = yIsBin ? "logistic" : y.type === "count" ? "poisson" : y.type === "continuous" ? "linear" : null;
      if (!model) throw userError(`"${lab(y)}" is ${y.type}. Regression here supports continuous, count or binary outcomes. Change the variable type in the Data tab, or use a non-parametric comparison.`);
      decision.push({ rule: "Model from outcome type", detail: `"${lab(y)}" is ${y.type} → ${({ linear: "multiple linear regression", logistic: "binary logistic regression", poisson: "Poisson regression" })[model]}.` });
    } else decision.push({ rule: "Model", detail: `You chose ${({ linear: "linear", logistic: "logistic", poisson: "Poisson", modpoisson: "modified Poisson (prevalence ratios)" })[model]} regression.` });
    if ((model === "logistic" || model === "modpoisson") && !yIsBin) throw userError(`${model === "logistic" ? "Logistic" : "Modified Poisson"} regression needs a binary outcome.`);
    if (model === "poisson" && yv.some(v => v < 0 || !Number.isInteger(v))) throw userError(`Poisson regression needs non-negative whole-number counts in "${lab(y)}".`);
    if (model === "linear" && yIsBin) throw userError("Linear regression needs a continuous outcome. Use logistic regression for a binary outcome.");
    const { names, X, terms } = designFor(ds, preds, rows);
    const k = names.length, n = rows.length;
    if (n <= k + 1) throw userError(`Only ${n} complete rows for ${k} predictor parameters — too few to fit the model.`);
    terms.forEach(t => { if (t.ref !== undefined) decision.push({ rule: "Reference category", detail: `"${lab(t.v)}": ${t.ref} is the reference; ${t.cols.length} indicator variable(s) created.` }); });

    // sample-size rule
    if (yIsBin) {
      const ev = S.sum(yv), minor = Math.min(ev, n - ev), epv = minor / k;
      decision.push({ rule: "Events per variable", detail: `${minor} events in the rarer outcome category for ${k} parameter(s) → EPV = ${f(epv, 1)} (${epv >= 10 ? "adequate" : "below 10: estimates may be biased and unstable; reduce predictors"})`, ok: epv >= 10 });
      if (epv < 10) warnings.push(`Events per variable is ${f(epv, 1)} (rule of thumb: ≥ 10). Consider fewer predictors.`);
      const prev = ev / n;
      if (model === "logistic" && prev > 0.10) decision.push({ rule: "Common outcome", detail: `${f(100 * prev, 1)}% have ${lab(y)} = ${y.levels[1]} (> 10%) → odds ratios will overstate prevalence/risk ratios. For cross-sectional or cohort data, consider the modified Poisson model (prevalence ratios).`, ok: false });
    } else {
      const ratio = n / k;
      decision.push({ rule: "Observations per parameter", detail: `${n} observations for ${k} parameter(s) → ${f(ratio, 1)} per parameter (${ratio >= 10 ? "adequate" : "below 10: risk of overfitting"})`, ok: ratio >= 10 });
    }
    const vifs = S.vif(X, names);
    if (vifs) {
      const worst = vifs.reduce((m, v) => v.vif > m.vif ? v : m, vifs[0]);
      decision.push({ rule: "Multicollinearity (VIF)", detail: `Highest VIF ${f(worst.vif)} (${worst.name}) → ${worst.vif > 10 ? "serious: predictors are redundant" : worst.vif > 5 ? "moderate: check these predictors" : "no concern"}`, ok: worst.vif <= 5 });
    }
    let fit;
    try {
      if (model === "linear") fit = S.ols(X, yv, names);
      else fit = S.glm(X, yv, names, model === "logistic" ? "binomial" : "poisson", { robust: model === "modpoisson" });
    } catch (e) {
      if (e.code === "singular") throw userError("The predictors are perfectly collinear (one is an exact combination of others, or a category has no variation). Remove a predictor or merge categories.");
      throw e;
    }
    let robust = model === "modpoisson";
    const tables = [], expLabel = { logistic: "OR", poisson: "IRR", modpoisson: "PR" }[model];
    if (model === "linear") {
      if (fit.bp) {
        const het = fit.bp.p < alpha;
        decision.push({ rule: "Homoscedasticity (Breusch–Pagan)", detail: `χ²(${fit.bp.df}) = ${f(fit.bp.lm)}, ${pAPA(fit.bp.p)} → ${het ? "unequal residual variance: robust (HC1) standard errors used" : "constant variance"}`, ok: !het });
        if (het && opts.robustAuto !== false) { fit = S.ols(X, yv, names, { robust: true }); robust = true; }
      }
      if (fit.residNormality) decision.push({ rule: "Normality of residuals", detail: `Shapiro–Wilk W = ${f(fit.residNormality.W, 3)}, ${pAPA(fit.residNormality.p)}${fit.residNormality.p < alpha ? (n >= 100 ? " → not normal, but with n ≥ 100 the coefficient tests remain reliable" : " → not normal; interpret p-values with caution or transform the outcome") : " → normal"}`, ok: fit.residNormality.p >= alpha || n >= 100 });
      decision.push({ rule: "Influential observations (Cook's distance)", detail: `${fit.nCooksHigh} observation(s) above 4/n = ${f(4 / n, 3)}; largest = ${f(fit.maxCook, 3)}${fit.maxCook > 1 ? " → at least one highly influential point; check it" : ""}`, ok: fit.maxCook <= 1 });
    } else if (model === "poisson") {
      const od = fit.dispersion > 1.5;
      decision.push({ rule: "Overdispersion", detail: `Pearson χ²/df = ${f(fit.dispersion)} → ${od ? "overdispersed: robust (sandwich) standard errors used; a negative binomial model is the usual alternative" : "no overdispersion"}`, ok: !od });
      if (od && opts.robustAuto !== false) { fit = S.glm(X, yv, names, "poisson", { robust: true }); robust = true; }
      const zerosObs = yv.filter(v => v === 0).length, zerosExp = S.sum(fit.fitted.map(m => Math.exp(-m)));
      if (zerosObs > 1.5 * zerosExp && zerosObs > 10) decision.push({ rule: "Excess zeros", detail: `${zerosObs} zeros observed vs ${f(zerosExp, 0)} expected → consider a zero-inflated model`, ok: false });
    } else {
      if (fit.separation) { decision.push({ rule: "Separation / convergence", detail: "The model did not converge cleanly or some predicted probabilities are 0 or 1 (separation). Estimates are unreliable; merge sparse categories or use Firth's penalised logistic regression.", ok: false }); warnings.push("Possible complete or quasi-complete separation: odds ratios are unreliable."); }
      else decision.push({ rule: "Convergence", detail: `Converged in ${fit.iterations} iterations; no separation detected.`, ok: true });
      if (model === "logistic") {
        const hl = S.hosmerLemeshow(yv, fit.fitted, 10);
        if (hl) decision.push({ rule: "Goodness of fit (Hosmer–Lemeshow)", detail: `χ²(${hl.df}) = ${f(hl.chi2)}, ${pAPA(hl.p)} → ${hl.p < alpha ? "poor fit" : "adequate fit"}`, ok: hl.p >= alpha });
        fit.hl = hl;
        fit.roc = S.roc(yv, fit.fitted);
        decision.push({ rule: "Discrimination (ROC)", detail: `AUC = ${f(fit.roc.auc, 3)} → ${fit.roc.auc >= .8 ? "excellent" : fit.roc.auc >= .7 ? "acceptable" : "poor"} discrimination`, ok: fit.roc.auc >= .7 });
      }
      if (model === "modpoisson") decision.push({ rule: "Standard errors", detail: "Modified Poisson always uses robust (sandwich, HC0) standard errors (Zou, 2004).", ok: true });
    }
    decision.push({ rule: "Model fitted", detail: ({ linear: "Multiple linear regression (OLS)", logistic: "Binary logistic regression", poisson: "Poisson regression", modpoisson: "Modified Poisson regression (log link, robust SE)" })[model] + (robust && model !== "modpoisson" ? " with robust standard errors" : ""), final: true });

    // crude (unadjusted) estimates, one predictor at a time
    let crude = null;
    if (preds.length > 1) {
      crude = {};
      preds.forEach(pv => {
        const rr = D.completeRows(ds, [y, pv]);
        const dd = designFor(ds, [pv], rr);
        const yy = rr.map(i => yIsBin ? (y.values[i] === y.levels[1] ? 1 : 0) : Number(y.values[i]));
        try {
          const ff = model === "linear" ? S.ols(dd.X, yy, dd.names, { robust }) : S.glm(dd.X, yy, dd.names, model === "logistic" ? "binomial" : "poisson", { robust });
          ff.coefs.slice(1).forEach(c => crude[c.name] = c);
        } catch (e) { /* leave blank */ }
      });
    }
    const coefRows = fit.coefs.slice(1).map(c => {
      const cr = crude && crude[c.name];
      if (model === "linear") return [c.name, ...(crude ? [cr ? `${f(cr.coef)} (${f(cr.ciLow)} to ${f(cr.ciHigh)})` : "–", cr ? pCell(cr.p) : "–"] : []), f(c.coef), f(c.se), `${f(c.ciLow)} to ${f(c.ciHigh)}`, pCell(c.p)];
      return [c.name, ...(crude ? [cr ? `${f(cr.exp)} (${f(cr.expLow)} to ${f(cr.expHigh)})` : "–", cr ? pCell(cr.p) : "–"] : []), f(c.exp), `${f(c.expLow)} to ${f(c.expHigh)}`, pCell(c.p)];
    });
    if (model === "linear") {
      tables.push({ title: "Regression coefficients", columns: ["Predictor", ...(crude ? ["Crude B (95% CI)", "p"] : []), crude ? "Adjusted B" : "B", "SE" + (robust ? " (robust)" : ""), "95% CI", "p"], rows: [["Intercept", ...(crude ? ["", ""] : []), f(fit.coefs[0].coef), f(fit.coefs[0].se), `${f(fit.coefs[0].ciLow)} to ${f(fit.coefs[0].ciHigh)}`, pCell(fit.coefs[0].p)], ...coefRows] });
      tables.push({ title: "Model fit", columns: ["n", "R²", "Adjusted R²", robust ? "Robust F" : "F", "df", "p", "RMSE"], rows: [[n, f(fit.r2, 3), f(fit.adjR2, 3), f(fit.F), `${k}, ${fit.df}`, pCell(fit.fP), f(fit.rmse)]] });
    } else {
      tables.push({ title: `${({ OR: "Odds ratios", IRR: "Incidence rate ratios", PR: "Prevalence ratios" })[expLabel]}`, columns: ["Predictor", ...(crude ? [`Crude ${expLabel} (95% CI)`, "p"] : []), crude ? `Adjusted ${expLabel}` : expLabel, "95% CI", "p"], rows: coefRows,
        note: robust ? "Robust (sandwich) standard errors." : "Wald confidence intervals." });
      tables.push({ title: "Model fit", columns: ["n", ...(yIsBin ? ["Events"] : []), "Log-likelihood", "LR χ²", "df", "p", ...(model === "logistic" ? ["McFadden R²", "AUC", "Hosmer–Lemeshow p"] : ["AIC"])],
        rows: [[n, ...(yIsBin ? [S.sum(yv)] : []), f(fit.ll), f(fit.lr), fit.lrDf, pCell(fit.lrP), ...(model === "logistic" ? [f(fit.mcFadden, 3), f(fit.roc.auc, 3), fit.hl ? pCell(fit.hl.p) : "–"] : [f(fit.aic, 1)])]],
        note: model === "modpoisson" ? "The likelihood of a Poisson model for a binary outcome is not meaningful; use the coefficient table." : null });
    }
    if (vifs) tables.push({ title: "Variance inflation factors", columns: ["Predictor", "VIF"], rows: vifs.map(v => [v.name, f(v.vif)]) });

    // write-up
    const sigC = fit.coefs.slice(1).filter(c => c.p < alpha);
    const adj = preds.length > 1 ? " after adjusting for the other variables in the model" : "";
    let writeup;
    if (model === "linear") {
      writeup = `A multiple linear regression${robust ? " with robust standard errors" : ""} was fitted for ${lab(y)} (n = ${n}). The model ${fit.fP < alpha ? "was statistically significant" : "was not statistically significant"}, F(${k}, ${fit.df}) = ${f(fit.F)}, ${pAPA(fit.fP)}, explaining ${f(100 * fit.r2, 1)}% of the variance (adjusted R² = ${noLead(f(fit.adjR2, 3))}).`;
      writeup += sigC.length ? " " + sigC.map(c => `${c.name} was significantly associated with ${lab(y)}${adj} (B = ${f(c.coef)}, 95% CI [${f(c.ciLow)}, ${f(c.ciHigh)}], ${pAPA(c.p)})`).join("; ") + "." : " No predictor was significantly associated with the outcome.";
    } else {
      const what = { OR: "odds", IRR: "rate", PR: "prevalence" }[expLabel], name = { OR: "AOR", IRR: "aIRR", PR: "aPR" }[expLabel], nm = preds.length > 1 ? name : expLabel;
      writeup = `${({ logistic: "Binary logistic regression", poisson: "Poisson regression", modpoisson: "Modified Poisson regression with robust standard errors" })[model]}${robust && model === "poisson" ? " with robust standard errors" : ""} was used to model ${lab(y)}${yIsBin ? ` (${y.levels[1]} vs ${y.levels[0]})` : ""} (n = ${n}).`;
      writeup += sigC.length ? " " + sigC.map(c => `${c.name} was associated with ${c.exp > 1 ? "higher" : "lower"} ${what} of ${lab(y)}${adj} (${nm} = ${f(c.exp)}, 95% CI [${f(c.expLow)}, ${f(c.expHigh)}], ${pAPA(c.p)})`).join("; ") + "." : " No predictor was significantly associated with the outcome.";
      if (model === "logistic") writeup += ` The model discriminated ${fit.roc.auc >= .8 ? "well" : fit.roc.auc >= .7 ? "acceptably" : "poorly"} (AUC = ${f(fit.roc.auc, 3)})${fit.hl ? ` and the Hosmer–Lemeshow test ${fit.hl.p < alpha ? "indicated poor" : "indicated adequate"} calibration, ${pAPA(fit.hl.p)}` : ""}.`;
    }
    let chart = null;
    if (model === "linear") chart = { type: "residuals", points: fit.fitted.slice(0, 3000).map((v, i) => [v, fit.resid[i]]) };
    else if (model === "logistic") chart = { type: "roc", points: fit.roc.points, auc: fit.roc.auc };
    else chart = { type: "forest", label: expLabel, rows: fit.coefs.slice(1).map(c => ({ name: c.name, est: c.exp, lo: c.expLow, hi: c.expHigh })) };
    return {
      kind: "regression", model, title: `${lab(y)} ~ ${preds.map(lab).join(" + ")}`, method: decision[decision.length - 1].detail, decision, tables, writeup, warnings, n, chart,
      spec: { kind: "regression", model, y: y.name, preds: preds.map(p => p.name), robust, crude: !!crude },
    };
  }

  /* =================================================================
     6. Table 1 — descriptive table, optionally by group with tests
     ================================================================= */
  function table1(ds, vars, group, opts) {
    opts = opts || {}; const alpha = opts.alpha || RULES.alpha;
    const levels = group ? group.levels.filter(l => ds.vars && group.values.some(v => v === l)) : [];
    const cols = ["Characteristic", ...(group ? [...levels.map(l => `${l}`), "Total"] : ["n (%) / summary"]), ...(group ? ["p", "Test"] : [])];
    const rows = [], decision = [], specVars = [];
    const groupN = group ? levels.map(l => group.values.filter(v => v === l).length) : [];
    if (group) cols.splice(1, levels.length + 1, ...levels.map((l, i) => `${l} (n = ${groupN[i]})`), `Total (n = ${S.sum(groupN)})`);
    vars.forEach(v => {
      if (group && v.name === group.name) return;
      const rr = D.completeRows(ds, group ? [v, group] : [v]);
      if (CONT.includes(v.type)) {
        const all = rr.map(i => Number(v.values[i]));
        const byG = group ? levels.map(l => rr.filter(i => group.values[i] === l).map(i => Number(v.values[i]))) : null;
        const checks = (byG || [all]).map((a, i) => judgeNormal(a, group ? levels[i] : lab(v), alpha));
        const normal = checks.every(c => c.ok);
        const summ = a => a.length ? (normal ? `${f(S.mean(a))} (${f(a.length > 1 ? S.sd(a) : NaN)})` : `${f(S.median(a))} (${f(S.quantile(a, .25))}–${f(S.quantile(a, .75))})`) : "–";
        let p = NaN, test = "", chosen = null;
        if (group && byG.filter(a => a.length >= 2).length >= 2) {
          const gg = byG.filter(a => a.length >= 2);
          if (normal) {
            const lev = S.leveneBF(gg), eq = lev.p >= alpha;
            if (gg.length === 2) { const t = eq ? S.tTestStudent(gg[0], gg[1]) : S.tTestWelch(gg[0], gg[1]); p = t.p; test = eq ? "t-test" : "Welch t-test"; chosen = eq ? "student" : "welch"; }
            else { const t = eq ? S.anova(gg) : S.welchAnova(gg); p = t.p; test = eq ? "ANOVA" : "Welch ANOVA"; chosen = eq ? "anova" : "welch_anova"; }
          } else if (gg.length === 2) { p = S.mannWhitney(gg[0], gg[1]).p; test = "Mann–Whitney"; chosen = "mwu"; }
          else { p = S.kruskal(gg).p; test = "Kruskal–Wallis"; chosen = "kruskal"; }
        }
        rows.push([`${lab(v)}, ${normal ? "mean (SD)" : "median (IQR)"}`, ...(group ? [...byG.map(summ), summ(all)] : [summ(all)]), ...(group ? [pCell(p), test] : [])]);
        decision.push({ rule: lab(v), detail: `${normal ? "Normal → mean (SD)" : "Not normal → median (IQR)"}${group ? `; ${test || "no test"}` : ""}` });
        specVars.push({ name: v.name, type: "cont", normal, chosen });
      } else if (CAT.includes(v.type)) {
        const lv = v.levels;
        let p = NaN, test = "", chosen = null;
        if (group) {
          const tab = lv.map(a => levels.map(b => rr.filter(i => v.values[i] === a && group.values[i] === b).length)).filter(r => S.sum(r) > 0);
          const keepC = levels.map((_, j) => S.sum(tab.map(r => r[j])) > 0);
          const t2 = tab.map(r => r.filter((_, j) => keepC[j]));
          if (t2.length >= 2 && t2[0].length >= 2) {
            const chi = S.chiSquare(t2), ok = chi.pctLow <= 20 && chi.minExpected >= 1;
            if (ok) { p = chi.p; test = "χ²"; chosen = "chi2"; } else { p = S.fisherExact(t2).p; test = "Fisher's exact"; chosen = "fisher"; }
          }
        }
        rows.push([`${lab(v)}, n (%)`, ...(group ? levels.map(() => "") : []), "", ...(group ? [pCell(p), test] : [])].slice(0, cols.length));
        lv.forEach(a => {
          const cell = sub => { const nn = sub.length, k = sub.filter(i => v.values[i] === a).length; return nn ? `${k} (${f(100 * k / nn, 1)})` : "–"; };
          const allCell = cell(rr);
          rows.push([`   ${a}`, ...(group ? [...levels.map(l => cell(rr.filter(i => group.values[i] === l))), allCell] : [allCell]), ...(group ? ["", ""] : [])]);
        });
        decision.push({ rule: lab(v), detail: `Categorical → n (column %)${group ? `; ${test || "no test"}` : ""}` });
        specVars.push({ name: v.name, type: "cat", chosen });
      }
    });
    const missingNote = vars.filter(v => v.nMissing > 0).map(v => `${lab(v)}: ${v.nMissing} missing`).join("; ");
    return {
      kind: "table1", title: group ? `Characteristics by ${lab(group)}` : "Characteristics of the sample", method: "Descriptive statistics" + (group ? " with group comparisons" : ""),
      decision: [{ rule: "Summary statistic rule", detail: "Continuous variables: mean (SD) when normal by the Shapiro–Wilk rule, otherwise median (IQR). Categorical: n (column %). Tests follow the same rules as the Compare tool." }, ...decision],
      tables: [{ title: group ? `Table 1. Characteristics by ${lab(group)}` : "Table 1. Characteristics of the sample", columns: cols, rows, note: (missingNote ? "Missing values: " + missingNote + ". " : "") + "Percentages are of non-missing values." }],
      writeup: "", warnings: [], n: ds.nRows,
      spec: { kind: "table1", vars: specVars, group: group ? group.name : null },
    };
  }

  /** Choose the right analysis for an outcome/exposure pair from their types. */
  function auto(ds, outcome, exposure, opts) {
    opts = opts || {};
    if (outcome.name === exposure.name) throw userError("Pick two different variables.");
    if (opts.paired) return paired(ds, outcome, exposure, opts);
    const o = outcome.type, e = exposure.type;
    const isOrd = t => ORDERED.includes(t), isCat = t => ["binary", "categorical"].includes(t);
    if (o === "id" || e === "id") throw userError("ID/text variables can't be analysed. Change the type in the Data tab if this is a mistake.");
    if (isOrd(o) && isCat(e)) return compareGroups(ds, outcome, exposure, opts);
    if (CONT.includes(o) && e === "ordinal") {
      const r = compareGroups(ds, outcome, exposure, opts);
      r.decision.unshift({ rule: "Ordinal exposure", detail: `"${lab(exposure)}" is ordinal, so its categories are compared as groups. To test a monotonic trend instead, use Spearman's correlation (set the exposure type to continuous, or choose "Force non-parametric" with two numeric variables).` });
      return r;
    }
    if (isCat(o) && isOrd(e) && o !== "ordinal") {
      if (e === "ordinal") return crosstab(ds, exposure, outcome, opts);
      const r = compareGroups(ds, exposure, outcome, opts);
      r.decision.unshift({ rule: "Direction", detail: `The outcome is categorical and the exposure numeric, so "${lab(exposure)}" is compared across the "${lab(outcome)}" groups (equivalent test of association). For adjusted effects use Regression.` });
      return r;
    }
    if (isOrd(o) && isOrd(e)) return correlate(ds, exposure, outcome, opts);
    if (CAT.includes(o) && CAT.includes(e)) return crosstab(ds, exposure, outcome, opts);
    throw userError("This combination of variable types isn't supported.");
  }

  function missingWarn(ds, rows, vars) { return rows.length < ds.nRows ? [`${ds.nRows - rows.length} row(s) with a missing value in ${vars.map(v => `"${lab(v)}"`).join(" or ")} were excluded (complete-case analysis).`] : []; }
  function userError(msg) { const e = new Error(msg); e.user = true; return e; }

  return { compareGroups, correlate, crosstab, paired, regression, table1, auto, RULES, fmt: { f, pAPA, pCell, noLead } };
})();
if (false) module.exports = Analysis;

/* =====================================================================
   Analysis extensions: ordinal, multinomial and mixed-effects regression,
   and survival analysis (Kaplan–Meier, log-rank, Cox).
   ===================================================================== */
(function (A) {
  "use strict";
  const S = (typeof Stats !== "undefined") ? Stats : require("./stats_ext.js");
  const D = (typeof Data !== "undefined") ? Data : require("./data.js");
  const { f, pAPA, pCell, noLead } = A.fmt;
  const lab = v => v.label && v.label !== v.name ? v.label : v.name;
  const CONT = ["continuous", "count"];
  const userError = m => Object.assign(new Error(m), { user: true });
  const missingWarn = (ds, rows, vars) => rows.length < ds.nRows ? [`${ds.nRows - rows.length} row(s) with a missing value in ${vars.map(v => `"${lab(v)}"`).join(" or ")} were excluded (complete-case analysis).`] : [];
  function design(preds, rows) {
    const terms = preds.map(v => {
      if (CONT.includes(v.type)) return { v, cols: [{ name: lab(v), level: null }] };
      const lv = v.levels.filter(l => rows.some(i => v.values[i] === l));
      return { v, ref: lv[0], cols: lv.slice(1).map(l => ({ name: `${lab(v)}: ${l} (vs ${lv[0]})`, level: l })) };
    });
    return { terms, names: terms.flatMap(t => t.cols.map(c => c.name)), X: rows.map(i => terms.flatMap(t => t.cols.map(c => c.level === null ? Number(t.v.values[i]) : (t.v.values[i] === c.level ? 1 : 0)))) };
  }
  function vifRule(X, names, decision) {
    const vifs = S.vif(X, names); if (!vifs) return null;
    const worst = vifs.reduce((m, v) => v.vif > m.vif ? v : m, vifs[0]);
    decision.push({ rule: "Multicollinearity (VIF)", detail: `Highest VIF ${f(worst.vif)} (${worst.name}) → ${worst.vif > 10 ? "serious: predictors are redundant" : worst.vif > 5 ? "moderate: check these predictors" : "no concern"}`, ok: worst.vif <= 5 });
    return vifs;
  }
  const refRules = (terms, decision) => terms.forEach(t => { if (t.ref !== undefined) decision.push({ rule: "Reference category", detail: `"${lab(t.v)}": ${t.ref} is the reference; ${t.cols.length} indicator variable(s) created.` }); });

  /* ---------------- ordinal / multinomial / mixed ---------------- */
  function regressionExt(ds, y, preds, opts) {
    opts = opts || {}; const alpha = opts.alpha || 0.05, model = opts.model;
    if (!preds.length) throw userError("Choose at least one predictor.");
    const cluster = opts.cluster ? ds.vars.find(v => v.name === opts.cluster) : null;
    if (model === "mixed" && !cluster) throw userError("A mixed model needs a cluster variable (for example clinic, village, school, or participant ID for repeated measures).");
    if (cluster && preds.some(p => p.name === cluster.name)) throw userError("The cluster variable cannot also be a predictor.");
    const rows = D.completeRows(ds, [y, ...preds]).filter(i => !cluster || cluster.values[i] !== null);
    const decision = [], warnings = missingWarn(ds, rows, [y, ...preds, ...(cluster ? [cluster] : [])]);
    const { names, X, terms } = design(preds, rows), k = names.length, n = rows.length;
    if (n <= k + 2) throw userError(`Only ${n} complete rows for ${k} predictor parameters — too few to fit the model.`);
    refRules(terms, decision);
    const tables = []; let fit, writeup, method, chart = null;
    const adj = preds.length > 1 ? " after adjusting for the other predictors" : "";
    if (model === "ordinal") {
      if (!y.levels || y.levels.length < 3) throw userError("Ordinal regression needs an outcome with at least three ordered categories. Set its type to ordinal in Data & variables and check the order.");
      const lv = y.levels.filter(l => rows.some(i => y.values[i] === l)), yy = rows.map(i => lv.indexOf(y.values[i]));
      decision.push({ rule: "Model", detail: `"${lab(y)}" has ${lv.length} ordered categories (${lv.join(" < ")}) → ordinal logistic regression (proportional odds).` });
      const minCat = Math.min(...lv.map((_, j) => yy.filter(v => v === j).length));
      decision.push({ rule: "Category sizes", detail: `Smallest category has ${minCat} observations${minCat < 10 ? " → sparse; consider merging adjacent categories" : " → adequate"}`, ok: minCat >= 10 });
      decision.push({ rule: "Events per parameter", detail: `${n} observations for ${k} parameter(s) → ${f(n / k, 1)} per parameter${n / k < 10 ? " (low)" : ""}`, ok: n / k >= 10 });
      const vifs = vifRule(X, names, decision);
      try { fit = S.ordinalLogit(X, yy, names, lv); } catch (e) { throw userError("The model could not be fitted. A predictor may perfectly separate the categories; merge sparse categories or remove a predictor."); }
      decision.push({ rule: "Proportional odds", detail: "Assumed: each predictor has the same odds ratio at every cut-point. QuantAI does not test this; the exported Stata (brant) and R code check it. If it fails, use a generalized ordinal or multinomial model.", ok: null });
      decision.push({ rule: "Model fitted", detail: "Ordinal logistic regression (proportional odds)", final: true });
      method = "Ordinal logistic regression";
      tables.push({ title: `Cumulative odds ratios (higher ${lab(y)} category)`, columns: ["Predictor", "OR", "95% CI", "p"], rows: fit.coefs.map(c => [c.name, f(c.exp), `${f(c.expLow)} to ${f(c.expHigh)}`, pCell(c.p)]), note: "OR > 1 means higher odds of being in a higher category. Wald confidence intervals." });
      tables.push({ title: "Cut-points (thresholds)", columns: ["Cut-point", "Estimate", "SE"], rows: fit.thresholds.map(t => [t.name, f(t.coef, 3), f(t.se, 3)]) });
      tables.push({ title: "Model fit", columns: ["n", "Log-likelihood", "LR χ²", "df", "p", "McFadden R²", "AIC"], rows: [[n, f(fit.ll), f(fit.lr), fit.lrDf, pCell(fit.lrP), f(fit.mcFadden, 3), f(fit.aic, 1)]] });
      if (vifs) tables.push({ title: "Variance inflation factors", columns: ["Predictor", "VIF"], rows: vifs.map(v => [v.name, f(v.vif)]) });
      const sig = fit.coefs.filter(c => c.p < alpha);
      writeup = `Ordinal logistic regression (proportional odds) was used to model ${lab(y)} (${lv.join(" < ")}; n = ${n}). The model ${fit.lrP < alpha ? "fitted significantly better than the null model" : "did not fit significantly better than the null model"}, χ²(${fit.lrDf}) = ${f(fit.lr)}, ${pAPA(fit.lrP)}.` + (sig.length ? " " + sig.map(c => `${c.name} was associated with ${c.exp > 1 ? "higher" : "lower"} odds of a higher ${lab(y)} category${adj} (OR = ${f(c.exp)}, 95% CI [${f(c.expLow)}, ${f(c.expHigh)}], ${pAPA(c.p)})`).join("; ") + "." : " No predictor was significantly associated with the outcome.");
      chart = { type: "forest", label: "OR", rows: fit.coefs.map(c => ({ name: c.name, est: c.exp, lo: c.expLow, hi: c.expHigh })) };
    } else if (model === "multinomial") {
      if (!y.levels || y.levels.length < 3) throw userError("Multinomial regression needs a categorical outcome with at least three categories.");
      const lv = y.levels.filter(l => rows.some(i => y.values[i] === l)), yy = rows.map(i => lv.indexOf(y.values[i]));
      decision.push({ rule: "Model", detail: `"${lab(y)}" has ${lv.length} unordered categories → multinomial logistic regression; ${lv[0]} is the reference outcome.` });
      const counts = lv.map((_, j) => yy.filter(v => v === j).length), epv = Math.min(...counts) / (k + 1);
      decision.push({ rule: "Events per variable", detail: `Smallest outcome category has ${Math.min(...counts)} cases for ${k + 1} parameters per equation → EPV = ${f(epv, 1)}${epv < 10 ? " (below 10: estimates may be unstable)" : " (adequate)"}`, ok: epv >= 10 });
      const vifs = vifRule(X, names, decision);
      try { fit = S.multinomialLogit(X, yy, names, lv); } catch (e) { throw userError("The model could not be fitted. A category may be too small or perfectly predicted; merge categories or remove a predictor."); }
      decision.push({ rule: "Convergence", detail: fit.separation ? "Possible separation: some estimates are very large. Merge sparse categories." : "Converged; no separation detected.", ok: !fit.separation });
      decision.push({ rule: "Model fitted", detail: "Multinomial logistic regression", final: true });
      method = "Multinomial logistic regression";
      fit.equations.forEach(eq => tables.push({ title: `${eq.level} vs ${fit.reference}: relative risk ratios`, columns: ["Predictor", "RRR", "95% CI", "p"], rows: eq.coefs.slice(1).map(c => [c.name, f(c.exp), `${f(c.expLow)} to ${f(c.expHigh)}`, pCell(c.p)]) }));
      tables.push({ title: "Model fit", columns: ["n", "Log-likelihood", "LR χ²", "df", "p", "McFadden R²"], rows: [[n, f(fit.ll), f(fit.lr), fit.lrDf, pCell(fit.lrP), f(fit.mcFadden, 3)]] });
      if (vifs) tables.push({ title: "Variance inflation factors", columns: ["Predictor", "VIF"], rows: vifs.map(v => [v.name, f(v.vif)]) });
      const sig = fit.equations.flatMap(eq => eq.coefs.slice(1).filter(c => c.p < alpha).map(c => `${c.name} for ${eq.level} vs ${fit.reference} (RRR = ${f(c.exp)}, 95% CI [${f(c.expLow)}, ${f(c.expHigh)}], ${pAPA(c.p)})`));
      writeup = `Multinomial logistic regression was used to model ${lab(y)} with ${fit.reference} as the reference category (n = ${n}), χ²(${fit.lrDf}) = ${f(fit.lr)}, ${pAPA(fit.lrP)}.` + (sig.length ? ` Significant associations${adj}: ${sig.join("; ")}.` : " No predictor was significantly associated with the outcome.");
    } else {
      if (!CONT.includes(y.type)) throw userError("The linear mixed model needs a continuous outcome.");
      const yy = rows.map(i => Number(y.values[i])), gg = rows.map(i => cluster.values[i]);
      const nG = new Set(gg).size;
      decision.push({ rule: "Model", detail: `Continuous outcome with observations grouped by "${lab(cluster)}" (${nG} clusters) → linear mixed model with a random intercept for each cluster (REML).` });
      decision.push({ rule: "Number of clusters", detail: `${nG} clusters${nG < 10 ? " → too few to estimate between-cluster variance reliably" : nG < 30 ? " → acceptable; variance estimates are imprecise below about 30" : " → adequate"}`, ok: nG >= 10 });
      const vifs = vifRule(X, names, decision);
      try { fit = S.mixedRandomIntercept(X, yy, gg, names); } catch (e) { throw userError("The model could not be fitted. Check that predictors vary and are not perfectly collinear."); }
      decision.push({ rule: "Clustering (ICC)", detail: `Intraclass correlation = ${f(fit.icc, 3)}: ${f(100 * fit.icc, 1)}% of the variance lies between clusters${fit.boundary ? ". The between-cluster variance is estimated at zero, so the model reduces to ordinary regression" : fit.icc < 0.01 ? " → negligible clustering" : " → clustering matters; ordinary regression would understate standard errors"}`, ok: true });
      decision.push({ rule: "Model fitted", detail: "Linear mixed model (random intercept, REML)", final: true });
      method = "Linear mixed model (random intercept)";
      tables.push({ title: "Fixed effects", columns: ["Predictor", "B", "SE", "95% CI", "p"], rows: fit.coefs.map(c => [c.name, f(c.coef), f(c.se), `${f(c.ciLow)} to ${f(c.ciHigh)}`, pCell(c.p)]), note: "Wald z-tests, as Stata mixed and statsmodels MixedLM report." });
      tables.push({ title: "Variance components", columns: ["Component", "Variance", "SD"], rows: [[`Between ${lab(cluster)} (random intercept)`, f(fit.tau2, 3), f(Math.sqrt(fit.tau2), 3)], ["Residual (within cluster)", f(fit.sigma2, 3), f(Math.sqrt(fit.sigma2), 3)], ["Intraclass correlation (ICC)", f(fit.icc, 3), ""]] });
      tables.push({ title: "Model information", columns: ["n", "Clusters", "Cluster size (min–max)", "REML log-likelihood"], rows: [[n, fit.nGroups, `${Math.min(...fit.groupSizes)}–${Math.max(...fit.groupSizes)}`, f(fit.reml)]] });
      if (vifs) tables.push({ title: "Variance inflation factors", columns: ["Predictor", "VIF"], rows: vifs.map(v => [v.name, f(v.vif)]) });
      const sig = fit.coefs.slice(1).filter(c => c.p < alpha);
      writeup = `A linear mixed model with a random intercept for ${lab(cluster)} (${fit.nGroups} clusters, n = ${n}) was fitted for ${lab(y)} using REML. The intraclass correlation was ${noLead(f(fit.icc, 2))}.` + (sig.length ? " " + sig.map(c => `${c.name} was significantly associated with ${lab(y)}${adj} (B = ${f(c.coef)}, 95% CI [${f(c.ciLow)}, ${f(c.ciHigh)}], ${pAPA(c.p)})`).join("; ") + "." : " No predictor was significantly associated with the outcome.");
    }
    return { kind: "regression", model, title: `${lab(y)} ~ ${preds.map(lab).join(" + ")}${cluster ? ` | ${lab(cluster)}` : ""}`, method, decision, tables, writeup, warnings, n, chart,
      spec: { kind: "regression", model, y: y.name, preds: preds.map(p => p.name), cluster: cluster ? cluster.name : null, robust: false, crude: false } };
  }

  /* ---------------- survival ---------------- */
  function survival(ds, timeV, eventV, groupV, covs, opts) {
    opts = opts || {}; const alpha = opts.alpha || 0.05; covs = covs || [];
    if (!CONT.includes(timeV.type)) throw userError("The time variable must be numeric (for example days or months of follow-up).");
    if (eventV.type !== "binary") throw userError("The event variable must be binary (event vs censored). Set its type to binary; the second category is the event.");
    const used = [timeV, eventV, ...(groupV ? [groupV] : []), ...covs];
    const rows = D.completeRows(ds, used).filter(i => Number(timeV.values[i]) >= 0);
    if (rows.length < 5) throw userError("Too few complete rows for survival analysis.");
    const time = rows.map(i => Number(timeV.values[i])), event = rows.map(i => eventV.values[i] === eventV.levels[1] ? 1 : 0);
    const decision = [], tables = [], warnings = missingWarn(ds, rows, used);
    const nEv = event.filter(Boolean).length;
    decision.push({ rule: "Variables", detail: `Time = "${lab(timeV)}"; event = "${lab(eventV)}" = ${eventV.levels[1]} (${eventV.levels[0]} is treated as censored). ${nEv} events among ${rows.length} participants.` });
    if (time.some(t => t === 0)) decision.push({ rule: "Zero times", detail: "Some follow-up times are 0; check they are correct.", ok: false });
    const groups = groupV ? groupV.levels.filter(l => rows.some(i => groupV.values[i] === l)) : [null];
    const kmRows = [], curves = [];
    groups.forEach(g => {
      const ix = rows.map((r, k) => k).filter(k => g === null || groupV.values[rows[k]] === g);
      const km = S.kaplanMeier(ix.map(k => time[k]), ix.map(k => event[k]));
      kmRows.push([g === null ? "All" : g, km.n, km.events, km.median === null ? "not reached" : f(km.median, 1), km.median === null ? "–" : `${km.medianLo === null ? "–" : f(km.medianLo, 1)} to ${km.medianHi === null ? "not reached" : f(km.medianHi, 1)}`]);
      curves.push({ label: g === null ? "All" : g, points: km.table.map(r => [r.t, r.s]) });
    });
    tables.push({ title: "Kaplan–Meier summary", columns: [groupV ? lab(groupV) : "Group", "n", "Events", "Median survival", "95% CI"], rows: kmRows, note: "Greenwood standard errors with log(−log) confidence intervals." });
    let method = "Kaplan–Meier", lr = null, writeup = `Of ${rows.length} participants, ${nEv} had the event.`;
    if (groupV && groups.length >= 2) {
      lr = S.logRank(rows.map((r, k) => time[k]), event, rows.map(i => groupV.values[i]));
      decision.push({ rule: "Comparing curves", detail: `${groups.length} groups → log-rank test, χ²(${lr.df}) = ${f(lr.chi2)}, ${pAPA(lr.p)}`, ok: true });
      tables.push({ title: "Log-rank test", columns: ["Group", "Observed", "Expected"], rows: groups.map(l => { const j = lr.levels.indexOf(l); return [l, lr.observed[j], f(lr.expected[j], 1)]; }), note: `χ²(${lr.df}) = ${f(lr.chi2)}, ${pAPA(lr.p)}` });
      method = "Kaplan–Meier with log-rank test";
      writeup += ` Survival ${lr.p < alpha ? "differed significantly" : "did not differ significantly"} between ${lab(groupV)} groups (log-rank χ²(${lr.df}) = ${f(lr.chi2)}, ${pAPA(lr.p)}).`;
    }
    const coxPreds = [...(groupV ? [groupV] : []), ...covs];
    if (coxPreds.length) {
      const { names, X, terms } = design(coxPreds, rows);
      refRules(terms, decision);
      const epv = nEv / names.length;
      decision.push({ rule: "Events per variable (Cox)", detail: `${nEv} events for ${names.length} parameter(s) → EPV = ${f(epv, 1)}${epv < 10 ? " (below 10: hazard ratios may be unstable)" : " (adequate)"}`, ok: epv >= 10 });
      let cox; try { cox = S.coxPH(X, time, event, names); } catch (e) { throw userError("The Cox model could not be fitted. A predictor may have no events in one category; merge categories or remove it."); }
      if (cox.separation) warnings.push("The Cox model shows signs of separation (very large hazard ratios); a category may have no events.");
      decision.push({ rule: "Proportional hazards", detail: "Assumed: hazard ratios are constant over time. QuantAI does not test this; the exported code runs the Schoenfeld-residual test (estat phtest in Stata, cox.zph in R). Crossing Kaplan–Meier curves are a warning sign.", ok: null });
      decision.push({ rule: "Model fitted", detail: "Cox proportional hazards regression (Efron ties)", final: true });
      method = coxPreds.length > 1 || covs.length ? "Cox proportional hazards regression" : method + " and Cox regression";
      tables.push({ title: "Hazard ratios", columns: ["Predictor", "HR", "95% CI", "p"], rows: cox.coefs.map(c => [c.name, f(c.exp), `${f(c.expLow)} to ${f(c.expHigh)}`, pCell(c.p)]), note: `Efron method for ties. Likelihood-ratio χ²(${cox.lrDf}) = ${f(cox.lr)}, ${pAPA(cox.lrP)}.` });
      const sig = cox.coefs.filter(c => c.p < alpha);
      writeup += sig.length ? " In Cox regression, " + sig.map(c => `${c.name} was associated with a ${c.exp > 1 ? "higher" : "lower"} hazard of ${lab(eventV)}${coxPreds.length > 1 ? " after adjustment" : ""} (HR = ${f(c.exp)}, 95% CI [${f(c.expLow)}, ${f(c.expHigh)}], ${pAPA(c.p)})`).join("; ") + "." : " No predictor was significantly associated with the hazard in Cox regression.";
    } else decision.push({ rule: "Method", detail: "No grouping variable or covariates → overall Kaplan–Meier estimate only.", final: true });
    return { kind: "survival", title: `Survival: ${lab(timeV)}, event ${lab(eventV)}${groupV ? ` by ${lab(groupV)}` : ""}`, method, decision, tables, writeup, warnings, n: rows.length,
      chart: { type: "km", curves, xlabel: lab(timeV) },
      spec: { kind: "survival", time: timeV.name, event: eventV.name, group: groupV ? groupV.name : null, covs: covs.map(c => c.name) } };
  }
  Object.assign(A, { regressionExt, survival });
})(typeof Analysis !== "undefined" ? Analysis : require("./analysis.js"));
if (false) module.exports = require("./analysis.js");

/* =====================================================================
   Code generator: turns analysis specs into runnable Stata, Python,
   R and SPSS scripts that read analysis_data.csv (the cleaned export),
   recode variables exactly as the app did, run the same assumption
   checks, and fit the same model.
   ===================================================================== */
const Codegen = (function () {
  "use strict";
  const LANGS = [
    { id: "stata", label: "Stata", ext: "do" },
    { id: "python", label: "Python", ext: "py" },
    { id: "r", label: "R", ext: "R" },
    { id: "spss", label: "SPSS", ext: "sps" },
  ];
  const CATT = ["binary", "categorical", "ordinal"];
  const qJ = s => JSON.stringify(String(s));                       // Python & R string literal
  const qStata = s => "`\"" + String(s) + "\"'";                    // Stata compound quotes
  const qSpss = s => "'" + String(s).replace(/'/g, "''") + "'";
  const numLit = s => String(Number(s));

  /* ---------- derived variable names ---------- */
  const catName = v => v.name + "_c";
  const isPlain01 = v => v.numeric && v.levels && v.levels[0] === "0" && v.levels[1] === "1" && v.levels.length === 2;
  const binName = v => isPlain01(v) ? v.name : v.name + "_01";
  /** Name for using a variable as a number (ordinal → its rank code). */
  const numName = (lang, v) => v.type === "ordinal" ? ((lang === "python" || lang === "r") ? v.name + "_n" : catName(v)) : v.name;

  function collectNeeds(ds, specs) {
    const by = n => ds.vars.find(v => v.name === n);
    const needs = new Map();
    const need = (name, what) => { const v = by(name); if (!v) return; const e = needs.get(name) || { v, cat: false, bin: false, num: false }; e[what] = true; if (what === "num" && v.type === "ordinal") e.cat = true; needs.set(name, e); };
    specs.forEach(s => {
      if (s.kind === "compare") { need(s.g, "cat"); if (by(s.y).type === "ordinal") need(s.y, "num"); }
      else if (s.kind === "correlate") { [s.x, s.y].forEach(n => { if (by(n).type === "ordinal") need(n, "num"); }); }
      else if (s.kind === "crosstab") { need(s.r, "cat"); need(s.c, "cat"); if (s.twoByTwo) { need(s.r, "bin"); need(s.c, "bin"); } }
      else if (s.kind === "mcnemar") { need(s.a, "bin"); need(s.b, "bin"); need(s.a, "cat"); need(s.b, "cat"); }
      else if (s.kind === "paired") { [s.a, s.b].forEach(n => { if (by(n).type === "ordinal") need(n, "num"); }); }
      else if (s.kind === "regression") { const y = by(s.y); if (s.model === "ordinal" || s.model === "multinomial") need(s.y, "cat"); else if (y.type === "binary") need(s.y, "bin"); s.preds.forEach(n => { if (CATT.includes(by(n).type)) need(n, "cat"); }); }
      else if (s.kind === "survival") { need(s.event, "bin"); if (s.group) need(s.group, "cat"); s.covs.forEach(n => { if (CATT.includes(by(n).type)) need(n, "cat"); }); }
      else if (s.kind === "table1") { if (s.group) need(s.group, "cat"); s.vars.forEach(x => { if (x.type === "cat") need(x.name, "cat"); }); }
    });
    return needs;
  }

  /* =================================================================
     Setup blocks
     ================================================================= */
  function header(lang, meta) {
    const c = lang === "python" || lang === "r" ? "#" : "*";
    const lines = [
      `QuantAI (modelanalysishub.com) — reproducible analysis script (${LANGS.find(l => l.id === lang).label})`,
      `Generated ${meta.date}${meta.source ? ` from "${meta.source}"` : ""}.`,
      `Reads analysis_data.csv (exported with this script: cleaned variable names, missing values left empty).`,
      `Every analysis below recodes variables exactly as the app did, runs the same assumption checks,`,
      `and fits the test the app chose under its rules (significance level α = ${meta.alpha}).`,
    ];
    if (lang === "spss") return lines.map(l => `* ${l.replace(/[.,]$/, "")}.`).join("\n") + "\n";
    return lines.map(l => `${c} ${l}`).join("\n") + "\n";
  }
  function setup(lang, ds, needs) {
    const vs = ds.vars;
    const strVars = vs.filter(v => !v.numeric || CATT.includes(v.type));
    const out = [];
    if (lang === "stata") {
      out.push("version 14", "clear all", "set more off", "",
        `import delimited using "analysis_data.csv", clear varnames(1) case(preserve) bindquote(strict) encoding("utf-8")`);
      out.push("", "* ---- Variable coding (identical to the app) ----");
      needs.forEach(({ v, cat, bin }) => {
        if (cat) {
          out.push(`* ${v.name}: ${v.levels.map((l, i) => `${i + 1} = ${l}`).join(", ")}  (1 = reference)`);
          out.push(`label define ${v.name}_lbl ${v.levels.map((l, i) => `${i + 1} ${qStata(l)}`).join(" ")}, replace`);
          if (v.numeric) { out.push(`recode ${v.name} ${v.levels.map((l, i) => `(${numLit(l)}=${i + 1})`).join(" ")}, generate(${catName(v)})`); out.push(`label values ${catName(v)} ${v.name}_lbl`); }
          else out.push(`encode ${v.name}, generate(${catName(v)}) label(${v.name}_lbl) noextend`);
        }
        if (bin && !isPlain01(v)) {
          out.push(`* ${binName(v)}: 1 = ${v.levels[1]}, 0 = ${v.levels[0]}`);
          out.push(`generate byte ${binName(v)} = .`);
          out.push(`replace ${binName(v)} = 0 if ${v.name} == ${v.numeric ? numLit(v.levels[0]) : qStata(v.levels[0])}`);
          out.push(`replace ${binName(v)} = 1 if ${v.name} == ${v.numeric ? numLit(v.levels[1]) : qStata(v.levels[1])}`);
        }
      });
    } else if (lang === "python") {
      out.push("# Requires: pip install pandas numpy scipy statsmodels",
        "import numpy as np", "import pandas as pd", "from scipy import stats", "import statsmodels.api as sm", "import statsmodels.formula.api as smf",
        "from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence",
        "from statsmodels.stats.diagnostic import het_breuschpagan", "from statsmodels.stats.oneway import anova_oneway", "# Needs scipy >= 1.11 (confidence intervals on t-tests)", "", "pd.set_option(\"display.width\", 160)", "",
        "# Only empty cells are missing; categorical columns are read as text so codes match exactly",
        `df = pd.read_csv("analysis_data.csv", keep_default_na=False, na_values=[""]${strVars.length ? `,\n                 dtype={${strVars.map(v => `${qJ(v.name)}: str`).join(", ")}}` : ""})`,
        "", "# ---- Variable coding (identical to the app; first category = reference) ----");
      needs.forEach(({ v, cat, bin, num }) => {
        if (cat) out.push(`df[${qJ(catName(v))}] = pd.Categorical(df[${qJ(v.name)}], categories=[${v.levels.map(qJ).join(", ")}])`);
        if (num && v.type === "ordinal") out.push(`df[${qJ(numName("python", v))}] = df[${qJ(catName(v))}].cat.codes.replace(-1, np.nan) + 1  # ordinal rank code`);
        if (bin) out.push(isPlain01(v) ? `df[${qJ(v.name)}] = pd.to_numeric(df[${qJ(v.name)}])  # already 0/1` : `df[${qJ(binName(v))}] = df[${qJ(v.name)}].map({${qJ(v.levels[0])}: 0, ${qJ(v.levels[1])}: 1})  # 1 = ${v.levels[1]}`);
      });
      out.push("", "def p_fmt(p):", "    return \"p < .001\" if p < 0.001 else f\"p = {p:.3f}\"");
    } else if (lang === "r") {
      out.push("# Base R only — no extra packages needed.", "options(width = 160)", "",
        "# Only empty cells are missing; categorical columns are read as text so codes match exactly",
        `dat <- read.csv("analysis_data.csv", na.strings = "", stringsAsFactors = FALSE${strVars.length ? `,\n                colClasses = c(${strVars.map(v => `${v.name} = "character"`).join(", ")})` : ""})`,
        "", "# ---- Variable coding (identical to the app; first level = reference) ----");
      needs.forEach(({ v, cat, bin, num }) => {
        if (cat) out.push(`dat$${catName(v)} <- factor(dat$${v.name}, levels = c(${v.levels.map(qJ).join(", ")}))`);
        if (num && v.type === "ordinal") out.push(`dat$${numName("r", v)} <- as.numeric(dat$${catName(v)})  # ordinal rank code`);
        if (bin) out.push(isPlain01(v) ? `dat$${v.name} <- as.numeric(dat$${v.name})  # already 0/1` : `dat$${binName(v)} <- ifelse(dat$${v.name} == ${qJ(v.levels[1])}, 1, ifelse(dat$${v.name} == ${qJ(v.levels[0])}, 0, NA))  # 1 = ${v.levels[1]}`);
      });
    } else if (lang === "spss") {
      const width = v => Math.max(1, ...v.values.filter(x => x !== null).map(x => new TextEncoder().encode(x).length));
      out.push(`GET DATA /TYPE=TXT /FILE="analysis_data.csv" /ENCODING='UTF8'`,
        `  /DELCASE=LINE /DELIMITERS="," /QUALIFIER='"' /ARRANGEMENT=DELIMITED /FIRSTCASE=2`,
        `  /VARIABLES=`,
        ...vs.map(v => `    ${v.name} ${v.numeric ? "F20.0" : "A" + width(v)}`),
        `.`, `DATASET NAME analysis WINDOW=FRONT.`);
      const numVars = vs.filter(v => v.numeric);
      if (numVars.length) out.push(`* Values with a decimal point keep their decimals; F20.0 only avoids implied decimals.`, `FORMATS ${numVars.map(v => v.name).join(" ")} (F12.3).`);
      out.push("", "* ---- Variable coding (identical to the app; code 1 = reference).");
      needs.forEach(({ v, cat, bin }) => {
        if (cat) {
          out.push(`RECODE ${v.name} ${v.levels.map((l, i) => `(${v.numeric ? numLit(l) : qSpss(l)}=${i + 1})`).join(" ")} INTO ${catName(v)}.`);
          out.push(`VALUE LABELS ${catName(v)} ${v.levels.map((l, i) => `${i + 1} ${qSpss(l)}`).join(" ")}.`);
        }
        if (bin && !isPlain01(v)) out.push(`RECODE ${v.name} (${v.numeric ? numLit(v.levels[0]) : qSpss(v.levels[0])}=0) (${v.numeric ? numLit(v.levels[1]) : qSpss(v.levels[1])}=1) INTO ${binName(v)}.`);
      });
      out.push("EXECUTE.");
    }
    return out.join("\n") + "\n";
  }

  /* =================================================================
     Sections
     ================================================================= */
  const by = (ds, n) => ds.vars.find(v => v.name === n);
  const banner = (lang, i, title) => {
    const t = `Analysis ${i}: ${title}`;
    if (lang === "python" || lang === "r") return `\n# ${"=".repeat(68)}\n# ${t}\n# ${"=".repeat(68)}`;
    if (lang === "stata") return `\n* ${"=".repeat(68)}\n* ${t}\n* ${"=".repeat(68)}`;
    return `\n* ${"=".repeat(68)}.\n* ${t}.\n* ${"=".repeat(68)}.`;
  };
  const cmt = (lang, s) => lang === "python" || lang === "r" ? `# ${s}` : lang === "stata" ? `* ${s}` : `* ${s}.`;

  function compareSection(lang, ds, s) {
    const y = by(ds, s.y), g = by(ds, s.g), yn = numName(lang, y), gc = catName(g), L = g.levels.length;
    const name = { student: "Independent-samples t-test", welch: "Welch's t-test", mwu: "Mann–Whitney U test", anova: "One-way ANOVA", welch_anova: "Welch's ANOVA", kruskal: "Kruskal–Wallis test" }[s.chosen];
    const ph = s.posthoc, o = [];
    if (lang === "stata") {
      if (!s.yOrdinal) o.push(cmt(lang, "Assumption checks: Shapiro–Wilk per group (Stata's swilk needs 4–2000 values per group), Brown–Forsythe (W50)"), `bysort ${gc}: swilk ${yn}`, `tabstat ${yn}, by(${gc}) statistics(n mean sd p50 p25 p75 skewness)`, `robvar ${yn}, by(${gc})`);
      o.push(cmt(lang, `Test chosen by the app's rules: ${name}`));
      if (s.chosen === "student") o.push(`ttest ${yn}, by(${gc})`);
      if (s.chosen === "welch") o.push(`ttest ${yn}, by(${gc}) unequal`);
      if (s.chosen === "mwu") o.push(cmt(lang, "Note: ranksum omits the continuity correction the app (and R, Python) apply, so p may differ in the 3rd decimal"), `ranksum ${yn}, by(${gc})`);
      if (s.chosen === "anova") o.push(`oneway ${yn} ${gc}, tabulate${ph ? " bonferroni" : ""}`, `quietly anova ${yn} ${gc}`, `estat esize`);
      if (s.chosen === "welch_anova") o.push(cmt(lang, "Stata has no built-in Welch ANOVA; computed from its formula (identical to R oneway.test)"),
        `preserve`, `keep if !missing(${yn}, ${gc})`, `collapse (mean) m=${yn} (sd) s=${yn} (count) n=${yn}, by(${gc})`,
        `generate w = n/s^2`, `quietly summarize w`, `scalar sw = r(sum)`, `scalar k = r(N)`, `generate wm = w*m`, `quietly summarize wm`, `scalar mw = r(sum)/sw`,
        `generate a = w*(m-mw)^2`, `quietly summarize a`, `scalar A = r(sum)/(k-1)`, `generate tt = (1-w/sw)^2/(n-1)`, `quietly summarize tt`, `scalar tmp = r(sum)`,
        `scalar F = A/(1 + 2*(k-2)/(k^2-1)*tmp)`, `scalar df2 = (k^2-1)/(3*tmp)`, `display "Welch F(" k-1 ", " %6.2f df2 ") = " %6.3f F ",  p = " %6.4f Ftail(k-1, df2, F)`, `restore`);
      if (s.chosen === "kruskal") o.push(`kwallis ${yn}, by(${gc})`);
      if (ph && ph !== "pooled-t") {
        o.push(cmt(lang, `Post-hoc: pairwise ${ph === "welch-t" ? "Welch t-tests" : "Mann–Whitney tests"}, Bonferroni-adjusted`), `local m = ${L * (L - 1) / 2}`,
          `forvalues i = 1/${L - 1} {`, `  forvalues j = \`=\`i'+1'/${L} {`,
          ph === "welch-t" ? `    quietly ttest ${yn} if inlist(${gc}, \`i', \`j'), by(${gc}) unequal` : `    quietly ranksum ${yn} if inlist(${gc}, \`i', \`j'), by(${gc})`,
          ph === "welch-t" ? `    local p = r(p)` : `    local p = 2*normal(-abs(r(z)))`,
          `    display "\`: label ${g.name}_lbl \`i'' vs \`: label ${g.name}_lbl \`j'':  p = " %6.4f \`p' "   Bonferroni p = " %6.4f min(1, \`p'*\`m')`, `  }`, `}`);
      }
    } else if (lang === "python") {
      o.push(`d = df[[${qJ(yn)}, ${qJ(gc)}]].dropna()`,
        `labels = [c for c in d[${qJ(gc)}].cat.categories if (d[${qJ(gc)}] == c).any()]`,
        `groups = [d.loc[d[${qJ(gc)}] == c, ${qJ(yn)}].to_numpy(dtype=float) for c in labels]`,
        `for lab, x in zip(labels, groups):`,
        `    print(f"{lab}: n={len(x)}, mean={x.mean():.2f}, SD={x.std(ddof=1):.2f}, median={np.median(x):.2f}, IQR={np.percentile(x,25):.2f}-{np.percentile(x,75):.2f}")`);
      if (!s.yOrdinal) o.push("# Assumption checks", `for lab, x in zip(labels, groups):`,
        `    sw = stats.shapiro(x) if 3 <= len(x) <= 5000 else None`,
        `    print(f"  {lab}: Shapiro-Wilk W={sw.statistic:.3f}, {p_fmt(sw.pvalue)}" if sw else f"  {lab}: n outside 3-5000", f"skewness={stats.skew(x):.2f}")`,
        `print("Brown-Forsythe (Levene, median):", stats.levene(*groups, center="median"))`);
      o.push(`# Test chosen by the app's rules: ${name}`);
      if (s.chosen === "student" || s.chosen === "welch") o.push(`r = stats.ttest_ind(groups[0], groups[1], equal_var=${s.chosen === "student" ? "True" : "False"})`, `ci = r.confidence_interval()`,
        `sp = np.sqrt(((len(groups[0])-1)*groups[0].var(ddof=1) + (len(groups[1])-1)*groups[1].var(ddof=1)) / (len(groups[0])+len(groups[1])-2))`,
        `print(f"t({r.df:.2f}) = {r.statistic:.2f}, {p_fmt(r.pvalue)}, mean difference = {groups[0].mean()-groups[1].mean():.2f}, 95% CI [{ci.low:.2f}, {ci.high:.2f}], d = {(groups[0].mean()-groups[1].mean())/sp:.2f}")`);
      if (s.chosen === "mwu") o.push(`r = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided", method="asymptotic", use_continuity=True)`, `print(f"U = {r.statistic:.1f}, {p_fmt(r.pvalue)}")`);
      if (s.chosen === "anova") o.push(`r = stats.f_oneway(*groups)`, `allv = np.concatenate(groups); ssb = sum(len(x)*(x.mean()-allv.mean())**2 for x in groups); ssw = sum(((x-x.mean())**2).sum() for x in groups)`,
        `print(f"F({len(groups)-1}, {len(allv)-len(groups)}) = {r.statistic:.2f}, {p_fmt(r.pvalue)}, eta squared = {ssb/(ssb+ssw):.3f}")`);
      if (s.chosen === "welch_anova") o.push(`r = anova_oneway(groups, use_var="unequal", welch_correction=True)`, `print(f"Welch F({r.df[0]:.0f}, {r.df[1]:.2f}) = {r.statistic:.2f}, {p_fmt(r.pvalue)}")`);
      if (s.chosen === "kruskal") o.push(`r = stats.kruskal(*groups)`, `N = sum(len(x) for x in groups)`, `print(f"H({len(groups)-1}) = {r.statistic:.2f}, {p_fmt(r.pvalue)}, epsilon squared = {r.statistic/((N**2-1)/(N+1)):.3f}")`);
      if (ph) {
        o.push(`# Post-hoc: Bonferroni-adjusted pairwise comparisons`, `pairs = [(i, j) for i in range(len(groups)) for j in range(i+1, len(groups))]`);
        if (ph === "pooled-t") o.push(`N = sum(len(x) for x in groups); k = len(groups); mse = sum(((x-x.mean())**2).sum() for x in groups)/(N-k)`,
          `for i, j in pairs:`, `    t = (groups[i].mean()-groups[j].mean())/np.sqrt(mse*(1/len(groups[i])+1/len(groups[j])))`, `    p = 2*stats.t.sf(abs(t), N-k)`,
          `    print(f"{labels[i]} vs {labels[j]}: t = {t:.2f}, {p_fmt(p)}, Bonferroni {p_fmt(min(1, p*len(pairs)))}")`);
        else o.push(`for i, j in pairs:`, ph === "welch-t" ? `    p = stats.ttest_ind(groups[i], groups[j], equal_var=False).pvalue` : `    p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided", method="asymptotic", use_continuity=True).pvalue`,
          `    print(f"{labels[i]} vs {labels[j]}: {p_fmt(p)}, Bonferroni {p_fmt(min(1, p*len(pairs)))}")`);
      }
    } else if (lang === "r") {
      o.push(`d <- na.omit(dat[, c("${yn}", "${gc}")]); d$${gc} <- droplevels(d$${gc})`,
        `aggregate(${yn} ~ ${gc}, data = d, FUN = function(x) c(n = length(x), mean = mean(x), sd = sd(x), median = median(x), q1 = unname(quantile(x, .25)), q3 = unname(quantile(x, .75))))`);
      if (!s.yOrdinal) o.push("# Assumption checks", `tapply(d$${yn}, d$${gc}, function(x) if (length(x) >= 3 && length(x) <= 5000) shapiro.test(x)$p.value else NA)  # Shapiro-Wilk p per group`,
        `anova(lm(abs(${yn} - ave(${yn}, ${gc}, FUN = median)) ~ ${gc}, data = d))  # Brown-Forsythe test`);
      o.push(`# Test chosen by the app's rules: ${name}`);
      if (s.chosen === "student") o.push(`t.test(${yn} ~ ${gc}, data = d, var.equal = TRUE)`);
      if (s.chosen === "welch") o.push(`t.test(${yn} ~ ${gc}, data = d, var.equal = FALSE)`);
      if (s.chosen === "mwu") o.push(`wilcox.test(${yn} ~ ${gc}, data = d, exact = FALSE, correct = TRUE)`);
      if (s.chosen === "anova") o.push(`fit <- aov(${yn} ~ ${gc}, data = d); summary(fit)`, `ss <- summary(fit)[[1]][["Sum Sq"]]; cat("eta squared =", round(ss[1]/sum(ss), 3), "\\n")`);
      if (s.chosen === "welch_anova") o.push(`oneway.test(${yn} ~ ${gc}, data = d, var.equal = FALSE)`);
      if (s.chosen === "kruskal") o.push(`kruskal.test(${yn} ~ ${gc}, data = d)`);
      if (ph === "pooled-t") o.push("# Post-hoc: pooled-SD t-tests, Bonferroni", `pairwise.t.test(d$${yn}, d$${gc}, pool.sd = TRUE, p.adjust.method = "bonferroni")`);
      if (ph === "welch-t") o.push("# Post-hoc: Welch t-tests, Bonferroni", `pairwise.t.test(d$${yn}, d$${gc}, pool.sd = FALSE, p.adjust.method = "bonferroni")`);
      if (ph === "mwu") o.push("# Post-hoc: Mann-Whitney tests, Bonferroni", `pairwise.wilcox.test(d$${yn}, d$${gc}, p.adjust.method = "bonferroni", exact = FALSE, correct = TRUE)`);
    } else if (lang === "spss") {
      if (!s.yOrdinal) o.push(cmt(lang, "Assumption checks: Shapiro-Wilk (Tests of Normality table) and Levene based on median (Test of Homogeneity of Variance)"),
        `EXAMINE VARIABLES=${yn} BY ${gc} /PLOT BOXPLOT NPPLOT SPREADLEVEL /STATISTICS DESCRIPTIVES /PERCENTILES(25,50,75) /NOTOTAL.`);
      o.push(cmt(lang, `Test chosen by the app's rules: ${name}`));
      if (s.chosen === "student" || s.chosen === "welch") o.push(cmt(lang, `Read the "${s.chosen === "student" ? "Equal variances assumed" : "Equal variances not assumed"}" row`), `T-TEST GROUPS=${gc}(1 2) /VARIABLES=${yn} /CRITERIA=CI(.95).`);
      if (s.chosen === "mwu") o.push(cmt(lang, "SPSS's asymptotic p omits the continuity correction, so it may differ in the 3rd decimal"), `NPAR TESTS /M-W=${yn} BY ${gc}(1 2).`);
      if (s.chosen === "anova" || s.chosen === "welch_anova") o.push(s.chosen === "welch_anova" ? cmt(lang, "Read the Welch row of Robust Tests of Equality of Means. SPSS's unequal-variance post-hoc is Games-Howell; the app uses Bonferroni-adjusted Welch t-tests") : cmt(lang, "Bonferroni post-hoc uses the pooled variance, as the app"),
        `ONEWAY ${yn} BY ${gc} /STATISTICS DESCRIPTIVES HOMOGENEITY WELCH${ph ? ` /POSTHOC=${ph === "pooled-t" ? "BONFERRONI" : "GH"} ALPHA(${s.alpha})` : ""}.`);
      if (s.chosen === "kruskal") {
        o.push(`NPAR TESTS /K-W=${yn} BY ${gc}(1 ${L}).`);
        if (ph) { o.push(cmt(lang, `Post-hoc: Mann-Whitney for each pair; multiply each p by ${L * (L - 1) / 2} (Bonferroni)`)); for (let i = 1; i < L; i++) for (let j = i + 1; j <= L; j++) o.push(`NPAR TESTS /M-W=${yn} BY ${gc}(${i} ${j}).`); }
      }
    }
    return o.join("\n");
  }

  function correlateSection(lang, ds, s) {
    const x = by(ds, s.x), y = by(ds, s.y), xn = numName(lang, x), yn = numName(lang, y), pear = s.chosen === "pearson", o = [];
    if (lang === "stata") {
      if (x.type !== "ordinal" && y.type !== "ordinal") o.push(cmt(lang, "Assumption check: Shapiro–Wilk on each variable"), `swilk ${xn} ${yn} if !missing(${xn}, ${yn})`);
      o.push(cmt(lang, `Test chosen by the app's rules: ${pear ? "Pearson" : "Spearman"} correlation`), pear ? `pwcorr ${xn} ${yn}, sig obs` : `spearman ${xn} ${yn}`);
    } else if (lang === "python") {
      o.push(`d = df[[${qJ(xn)}, ${qJ(yn)}]].dropna().astype(float)`);
      if (x.type !== "ordinal" && y.type !== "ordinal") o.push(`print("Shapiro-Wilk:", stats.shapiro(d[${qJ(xn)}]), stats.shapiro(d[${qJ(yn)}]))`);
      o.push(`# Test chosen by the app's rules: ${pear ? "Pearson" : "Spearman"} correlation`);
      o.push(pear ? `r = stats.pearsonr(d[${qJ(xn)}], d[${qJ(yn)}]); ci = r.confidence_interval()\nprint(f"r({len(d)-2}) = {r.statistic:.3f}, {p_fmt(r.pvalue)}, 95% CI [{ci.low:.3f}, {ci.high:.3f}], n = {len(d)}")`
        : `r = stats.spearmanr(d[${qJ(xn)}], d[${qJ(yn)}])\nprint(f"rho({len(d)-2}) = {r.statistic:.3f}, {p_fmt(r.pvalue)}, n = {len(d)}")`);
    } else if (lang === "r") {
      o.push(`d <- na.omit(dat[, c("${xn}", "${yn}")])`);
      if (x.type !== "ordinal" && y.type !== "ordinal") o.push(`shapiro.test(d$${xn}); shapiro.test(d$${yn})`);
      o.push(`# Test chosen by the app's rules: ${pear ? "Pearson" : "Spearman"} correlation`, pear ? `cor.test(d$${xn}, d$${yn}, method = "pearson")` : `cor.test(d$${xn}, d$${yn}, method = "spearman", exact = FALSE)`);
    } else {
      if (x.type !== "ordinal" && y.type !== "ordinal") o.push(`EXAMINE VARIABLES=${xn} ${yn} /PLOT NPPLOT /MISSING=LISTWISE.`);
      o.push(cmt(lang, `Test chosen by the app's rules: ${pear ? "Pearson" : "Spearman"} correlation`), pear ? `CORRELATIONS /VARIABLES=${xn} ${yn} /PRINT=TWOTAIL NOSIG /MISSING=LISTWISE.` : `NONPAR CORR /VARIABLES=${xn} ${yn} /PRINT=SPEARMAN TWOTAIL NOSIG /MISSING=LISTWISE.`);
    }
    return o.join("\n");
  }

  function crosstabSection(lang, ds, s) {
    const r = by(ds, s.r), c = by(ds, s.c), rc = catName(r), cc = catName(c), fisher = s.chosen === "fisher", o = [];
    const testName = fisher ? "Fisher's exact test" : "Pearson chi-square (no continuity correction)";
    if (lang === "stata") {
      o.push(cmt(lang, "Counts, row %, and expected counts (Cochran's rule: ≥ 80% of expected counts ≥ 5, none < 1)"), `tabulate ${rc} ${cc}, row expected chi2${fisher ? " exact" : ""}`, cmt(lang, `Test chosen by the app's rules: ${testName}${fisher ? " — read 'Fisher's exact' (two-sided)" : ""}`));
      if (s.twoByTwo) o.push(cmt(lang, `Odds ratio (Woolf CI) and risk/prevalence ratio: ${binName(r)} = exposure, ${binName(c)} = outcome`), `cc ${binName(c)} ${binName(r)}, woolf`, `cs ${binName(c)} ${binName(r)}`);
    } else if (lang === "python") {
      o.push(`d = df[[${qJ(rc)}, ${qJ(cc)}]].dropna()`, `d[${qJ(rc)}] = d[${qJ(rc)}].cat.remove_unused_categories(); d[${qJ(cc)}] = d[${qJ(cc)}].cat.remove_unused_categories()`, `tab = pd.crosstab(d[${qJ(rc)}], d[${qJ(cc)}])`, `print(tab)`, `print((pd.crosstab(d[${qJ(rc)}], d[${qJ(cc)}], normalize="index")*100).round(1))`,
        `chi2, p, dof, expected = stats.chi2_contingency(tab, correction=False)`, `print("Expected counts:\\n", np.round(expected, 2))`,
        `print(f"min expected = {expected.min():.2f}; % cells < 5 = {100*(expected < 5).mean():.0f}%")`, `# Test chosen by the app's rules: ${testName}`);
      if (!fisher) o.push(`n = tab.to_numpy().sum(); V = np.sqrt(chi2/(n*(min(tab.shape)-1)))`, `print(f"chi2({dof}, N={n}) = {chi2:.2f}, {p_fmt(p)}, Cramer's V = {V:.3f}")`);
      else if (s.twoByTwo) o.push(`print("Fisher's exact:", p_fmt(stats.fisher_exact(tab.to_numpy())[1]))`);
      else o.push(`# scipy has no Fisher's exact test beyond 2x2 tables: run the R or Stata script for this p-value.`, `print("(chi-square shown for reference only)", p_fmt(p))`);
      if (s.twoByTwo) o.push(`# Odds ratio and risk/prevalence ratio: exposed row = ${r.levels[1]}, outcome column = ${c.levels[1]}`,
        `t2 = sm.stats.Table2x2(tab.to_numpy()[::-1, ::-1], shift_zeros=True)`, `print(f"OR = {t2.oddsratio:.2f}, 95% CI {np.round(t2.oddsratio_confint(), 2)}; RR = {t2.riskratio:.2f}, 95% CI {np.round(t2.riskratio_confint(), 2)}")`);
    } else if (lang === "r") {
      o.push(`d <- na.omit(dat[, c("${rc}", "${cc}")])`, `tab <- table(droplevels(d$${rc}), droplevels(d$${cc}), dnn = c("${r.name}", "${c.name}")); tab`, `round(prop.table(tab, 1) * 100, 1)`,
        `ct <- suppressWarnings(chisq.test(tab, correct = FALSE)); round(ct$expected, 2)`, `cat("min expected =", round(min(ct$expected), 2), "; % cells < 5 =", round(100 * mean(ct$expected < 5)), "%\\n")`, `# Test chosen by the app's rules: ${testName}`);
      if (!fisher) o.push(`ct`, `cat("Cramer's V =", round(sqrt(ct$statistic / (sum(tab) * (min(dim(tab)) - 1))), 3), "\\n")`);
      else o.push(s.exact ? `fisher.test(tab)` : `fisher.test(tab, simulate.p.value = TRUE, B = 20000)  # table too large for full enumeration`);
      if (s.twoByTwo) o.push(`# Odds ratio and risk ratio (Woolf/log CIs; 0.5 added to every cell if any is zero)`,
        `a <- tab[2, 2]; b <- tab[2, 1]; c <- tab[1, 2]; dd <- tab[1, 1]`, `if (any(c(a, b, c, dd) == 0)) { a <- a + .5; b <- b + .5; c <- c + .5; dd <- dd + .5 }`,
        `or <- a * dd / (b * c); se <- sqrt(1/a + 1/b + 1/c + 1/dd)`, `rr <- (a/(a+b)) / (c/(c+dd)); se_rr <- sqrt(1/a - 1/(a+b) + 1/c - 1/(c+dd))`,
        `cat(sprintf("OR = %.2f (95%% CI %.2f to %.2f); RR = %.2f (95%% CI %.2f to %.2f)\\n", or, exp(log(or) - 1.959964*se), exp(log(or) + 1.959964*se), rr, exp(log(rr) - 1.959964*se_rr), exp(log(rr) + 1.959964*se_rr)))`);
    } else {
      o.push(cmt(lang, `Test chosen by the app's rules: ${testName}. Read the "Pearson Chi-Square" or "Fisher's Exact Test" row`),
        `CROSSTABS /TABLES=${rc} BY ${cc} /STATISTICS=CHISQ PHI${s.twoByTwo ? " RISK" : ""} /CELLS=COUNT ROW EXPECTED${fisher && !s.twoByTwo ? " /METHOD=EXACT TIMER(5)" : ""}.`);
      if (s.twoByTwo) o.push(cmt(lang, "The Risk Estimate table's odds ratio is for the exposed row vs the reference row, as in the app"));
    }
    return o.join("\n");
  }

  function pairedSection(lang, ds, s) {
    const a = by(ds, s.a), b = by(ds, s.b), o = [];
    if (s.kind === "mcnemar") {
      const an = binName(a), bn = binName(b);
      if (lang === "stata") o.push(cmt(lang, `McNemar's test (${s.exact ? "read the exact McNemar significance" : "chi-square"}). Note: Stata's McNemar chi2 has no continuity correction`), `tabulate ${an} ${bn}`, `mcc ${an} ${bn}`);
      if (lang === "python") o.push(`from statsmodels.stats.contingency_tables import mcnemar`, `d = df[[${qJ(an)}, ${qJ(bn)}]].dropna()`, `tab = pd.crosstab(d[${qJ(an)}], d[${qJ(bn)}]).reindex(index=[0, 1], columns=[0, 1], fill_value=0); print(tab)`,
        `print("McNemar (chi-square, continuity-corrected):", mcnemar(tab.to_numpy(), exact=False, correction=True))`, `print("McNemar (exact binomial):", mcnemar(tab.to_numpy(), exact=True))`, `# The app reports the ${s.exact ? "exact" : "chi-square"} version (rule: exact when discordant pairs < 25)`);
      if (lang === "r") o.push(`d <- na.omit(dat[, c("${an}", "${bn}")])`, `tab <- table(factor(d$${an}, levels = 0:1), factor(d$${bn}, levels = 0:1)); tab`, `mcnemar.test(tab, correct = TRUE)`, `binom.test(min(tab[1, 2], tab[2, 1]), tab[1, 2] + tab[2, 1])  # exact version`, `# The app reports the ${s.exact ? "exact" : "chi-square"} version (rule: exact when discordant pairs < 25)`);
      if (lang === "spss") o.push(`NPAR TESTS /MCNEMAR=${an} WITH ${bn} (PAIRED).`, `CROSSTABS /TABLES=${an} BY ${bn} /STATISTICS=MCNEMAR /CELLS=COUNT.`);
      return o.join("\n");
    }
    const an = numName(lang, a), bn = numName(lang, b), t = s.chosen === "paired_t";
    if (lang === "stata") o.push(`generate double _diff = ${an} - ${bn}`, cmt(lang, "Assumption check: normality of the differences"), `swilk _diff`, cmt(lang, `Test chosen by the app's rules: ${t ? "paired t-test" : "Wilcoxon signed-rank (Stata keeps zero differences and omits the continuity correction, so p may differ slightly)"}`), t ? `ttest ${an} == ${bn}` : `signrank ${an} = ${bn}`, `drop _diff`);
    if (lang === "python") o.push(`d = df[[${qJ(an)}, ${qJ(bn)}]].dropna().astype(float)`, `diff = d[${qJ(an)}] - d[${qJ(bn)}]`, `print("Shapiro-Wilk on differences:", stats.shapiro(diff))`,
      t ? `r = stats.ttest_rel(d[${qJ(an)}], d[${qJ(bn)}]); ci = r.confidence_interval()\nprint(f"t({r.df}) = {r.statistic:.2f}, {p_fmt(r.pvalue)}, mean difference = {diff.mean():.2f}, 95% CI [{ci.low:.2f}, {ci.high:.2f}], dz = {diff.mean()/diff.std(ddof=1):.2f}")`
        : `r = stats.wilcoxon(d[${qJ(an)}], d[${qJ(bn)}], zero_method="wilcox", correction=True, method="approx")\nprint(f"Wilcoxon signed-rank: {p_fmt(r.pvalue)}")`);
    if (lang === "r") o.push(`d <- na.omit(dat[, c("${an}", "${bn}")])`, `shapiro.test(d$${an} - d$${bn})`, t ? `t.test(d$${an}, d$${bn}, paired = TRUE)` : `wilcox.test(d$${an}, d$${bn}, paired = TRUE, exact = FALSE, correct = TRUE)`);
    if (lang === "spss") o.push(`COMPUTE diff_ = ${an} - ${bn}.`, `EXAMINE VARIABLES=diff_ /PLOT NPPLOT.`, t ? `T-TEST PAIRS=${an} WITH ${bn} (PAIRED) /CRITERIA=CI(.95).` : `NPAR TESTS /WILCOXON=${an} WITH ${bn} (PAIRED).`);
    return o.join("\n");
  }

  function spssDummies(preds, isCat) {
    const dum = [], terms = [];
    preds.forEach(v => { if (isCat(v)) v.levels.slice(1).forEach((l, i) => { const nm = `${v.name}_d${i + 2}`; dum.push(`COMPUTE ${nm} = (${catName(v)} = ${i + 2}).`); terms.push(nm); }); else terms.push(v.name); });
    return { dum, terms };
  }
  function extRegressionSection(lang, ds, s) {
    const y = by(ds, s.y), preds = s.preds.map(n => by(ds, n)), isCat = v => CATT.includes(v.type), o = [];
    const pv = v => isCat(v) ? catName(v) : v.name, cl = s.cluster;
    const vars = [s.model === "mixed" ? y.name : catName(y), ...preds.map(pv), ...(cl ? [cl] : [])];
    if (lang === "stata") {
      const rhs = preds.map(v => isCat(v) ? `i.${catName(v)}` : v.name).join(" ");
      if (s.model === "ordinal") o.push(cmt(lang, "Ordinal logistic regression (proportional odds); or = cumulative odds ratios"), `ologit ${catName(y)} ${rhs}, or`, cmt(lang, "Proportional-odds check (Brant test): ssc install spost13_ado, then:"), `* brant, detail`);
      if (s.model === "multinomial") o.push(cmt(lang, `Multinomial logistic regression; reference outcome = ${y.levels[0]}`), `mlogit ${catName(y)} ${rhs}, baseoutcome(1) rrr`);
      if (s.model === "mixed") o.push(cmt(lang, "Linear mixed model with a random intercept per cluster (REML)"), `egen long _cl = group(${cl})`, `mixed ${y.name} ${rhs} || _cl:, reml`, `estat icc`, `drop _cl`);
    } else if (lang === "python") {
      o.push(`d = df[[${vars.map(qJ).join(", ")}]].dropna()`);
      if (s.model === "mixed") {
        const rhs = preds.map(v => isCat(v) ? `C(${catName(v)})` : v.name).join(" + ");
        o.push(`# Random-intercept model (REML). statsmodels derives fixed-effect SEs from the joint Hessian; Stata, R (nlme) and QuantAI use (X'V^-1X)^-1, so SEs can differ in the 3rd-4th decimal.`,
          `m = smf.mixedlm("${y.name} ~ ${rhs}", d, groups=d[${qJ(cl)}]).fit(reml=True)`, `print(m.summary())`, `tau2, s2 = float(m.cov_re.iloc[0, 0]), m.scale`, `print(f"ICC = {tau2/(tau2+s2):.3f}")`);
      } else {
        o.push(`X = pd.get_dummies(d[[${preds.map(v => qJ(pv(v))).join(", ")}]], drop_first=True).astype(float)  # first category = reference`);
        if (s.model === "ordinal") o.push(`from statsmodels.miscmodels.ordinal_model import OrderedModel`, `m = OrderedModel(d[${qJ(catName(y))}].cat.as_ordered(), X, distr="logit").fit(method="bfgs", maxiter=5000, disp=False)`, `print(m.summary())`, `k = X.shape[1]`, `print(pd.DataFrame({"OR": np.exp(m.params[:k]), "2.5%": np.exp(m.conf_int().iloc[:k, 0]), "97.5%": np.exp(m.conf_int().iloc[:k, 1]), "p": m.pvalues[:k]}).round(4))`);
        if (s.model === "multinomial") o.push(`m = sm.MNLogit(d[${qJ(catName(y))}].cat.codes, sm.add_constant(X)).fit(disp=0, maxiter=200)  # code 0 = ${y.levels[0]} (reference)`, `print(m.summary())`, `print(np.exp(m.params).round(4))  # relative risk ratios`);
      }
    } else if (lang === "r") {
      o.push(`d <- na.omit(dat[, c(${vars.map(qJ).join(", ")})])`);
      const rhs = preds.map(pv).join(" + ");
      if (s.model === "ordinal") o.push(`library(MASS)  # ships with R`, `d$${catName(y)} <- factor(d$${catName(y)}, levels = levels(d$${catName(y)}), ordered = TRUE)`, `m <- polr(${catName(y)} ~ ${rhs}, data = d, Hess = TRUE)`, `summary(m)`,
        `b <- coef(m); se <- sqrt(diag(vcov(m)))[names(b)]`, `round(cbind(OR = exp(b), lo = exp(b - 1.959964*se), hi = exp(b + 1.959964*se), p = 2*pnorm(-abs(b/se))), 4)`, `# Proportional-odds check: install.packages("brant"); brant::brant(m)`);
      if (s.model === "multinomial") o.push(`library(nnet)  # ships with R`, `m <- multinom(${catName(y)} ~ ${rhs}, data = d, trace = FALSE)  # reference = ${y.levels[0]}`, `sm <- summary(m); z <- sm$coefficients / sm$standard.errors`, `round(exp(coef(m)), 4)  # relative risk ratios`, `round(2 * pnorm(-abs(z)), 4)  # p-values`);
      if (s.model === "mixed") o.push(`library(nlme)  # ships with R`, `m <- lme(${y.name} ~ ${rhs}, random = ~ 1 | ${cl}, data = d, method = "REML")`, `summary(m)  # nlme reports t-tests; QuantAI and Stata report z-tests (same estimates and SEs)`, `vc <- as.numeric(VarCorr(m)[, "Variance"]); cat("ICC =", round(vc[1] / sum(vc), 3), "\\n")`);
    } else {
      const { dum, terms } = spssDummies(preds, isCat);
      if (dum.length) o.push(cmt(lang, "Indicator variables (reference = code 1)"), ...dum, "EXECUTE.");
      if (s.model === "ordinal") o.push(`PLUM ${catName(y)} WITH ${terms.join(" ")} /LINK=LOGIT /PRINT=PARAMETER SUMMARY TPARALLEL.`, cmt(lang, "Exponentiate the location estimates for odds ratios; TPARALLEL tests proportional odds"));
      if (s.model === "multinomial") o.push(`NOMREG ${catName(y)} (BASE=FIRST ORDER=ASCENDING) WITH ${terms.join(" ")} /PRINT=PARAMETER SUMMARY LRT FIT.`);
      if (s.model === "mixed") o.push(`MIXED ${y.name} WITH ${terms.join(" ")} /FIXED=${terms.join(" ")} /RANDOM=INTERCEPT | SUBJECT(${cl}) COVTYPE(VC) /METHOD=REML /PRINT=SOLUTION TESTCOV.`);
    }
    return o.join("\n");
  }
  function survivalSection(lang, ds, s) {
    const ev = by(ds, s.event), g = s.group ? by(ds, s.group) : null, covs = s.covs.map(n => by(ds, n)), isCat = v => CATT.includes(v.type), o = [];
    const preds = [...(g ? [g] : []), ...covs], pv = v => isCat(v) ? catName(v) : v.name, en = binName(ev);
    const vars = [s.time, en, ...preds.map(pv)];
    if (lang === "stata") {
      o.push(cmt(lang, `Event = ${ev.levels[1]}; ${ev.levels[0]} = censored`), `stset ${s.time}, failure(${en})`, g ? `sts list, by(${catName(g)}) at(0) compare` : `sts list`, g ? `stci, by(${catName(g)})   // median survival` : `stci`);
      if (g) o.push(`sts test ${catName(g)}   // log-rank`, `sts graph, by(${catName(g)})`); else o.push(`sts graph`);
      if (preds.length) o.push(cmt(lang, "Cox regression with Efron ties, as QuantAI (Stata's default is Breslow)"), `stcox ${preds.map(v => isCat(v) ? `i.${catName(v)}` : v.name).join(" ")}, efron`, `estat phtest, detail   // proportional-hazards check`);
    } else if (lang === "python") {
      o.push(`from statsmodels.duration.survfunc import SurvfuncRight, survdiff`, `d = df[[${vars.map(qJ).join(", ")}]].dropna()`);
      if (g) o.push(`for lvl, sub in d.groupby(${qJ(catName(g))}, observed=True):`, `    sf = SurvfuncRight(sub[${qJ(s.time)}], sub[${qJ(en)}])`, `    print(lvl, "n =", len(sub), "events =", int(sub[${qJ(en)}].sum()), "median =", sf.quantile(0.5))`,
        `stat, p = survdiff(d[${qJ(s.time)}], d[${qJ(en)}], d[${qJ(catName(g))}].astype(str))`, `print(f"Log-rank chi2 = {stat:.2f}, {p_fmt(p)}")`);
      else o.push(`sf = SurvfuncRight(d[${qJ(s.time)}], d[${qJ(en)}]); print(sf.summary().head(20)); print("median =", sf.quantile(0.5))`);
      if (preds.length) o.push(`m = sm.PHReg.from_formula("${s.time} ~ ${preds.map(v => isCat(v) ? `C(${catName(v)})` : v.name).join(" + ")}", d, status=d[${qJ(en)}].to_numpy(), ties="efron").fit()`, `print(m.summary())`, `print(np.exp(m.params).round(4))  # hazard ratios`);
    } else if (lang === "r") {
      o.push(`library(survival)  # ships with R`, `d <- na.omit(dat[, c(${vars.map(qJ).join(", ")})])`, `km <- survfit(Surv(${s.time}, ${en}) ~ ${g ? catName(g) : "1"}, data = d, conf.type = "log-log"); print(km)`, `plot(km, col = 1:6, xlab = "${s.time}", ylab = "Survival probability")`);
      if (g) o.push(`survdiff(Surv(${s.time}, ${en}) ~ ${catName(g)}, data = d)  # log-rank`);
      if (preds.length) o.push(`m <- coxph(Surv(${s.time}, ${en}) ~ ${preds.map(pv).join(" + ")}, data = d, ties = "efron")`, `summary(m)`, `cox.zph(m)  # proportional-hazards check`);
    } else {
      o.push(cmt(lang, `Event = ${ev.levels[1]} (coded 1)`), g ? `KM ${s.time} BY ${catName(g)} /STATUS=${en}(1) /PRINT=TABLE MEAN /PLOT=SURVIVAL /TEST=LOGRANK /COMPARE=OVERALL POOLED.` : `KM ${s.time} /STATUS=${en}(1) /PRINT=TABLE MEAN /PLOT=SURVIVAL.`);
      if (preds.length) { const cats = preds.filter(isCat); o.push(cmt(lang, "SPSS uses Breslow ties, so results can differ slightly from QuantAI's Efron method"), `COXREG ${s.time} /STATUS=${en}(1) /METHOD=ENTER ${preds.map(pv).join(" ")}${cats.length ? ` /CATEGORICAL=${cats.map(catName).join(" ")} ${cats.map(v => `/CONTRAST (${catName(v)})=Indicator(1)`).join(" ")}` : ""} /PRINT=CI(95).`); }
    }
    return o.join("\n");
  }
  function regressionSection(lang, ds, s) {
    if (["ordinal", "multinomial", "mixed"].includes(s.model)) return extRegressionSection(lang, ds, s);
    const y = by(ds, s.y), preds = s.preds.map(n => by(ds, n));
    const yv = s.model === "logistic" || s.model === "modpoisson" ? binName(y) : y.name;
    const isCat = v => CATT.includes(v.type);
    const o = [], name = { linear: "Multiple linear regression", logistic: "Binary logistic regression", poisson: "Poisson regression", modpoisson: "Modified Poisson regression (prevalence ratios, robust SE)" }[s.model];
    const allVars = [yv, ...preds.map(v => isCat(v) ? catName(v) : v.name)];
    if (lang === "stata") {
      const rhs = preds.map(v => isCat(v) ? `i.${catName(v)}` : v.name).join(" ");
      o.push(`preserve`);
      if (s.crude) {
        const cmd = s.model === "linear" ? "regress" : s.model === "logistic" ? "logistic" : "poisson";
        const opt = s.model === "linear" ? (s.robust ? ", vce(robust)" : "") : s.model === "logistic" ? "" : `, irr${s.robust ? " vce(robust)" : ""}`;
        o.push(cmt(lang, "Crude (unadjusted) estimates, each on all rows with that predictor and the outcome"), ...preds.map(v => `${cmd} ${yv} ${isCat(v) ? "i." + catName(v) : v.name}${opt}`));
      }
      o.push(cmt(lang, "Complete cases for the full model"), `keep if !missing(${allVars.join(", ")})`);
      if (s.model === "linear") {
        o.push(cmt(lang, "Diagnostics on the ordinary model"), `quietly regress ${yv} ${rhs}`, `estat vif`, `estat hettest, rhs iid   // Breusch-Pagan (Koenker), as the app`,
          `predict double _res, residuals`, `swilk _res`, `predict double _cook, cooksd`, `count if _cook > 4/e(N) & !missing(_cook)`, `summarize _cook`);
        o.push(cmt(lang, `Final model${s.robust ? " with robust (HC1) standard errors — Breusch-Pagan was significant" : ""}`), `regress ${yv} ${rhs}${s.robust ? ", vce(robust)" : ""}`);
      } else if (s.model === "logistic") {
        o.push(cmt(lang, "VIF depends only on the predictors"), `quietly regress ${yv} ${rhs}`, `estat vif`);
        o.push(cmt(lang, "Adjusted odds ratios"), `logistic ${yv} ${rhs}`, `estat gof, group(10) table   // Hosmer-Lemeshow (Stata's grouping of ties can differ slightly)`, `lroc, nograph`);
      } else {
        o.push(cmt(lang, "VIF depends only on the predictors"), `quietly regress ${yv} ${rhs}`, `estat vif`);
        if (s.model === "poisson") o.push(`quietly poisson ${yv} ${rhs}`, `predict double _mu, n`, `generate double _pr = (${yv} - _mu)^2/_mu`, `quietly summarize _pr`, `display "Pearson chi2/df (dispersion) = " r(sum)/(e(N) - e(k))`);
        const vce = s.robust ? " vce(robust)" : "";
        o.push(cmt(lang, s.robust ? "Robust SEs (Stata multiplies by n/(n-1) — results differ from the app only in the 3rd-4th decimal)" : "Final model"), `poisson ${yv} ${rhs}, irr${vce}`);
      }
      o.push(`restore`);
    } else if (lang === "python") {
      const rhs = preds.map(v => isCat(v) ? `C(${catName(v)})` : v.name).join(" + ");
      o.push(`d = df[[${allVars.map(qJ).join(", ")}]].dropna()`);
      const fitExpr = (rhsX, data = "d") => s.model === "linear" ? `smf.ols("${yv} ~ ${rhsX}", data=${data}).fit(${s.robust ? 'cov_type="HC1"' : ""})`
        : s.model === "logistic" ? `smf.logit("${yv} ~ ${rhsX}", data=${data}).fit(disp=0)`
        : `smf.glm("${yv} ~ ${rhsX}", data=${data}, family=sm.families.Poisson()).fit(${s.robust ? 'cov_type="HC0"' : ""})`;
      if (s.model === "linear") {
        o.push(`ols0 = smf.ols("${yv} ~ ${rhs}", data=d).fit()`, `X = ols0.model.exog`,
          `print("VIF:", {n: round(variance_inflation_factor(X, i), 2) for i, n in enumerate(ols0.model.exog_names) if n != "Intercept"})`,
          `bp = het_breuschpagan(ols0.resid, X); print(f"Breusch-Pagan (Koenker): LM = {bp[0]:.2f}, {p_fmt(bp[1])}")`,
          `print("Shapiro-Wilk on residuals:", stats.shapiro(ols0.resid) if len(d) <= 5000 else "n > 5000")`,
          `cook = OLSInfluence(ols0).cooks_distance[0]; print(f"Cook's D > 4/n: {(cook > 4/len(d)).sum()}, max = {cook.max():.3f}")`);
      } else {
        o.push(`X = smf.ols("${yv} ~ ${rhs}", data=d).fit().model`, `print("VIF:", {n: round(variance_inflation_factor(X.exog, i), 2) for i, n in enumerate(X.exog_names) if n != "Intercept"})`);
      }
      if (s.crude) {
        o.push("# Crude (unadjusted) estimates, one predictor at a time, on all rows with that predictor and the outcome");
        preds.forEach(v => { const vn = isCat(v) ? catName(v) : v.name, t = isCat(v) ? `C(${vn})` : vn, dd = `df[[${qJ(yv)}, ${qJ(vn)}]].dropna()`; o.push(s.model === "linear" ? `print(${fitExpr(t, dd)}.summary().tables[1])` : `m = ${fitExpr(t, dd)}; print(pd.DataFrame({"ratio": np.exp(m.params), "lo": np.exp(m.conf_int()[0]), "hi": np.exp(m.conf_int()[1]), "p": m.pvalues}).round(4))`); });
      }
      o.push(`# Final model: ${name}`, `model = ${fitExpr(rhs)}`, `print(model.summary())`);
      if (s.model !== "linear") o.push(`print(pd.DataFrame({"${({ logistic: "OR", poisson: "IRR", modpoisson: "PR" })[s.model]}": np.exp(model.params), "2.5%": np.exp(model.conf_int()[0]), "97.5%": np.exp(model.conf_int()[1]), "p": model.pvalues}).round(4))`);
      if (s.model === "poisson") o.push(`m0 = smf.glm("${yv} ~ ${rhs}", data=d, family=sm.families.Poisson()).fit()`, `print(f"Pearson chi2/df (dispersion) = {m0.pearson_chi2/m0.df_resid:.2f}")`);
      if (s.model === "logistic") o.push(`# Hosmer-Lemeshow (10 groups at quantiles of the fitted risk, as R ResourceSelection)`,
        `p = model.predict(); yy = d[${qJ(yv)}].to_numpy()`, `br = np.unique(np.quantile(p, np.linspace(0, 1, 11)))`, `g = np.clip(np.searchsorted(br, p, side="left") - 1, 0, len(br) - 2)`,
        `o1, e1, cnt = np.bincount(g, yy), np.bincount(g, p), np.bincount(g)`, `hl = (((o1-e1)**2/e1) + ((cnt-o1)-(cnt-e1))**2/(cnt-e1)).sum()`, `print(f"Hosmer-Lemeshow chi2({len(br)-3}) = {hl:.2f}, {p_fmt(stats.chi2.sf(hl, len(br)-3))}")`,
        `u = stats.mannwhitneyu(p[yy == 1], p[yy == 0]).statistic; print(f"AUC = {u/((yy == 1).sum()*(yy == 0).sum()):.3f}")`);
    } else if (lang === "r") {
      const rhs = preds.map(v => isCat(v) ? catName(v) : v.name).join(" + ");
      o.push(`d <- na.omit(dat[, c(${allVars.map(qJ).join(", ")})])`,
        `vif_base <- function(f, data) { X <- model.matrix(f, data)[, -1, drop = FALSE]; if (ncol(X) < 2) return(NA); setNames(sapply(seq_len(ncol(X)), function(j) 1/(1 - summary(lm(X[, j] ~ X[, -j]))$r.squared)), colnames(X)) }`,
        `round(vif_base(~ ${rhs}, d), 2)  # VIF`);
      if (s.model === "linear") {
        o.push(`fit <- lm(${yv} ~ ${rhs}, data = d)`, `aux <- lm(resid(fit)^2 ~ model.matrix(fit)[, -1]); bp <- nobs(fit) * summary(aux)$r.squared`,
          `cat("Breusch-Pagan (Koenker): LM =", round(bp, 2), " p =", signif(pchisq(bp, ncol(model.matrix(fit)) - 1, lower.tail = FALSE), 3), "\\n")`,
          `if (nobs(fit) <= 5000) shapiro.test(resid(fit))`, `cook <- cooks.distance(fit); cat("Cook's D > 4/n:", sum(cook > 4/nobs(fit)), " max =", round(max(cook), 3), "\\n")`,
          `robust_table <- function(fit) { X <- model.matrix(fit); e <- resid(fit); n <- nrow(X); k <- ncol(X); B <- solve(crossprod(X)); V <- B %*% crossprod(X * e) %*% B * n/(n - k); se <- sqrt(diag(V)); b <- coef(fit); tq <- qt(.975, n - k); cbind(B = b, SE = se, lo = b - tq*se, hi = b + tq*se, p = 2*pt(-abs(b/se), n - k)) }`);
        if (s.crude) o.push("# Crude estimates (all rows with that predictor and the outcome)", ...preds.map(v => { const vn = isCat(v) ? catName(v) : v.name, dd = `na.omit(dat[, c("${yv}", "${vn}")])`; return s.robust ? `round(robust_table(lm(${yv} ~ ${vn}, data = ${dd})), 4)` : `{ m <- lm(${yv} ~ ${vn}, data = ${dd}); cbind(coef(summary(m)), confint(m)) }`; }));
        o.push(`# Final model: ${name}${s.robust ? " with robust (HC1) SEs" : ""}`, s.robust ? `round(robust_table(fit), 4)` : `summary(fit); confint(fit)`);
      } else {
        const fam = s.model === "logistic" ? "binomial" : "poisson";
        o.push(`ratio_table <- function(fit, robust = FALSE) { b <- coef(fit); V <- vcov(fit); if (robust) { X <- model.matrix(fit); V <- V %*% crossprod(X * (fit$y - fitted(fit))) %*% V }; se <- sqrt(diag(V)); round(cbind(ratio = exp(b), lo = exp(b - 1.959964*se), hi = exp(b + 1.959964*se), p = 2*pnorm(-abs(b/se))), 4) }`);
        if (s.crude) o.push("# Crude ratios (all rows with that predictor and the outcome)", ...preds.map(v => { const vn = isCat(v) ? catName(v) : v.name; return `ratio_table(glm(${yv} ~ ${vn}, family = ${fam}, data = na.omit(dat[, c("${yv}", "${vn}")])), robust = ${s.robust ? "TRUE" : "FALSE"})`; }));
        o.push(`# Final model: ${name}`, `fit <- glm(${yv} ~ ${rhs}, family = ${fam}, data = d)`, `summary(fit)`, `ratio_table(fit, robust = ${s.robust ? "TRUE" : "FALSE"})  # Wald CIs${s.robust ? ", robust (HC0) SEs" : ""}`);
        if (s.model === "poisson") o.push(`cat("Pearson chi2/df (dispersion) =", round(sum(residuals(fit, type = "pearson")^2) / df.residual(fit), 2), "\\n")`);
        if (s.model === "logistic") o.push(`# Hosmer-Lemeshow (same grouping as ResourceSelection::hoslem.test, g = 10)`,
          `p <- fitted(fit); br <- unique(quantile(p, seq(0, 1, .1))); g <- cut(p, br, include.lowest = TRUE)`,
          `o1 <- tapply(fit$y, g, sum); e1 <- tapply(p, g, sum); nn <- table(g); hl <- sum((o1-e1)^2/e1 + ((nn-o1)-(nn-e1))^2/(nn-e1))`,
          `cat("Hosmer-Lemeshow chi2(", length(br) - 3, ") =", round(hl, 2), " p =", signif(pchisq(hl, length(br) - 3, lower.tail = FALSE), 3), "\\n")`,
          `r <- rank(p); n1 <- sum(fit$y == 1); n0 <- sum(fit$y == 0); cat("AUC =", round((sum(r[fit$y == 1]) - n1*(n1+1)/2)/(n1*n0), 3), "\\n")`);
      }
    } else if (lang === "spss") {
      if (s.model === "linear") {
        const dum = [], terms = [];
        preds.forEach(v => { if (isCat(v)) v.levels.slice(1).forEach((l, i) => { const nm = `${v.name}_d${i + 2}`; dum.push(`COMPUTE ${nm} = (${catName(v)} = ${i + 2}).`); terms.push(nm); }); else terms.push(v.name); });
        if (dum.length) o.push(cmt(lang, "Indicator variables for categorical predictors (reference = code 1)"), ...dum, "EXECUTE.");
        o.push(`REGRESSION /MISSING=LISTWISE /STATISTICS=COEFF OUTS CI(95) R ANOVA COLLIN TOL /DEPENDENT=${yv} /METHOD=ENTER ${terms.join(" ")} /SAVE=RESID(res_) COOK(cook_).`,
          `EXAMINE VARIABLES=res_ /PLOT NPPLOT.`,
          cmt(lang, "Breusch-Pagan (Koenker): n × R² of this auxiliary regression, chi-square with df = number of predictors"), `COMPUTE res2_ = res_**2.`, `REGRESSION /DEPENDENT=res2_ /METHOD=ENTER ${terms.join(" ")}.`);
        if (s.robust) o.push(cmt(lang, "Robust standard errors (HC1)"), `UNIANOVA ${yv} WITH ${terms.join(" ")} /ROBUST=HC1 /PRINT=PARAMETER /DESIGN=${terms.join(" ")}.`);
      } else if (s.model === "logistic") {
        const cats = preds.filter(isCat);
        o.push(`LOGISTIC REGRESSION VARIABLES ${yv} /METHOD=ENTER ${preds.map(v => isCat(v) ? catName(v) : v.name).join(" ")}`,
          ...(cats.length ? [`  /CATEGORICAL=${cats.map(catName).join(" ")}`, ...cats.map(v => `  /CONTRAST (${catName(v)})=Indicator(1)`)] : []),
          `  /SAVE=PRED(phat_) /PRINT=GOODFIT CI(95) /CRITERIA=ITERATE(100).`, `ROC phat_ BY ${yv} (1) /PRINT=SE.`);
      } else {
        const cats = preds.filter(isCat), conts = preds.filter(v => !isCat(v));
        o.push(cmt(lang, "ORDER=DESCENDING makes code 1 the reference category, as in the app"),
          `GENLIN ${yv}${cats.length ? ` BY ${cats.map(catName).join(" ")} (ORDER=DESCENDING)` : ""}${conts.length ? ` WITH ${conts.map(v => v.name).join(" ")}` : ""}`,
          `  /MODEL ${preds.map(v => isCat(v) ? catName(v) : v.name).join(" ")} INTERCEPT=YES DISTRIBUTION=POISSON LINK=LOG`,
          `  /CRITERIA COVB=${s.robust ? "ROBUST" : "MODEL"} CILEVEL=95 /PRINT FIT SOLUTION (EXPONENTIATED).`);
      }
    }
    return o.join("\n");
  }

  function table1Section(lang, ds, s) {
    const g = s.group ? by(ds, s.group) : null, gc = g ? catName(g) : null, o = [];
    s.vars.forEach(x => {
      const v = by(ds, x.name);
      if (x.type === "cont") {
        if (lang === "stata") { o.push(g ? `tabstat ${v.name}, by(${gc}) statistics(n ${x.normal ? "mean sd" : "p50 p25 p75"})` : `tabstat ${v.name}, statistics(n ${x.normal ? "mean sd" : "p50 p25 p75"})`);
          if (g && x.chosen) o.push({ student: `ttest ${v.name}, by(${gc})`, welch: `ttest ${v.name}, by(${gc}) unequal`, mwu: `ranksum ${v.name}, by(${gc})`, anova: `oneway ${v.name} ${gc}`, welch_anova: `* Welch ANOVA: see the Compare section template`, kruskal: `kwallis ${v.name}, by(${gc})` }[x.chosen]); }
        if (lang === "python") { o.push(g ? `print(df.groupby(${qJ(gc)}, observed=True)[${qJ(v.name)}].describe())` : `print(df[${qJ(v.name)}].describe())`);
          if (g && x.chosen) o.push(`gs = [x.dropna().to_numpy(dtype=float) for _, x in df.groupby(${qJ(gc)}, observed=True)[${qJ(v.name)}]]`, `print(${qJ(v.name)}, ${({ student: "stats.ttest_ind(*gs)", welch: "stats.ttest_ind(*gs, equal_var=False)", mwu: 'stats.mannwhitneyu(*gs, method="asymptotic")', anova: "stats.f_oneway(*gs)", welch_anova: 'anova_oneway(gs, use_var="unequal", welch_correction=True)', kruskal: "stats.kruskal(*gs)" })[x.chosen]})`); }
        if (lang === "r") { o.push(g ? `tapply(dat$${v.name}, dat$${gc}, function(x) summary(x))` : `summary(dat$${v.name}); sd(dat$${v.name}, na.rm = TRUE)`);
          if (g && x.chosen) o.push({ student: `t.test(${v.name} ~ ${gc}, data = dat, var.equal = TRUE)`, welch: `t.test(${v.name} ~ ${gc}, data = dat)`, mwu: `wilcox.test(${v.name} ~ ${gc}, data = dat, exact = FALSE)`, anova: `summary(aov(${v.name} ~ ${gc}, data = dat))`, welch_anova: `oneway.test(${v.name} ~ ${gc}, data = dat)`, kruskal: `kruskal.test(${v.name} ~ ${gc}, data = dat)` }[x.chosen]); }
        if (lang === "spss") { o.push(g ? `MEANS TABLES=${v.name} BY ${gc} /CELLS=COUNT MEAN STDDEV MEDIAN.` : `DESCRIPTIVES VARIABLES=${v.name} /STATISTICS=MEAN STDDEV MIN MAX.`);
          if (g && x.chosen) o.push({ student: `T-TEST GROUPS=${gc}(1 2) /VARIABLES=${v.name}.`, welch: `T-TEST GROUPS=${gc}(1 2) /VARIABLES=${v.name}.`, mwu: `NPAR TESTS /M-W=${v.name} BY ${gc}(1 2).`, anova: `ONEWAY ${v.name} BY ${gc}.`, welch_anova: `ONEWAY ${v.name} BY ${gc} /STATISTICS WELCH.`, kruskal: `NPAR TESTS /K-W=${v.name} BY ${gc}(1 ${g.levels.length}).` }[x.chosen]); }
      } else {
        const vc = catName(v);
        if (lang === "stata") o.push(g ? `tabulate ${vc} ${gc}, column${x.chosen === "fisher" ? " exact" : x.chosen ? " chi2" : ""}` : `tabulate ${vc}`);
        if (lang === "python") { o.push(g ? `print(pd.crosstab(df[${qJ(vc)}], df[${qJ(gc)}], margins=True)); print((pd.crosstab(df[${qJ(vc)}], df[${qJ(gc)}], normalize="columns")*100).round(1))` : `print(df[${qJ(vc)}].value_counts(sort=False)); print((df[${qJ(vc)}].value_counts(normalize=True, sort=False)*100).round(1))`);
          if (g && x.chosen) o.push(x.chosen === "chi2" ? `print(${qJ(v.name)}, stats.chi2_contingency(pd.crosstab(df[${qJ(vc)}], df[${qJ(gc)}]), correction=False)[:2])` : `# Fisher's exact for ${v.name}: see the R or Stata script for tables larger than 2x2`); }
        if (lang === "r") { o.push(g ? `tab <- table(dat$${vc}, dat$${gc}); tab; round(prop.table(tab, 2) * 100, 1)` : `table(dat$${vc}); round(prop.table(table(dat$${vc})) * 100, 1)`);
          if (g && x.chosen) o.push(x.chosen === "chi2" ? `chisq.test(tab, correct = FALSE)` : `fisher.test(tab, simulate.p.value = (prod(dim(tab)) > 9), B = 20000)`); }
        if (lang === "spss") o.push(g ? `CROSSTABS /TABLES=${vc} BY ${gc} /CELLS=COUNT COLUMN${x.chosen ? " /STATISTICS=CHISQ" : ""}.` : `FREQUENCIES VARIABLES=${vc}.`);
      }
    });
    const note = lang === "stata" ? cmt(lang, "Stata's p25/p75 use a different percentile definition from the app (type 7, as R and Python); medians agree") : lang === "spss" ? cmt(lang, "SPSS percentiles use a different definition from the app (type 7, as R and Python); medians agree") : null;
    return [cmt(lang, "Summary rule: mean (SD) if normal by Shapiro–Wilk, otherwise median (IQR); categorical n (column %)"), ...(note ? [note] : []), ...o].join("\n");
  }

  function section(lang, ds, spec, i, title) {
    const body = spec.kind === "compare" ? compareSection(lang, ds, spec) : spec.kind === "correlate" ? correlateSection(lang, ds, spec)
      : spec.kind === "crosstab" ? crosstabSection(lang, ds, spec) : spec.kind === "mcnemar" || spec.kind === "paired" ? pairedSection(lang, ds, spec)
      : spec.kind === "regression" ? regressionSection(lang, ds, spec) : spec.kind === "table1" ? table1Section(lang, ds, spec)
      : spec.kind === "survival" ? survivalSection(lang, ds, spec) : "";
    return banner(lang, i, title) + "\n" + body + "\n";
  }

  /** Full script for one language. entries: [{spec, title}] */
  function script(lang, ds, entries, meta) {
    const needs = collectNeeds(ds, entries.map(e => e.spec));
    let s = header(lang, meta) + "\n" + setup(lang, ds, needs);
    entries.forEach((e, i) => s += section(lang, ds, e.spec, i + 1, e.title));
    if (lang === "spss") s += "\nEXECUTE.\n";
    return s;
  }
  return { script, LANGS, collectNeeds };
})();
if (false) module.exports = Codegen;

/* =====================================================================
   Methodology engine: fixed, textbook rules. The same answers always
   give the same recommendation; nothing here calls an AI.
   ===================================================================== */
const Method = (function () {
  "use strict";

  /* =================================================================
     Study designs
     ================================================================= */
  const DESIGNS = {
    cs_desc: { name: "Descriptive cross-sectional survey", family: "Observational · descriptive",
      what: "Measures the outcome (and characteristics) in a sample at one point in time to estimate how common it is.",
      strengths: ["Quick and relatively cheap", "Gives prevalence directly", "Good for planning services and generating hypotheses"],
      limitations: ["Cannot show cause and effect", "Captures existing (prevalent) cases, so long-lasting conditions are over-represented", "Sensitive to non-response"],
      biases: ["Selection bias from the sampling frame → use probability sampling (simple random, systematic, stratified or multistage cluster)", "Non-response bias → report response rate and compare responders with non-responders", "Measurement/information bias → validated, pre-tested tools and trained data collectors"],
      measure: "Prevalence (%) with 95% CI", analysis: "Proportions with 95% CI, adjusted for the survey design (weights, clusters) if complex sampling was used", sampling: "Probability sampling; single-proportion sample size (Cochran) × design effect", guideline: "STROBE (cross-sectional)", evidence: 4 },
    cs_anal: { name: "Analytical cross-sectional study", family: "Observational · analytic",
      what: "Measures exposures and outcome at the same time to estimate associations (e.g. factors associated with hypertension).",
      strengths: ["Fast, affordable, single data collection", "Can study many exposures and outcomes", "Estimates prevalence and prevalence ratios together"],
      limitations: ["Temporality is unknown: you cannot tell whether exposure came before the outcome", "Not suited to rare outcomes or short-duration conditions", "Associations, not causes"],
      biases: ["Reverse causation → word conclusions as associations", "Confounding → collect known confounders and adjust in multivariable models", "Selection and non-response bias → probability sampling, report response rate"],
      measure: "Prevalence ratio (preferred when the outcome is common) or prevalence odds ratio", analysis: "Chi-square/t-tests for bivariate analysis; modified Poisson (prevalence ratios) or logistic regression for adjusted associations", sampling: "Probability sampling; size for the prevalence estimate or for comparing two proportions, whichever is larger", guideline: "STROBE (cross-sectional)", evidence: 4 },
    cohort_pro: { name: "Prospective cohort study", family: "Observational · analytic",
      what: "Recruits people free of the outcome, classifies them by exposure, and follows them forward to see who develops the outcome.",
      strengths: ["Exposure is measured before the outcome (clear temporality)", "Gives incidence and risk ratios directly", "Can study several outcomes of one exposure", "Good for rare exposures"],
      limitations: ["Slow and costly", "Loss to follow-up", "Inefficient for rare outcomes or long latency"],
      biases: ["Loss to follow-up (attrition) bias → tracing procedures, report retention, compare completers and non-completers", "Confounding → measure and adjust", "Information bias if outcome assessment differs by exposure → blinded outcome assessment"],
      measure: "Incidence, risk ratio (RR), incidence rate ratio, hazard ratio", analysis: "Risk/rate ratios; Poisson regression with person-time; Kaplan–Meier and Cox regression for time-to-event", sampling: "Select by exposure status; size for comparing two proportions or for a hazard ratio, inflated for loss to follow-up", guideline: "STROBE (cohort)", evidence: 3 },
    cohort_retro: { name: "Retrospective (historical) cohort study", family: "Observational · analytic",
      what: "Uses existing records to identify a past cohort, classify their exposure, and trace outcomes up to now.",
      strengths: ["Much faster and cheaper than a prospective cohort", "Temporality preserved if records captured exposure first", "Uses routine data (registers, clinic records, DHIS2)"],
      limitations: ["Limited to variables already recorded", "Missing or inconsistent records", "Confounders may not have been measured"],
      biases: ["Information bias from record quality → data abstraction form, double abstraction of a sample", "Selection bias if records are incomplete for some groups", "Unmeasured confounding → state as limitation; sensitivity analysis"],
      measure: "Risk ratio, rate ratio, hazard ratio", analysis: "As for prospective cohort: Poisson or Cox regression", sampling: "All eligible records or a random sample of them", guideline: "STROBE (cohort) + RECORD for routinely collected data", evidence: 3 },
    case_control: { name: "Case–control study", family: "Observational · analytic",
      what: "Starts from people with the outcome (cases) and comparable people without it (controls), then compares past exposure.",
      strengths: ["Efficient for rare outcomes and long latency", "Quick and relatively cheap", "Can study many exposures for one outcome"],
      limitations: ["Cannot estimate incidence or prevalence", "Recall bias", "Choosing appropriate controls is difficult"],
      biases: ["Selection bias in controls → controls must come from the same population that produced the cases", "Recall bias → objective records where possible, same interview for cases and controls", "Confounding → matching and/or adjustment (conditional logistic regression if matched)"],
      measure: "Odds ratio (OR)", analysis: "Chi-square and crude OR; multivariable logistic regression (conditional logistic if individually matched)", sampling: "Unmatched or matched (1–4 controls per case); size from expected OR and exposure prevalence among controls", guideline: "STROBE (case–control)", evidence: 3 },
    ecological: { name: "Ecological study", family: "Observational · analytic",
      what: "Compares groups (districts, regions, countries) rather than individuals, using aggregate data.",
      strengths: ["Uses existing aggregate data; very cheap", "Good for hypothesis generation and policy-level exposures"],
      limitations: ["Ecological fallacy: group-level associations may not hold for individuals", "Little control of confounding"],
      biases: ["Ecological fallacy → do not draw individual-level conclusions", "Data quality differs across areas"],
      measure: "Correlation or regression coefficients between area-level rates", analysis: "Correlation, linear or Poisson regression on area-level data; spatial analysis (GIS) where useful", sampling: "All areas with data", guideline: "STROBE (adapted)", evidence: 5 },
    rct: { name: "Randomised controlled trial (individual)", family: "Experimental",
      what: "Individuals are randomly allocated to the intervention or control and followed up for the outcome.",
      strengths: ["Strongest design for cause and effect", "Randomisation balances known and unknown confounders", "Allows blinding"],
      limitations: ["Expensive and time-consuming", "Ethical limits on what can be randomised", "Strict eligibility can limit generalisability"],
      biases: ["Selection bias → concealed allocation (sealed envelopes or central randomisation)", "Performance and detection bias → blinding where possible", "Attrition bias → intention-to-treat analysis, report flow diagram"],
      measure: "Risk ratio, risk difference, mean difference; NNT", analysis: "Intention-to-treat comparison of outcomes (t-test/chi-square or regression adjusted for baseline values)", sampling: "Sample size for comparing two means or proportions, inflated for attrition; register the trial before enrolment", guideline: "CONSORT 2010 (protocol: SPIRIT)", evidence: 2 },
    crct: { name: "Cluster randomised trial", family: "Experimental",
      what: "Groups (clinics, schools, communities) rather than individuals are randomised, because the intervention is delivered at group level or contamination is likely.",
      strengths: ["Fits interventions delivered to whole facilities or communities", "Reduces contamination between arms"],
      limitations: ["Needs many clusters; larger total sample than an individual RCT", "Analysis must account for clustering"],
      biases: ["Identification/recruitment bias after clusters are randomised → recruit participants before allocation or blind recruiters", "Imbalance with few clusters → stratified or matched randomisation"],
      measure: "Risk ratio, mean difference (cluster-adjusted)", analysis: "Mixed-effects models or GEE with robust SEs; intra-cluster correlation (ICC) reported", sampling: "Individual-RCT sample size × design effect [1 + (m − 1) × ICC]", guideline: "CONSORT extension for cluster trials", evidence: 2 },
    quasi: { name: "Quasi-experimental study (non-randomised controlled before–after)", family: "Experimental (non-randomised)",
      what: "Compares an intervention group with a comparison group, before and after, without random allocation.",
      strengths: ["Feasible when randomisation is impossible or unethical", "Evaluates real-world programmes"],
      limitations: ["Groups may differ in ways that affect the outcome", "Weaker causal claims than an RCT"],
      biases: ["Selection bias / baseline imbalance → measure baseline, use difference-in-differences", "Secular trends and co-interventions → comparison group, document context"],
      measure: "Difference-in-differences; adjusted RR or mean difference", analysis: "Difference-in-differences regression (group × time interaction); propensity-score methods", sampling: "As for two means/proportions, inflated for design effect if clustered", guideline: "TREND statement", evidence: 3 },
    its: { name: "Interrupted time series", family: "Experimental (non-randomised)",
      what: "Uses many routine measurements before and after an intervention or policy to see if level or trend changed.",
      strengths: ["Uses routine data (e.g. monthly DHIS2 indicators)", "Controls for pre-existing trends", "Strong quasi-experimental design"],
      limitations: ["Needs enough time points (≥ 8–12 before and after)", "Other events at the same time can confound"],
      biases: ["Co-occurring events → control series if available", "Changes in data recording → check data quality around the interruption"],
      measure: "Change in level and slope", analysis: "Segmented regression (with autocorrelation check) or ARIMA", sampling: "All time points available", guideline: "No single guideline; follow Cochrane EPOC criteria", evidence: 3 },
    diag: { name: "Diagnostic accuracy study", family: "Observational · diagnostic",
      what: "A new test (index test) and a reference standard are applied to the same people to measure agreement.",
      strengths: ["Directly estimates sensitivity, specificity, predictive values"],
      limitations: ["Needs a credible reference standard", "Spectrum of patients affects results"],
      biases: ["Verification bias → apply the reference standard to everyone", "Review bias → read index test blind to the reference result", "Spectrum bias → recruit a consecutive series from the intended-use setting"],
      measure: "Sensitivity, specificity, PPV, NPV, likelihood ratios, AUC", analysis: "2×2 table against reference standard with exact (Clopper–Pearson) CIs; ROC curve for continuous tests", sampling: "Size for the expected sensitivity (among diseased) and specificity (among non-diseased)", guideline: "STARD 2015", evidence: 3 },
    qual: { name: "Qualitative study", family: "Qualitative",
      what: "Explores experiences, meanings, perceptions or processes in depth using interviews, focus groups or observation.",
      strengths: ["Explains how and why", "Captures context and perspectives", "Generates hypotheses and informs intervention design"],
      limitations: ["Findings are not statistically generalisable", "Researcher influence on data and interpretation"],
      biases: ["Researcher bias → reflexivity, memo-writing, multiple coders", "Social desirability → private setting, skilled facilitators", "Transferability → thick description of setting and participants"],
      measure: "Themes (no statistical measure)", analysis: "Thematic analysis (Braun & Clarke), framework analysis, or approach-specific methods; NVivo/Atlas.ti optional", sampling: "Purposive sampling until data saturation (often 12–30 interviews, 4–8 focus groups)", guideline: "COREQ (interviews/FGDs) or SRQR", evidence: null },
    mixed: { name: "Mixed-methods study", family: "Mixed methods",
      what: "Combines quantitative and qualitative strands and integrates them.",
      strengths: ["Numbers and explanations together", "Qualitative strand can explain or build on quantitative results"],
      limitations: ["Time- and skill-intensive", "Integration is often weak if not planned"],
      biases: ["Each strand carries its own biases → apply the safeguards of both", "Weak integration → plan a joint display and integration point"],
      measure: "Measures of each strand plus integrated findings", analysis: "Explanatory sequential (quant → qual), exploratory sequential (qual → quant), or convergent (both at once)", sampling: "Probability sample for the quantitative strand; purposive sub-sample for the qualitative strand", guideline: "GRAMMS; plus STROBE/COREQ for each strand", evidence: null },
    sr: { name: "Systematic review (± meta-analysis)", family: "Evidence synthesis",
      what: "Systematically searches, appraises and synthesises all studies on a focused question.",
      strengths: ["Highest level of evidence when based on good studies", "Transparent and reproducible"],
      limitations: ["Only as good as the included studies", "Publication bias", "Time-consuming"],
      biases: ["Publication bias → search grey literature, funnel plot", "Selection bias → two independent reviewers", "Risk of bias in included studies → RoB 2, ROBINS-I, Newcastle–Ottawa"],
      measure: "Pooled effect (RR, OR, mean difference) with heterogeneity (I²)", analysis: "Random-effects meta-analysis if studies are similar enough; narrative synthesis otherwise", sampling: "All eligible studies; register protocol on PROSPERO", guideline: "PRISMA 2020", evidence: 1 },
    scoping: { name: "Scoping review", family: "Evidence synthesis",
      what: "Maps the extent and type of evidence on a broad topic, without appraising quality in depth.",
      strengths: ["Good for broad or emerging topics", "Identifies gaps for future research"],
      limitations: ["No pooled effect estimate", "Usually no formal quality appraisal"],
      biases: ["Incomplete search → broad databases + grey literature", "Selection bias → two reviewers"],
      measure: "Descriptive mapping (counts, charts, themes)", analysis: "Charting and narrative/thematic summary (Arksey & O'Malley; JBI)", sampling: "All eligible sources", guideline: "PRISMA-ScR", evidence: null },
  };

  const DESIGN_QUESTIONS = [
    { id: "aim", q: "What is the main aim of the study?", options: [
      ["prevalence", "Measure how common something is, or describe a population"],
      ["association", "Find factors associated with an outcome (risk factors)"],
      ["effect", "Test whether an intervention or programme works"],
      ["diagnostic", "Check how accurate a test or screening tool is"],
      ["explore", "Explore experiences, perceptions or reasons (how/why)"],
      ["both", "Measure something AND explain it in depth"],
      ["synthesis", "Summarise existing studies"],
      ["areas", "Compare districts/regions using aggregate data"]] },
    { id: "assign", q: "Can you decide who receives the intervention?", show: a => a.aim === "effect", options: [
      ["random", "Yes, and allocation can be random"], ["norandom", "We control it, but cannot randomise"], ["policy", "No: it is an existing programme or policy with routine data over time"]] },
    { id: "level", q: "Is the intervention delivered to individuals or to groups (clinics, schools, communities)?", show: a => a.aim === "effect" && a.assign === "random", options: [["individual", "Individuals"], ["group", "Groups"]] },
    { id: "rare", q: "How common is the outcome in the population?", show: a => a.aim === "association", options: [["common", "Common (roughly 10% or more)"], ["rare", "Rare, or takes years to develop"]] },
    { id: "time", q: "Can you follow participants over time?", show: a => a.aim === "association", options: [
      ["follow", "Yes, we can follow people forward"], ["records", "We have past records showing exposure before the outcome"], ["once", "No, one data collection only"]] },
    { id: "exposure", q: "Is the exposure rare (e.g. a specific occupation or treatment)?", show: a => a.aim === "association" && a.time !== "once", options: [["no", "No"], ["yes", "Yes"]] },
    { id: "qualtype", q: "What do you want to understand?", show: a => a.aim === "explore" || a.aim === "both", options: [
      ["lived", "People's lived experience of something (phenomenology)"], ["theory", "A process, to build a theory (grounded theory)"], ["culture", "Practices and beliefs of a community (ethnography)"], ["case", "One programme, facility or case in depth (case study)"], ["describe", "Straightforward description of views (qualitative descriptive)"]] },
    { id: "breadth", q: "How focused is the review question?", show: a => a.aim === "synthesis", options: [["focused", "Focused (specific population, intervention/exposure, outcome)"], ["broad", "Broad: mapping what research exists"]] },
    { id: "constraint", q: "Main practical constraint", options: [["none", "No major constraint"], ["time", "Limited time (a few months)"], ["budget", "Limited budget"], ["secondary", "Only existing/secondary data available"]] },
  ];

  const QUAL_APPROACH = { lived: "Phenomenology: in-depth interviews with people who have lived the experience; analyse for essential meanings.", theory: "Grounded theory: theoretical sampling, constant comparison and coding (open, axial, selective) until saturation.", culture: "Ethnography: participant observation, field notes and informal interviews over an extended period.", case: "Case study: multiple data sources (interviews, documents, observation) about one bounded case.", describe: "Qualitative description: semi-structured interviews or focus groups with thematic analysis." };

  function decideDesign(a) {
    const why = [], alts = [];
    let id;
    switch (a.aim) {
      case "prevalence": id = "cs_desc"; why.push("The aim is to estimate how common something is, which a single survey of a representative sample answers directly."); alts.push(["cs_anal", "if you also want factors associated with the outcome"]); break;
      case "association":
        if (a.time === "once") { id = "cs_anal"; why.push("Only one data collection is possible, so exposure and outcome are measured together."); if (a.rare === "rare") { why.push("Because the outcome is rare, a case–control design would be more efficient if cases can be identified (e.g. from clinic registers)."); alts.push(["case_control", "more efficient for a rare outcome: recruit known cases and comparable controls"]); } else alts.push(["case_control", "if the outcome turns out to be rare"]); alts.push(["cohort_retro", "if past records show exposure before the outcome"]); }
        else if (a.rare === "rare") { id = "case_control"; why.push("The outcome is rare or slow to develop, so starting from cases is far more efficient than following a large cohort."); alts.push(["cohort_retro", "if complete records exist, it gives risk directly"]); alts.push(["cs_anal", "if time is very short, accepting that temporality is unknown"]); }
        else if (a.time === "records") { id = "cohort_retro"; why.push("Existing records show exposure before the outcome, so a historical cohort keeps temporality without years of follow-up."); alts.push(["cohort_pro", "if records lack key variables"]); alts.push(["case_control", "if the outcome is rarer than expected"]); }
        else { id = "cohort_pro"; why.push("You can follow people forward, so exposure is measured before the outcome — the clearest observational evidence of temporality."); if (a.exposure === "yes") why.push("A cohort is also the natural choice for a rare exposure, because you select people by exposure status."); alts.push(["cohort_retro", "faster if suitable records exist"]); alts.push(["cs_anal", "if follow-up is not affordable"]); }
        break;
      case "effect":
        if (a.assign === "random") { id = a.level === "group" ? "crct" : "rct"; why.push("You can allocate the intervention at random, which gives the strongest evidence of cause and effect."); if (a.level === "group") why.push("Because the intervention is delivered to groups, clusters are randomised and the analysis must account for clustering."); alts.push(["quasi", "if randomisation proves unacceptable to stakeholders"]); }
        else if (a.assign === "policy") { id = "its"; why.push("The programme already exists and routine data over time are available, so changes in level and trend after its start can be measured."); alts.push(["quasi", "if a comparison area with before/after data exists"]); alts.push(["cohort_retro", "comparing outcomes of exposed and unexposed individuals from records"]); }
        else { id = "quasi"; why.push("You control who gets the intervention but cannot randomise, so a comparison group with before-and-after measurements is the strongest feasible design."); alts.push(["its", "if many routine time points are available"]); alts.push(["rct", "if randomisation becomes possible"]); }
        break;
      case "diagnostic": id = "diag"; why.push("Accuracy questions need the index test and a reference standard applied to the same people."); break;
      case "explore": id = "qual"; why.push("How/why questions about experiences and meanings need qualitative methods."); if (a.qualtype) why.push(QUAL_APPROACH[a.qualtype]); alts.push(["mixed", "if you also need to measure how common the views are"]); break;
      case "both": id = "mixed"; why.push("You need both numbers and in-depth explanation, so the two strands are combined and integrated."); if (a.qualtype) why.push("Qualitative strand: " + QUAL_APPROACH[a.qualtype]); alts.push(["cs_anal", "quantitative strand only, if time is short"]); alts.push(["qual", "qualitative strand only"]); break;
      case "synthesis": id = a.breadth === "broad" ? "scoping" : "sr"; why.push(a.breadth === "broad" ? "A broad mapping question suits a scoping review." : "A focused question with defined population, exposure/intervention and outcome suits a systematic review."); alts.push([a.breadth === "broad" ? "sr" : "scoping", a.breadth === "broad" ? "once the question is narrowed" : "if the evidence base is too varied to synthesise"]); break;
      case "areas": id = "ecological"; why.push("The units are areas with aggregate data, which is an ecological design."); alts.push(["cs_anal", "if individual-level data can be collected"]); break;
      default: return null;
    }
    const c = a.constraint;
    if (c === "time" && ["cohort_pro", "rct", "crct"].includes(id)) why.push("Note: with limited time, this design may not finish; the alternatives below are faster.");
    if (c === "secondary" && !["cohort_retro", "its", "ecological", "sr", "scoping", "cs_desc", "cs_anal"].includes(id)) { why.push("Note: with only secondary data, consider a retrospective cohort, analytical cross-sectional analysis of an existing survey (e.g. DHS), or an ecological study."); alts.unshift(["cohort_retro", "uses existing records"]); }
    if (c === "budget" && ["cohort_pro", "crct", "rct"].includes(id)) why.push("Note: this is among the most expensive designs; budget carefully for follow-up.");
    const seen = new Set([id]);
    return { id, design: DESIGNS[id], why, alternatives: alts.filter(([k]) => !seen.has(k) && seen.add(k)).map(([k, when]) => ({ id: k, design: DESIGNS[k], when })) };
  }

  /* =================================================================
     Sample size calculators
     ================================================================= */
  const SS = (typeof Stats !== "undefined" ? Stats : require("./stats.js")).SampleSize;
  const r3 = x => Math.round(x * 1000) / 1000;
  const SAMPLE = [
    { id: "prev", name: "Single proportion (prevalence survey)", use: "Descriptive cross-sectional surveys estimating a prevalence.",
      fields: [["p", "Expected prevalence (0–1)", 0.5, "Use 0.5 if unknown: it gives the largest sample"], ["d", "Margin of error (absolute, 0–1)", 0.05], ["conf", "Confidence level", 0.95]],
      formula: "n = Z² × p(1 − p) / d²", calc: v => { const r = SS.oneProportion(v); return { n: r.n, parts: [["Z", r3(r.za)]] }; },
      methods: (v, n) => `The sample size was calculated using Cochran's formula for a single proportion, n = Z²p(1 − p)/d², assuming an expected prevalence of ${v.p * 100}%, a ${v.conf * 100}% confidence level (Z = ${r3(SS.oneProportion(v).za)}) and a margin of error of ${v.d * 100}%, giving ${n.base}` ,
      code: v => ({ stata: `* n = Z^2 p(1-p) / d^2\ndisplay ceil(invnormal(${1 - (1 - v.conf) / 2})^2 * ${v.p}*(1-${v.p}) / ${v.d}^2)`, python: `from scipy.stats import norm\nimport math\nz = norm.ppf(${1 - (1 - v.conf) / 2})\nprint(math.ceil(z**2 * ${v.p}*(1-${v.p}) / ${v.d}**2))`, r: `z <- qnorm(${1 - (1 - v.conf) / 2})\nceiling(z^2 * ${v.p}*(1-${v.p}) / ${v.d}^2)`, spss: `COMPUTE n = (IDF.NORMAL(${1 - (1 - v.conf) / 2},0,1)**2 * ${v.p}*(1-${v.p})) / ${v.d}**2.\nEXECUTE.` }) },
    { id: "mean1", name: "Single mean (estimate an average)", use: "Estimating a mean (e.g. average systolic BP) with a given precision.",
      fields: [["sd", "Expected standard deviation", 15], ["d", "Margin of error (same units)", 3], ["conf", "Confidence level", 0.95]],
      formula: "n = (Z × σ / d)²", calc: v => { const r = SS.oneMean(v); return { n: r.n, parts: [["Z", r3(r.za)]] }; },
      methods: (v, n) => `The sample size required to estimate the mean within ±${v.d} units with ${v.conf * 100}% confidence, assuming a standard deviation of ${v.sd}, was ${n.base} (n = (Zσ/d)²)`,
      code: v => ({ stata: `display ceil((invnormal(${1 - (1 - v.conf) / 2})*${v.sd}/${v.d})^2)`, python: `from scipy.stats import norm\nimport math\nprint(math.ceil((norm.ppf(${1 - (1 - v.conf) / 2})*${v.sd}/${v.d})**2))`, r: `ceiling((qnorm(${1 - (1 - v.conf) / 2})*${v.sd}/${v.d})^2)`, spss: `COMPUTE n = (IDF.NORMAL(${1 - (1 - v.conf) / 2},0,1)*${v.sd}/${v.d})**2.\nEXECUTE.` }) },
    { id: "prop2", name: "Compare two proportions", use: "Cohort studies, trials or analytical cross-sectional studies comparing the proportion with an outcome in two groups.",
      fields: [["p1", "Proportion in group 1 (e.g. exposed)", 0.3], ["p2", "Proportion in group 2 (e.g. unexposed)", 0.15], ["alpha", "Significance level α (two-sided)", 0.05], ["power", "Power (1 − β)", 0.8], ["ratio", "Group 2 : group 1 ratio", 1]],
      formula: "n₁ = [Z₁₋α/₂ √((1 + 1/k) p̄(1 − p̄)) + Z₁₋β √(p₁(1 − p₁) + p₂(1 − p₂)/k)]² / (p₁ − p₂)²,  p̄ = (p₁ + k p₂)/(1 + k)", perGroup: true,
      calc: v => { const r = SS.twoProportions(v); return { n: r.n1, n2: r.n2, parts: [["Z₁₋α/₂", r3(r.za)], ["Z₁₋β", r3(r.zb)]] }; },
      methods: (v, n) => `The sample size for comparing two independent proportions (${v.p1 * 100}% vs ${v.p2 * 100}%) with ${v.power * 100}% power at a two-sided significance level of ${v.alpha} was ${n.base1} in group 1 and ${n.base2} in group 2 (Fleiss formula without continuity correction)`,
      code: v => ({ stata: `power twoproportions ${v.p1} ${v.p2}, alpha(${v.alpha}) power(${v.power})${v.ratio !== 1 ? ` nratio(${v.ratio})` : ""}`, python: `from scipy.stats import norm\nimport math\np1, p2, k = ${v.p1}, ${v.p2}, ${v.ratio}\nza, zb = norm.ppf(1-${v.alpha}/2), norm.ppf(${v.power})\npb = (p1 + k*p2)/(1 + k)\nn1 = (za*math.sqrt((1+1/k)*pb*(1-pb)) + zb*math.sqrt(p1*(1-p1) + p2*(1-p2)/k))**2 / (p1-p2)**2\nprint(math.ceil(n1), math.ceil(n1*k))`, r: `p1 <- ${v.p1}; p2 <- ${v.p2}; k <- ${v.ratio}\nza <- qnorm(1-${v.alpha}/2); zb <- qnorm(${v.power}); pb <- (p1 + k*p2)/(1 + k)\nn1 <- (za*sqrt((1+1/k)*pb*(1-pb)) + zb*sqrt(p1*(1-p1) + p2*(1-p2)/k))^2 / (p1-p2)^2\nc(ceiling(n1), ceiling(n1*k))`, spss: `* SPSS 27+: POWER PROPORTIONS INDEPENDENT /PARAMETERS TEST=NONDIRECTIONAL SIGNIFICANCE=${v.alpha} POWER=${v.power} PROPORTIONS=${v.p1} ${v.p2}.` }) },
    { id: "mean2", name: "Compare two means", use: "Trials or observational studies comparing a continuous outcome (e.g. BP) between two groups.",
      fields: [["sd", "Common standard deviation", 15], ["diff", "Smallest difference worth detecting", 5], ["alpha", "Significance level α (two-sided)", 0.05], ["power", "Power (1 − β)", 0.8], ["ratio", "Group 2 : group 1 ratio", 1]],
      formula: "n₁ = (1 + 1/k)(t₁₋α/₂ + t₁₋β)² σ² / Δ²  (iterated with t; normal-approximation value also shown)", perGroup: true,
      calc: v => { const r = SS.twoMeans(v); return { n: r.n1, n2: r.n2, parts: [["Z₁₋α/₂", r3(r.za)], ["Z₁₋β", r3(r.zb)], ["Normal-approximation n₁", Math.ceil(r.nNormal)]] }; },
      methods: (v, n) => `To detect a difference of ${v.diff} units (SD ${v.sd}; effect size d = ${r3(v.diff / v.sd)}) with ${v.power * 100}% power at a two-sided α of ${v.alpha}, ${n.base1} participants were required in group 1 and ${n.base2} in group 2`,
      code: v => ({ stata: `power twomeans 0 ${v.diff}, sd(${v.sd}) alpha(${v.alpha}) power(${v.power})${v.ratio !== 1 ? ` nratio(${v.ratio})` : ""}`, python: `from statsmodels.stats.power import TTestIndPower\nimport math\nn1 = TTestIndPower().solve_power(effect_size=${v.diff}/${v.sd}, alpha=${v.alpha}, power=${v.power}, ratio=${v.ratio})\nprint(math.ceil(n1), math.ceil(n1*${v.ratio}))`, r: `power.t.test(delta = ${v.diff}, sd = ${v.sd}, sig.level = ${v.alpha}, power = ${v.power})  # equal groups`, spss: `* SPSS 27+: POWER MEANS INDEPENDENT /PARAMETERS TEST=NONDIRECTIONAL SIGNIFICANCE=${v.alpha} POWER=${v.power} MEAN=0 ${v.diff} SD=${v.sd} ${v.sd}.` }) },
    { id: "cc", name: "Unmatched case–control (odds ratio)", use: "Case–control studies.",
      fields: [["p0", "Exposure prevalence among controls (0–1)", 0.2], ["or", "Smallest odds ratio worth detecting", 2], ["alpha", "Significance level α (two-sided)", 0.05], ["power", "Power (1 − β)", 0.8], ["ratio", "Controls per case", 1]],
      formula: "p₁ = OR·p₀ / (1 + p₀(OR − 1)); then the two-proportion formula with k controls per case", perGroup: true, groupNames: ["cases", "controls"],
      calc: v => { const r = SS.caseControl(v); return { n: r.cases, n2: r.controls, parts: [["Exposure among cases p₁", r3(r.p1)], ["Z₁₋α/₂", r3(r.za)], ["Z₁₋β", r3(r.zb)]] }; },
      methods: (v, n) => `Assuming ${v.p0 * 100}% exposure among controls, ${v.ratio} control(s) per case, ${v.power * 100}% power and a two-sided α of ${v.alpha}, ${n.base1} cases and ${n.base2} controls were needed to detect an odds ratio of ${v.or}`,
      code: v => ({ stata: `* exposure among cases implied by OR\nlocal p1 = ${v.or}*${v.p0}/(1 + ${v.p0}*(${v.or}-1))\npower twoproportions ${v.p0} \`p1', alpha(${v.alpha}) power(${v.power}) nratio(${1 / v.ratio})`, python: `p0, OR, k = ${v.p0}, ${v.or}, ${v.ratio}\np1 = OR*p0/(1 + p0*(OR-1))\n# then use the two-proportion formula with p1 (cases) and p0 (controls), ratio k\nprint(p1)`, r: `p0 <- ${v.p0}; OR <- ${v.or}; k <- ${v.ratio}\np1 <- OR*p0/(1 + p0*(OR-1))\nza <- qnorm(1-${v.alpha}/2); zb <- qnorm(${v.power}); pb <- (p1 + k*p0)/(1 + k)\nn1 <- (za*sqrt((1+1/k)*pb*(1-pb)) + zb*sqrt(p1*(1-p1) + p0*(1-p0)/k))^2/(p1-p0)^2\nc(cases = ceiling(n1), controls = ceiling(n1*k))`, spss: `* Compute p1 = OR*p0/(1+p0*(OR-1)) and use POWER PROPORTIONS INDEPENDENT with p1 and p0.` }) },
    { id: "corr", name: "Correlation", use: "Detecting a correlation coefficient different from zero.",
      fields: [["r", "Expected correlation r", 0.3], ["alpha", "Significance level α (two-sided)", 0.05], ["power", "Power (1 − β)", 0.8]],
      formula: "n = [(Z₁₋α/₂ + Z₁₋β) / atanh(r)]² + 3", calc: v => { const r = SS.correlation(v); return { n: r.n, parts: [["Z₁₋α/₂", r3(r.za)], ["Z₁₋β", r3(r.zb)]] }; },
      methods: (v, n) => `To detect a correlation of r = ${v.r} with ${v.power * 100}% power at a two-sided α of ${v.alpha}, ${n.base} participants were required (Fisher z method)`,
      code: v => ({ stata: `power onecorrelation 0 ${v.r}, alpha(${v.alpha}) power(${v.power})`, python: `from scipy.stats import norm\nimport math\nprint(math.ceil(((norm.ppf(1-${v.alpha}/2)+norm.ppf(${v.power}))/math.atanh(${v.r}))**2 + 3))`, r: `ceiling(((qnorm(1-${v.alpha}/2)+qnorm(${v.power}))/atanh(${v.r}))^2 + 3)`, spss: `* SPSS 27+: POWER PEARSON ONESAMPLE /PARAMETERS TEST=NONDIRECTIONAL SIGNIFICANCE=${v.alpha} POWER=${v.power} NULL=0 ALTERNATIVE=${v.r}.` }) },
  ];
  function sampleSize(id, v, adj) {
    const c = SAMPLE.find(s => s.id === id); const r = c.calc(v);
    const out = { calc: c, parts: r.parts, steps: [] };
    const doAdj = n => SS.adjust(n, { N: adj.N, deff: adj.deff, nonResponse: adj.nonResponse });
    if (c.perGroup) {
      const a1 = doAdj(r.n), a2 = doAdj(r.n2);
      out.base1 = Math.ceil(r.n - 1e-9); out.base2 = Math.ceil(r.n2 - 1e-9);
      out.final1 = Math.ceil(a1.n - 1e-9); out.final2 = Math.ceil(a2.n - 1e-9); out.total = out.final1 + out.final2;
      out.steps = a1.steps.map((s, i) => ({ label: s.label, value: `${Math.ceil(s.value - 1e-9)} + ${Math.ceil(a2.steps[i].value - 1e-9)}` }));
    } else {
      const a = doAdj(r.n); out.base = Math.ceil(r.n - 1e-9); out.final = Math.ceil(a.n - 1e-9); out.total = out.final;
      out.steps = a.steps.map(s => ({ label: s.label, value: Math.ceil(s.value - 1e-9) }));
    }
    let txt = c.methods(v, out);
    const extra = [];
    if (adj.deff && adj.deff !== 1) extra.push(`multiplied by a design effect of ${adj.deff}`);
    if (adj.N) extra.push(`corrected for a finite population of ${adj.N.toLocaleString()}`);
    if (adj.nonResponse) extra.push(`inflated by ${adj.nonResponse * 100}% for non-response`);
    txt += extra.length ? `. This was ${extra.join(", ")}, giving a final sample of ${c.perGroup ? `${out.final1} + ${out.final2} = ${out.total}` : out.final}.` : ".";
    out.methodsText = txt;
    out.code = c.code(v);
    return out;
  }

  /* =================================================================
     Statistical test selector (with code templates)
     ================================================================= */
  const T = (name, when, assumptions, effect, code, extra) => Object.assign({ name, when, assumptions, effect, code }, extra || {});
  const TESTS = {
    one_t: T("One-sample t-test", "Compare a mean with a known value.", ["Approximately normal (or n ≥ 30)"], "Cohen's d", { stata: "ttest y == 120", python: "stats.ttest_1samp(df['y'].dropna(), 120)", r: "t.test(dat$y, mu = 120)", spss: "T-TEST /TESTVAL=120 /VARIABLES=y." }, { alt: "wilcoxon_1" }),
    wilcoxon_1: T("One-sample Wilcoxon signed-rank", "Compare a median with a known value when data are skewed.", ["Symmetric distribution around the median"], "r = z/√n", { stata: "signrank y = 120", python: "stats.wilcoxon(df['y'].dropna() - 120)", r: "wilcox.test(dat$y, mu = 120, exact = FALSE)", spss: "NPTESTS /ONESAMPLE TEST (y) WILCOXON(TESTVALUE=120)." }),
    binom: T("One-sample proportion test (binomial)", "Compare a proportion with a known value.", ["Independent observations"], "Difference in proportions", { stata: "bitest y == 0.5", python: "stats.binomtest(k=df['y'].sum(), n=df['y'].count(), p=0.5)", r: "binom.test(sum(dat$y, na.rm = TRUE), sum(!is.na(dat$y)), p = 0.5)", spss: "NPAR TESTS /BINOMIAL (0.5)=y." }),
    student: T("Independent-samples t-test", "Compare means of two independent groups.", ["Normal in each group (Shapiro–Wilk, or n ≥ 30)", "Equal variances (Levene/Brown–Forsythe); otherwise Welch's t-test", "Independent observations"], "Cohen's d", { stata: "ttest y, by(group)", python: "stats.ttest_ind(g1, g2)", r: "t.test(y ~ group, data = dat, var.equal = TRUE)", spss: "T-TEST GROUPS=group(1 2) /VARIABLES=y." }, { alt: "mwu", variant: "welch" }),
    welch: T("Welch's t-test", "Compare means of two independent groups with unequal variances.", ["Normal in each group (or n ≥ 30)"], "Cohen's d", { stata: "ttest y, by(group) unequal", python: "stats.ttest_ind(g1, g2, equal_var=False)", r: "t.test(y ~ group, data = dat)", spss: "T-TEST GROUPS=group(1 2) /VARIABLES=y.  * read 'Equal variances not assumed'" }),
    mwu: T("Mann–Whitney U test", "Compare two independent groups when data are skewed or ordinal.", ["Independent observations", "Similar distribution shapes if interpreting as a difference in medians"], "r = z/√N", { stata: "ranksum y, by(group)", python: "stats.mannwhitneyu(g1, g2, method='asymptotic')", r: "wilcox.test(y ~ group, data = dat, exact = FALSE)", spss: "NPAR TESTS /M-W=y BY group(1 2)." }),
    paired_t: T("Paired t-test", "Compare two measurements on the same people (before/after).", ["Differences approximately normal"], "Cohen's dz", { stata: "ttest before == after", python: "stats.ttest_rel(df['before'], df['after'])", r: "t.test(dat$before, dat$after, paired = TRUE)", spss: "T-TEST PAIRS=before WITH after (PAIRED)." }, { alt: "wilcoxon" }),
    wilcoxon: T("Wilcoxon signed-rank test", "Paired measurements when differences are skewed or data are ordinal.", ["Differences roughly symmetric"], "r = z/√n", { stata: "signrank before = after", python: "stats.wilcoxon(df['before'], df['after'])", r: "wilcox.test(dat$before, dat$after, paired = TRUE, exact = FALSE)", spss: "NPAR TESTS /WILCOXON=before WITH after (PAIRED)." }),
    anova: T("One-way ANOVA", "Compare means of three or more independent groups.", ["Normal in each group", "Equal variances; otherwise Welch's ANOVA", "Post-hoc tests only if the overall F is significant (Tukey or Bonferroni)"], "η²", { stata: "oneway y group, bonferroni tabulate", python: "stats.f_oneway(*groups)", r: "summary(aov(y ~ group, data = dat)); TukeyHSD(aov(y ~ group, data = dat))", spss: "ONEWAY y BY group /POSTHOC=TUKEY BONFERRONI ALPHA(0.05)." }, { alt: "kruskal", variant: "welch_anova" }),
    welch_anova: T("Welch's ANOVA", "Three or more groups with unequal variances.", ["Normal in each group"], "ω²", { stata: "* user-written: ssc install wtest\nwtest y group", python: "from statsmodels.stats.oneway import anova_oneway\nanova_oneway(groups, use_var='unequal')", r: "oneway.test(y ~ group, data = dat); pairwise.t.test(dat$y, dat$group, pool.sd = FALSE, p.adjust.method = 'bonferroni')", spss: "ONEWAY y BY group /STATISTICS WELCH /POSTHOC=GH." }),
    kruskal: T("Kruskal–Wallis test", "Three or more independent groups, skewed or ordinal data.", ["Independent observations", "Post-hoc: Dunn's test or pairwise Mann–Whitney with Bonferroni"], "ε²", { stata: "kwallis y, by(group)\n* post-hoc: ssc install dunntest", python: "stats.kruskal(*groups)", r: "kruskal.test(y ~ group, data = dat); pairwise.wilcox.test(dat$y, dat$group, p.adjust.method = 'bonferroni', exact = FALSE)", spss: "NPAR TESTS /K-W=y BY group(1 3)." }),
    rm_anova: T("Repeated-measures ANOVA", "The same people measured three or more times (or conditions).", ["Normality of residuals", "Sphericity (Mauchly's test); otherwise Greenhouse–Geisser correction", "No missing time points (else use a mixed model)"], "Partial η²", { stata: "* long format: id time y\nanova y id time, repeated(time)", python: "from statsmodels.stats.anova import AnovaRM\nprint(AnovaRM(long_df, 'y', 'id', within=['time']).fit())", r: "summary(aov(y ~ time + Error(id/time), data = long))", spss: "GLM t1 t2 t3 /WSFACTOR=time 3 Polynomial /PRINT=DESCRIPTIVE ETASQ." }, { alt: "friedman" }),
    friedman: T("Friedman test", "Three or more repeated measurements, skewed or ordinal.", ["Same subjects across conditions"], "Kendall's W", { stata: "* ssc install emh\nfriedman t1 t2 t3", python: "stats.friedmanchisquare(df['t1'], df['t2'], df['t3'])", r: "friedman.test(as.matrix(dat[, c('t1','t2','t3')]))", spss: "NPAR TESTS /FRIEDMAN=t1 t2 t3." }),
    pearson: T("Pearson correlation", "Linear relationship between two continuous variables.", ["Both approximately normal", "Linear relationship (check scatter plot)", "No extreme outliers"], "r (r² = shared variance)", { stata: "pwcorr x y, sig obs", python: "stats.pearsonr(df['x'], df['y'])", r: "cor.test(dat$x, dat$y)", spss: "CORRELATIONS /VARIABLES=x y /PRINT=TWOTAIL." }, { alt: "spearman" }),
    spearman: T("Spearman rank correlation", "Monotonic relationship; skewed or ordinal variables.", ["Monotonic relationship"], "ρ", { stata: "spearman x y", python: "stats.spearmanr(df['x'], df['y'])", r: "cor.test(dat$x, dat$y, method = 'spearman', exact = FALSE)", spss: "NONPAR CORR /VARIABLES=x y /PRINT=SPEARMAN TWOTAIL." }),
    chi2: T("Chi-square test of independence", "Association between two categorical variables.", ["Independent observations", "Expected counts ≥ 5 in ≥ 80% of cells and none < 1; otherwise Fisher's exact test"], "Cramér's V; OR/RR for 2×2", { stata: "tabulate a b, chi2 expected row", python: "stats.chi2_contingency(pd.crosstab(df['a'], df['b']), correction=False)", r: "chisq.test(table(dat$a, dat$b), correct = FALSE)", spss: "CROSSTABS /TABLES=a BY b /STATISTICS=CHISQ PHI /CELLS=COUNT ROW EXPECTED." }, { alt: "fisher" }),
    fisher: T("Fisher's exact test", "Categorical association with small expected counts.", ["Fixed margins (conditional test)"], "OR for 2×2", { stata: "tabulate a b, exact", python: "stats.fisher_exact(pd.crosstab(df['a'], df['b']))  # 2x2 only", r: "fisher.test(table(dat$a, dat$b))", spss: "CROSSTABS /TABLES=a BY b /STATISTICS=CHISQ /METHOD=EXACT." }),
    mcnemar: T("McNemar's test", "Paired binary data (before/after, matched pairs).", ["Exact version when discordant pairs < 25"], "Difference in proportions; matched OR = b/c", { stata: "mcc before after", python: "from statsmodels.stats.contingency_tables import mcnemar\nmcnemar(pd.crosstab(df['before'], df['after']), exact=False, correction=True)", r: "mcnemar.test(table(dat$before, dat$after))", spss: "NPAR TESTS /MCNEMAR=before WITH after (PAIRED)." }),
    cochran_q: T("Cochran's Q test", "Binary outcome measured three or more times on the same people.", ["Post-hoc: pairwise McNemar with Bonferroni"], "Kendall's W", { stata: "* ssc install cochran\ncochran t1 t2 t3", python: "from statsmodels.stats.contingency_tables import cochrans_q\ncochrans_q(df[['t1','t2','t3']])", r: "# install.packages('RVAideMemoire')\nRVAideMemoire::cochran.qtest(y ~ time | id, data = long)", spss: "NPAR TESTS /COCHRAN=t1 t2 t3." }),
    linear: T("Multiple linear regression", "Continuous outcome with several predictors (adjusted effects).", ["Linearity", "Independent errors", "Homoscedasticity (Breusch–Pagan); otherwise robust SEs", "Normal residuals (less important when n is large)", "No multicollinearity (VIF < 5–10)", "≥ 10–20 observations per predictor"], "B with 95% CI; R²", { stata: "regress y x1 x2 i.group\nestat vif\nestat hettest", python: "smf.ols('y ~ x1 + x2 + C(group)', data=df).fit().summary()", r: "summary(lm(y ~ x1 + x2 + group, data = dat))", spss: "REGRESSION /STATISTICS COEFF CI(95) R ANOVA COLLIN /DEPENDENT y /METHOD=ENTER x1 x2." }),
    logistic: T("Binary logistic regression", "Binary outcome with several predictors.", ["≥ 10 events per predictor", "Linearity of continuous predictors with the log-odds", "No multicollinearity", "Goodness of fit (Hosmer–Lemeshow), discrimination (AUC)", "If the outcome is common (> 10%) and you want prevalence/risk ratios: modified Poisson"], "Adjusted OR with 95% CI", { stata: "logistic y x1 x2 i.group\nestat gof, group(10)\nlroc", python: "m = smf.logit('y ~ x1 + x2 + C(group)', data=df).fit(); np.exp(m.params)", r: "m <- glm(y ~ x1 + x2 + group, family = binomial, data = dat); exp(cbind(coef(m), confint.default(m)))", spss: "LOGISTIC REGRESSION VARIABLES y /METHOD=ENTER x1 x2 group /CATEGORICAL=group /PRINT=GOODFIT CI(95)." }),
    modpoisson: T("Modified Poisson regression (prevalence/risk ratios)", "Binary outcome that is common, in cross-sectional or cohort studies.", ["Robust (sandwich) standard errors always", "Same predictor checks as logistic regression"], "Adjusted PR/RR with 95% CI", { stata: "poisson y x1 x2 i.group, irr vce(robust)", python: "smf.glm('y ~ x1 + x2 + C(group)', data=df, family=sm.families.Poisson()).fit(cov_type='HC0')", r: "m <- glm(y ~ x1 + x2 + group, family = poisson, data = dat)\nlibrary(sandwich); lmtest::coeftest(m, vcov = vcovHC(m, 'HC0'))", spss: "GENLIN y BY group WITH x1 x2 /MODEL group x1 x2 DISTRIBUTION=POISSON LINK=LOG /CRITERIA COVB=ROBUST /PRINT SOLUTION (EXPONENTIATED)." }),
    multinomial: T("Multinomial logistic regression", "Nominal outcome with three or more unordered categories.", ["Independence of irrelevant alternatives", "Adequate events in every category"], "Relative risk ratios", { stata: "mlogit y x1 x2, rrr baseoutcome(1)", python: "smf.mnlogit('y ~ x1 + x2', data=df).fit().summary()", r: "nnet::multinom(y ~ x1 + x2, data = dat)", spss: "NOMREG y (BASE=FIRST) WITH x1 x2 /PRINT=PARAMETER SUMMARY." }),
    ordinal: T("Ordinal logistic regression", "Ordered outcome (e.g. Likert, disease stage).", ["Proportional odds (Brant test); otherwise generalized ordinal model"], "Cumulative OR", { stata: "ologit y x1 x2, or\n* ssc install spost13_ado; brant", python: "from statsmodels.miscmodels.ordinal_model import OrderedModel\nOrderedModel(df['y'], df[['x1','x2']], distr='logit').fit(method='bfgs').summary()", r: "MASS::polr(factor(y, ordered = TRUE) ~ x1 + x2, data = dat, Hess = TRUE)", spss: "PLUM y WITH x1 x2 /LINK=LOGIT /PRINT=PARAMETER SUMMARY TPARALLEL." }),
    poisson: T("Poisson regression", "Count outcome (visits, episodes) or rates with person-time.", ["Mean ≈ variance: check dispersion (Pearson χ²/df); if > 1.5 use robust SEs or negative binomial", "Use log(person-time) as offset for rates"], "Incidence rate ratio", { stata: "poisson y x1 x2, irr\nestat gof", python: "smf.glm('y ~ x1 + x2', data=df, family=sm.families.Poisson()).fit()", r: "glm(y ~ x1 + x2, family = poisson, data = dat)", spss: "GENLIN y WITH x1 x2 /MODEL x1 x2 DISTRIBUTION=POISSON LINK=LOG /PRINT FIT SOLUTION (EXPONENTIATED)." }, { variant: "negbin" }),
    negbin: T("Negative binomial regression", "Overdispersed counts.", ["Overdispersion present (α > 0)"], "Incidence rate ratio", { stata: "nbreg y x1 x2, irr", python: "smf.negativebinomial('y ~ x1 + x2', data=df).fit().summary()", r: "MASS::glm.nb(y ~ x1 + x2, data = dat)", spss: "GENLIN y WITH x1 x2 /MODEL x1 x2 DISTRIBUTION=NEGBIN(MLE) LINK=LOG /PRINT SOLUTION (EXPONENTIATED)." }),
    km: T("Kaplan–Meier with log-rank test", "Compare time-to-event (survival) between groups.", ["Non-informative censoring", "Proportional hazards for the log-rank test to be most powerful"], "Median survival; hazard ratio (from Cox)", { stata: "stset time, failure(event)\nsts graph, by(group)\nsts test group", python: "# pip install lifelines\nfrom lifelines.statistics import logrank_test", r: "library(survival)\nsurvdiff(Surv(time, event) ~ group, data = dat)", spss: "KM time BY group /STATUS=event(1) /TEST=LOGRANK." }, { alt: "cox" }),
    cox: T("Cox proportional hazards regression", "Time-to-event outcome with several predictors.", ["Proportional hazards (Schoenfeld residuals)", "≥ 10 events per predictor"], "Adjusted hazard ratio", { stata: "stset time, failure(event)\nstcox x1 x2 i.group\nestat phtest, detail", python: "# pip install lifelines\nfrom lifelines import CoxPHFitter\nCoxPHFitter().fit(df, 'time', 'event').print_summary()", r: "library(survival)\nm <- coxph(Surv(time, event) ~ x1 + x2 + group, data = dat); summary(m); cox.zph(m)", spss: "COXREG time /STATUS=event(1) /METHOD=ENTER x1 x2 group /CATEGORICAL=group /PRINT=CI(95)." }),
    kappa: T("Cohen's kappa", "Agreement between two raters/tests on a categorical result.", ["Same categories for both raters"], "κ (≤ .20 slight, .21–.40 fair, .41–.60 moderate, .61–.80 substantial, > .80 almost perfect)", { stata: "kap rater1 rater2", python: "from sklearn.metrics import cohen_kappa_score\ncohen_kappa_score(df['r1'], df['r2'])", r: "# install.packages('irr')\nirr::kappa2(dat[, c('r1','r2')])", spss: "CROSSTABS /TABLES=r1 BY r2 /STATISTICS=KAPPA." }),
    icc: T("Intraclass correlation (ICC)", "Agreement/reliability of continuous measurements.", ["Choose model (one-way/two-way), type (agreement/consistency) and unit (single/average)"], "ICC with 95% CI (< .5 poor, .5–.75 moderate, .75–.9 good, > .9 excellent)", { stata: "icc y target rater, absolute", python: "# pip install pingouin\nimport pingouin as pg\npg.intraclass_corr(long_df, targets='id', raters='rater', ratings='y')", r: "# install.packages('irr')\nirr::icc(dat[, c('r1','r2')], model = 'twoway', type = 'agreement')", spss: "RELIABILITY /VARIABLES=r1 r2 /ICC=MODEL(MIXED) TYPE(ABSOLUTE) CIN=95." }),
    diag_acc: T("Sensitivity, specificity, predictive values", "Accuracy of a test against a reference standard.", ["Reference standard applied to everyone", "Exact (Clopper–Pearson) CIs"], "Sensitivity, specificity, PPV, NPV, LR+, LR−; AUC for continuous tests", { stata: "* ssc install diagt\ndiagt reference test", python: "tab = pd.crosstab(df['test'], df['reference'])\nsens = tab.loc[1,1]/tab[1].sum(); spec = tab.loc[0,0]/tab[0].sum()", r: "tab <- table(dat$test, dat$reference)\nbinom.test(tab['1','1'], sum(tab[, '1']))  # sensitivity with exact CI", spss: "CROSSTABS /TABLES=test BY reference /CELLS=COUNT COLUMN.\nROC score BY reference (1) /PRINT=SE COORDINATES." }),
  };
  const TEST_QUESTIONS = [
    { id: "goal", q: "What do you want to do?", options: [["compare", "Compare groups or time points"], ["relate", "Measure the relationship between two variables"], ["predict", "Model an outcome with several predictors (adjusted effects)"], ["agree", "Measure agreement or test accuracy"], ["onesample", "Compare one sample with a known value"]] },
    { id: "outcome", q: "What type is the outcome (dependent) variable?", show: a => a.goal && a.goal !== "relate" && a.goal !== "agree", options: [["cont", "Continuous (BP, weight, score)"], ["ord", "Ordinal (Likert, stage)"], ["bin", "Binary (yes/no)"], ["nom", "Nominal with 3+ categories"], ["count", "Count (visits, episodes)"], ["time", "Time to an event (survival)"]] },
    { id: "groups", q: "How many groups or time points?", show: a => a.goal === "compare", options: [["2", "Two"], ["3", "Three or more"]] },
    { id: "paired", q: "Are the groups independent or paired?", show: a => a.goal === "compare" && a.outcome !== "time", options: [["ind", "Independent (different people)"], ["pair", "Paired (same people measured repeatedly, or matched)"]] },
    { id: "normal", q: "Are the data approximately normal? (Shapiro–Wilk p ≥ .05, or n ≥ 30 per group with mild skew)", show: a => (a.goal === "compare" || a.goal === "onesample") && a.outcome === "cont", options: [["yes", "Yes"], ["no", "No / skewed"], ["unsure", "Not sure yet"]] },
    { id: "vars", q: "What types are the two variables?", show: a => a.goal === "relate", options: [["cc", "Both continuous"], ["oo", "At least one ordinal, or skewed"], ["catcat", "Both categorical"], ["catcont", "One categorical, one continuous"]] },
    { id: "common", q: "Is the outcome common (more than about 10%) and do you want prevalence/risk ratios?", show: a => a.goal === "predict" && a.outcome === "bin", options: [["no", "Rare, or odds ratios are fine (e.g. case–control)"], ["yes", "Common, cross-sectional or cohort: prevalence/risk ratios"]] },
    { id: "agreetype", q: "What kind of agreement?", show: a => a.goal === "agree", options: [["cat", "Two raters/tests, categorical result"], ["cont", "Continuous measurements"], ["acc", "Test vs reference standard"]] },
  ];
  function selectTest(a) {
    const g = a.goal, o = a.outcome, why = [];
    let id = null, alt = null;
    const normal = a.normal === "yes";
    if (g === "onesample") { id = o === "bin" ? "binom" : (o === "cont" && normal) ? "one_t" : "wilcoxon_1"; if (o === "cont" && a.normal === "unsure") alt = "wilcoxon_1"; }
    else if (g === "compare") {
      if (o === "time") id = "km";
      else if (o === "bin" || o === "nom") id = a.paired === "pair" ? (a.groups === "2" && o === "bin" ? "mcnemar" : "cochran_q") : "chi2";
      else if (o === "count") id = a.paired === "pair" ? (a.groups === "2" ? "wilcoxon" : "friedman") : (a.groups === "2" ? "mwu" : "kruskal");
      else {
        const par = o === "cont" && normal;
        if (a.paired === "pair") id = a.groups === "2" ? (par ? "paired_t" : "wilcoxon") : (par ? "rm_anova" : "friedman");
        else id = a.groups === "2" ? (par ? "student" : "mwu") : (par ? "anova" : "kruskal");
        if (o === "cont" && a.normal === "unsure") why.push("Check normality first (Shapiro–Wilk per group, histogram/Q–Q plot). The parametric test is shown with its non-parametric alternative.");
        if (o === "cont" && a.normal === "unsure") { id = a.paired === "pair" ? (a.groups === "2" ? "paired_t" : "rm_anova") : (a.groups === "2" ? "student" : "anova"); }
      }
      if (o === "ord") why.push("Ordinal outcomes use rank-based tests because the distance between categories is not meaningful.");
    }
    else if (g === "relate") id = a.vars === "cc" ? "pearson" : a.vars === "oo" ? "spearman" : a.vars === "catcat" ? "chi2" : a.vars === "catcont" ? null : null;
    else if (g === "predict") id = { cont: "linear", bin: a.common === "yes" ? "modpoisson" : "logistic", nom: "multinomial", ord: "ordinal", count: "poisson", time: "cox" }[o];
    else if (g === "agree") id = { cat: "kappa", cont: "icc", acc: "diag_acc" }[a.agreetype];
    if (g === "relate" && a.vars === "catcont") { why.push("With one categorical and one continuous variable, the question is a group comparison: use 'Compare groups' with the continuous variable as the outcome."); return { id: null, why }; }
    if (!id) return null;
    const t = TESTS[id];
    const altId = alt || t.alt || null, variantId = t.variant || null;
    return { id, test: t, why, alternative: altId ? { id: altId, test: TESTS[altId] } : null, variant: variantId ? { id: variantId, test: TESTS[variantId] } : null };
  }

  /* =================================================================
     Research question builder
     ================================================================= */
  const FRAMEWORKS = {
    PECO: { name: "PECO (observational)", fields: [["P", "Population", "adults aged 18+ in Tamale Metropolis"], ["E", "Exposure", "high dietary salt intake"], ["C", "Comparison", "low salt intake"], ["O", "Outcome", "hypertension"]] },
    PICO: { name: "PICO (intervention)", fields: [["P", "Population", "adults with hypertension attending CHPS compounds"], ["I", "Intervention", "SMS medication reminders"], ["C", "Comparison", "usual care"], ["O", "Outcome", "blood pressure control at 6 months"]] },
    SPIDER: { name: "SPIDER (qualitative)", fields: [["S", "Sample", "nurses in rural health facilities"], ["PI", "Phenomenon of interest", "managing hypertension with limited equipment"], ["D", "Design", "in-depth interviews"], ["E", "Evaluation", "experiences and coping strategies"], ["R", "Research type", "qualitative"]] },
    PCC: { name: "PCC (scoping review)", fields: [["P", "Population", "adults in sub-Saharan Africa"], ["C", "Concept", "digital interventions for hypertension"], ["Ctx", "Context", "primary health care"]] },
  };
  function buildQuestion(fw, v, setting) {
    const at = setting ? ` in ${setting}` : "";
    if (fw === "PECO") return {
      question: `Among ${v.P}${at}, is ${v.E}, compared with ${v.C}, associated with ${v.O}?`,
      general: `To assess the association between ${v.E} and ${v.O} among ${v.P}${at}.`,
      specific: [`To determine the prevalence of ${v.O} among ${v.P}${at}.`, `To describe the distribution of ${v.E} among ${v.P}.`, `To compare ${v.O} between those with ${v.E} and those with ${v.C}.`, `To identify other factors associated with ${v.O} after adjusting for confounders.`],
      h0: `There is no association between ${v.E} and ${v.O} among ${v.P}.`, h1: `There is an association between ${v.E} and ${v.O} among ${v.P}.` };
    if (fw === "PICO") return {
      question: `Among ${v.P}${at}, does ${v.I}, compared with ${v.C}, improve ${v.O}?`,
      general: `To evaluate the effect of ${v.I} on ${v.O} among ${v.P}${at}.`,
      specific: [`To measure ${v.O} at baseline in the intervention and comparison groups.`, `To compare ${v.O} between ${v.I} and ${v.C} at follow-up.`, `To assess the acceptability and fidelity of ${v.I}.`, `To identify factors that modify the effect of ${v.I}.`],
      h0: `${cap(v.I)} has no effect on ${v.O} compared with ${v.C}.`, h1: `${cap(v.I)} changes ${v.O} compared with ${v.C}.` };
    if (fw === "SPIDER") return {
      question: `What are the ${v.E} of ${v.S}${at} regarding ${v.PI}?`,
      general: `To explore the ${v.E} of ${v.S} regarding ${v.PI}${at}.`,
      specific: [`To describe how ${v.S} understand ${v.PI}.`, `To explore barriers and facilitators ${v.S} face regarding ${v.PI}.`, `To identify strategies ${v.S} use or recommend.`],
      h0: null, h1: null, note: `Qualitative studies use research questions, not hypotheses. Planned design: ${v.D} (${v.R}).` };
    return {
      question: `What is known about ${v.C} among ${v.P} in ${v.Ctx}?`,
      general: `To map the evidence on ${v.C} among ${v.P} in ${v.Ctx}.`,
      specific: [`To identify the types of ${v.C} that have been studied.`, `To describe the study designs and outcomes reported.`, `To identify gaps in the evidence.`],
      h0: null, h1: null, note: "Scoping reviews map evidence and do not test hypotheses." };
  }
  const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;
  const FINER = [["Feasible", "Enough participants, time, money, skills and access"], ["Interesting", "Matters to you, your supervisor and the field"], ["Novel", "Confirms, refutes or extends previous findings, or fills a gap"], ["Ethical", "Can be approved by an ethics committee; risks are minimal and justified"], ["Relevant", "Results could change practice, policy or future research"]];

  /* =================================================================
     Reporting checklists (paraphrased; see the official statements)
     ================================================================= */
  const CHECKLISTS = {
    STROBE: { name: "STROBE — observational studies", url: "https://www.strobe-statement.org", sections: [
      ["Title and abstract", ["Indicate the study design in the title or abstract", "Give an informative, balanced summary of what was done and found"]],
      ["Introduction", ["Explain the scientific background and rationale", "State specific objectives, including any pre-specified hypotheses"]],
      ["Methods", ["Present key elements of study design early", "Describe setting, locations, dates of recruitment, exposure, follow-up and data collection", "Give eligibility criteria and methods of selecting participants (and matching criteria / follow-up methods where relevant)", "Define all outcomes, exposures, predictors, potential confounders and effect modifiers", "Give sources of data and measurement methods for each variable; describe comparability across groups", "Describe efforts to address potential sources of bias", "Explain how the study size was arrived at", "Explain how quantitative variables were handled and grouped", "Describe all statistical methods, including control for confounding, subgroups, interactions, missing data, loss to follow-up and sensitivity analyses"]],
      ["Results", ["Report numbers at each stage (eligible, examined, included, analysed) with reasons for non-participation; consider a flow diagram", "Give participant characteristics and number with missing data for each variable", "Report numbers of outcome events or summary measures", "Give unadjusted and confounder-adjusted estimates with precision (95% CI) and state which confounders were adjusted for", "Report other analyses (subgroups, interactions, sensitivity)"]],
      ["Discussion", ["Summarise key results with reference to objectives", "Discuss limitations, including sources of bias or imprecision, and their direction and magnitude", "Give a cautious overall interpretation considering objectives, limitations, multiplicity and other evidence", "Discuss generalisability (external validity)"]],
      ["Other", ["Give the source of funding and the role of funders"]]] },
    CONSORT: { name: "CONSORT 2010 — randomised trials", url: "https://www.consort-spirit.org", sections: [
      ["Title and abstract", ["Identify as a randomised trial in the title", "Structured summary of design, methods, results and conclusions"]],
      ["Introduction", ["Scientific background and rationale", "Specific objectives or hypotheses"]],
      ["Methods", ["Trial design (parallel, factorial, cluster) and allocation ratio; any changes after start", "Eligibility criteria and settings", "Interventions for each group in enough detail to replicate", "Pre-specified primary and secondary outcomes, how and when assessed", "How sample size was determined; interim analyses and stopping rules", "Method of generating the random sequence and type of randomisation", "Allocation concealment mechanism", "Who generated the sequence, enrolled participants and assigned interventions", "Who was blinded and how", "Statistical methods for primary and secondary outcomes and additional analyses"]],
      ["Results", ["Participant flow diagram with numbers randomised, treated and analysed in each group", "Losses and exclusions after randomisation with reasons", "Dates of recruitment and follow-up; why the trial ended", "Baseline demographic and clinical characteristics by group (Table 1)", "Numbers analysed in each group and whether by original assigned groups (intention-to-treat)", "For each outcome: results by group, effect size and 95% CI (absolute and relative for binary outcomes)", "Other analyses, distinguishing pre-specified from exploratory", "All important harms or unintended effects"]],
      ["Discussion", ["Limitations, sources of bias, imprecision and multiplicity", "Generalisability", "Interpretation consistent with results, balancing benefits and harms"]],
      ["Other", ["Registration number and registry", "Where the full protocol can be accessed", "Sources of funding and role of funders"]]] },
    STARD: { name: "STARD 2015 — diagnostic accuracy", url: "https://www.equator-network.org/reporting-guidelines/stard/", sections: [
      ["Title and abstract", ["Identify as a diagnostic accuracy study with at least one accuracy measure", "Structured summary"]],
      ["Introduction", ["Scientific and clinical background, including intended use and role of the index test", "Study objectives and hypotheses"]],
      ["Methods", ["Whether data collection was planned before (prospective) or after (retrospective) the tests", "Eligibility criteria, how participants were identified and recruited, setting, dates", "Whether participants were a consecutive, random or convenience series", "Index test and reference standard in enough detail to replicate, including rationale for the reference standard", "Definition and rationale for test positivity cut-offs (pre-specified or exploratory)", "Whether readers of each test were blinded to the other", "Methods for estimating accuracy measures and their precision (95% CI)", "Handling of indeterminate results and missing data", "Intended sample size and how it was determined"]],
      ["Results", ["Flow of participants (diagram)", "Baseline characteristics and distribution of severity / alternative diagnoses", "Time interval and any interventions between index test and reference standard", "Cross-tabulation of index test results by reference standard results", "Estimates of accuracy (sensitivity, specificity, etc.) with 95% CI", "Any adverse events from performing the tests"]],
      ["Discussion", ["Limitations, including sources of bias and uncertainty", "Implications for practice, including intended use and clinical role"]],
      ["Other", ["Registration, protocol access, and funding"]]] },
    COREQ: { name: "COREQ — interviews and focus groups", url: "https://www.equator-network.org/reporting-guidelines/coreq/", sections: [
      ["Research team and reflexivity", ["Who conducted interviews/FGDs, with their credentials, occupation, gender, experience and training", "Relationship with participants before the study and what participants knew about the researcher", "Interviewer characteristics that may influence the research (assumptions, interests)"]],
      ["Study design", ["Methodological orientation (e.g. grounded theory, phenomenology, content analysis)", "Sampling method, how participants were approached, sample size, and number who refused or dropped out", "Setting of data collection, presence of non-participants, and participant characteristics", "Interview guide (pilot tested?), repeat interviews, audio/visual recording, field notes", "Duration of interviews/FGDs, whether data saturation was discussed, and whether transcripts were returned to participants"]],
      ["Analysis and findings", ["Number of coders and description of the coding tree", "Whether themes were identified in advance or derived from the data", "Software used, and whether participants gave feedback on findings", "Participant quotations presented and identified", "Consistency between data and findings; clarity of major and minor themes"]]] },
    PRISMA: { name: "PRISMA 2020 — systematic reviews", url: "https://www.prisma-statement.org", sections: [
      ["Title and abstract", ["Identify the report as a systematic review", "Structured abstract (PRISMA for Abstracts)"]],
      ["Introduction", ["Rationale in the context of existing knowledge", "Explicit objectives or questions"]],
      ["Methods", ["Eligibility criteria and how studies were grouped for synthesis", "All information sources searched with dates", "Full search strategies for all databases", "Selection process (number of reviewers, independence, automation tools)", "Data collection process and data items sought", "Risk-of-bias assessment methods and tools", "Effect measures for each outcome", "Synthesis methods, heterogeneity exploration and sensitivity analyses", "Reporting bias assessment and certainty assessment (e.g. GRADE)"]],
      ["Results", ["Study selection with a PRISMA flow diagram", "Characteristics of included studies", "Risk of bias in each study", "Results of individual studies and of syntheses (pooled estimate, CI, heterogeneity)", "Reporting biases and certainty of evidence"]],
      ["Discussion", ["Interpretation in the context of other evidence", "Limitations of the evidence and of the review process", "Implications for practice, policy and research"]],
      ["Other", ["Registration (e.g. PROSPERO) and protocol", "Support/funding and competing interests", "Availability of data, code and materials"]]] },
  };

  return { DESIGNS, DESIGN_QUESTIONS, decideDesign, SAMPLE, sampleSize, TESTS, TEST_QUESTIONS, selectTest, FRAMEWORKS, buildQuestion, FINER, CHECKLISTS };
})();
if (false) module.exports = Method;

const EXAMPLE_CSV = "Participant ID,Age (years),Sex,District,Residence,Education level,Salt intake,BMI,Systolic BP,Systolic BP 3 months,Hypertension,Smoker,Clinic visits (12m),Aware of status (baseline),Aware of status (3 months),Clinic,Systolic BP 6 months,Follow-up (months),Lost to follow-up,Self-rated health,Usual source of care\nGH-0001,55,Female,Tamale,Rural,Primary,High,17.2,152,149,Yes,No,4,No,Yes,TAM-2,137,10.6,No,Poor,CHPS compound\nGH-0002,36,Female,Kumasi,Urban,Primary,Low,20.7,124,114,No,No,2,Yes,Yes,KUM-2,117,5.7,Yes,Very good,Hospital\nGH-0003,38,Male,Kumasi,Urban,Secondary,Moderate,49.0,145,143,Yes,No,2,Yes,No,KUM-1,141,12.1,No,Good,CHPS compound\nGH-0004,55,Male,Accra,Rural,Secondary,Moderate,25.1,121,113,No,No,5,Yes,Yes,ACC-2,124,7.6,No,Fair,CHPS compound\nGH-0005,53,Female,Accra,Urban,Secondary,Moderate,24.0,115,,No,No,1,No,Yes,ACC-2,110,12.5,No,Good,Hospital\nGH-0006,43,Female,Ho,Rural,Primary,High,12.2,115,109,No,No,1,No,No,HO-2,104,11.2,No,Fair,CHPS compound\nGH-0007,62,Female,Cape Coast,Urban,None,High,15.6,113,101,No,No,3,No,No,CAP-4,107,19.4,Yes,Fair,Hospital\nGH-0008,45,Female,Tamale,Rural,Primary,Moderate,34.0,143,156,Yes,No,1,Yes,Yes,TAM-1,133,16.4,Yes,Very good,CHPS compound\nGH-0009,21,Male,Kumasi,Urban,Primary,Moderate,42.4,134,134,No,No,0,No,No,KUM-1,125,23.2,No,Good,CHPS compound\nGH-0010,36,Male,Accra,Urban,None,High,26.0,110,106,No,No,0,No,Yes,ACC-3,101,2.6,Yes,Good,Pharmacy\nGH-0011,27,Male,Cape Coast,Rural,None,Low,19.1,107,105,No,No,0,Yes,Yes,CAP-1,96,24.4,No,Fair,CHPS compound\nGH-0012,61,Female,Cape Coast,Rural,Secondary,High,20.7,144,,Yes,No,0,No,No,CAP-2,148,2.1,Yes,Good,Hospital\nGH-0013,53,Female,Kumasi,Urban,Primary,Moderate,21.2,141,127,Yes,No,1,No,Yes,KUM-1,132,4.2,Yes,Very good,Hospital\nGH-0014,48,Male,Ho,Urban,Tertiary,Low,25.8,131,127,No,No,5,Yes,Yes,HO-3,125,35.3,No,Good,Hospital\nGH-0015,18,Female,Kumasi,Urban,Secondary,Moderate,27.7,121,113,No,No,0,No,No,KUM-1,111,20.5,No,Poor,CHPS compound\nGH-0016,27,Female,Ho,Urban,None,Moderate,46.1,138,131,No,No,2,No,No,HO-1,140,10.1,No,Fair,Pharmacy\nGH-0017,44,Male,Accra,Urban,Secondary,High,25.1,125,113,No,Yes,3,Yes,Yes,ACC-1,105,8.6,No,Good,CHPS compound\nGH-0018,23,Female,Accra,Rural,None,Low,17.6,106,96,No,No,0,Yes,Yes,ACC-2,106,24.7,No,Fair,Pharmacy\nGH-0019,58,Female,Accra,Urban,Primary,Moderate,23.1,132,115,No,No,3,No,No,ACC-3,134,32.0,No,Fair,Hospital\nGH-0020,45,Female,Tamale,Rural,Primary,High,33.9,129,116,No,No,1,No,No,TAM-2,116,26.2,No,Good,Hospital\nGH-0021,44,Male,Kumasi,Urban,None,High,22.1,132,134,No,No,3,Yes,Yes,KUM-1,145,3.2,Yes,Poor,CHPS compound\nGH-0022,45,Female,Accra,Rural,Tertiary,Moderate,12.3,113,107,No,No,1,No,No,ACC-4,105,18.2,No,Fair,Hospital\nGH-0023,34,Male,Kumasi,Urban,Primary,High,13.4,120,108,No,No,1,No,No,KUM-3,92,11.5,No,Fair,CHPS compound\nGH-0024,32,Female,Tamale,Rural,Secondary,Low,21.5,112,107,No,No,2,No,No,TAM-4,97,18.8,No,Poor,CHPS compound\nGH-0025,21,Female,Accra,Urban,Primary,Moderate,20.3,108,129,No,No,0,No,No,ACC-1,80,23.5,No,Good,CHPS compound\nGH-0026,46,Female,Accra,Urban,Secondary,Low,30.1,104,84,No,No,4,Yes,Yes,ACC-1,89,18.6,No,Good,CHPS compound\nGH-0027,51,Male,Tamale,Urban,Tertiary,Low,30.4,164,156,Yes,Yes,0,Yes,Yes,TAM-1,169,34.2,No,Very good,CHPS compound\nGH-0028,43,Female,Kumasi,Urban,None,High,44.8,155,149,Yes,No,1,No,No,KUM-3,154,2.4,Yes,Fair,Hospital\nGH-0029,42,Male,Accra,Urban,Primary,High,21.8,133,126,No,No,3,No,No,ACC-2,128,2.7,Yes,Very good,Hospital\nGH-0030,34,Male,Accra,Urban,Secondary,Moderate,25.8,146,150,Yes,No,0,Yes,Yes,ACC-1,138,12.6,No,Very good,CHPS compound\nGH-0031,49,Male,Tamale,Urban,None,Moderate,36.5,154,150,Yes,No,4,Yes,Yes,TAM-2,119,31.2,No,Good,Hospital\nGH-0032,18,Female,Cape Coast,Rural,Secondary,High,19.7,115,114,No,No,0,Yes,Yes,CAP-2,113,23.3,No,Good,CHPS compound\nGH-0033,32,Female,Ho,Rural,Tertiary,Moderate,19.0,121,125,No,No,3,Yes,Yes,HO-3,118,23.7,No,Very good,CHPS compound\nGH-0034,37,Male,Kumasi,Urban,Secondary,Moderate,16.8,138,133,No,No,1,No,Yes,KUM-1,125,13.4,Yes,Fair,CHPS compound\nGH-0035,39,Female,Accra,Rural,Secondary,Low,34.3,131,124,No,No,2,No,No,ACC-2,128,7.3,Yes,Very good,CHPS compound\nGH-0036,49,Female,Cape Coast,Rural,Primary,Moderate,28.7,113,,No,No,1,No,No,CAP-2,100,14.6,No,Fair,CHPS compound\nGH-0037,43,Male,Kumasi,Rural,None,Low,20.3,101,95,No,Yes,3,Yes,Yes,KUM-1,115,6.3,No,Good,CHPS compound\nGH-0038,48,Male,Accra,Rural,Tertiary,Moderate,15.8,139,136,No,No,2,No,No,ACC-1,125,30.3,No,Very good,CHPS compound\nGH-0039,62,Female,Cape Coast,Rural,Secondary,Low,34.1,138,129,No,No,1,No,No,CAP-2,133,0.8,Yes,Fair,CHPS compound\nGH-0040,64,Female,Accra,Rural,Secondary,Moderate,37.6,154,141,Yes,No,6,No,No,ACC-1,137,13.5,Yes,Good,CHPS compound\nGH-0041,54,Male,Ho,Rural,Primary,Moderate,18.7,126,113,No,No,2,No,Yes,HO-3,107,4.9,Yes,Fair,CHPS compound\nGH-0042,62,Male,Accra,Rural,None,High,19.1,142,140,Yes,No,2,No,No,ACC-3,138,16.2,Yes,Very good,Hospital\nGH-0043,68,Male,Accra,Rural,Secondary,High,36.8,144,139,Yes,No,2,No,No,ACC-4,134,0.5,Yes,Very good,CHPS compound\nGH-0044,51,Female,Ho,Urban,Secondary,Low,19.2,127,126,No,No,2,Yes,Yes,HO-2,107,8.9,Yes,Fair,Hospital\nGH-0045,28,Female,Tamale,Urban,None,High,21.8,87,64,No,No,0,No,Yes,TAM-4,98,28.2,No,Good,Hospital\nGH-0046,59,Female,Kumasi,Rural,Primary,High,15.0,124,111,No,No,0,No,No,KUM-3,132,27.0,No,Poor,CHPS compound\nGH-0047,33,Female,Kumasi,Urban,Secondary,Moderate,25.0,101,86,No,Yes,0,No,No,KUM-4,94,27.9,No,Good,Hospital\nGH-0048,70,Male,Tamale,Rural,None,Moderate,36.5,152,157,Yes,No,0,No,Yes,TAM-4,153,12.1,Yes,Good,Hospital\nGH-0049,68,Female,Ho,Rural,Primary,Low,40.3,150,143,Yes,No,4,No,Yes,HO-4,157,8.5,No,Good,Hospital\nGH-0050,39,Female,Kumasi,Urban,Primary,Low,38.4,129,112,No,No,1,No,No,KUM-2,130,19.2,Yes,Very good,Hospital\nGH-0051,32,Female,Kumasi,Urban,Secondary,Moderate,32.0,126,133,No,No,0,Yes,Yes,KUM-2,118,26.3,No,Good,Hospital\nGH-0052,37,Male,Accra,Urban,Primary,Moderate,39.2,114,111,No,Yes,1,No,No,ACC-1,107,34.2,No,Good,Hospital\nGH-0053,45,Female,Accra,Rural,Secondary,Low,19.9,104,117,No,No,0,No,No,ACC-2,96,29.6,Yes,Good,CHPS compound\nGH-0054,63,Male,Kumasi,Urban,Primary,High,39.4,152,121,Yes,Yes,6,Yes,Yes,KUM-2,163,2.7,Yes,Fair,Pharmacy\nGH-0055,30,Female,Kumasi,Rural,Primary,High,16.4,117,111,No,No,1,Yes,Yes,KUM-2,125,18.1,No,Very good,CHPS compound\nGH-0056,57,Male,Kumasi,Urban,None,Moderate,26.0,139,139,No,No,3,Yes,Yes,KUM-2,134,2.7,Yes,Good,Pharmacy\nGH-0057,19,Male,Kumasi,Rural,Primary,High,29.6,99,87,No,Yes,3,No,Yes,KUM-2,97,20.4,No,Good,CHPS compound\nGH-0058,62,Male,Accra,Urban,Tertiary,Moderate,31.7,160,159,Yes,No,10,No,No,ACC-2,161,10.9,Yes,Good,Hospital\nGH-0059,45,Female,Tamale,Rural,Secondary,Low,19.9,116,106,No,No,0,Yes,Yes,TAM-2,101,12.2,Yes,Poor,Pharmacy\nGH-0060,49,Male,Ho,Urban,Secondary,Moderate,28.7,154,154,Yes,No,4,No,No,HO-1,156,7.0,Yes,Good,Hospital\nGH-0061,45,Female,Kumasi,Urban,Secondary,Low,40.3,145,131,Yes,No,2,No,No,KUM-1,152,17.7,No,Very good,Pharmacy\nGH-0062,58,Female,Accra,Rural,None,Moderate,,152,154,Yes,No,2,No,No,ACC-3,133,2.1,Yes,Good,Pharmacy\nGH-0063,51,Female,Kumasi,Urban,Secondary,High,39.3,151,145,Yes,No,4,No,No,KUM-2,150,20.0,No,Good,Hospital\nGH-0064,68,Female,Tamale,Urban,Secondary,Low,16.3,121,126,No,No,2,No,No,TAM-1,134,7.7,Yes,Very good,CHPS compound\nGH-0065,23,Female,Cape Coast,Rural,Secondary,Low,,96,90,No,No,2,No,No,CAP-4,87,18.7,No,Good,CHPS compound\nGH-0066,34,Male,Cape Coast,Rural,Tertiary,Low,15.8,122,119,No,No,0,No,Yes,CAP-2,127,17.7,No,Very good,CHPS compound\nGH-0067,29,Female,Tamale,Urban,None,Low,20.2,104,111,No,No,2,Yes,Yes,TAM-3,86,29.0,No,Very good,CHPS compound\nGH-0068,55,Female,Accra,Rural,None,Low,30.8,108,88,No,No,3,No,Yes,ACC-1,102,3.1,Yes,Poor,Hospital\nGH-0069,24,Male,Ho,Rural,Tertiary,,29.1,123,121,No,No,1,No,Yes,HO-1,108,12.0,No,Very good,CHPS compound\nGH-0070,78,Female,Accra,Rural,Primary,Low,9.3,129,125,No,No,2,No,No,ACC-3,110,14.0,No,Fair,Hospital\nGH-0071,47,Male,Kumasi,Urban,Tertiary,Moderate,30.3,139,126,No,No,4,Yes,Yes,KUM-3,127,8.1,No,Very good,Pharmacy\nGH-0072,62,Female,Accra,Urban,Primary,Low,32.2,147,151,Yes,No,1,No,No,ACC-4,134,6.5,Yes,Very good,Hospital\nGH-0073,33,Male,Ho,Rural,Primary,Moderate,25.8,148,153,Yes,Yes,1,No,No,HO-4,155,18.2,No,Very good,Hospital\nGH-0074,49,Male,Cape Coast,Rural,Secondary,Moderate,24.0,110,112,No,No,1,Yes,Yes,CAP-3,115,8.1,No,Poor,Pharmacy\nGH-0075,54,Male,Cape Coast,Rural,Primary,High,15.5,150,136,Yes,No,4,No,No,CAP-4,138,10.3,Yes,Poor,CHPS compound\nGH-0076,37,Male,Kumasi,Urban,None,Moderate,29.2,136,,No,No,0,Yes,Yes,KUM-1,129,6.5,Yes,Very good,Hospital\nGH-0077,43,Male,Tamale,Urban,Primary,Moderate,20.4,122,94,No,No,4,Yes,Yes,TAM-4,112,33.6,No,Poor,Hospital\nGH-0078,66,Female,Cape Coast,Urban,Primary,Moderate,38.1,130,124,No,No,0,Yes,Yes,CAP-1,121,12.0,No,Very good,Hospital\nGH-0079,62,Female,Tamale,Urban,Tertiary,Low,23.5,131,123,No,No,3,No,No,TAM-2,119,9.8,Yes,Poor,Hospital\nGH-0080,54,Female,Accra,Urban,Tertiary,Low,28.3,138,129,No,No,2,No,Yes,ACC-1,124,4.2,Yes,Good,Hospital\nGH-0081,72,Male,Accra,Urban,None,Low,26.0,140,141,Yes,No,0,No,No,ACC-4,128,14.4,Yes,Good,Hospital\nGH-0082,21,Female,Kumasi,Urban,Primary,Low,23.9,102,98,No,No,0,No,No,KUM-1,89,34.2,No,Very good,CHPS compound\nGH-0083,50,Male,Kumasi,Urban,Secondary,,31.0,128,129,No,No,1,No,No,KUM-3,108,28.0,No,Fair,Hospital\nGH-0084,41,Female,Tamale,Rural,None,Moderate,35.6,122,118,No,No,0,Yes,Yes,TAM-1,120,16.7,No,Very good,Pharmacy\nGH-0085,65,Female,Tamale,Urban,None,High,16.9,140,138,Yes,No,5,No,No,TAM-1,136,6.5,Yes,Fair,Hospital\nGH-0086,60,Female,Kumasi,Urban,None,Low,30.1,142,125,Yes,No,3,No,No,KUM-4,138,2.3,Yes,Very good,Hospital\nGH-0087,61,Female,Kumasi,Urban,Tertiary,Moderate,41.3,160,154,Yes,No,0,No,No,KUM-1,173,19.9,No,Good,Pharmacy\nGH-0088,38,Female,Kumasi,Urban,Primary,Moderate,16.8,103,98,No,No,0,No,No,KUM-2,101,23.0,No,Fair,Pharmacy\nGH-0089,53,Female,Kumasi,Urban,None,High,,148,145,Yes,No,5,No,No,KUM-2,165,3.3,Yes,Fair,CHPS compound\nGH-0090,60,Male,Cape Coast,Rural,Primary,High,21.7,169,171,Yes,No,1,No,No,CAP-1,157,28.9,No,Good,CHPS compound\nGH-0091,63,Female,Cape Coast,Urban,None,Moderate,26.4,137,,No,No,0,No,No,CAP-2,130,29.8,No,Poor,Pharmacy\nGH-0092,34,Female,Tamale,Urban,Secondary,Moderate,21.2,130,128,No,No,5,No,No,TAM-1,124,20.5,No,Good,Hospital\nGH-0093,37,Male,Cape Coast,Rural,Primary,High,21.5,106,91,No,No,0,No,No,CAP-4,106,6.1,Yes,Very good,CHPS compound\nGH-0094,49,Female,Ho,Urban,Secondary,Low,21.5,152,143,Yes,No,5,No,No,HO-3,140,5.8,Yes,Fair,Hospital\nGH-0095,64,Female,Kumasi,Urban,Tertiary,Moderate,39.7,156,157,Yes,No,4,Yes,Yes,KUM-1,144,1.3,Yes,Good,Pharmacy\nGH-0096,54,Male,Kumasi,Rural,Secondary,High,33.3,159,162,Yes,No,2,No,No,KUM-4,154,14.7,No,Good,CHPS compound\nGH-0097,64,Female,Kumasi,Rural,Primary,High,25.1,142,140,Yes,No,2,Yes,Yes,KUM-4,141,12.1,No,Poor,Pharmacy\nGH-0098,21,Male,Cape Coast,Rural,None,Low,26.3,92,78,No,No,2,No,No,CAP-4,88,21.8,No,Fair,Hospital\nGH-0099,42,Female,Tamale,Rural,Primary,Moderate,20.2,125,112,No,No,1,No,No,TAM-4,117,26.1,No,Very good,Hospital\nGH-0100,52,Female,Kumasi,Urban,Primary,Moderate,35.6,138,137,No,No,2,No,No,KUM-1,133,21.7,Yes,Fair,Hospital\nGH-0101,50,Male,Ho,Rural,Primary,Low,18.7,134,135,No,No,0,Yes,No,HO-3,125,20.3,No,Poor,Pharmacy\nGH-0102,21,Male,Tamale,Rural,None,High,29.7,111,102,No,Yes,2,No,No,TAM-3,112,34.5,No,Fair,CHPS compound\nGH-0103,60,Male,Kumasi,Rural,Secondary,Moderate,,101,104,No,Yes,3,No,No,KUM-3,100,12.3,Yes,Good,CHPS compound\nGH-0104,50,Male,Accra,Urban,Secondary,Moderate,32.3,136,134,No,Yes,2,No,No,ACC-4,115,23.9,Yes,Good,Hospital\nGH-0105,40,Female,Ho,Urban,Secondary,Moderate,19.7,113,94,No,No,1,No,No,HO-1,118,8.7,No,Poor,Pharmacy\nGH-0106,62,Male,Tamale,Rural,Primary,Low,27.2,146,147,Yes,Yes,2,Yes,Yes,TAM-4,136,6.2,Yes,Good,CHPS compound\nGH-0107,24,Male,Kumasi,Urban,Primary,High,16.2,100,95,No,No,2,No,No,KUM-3,102,23.0,No,Good,CHPS compound\nGH-0108,39,Male,Cape Coast,Rural,Secondary,High,28.9,126,122,No,No,1,No,No,CAP-2,111,11.8,No,Very good,CHPS compound\nGH-0109,57,Female,Kumasi,Rural,Primary,Low,24.7,123,111,No,No,1,Yes,Yes,KUM-3,118,3.2,Yes,Fair,Hospital\nGH-0110,51,Male,Accra,Urban,Primary,Moderate,16.2,113,107,No,No,1,No,Yes,ACC-4,113,18.9,No,Good,Pharmacy\nGH-0111,43,Male,Kumasi,Urban,Tertiary,Low,32.3,138,146,No,Yes,0,No,Yes,KUM-2,128,10.0,No,Poor,CHPS compound\nGH-0112,44,Male,Accra,Urban,Tertiary,Low,24.5,120,125,No,No,1,Yes,No,ACC-2,142,29.1,No,Good,Hospital\nGH-0113,43,Female,Kumasi,Rural,Secondary,Low,28.0,117,108,No,No,0,Yes,Yes,KUM-1,121,30.5,No,Very good,CHPS compound\nGH-0114,57,Male,Tamale,Rural,Tertiary,High,23.3,127,123,No,No,0,No,No,TAM-4,113,8.2,No,Poor,Pharmacy\nGH-0115,42,Male,Accra,Urban,Secondary,High,20.5,121,121,No,No,4,Yes,Yes,ACC-3,121,14.3,No,Very good,Hospital\nGH-0116,55,Female,Tamale,Urban,Primary,Moderate,29.6,154,162,Yes,No,1,No,Yes,TAM-2,149,27.4,No,Fair,Pharmacy\nGH-0117,19,Female,Tamale,Urban,Secondary,Low,16.3,97,90,No,No,0,Yes,Yes,TAM-3,96,30.0,No,Poor,CHPS compound\nGH-0118,62,Female,Kumasi,Urban,,Low,29.5,127,127,No,No,0,Yes,Yes,KUM-2,135,10.2,No,Good,Pharmacy\nGH-0119,78,Male,Tamale,Rural,None,Low,16.9,137,127,No,No,1,No,Yes,TAM-1,135,4.7,Yes,Poor,Pharmacy\nGH-0120,52,Male,Kumasi,Rural,Tertiary,High,28.2,148,141,Yes,No,4,No,No,KUM-3,155,33.6,No,Very good,CHPS compound\nGH-0121,45,Male,Kumasi,Urban,Tertiary,Moderate,20.2,123,127,No,No,1,No,Yes,KUM-2,123,13.5,No,Fair,CHPS compound\nGH-0122,39,Male,Cape Coast,Rural,Primary,Low,28.3,130,137,No,No,0,No,Yes,CAP-3,122,26.8,No,Very good,CHPS compound\nGH-0123,55,Male,Accra,Rural,Primary,High,32.8,152,153,Yes,Yes,1,No,Yes,ACC-3,138,3.4,Yes,Good,Pharmacy\nGH-0124,38,Female,Accra,Urban,Primary,High,16.8,127,120,No,No,3,Yes,Yes,ACC-4,128,11.5,No,Very good,Hospital\nGH-0125,35,Male,Accra,Urban,Secondary,Moderate,15.5,116,106,No,Yes,0,No,No,ACC-2,114,27.6,No,Good,Hospital\nGH-0126,18,Female,Tamale,Rural,Primary,Moderate,29.3,96,109,No,No,2,No,Yes,TAM-4,97,0.5,Yes,Very good,Hospital\nGH-0127,49,Female,Kumasi,Urban,None,Low,25.5,123,107,No,No,1,No,No,KUM-2,127,23.4,No,Poor,Pharmacy\nGH-0128,42,Female,Ho,Urban,Tertiary,Moderate,20.5,110,104,No,No,2,No,No,HO-2,103,27.6,No,Fair,Hospital\nGH-0129,43,Female,Kumasi,Urban,Secondary,Low,28.6,134,133,No,No,0,No,No,KUM-2,123,14.9,No,Fair,Pharmacy\nGH-0130,46,Male,Kumasi,Urban,Tertiary,Low,21.1,139,152,No,No,1,No,No,KUM-2,133,6.3,Yes,Good,Hospital\nGH-0131,37,Male,Cape Coast,Urban,Secondary,High,17.9,115,120,No,No,3,Yes,Yes,CAP-4,108,6.5,No,Fair,CHPS compound\nGH-0132,48,Female,Ho,Rural,Primary,Low,11.6,121,107,No,No,4,No,Yes,HO-4,106,10.0,No,Fair,Hospital\nGH-0133,32,Female,Ho,Rural,Secondary,Low,34.9,129,124,No,No,0,No,Yes,HO-4,134,3.8,Yes,Very good,Hospital\nGH-0134,44,Male,Cape Coast,Urban,Primary,Low,38.2,133,128,No,No,1,Yes,Yes,CAP-1,123,19.6,No,Poor,Hospital\nGH-0135,42,Male,Tamale,Rural,Tertiary,Low,35.1,151,150,Yes,No,2,No,No,TAM-2,135,3.5,Yes,Good,CHPS compound\nGH-0136,24,Male,Kumasi,Urban,Primary,Low,34.7,116,116,No,No,1,No,Yes,KUM-2,111,9.1,No,Good,Hospital\nGH-0137,30,Male,Ho,Rural,Primary,High,23.0,112,111,No,No,3,No,Yes,HO-1,108,20.3,No,Good,CHPS compound\nGH-0138,38,Female,Cape Coast,Rural,Secondary,High,43.9,130,118,No,Yes,0,No,No,CAP-4,127,25.2,No,Fair,CHPS compound\nGH-0139,39,Female,Accra,Rural,Primary,High,20.4,134,122,No,No,1,Yes,Yes,ACC-2,122,20.0,No,Good,CHPS compound\nGH-0140,49,Female,Ho,Rural,Primary,Moderate,33.5,125,115,No,No,1,No,No,HO-4,118,24.9,No,Good,CHPS compound\nGH-0141,41,Male,Kumasi,Urban,Primary,High,32.3,125,123,No,Yes,5,Yes,Yes,KUM-2,127,30.0,No,Very good,Hospital\nGH-0142,45,Female,Accra,Urban,None,High,24.8,142,134,Yes,No,0,Yes,Yes,ACC-1,128,10.1,No,Good,Hospital\nGH-0143,41,Female,Tamale,Urban,Secondary,Moderate,43.4,143,125,Yes,No,0,No,Yes,TAM-2,120,7.7,Yes,Very good,Pharmacy\nGH-0144,18,Male,Accra,Urban,Secondary,,,123,111,No,No,0,No,Yes,ACC-4,123,30.8,No,Fair,CHPS compound\nGH-0145,44,Female,Cape Coast,Urban,Primary,Moderate,32.5,120,104,No,No,0,Yes,Yes,CAP-1,99,12.2,No,Good,CHPS compound\nGH-0146,35,Female,Tamale,Rural,Primary,Low,25.8,120,128,No,No,0,No,No,TAM-3,102,20.6,No,Fair,CHPS compound\nGH-0147,44,Male,Tamale,Urban,None,High,21.5,113,126,No,Yes,2,No,No,TAM-2,95,13.4,Yes,Good,Hospital\nGH-0148,36,Male,Accra,Urban,Primary,Moderate,22.8,123,112,No,Yes,2,No,Yes,ACC-4,105,25.3,No,Poor,Hospital\nGH-0149,55,Male,Tamale,Rural,,Moderate,14.4,125,122,No,Yes,1,Yes,Yes,TAM-4,104,19.3,No,Good,CHPS compound\nGH-0150,45,Male,Accra,Urban,Primary,Moderate,13.7,116,,No,No,2,Yes,Yes,ACC-1,106,14.1,No,Very good,Hospital\nGH-0151,55,Male,Cape Coast,Rural,None,Moderate,18.1,129,118,No,No,2,No,Yes,CAP-4,123,34.7,No,Poor,Pharmacy\nGH-0152,56,Male,Accra,Urban,None,Low,35.2,148,134,Yes,No,1,No,No,ACC-1,137,6.4,Yes,Poor,Pharmacy\nGH-0153,39,Male,Kumasi,Urban,Primary,Moderate,20.7,97,90,No,Yes,1,Yes,Yes,KUM-2,97,14.0,No,Good,Hospital\nGH-0154,57,Male,Tamale,Rural,Primary,Low,30.0,122,116,No,No,1,No,No,TAM-4,118,21.2,No,Poor,Hospital\nGH-0155,49,Female,Kumasi,Urban,Primary,Moderate,17.7,129,138,No,No,1,Yes,Yes,KUM-3,129,25.0,No,Fair,CHPS compound\nGH-0156,32,Female,Tamale,Rural,Secondary,Low,25.6,113,,No,No,0,No,No,TAM-2,99,29.0,No,Fair,CHPS compound\nGH-0157,44,Female,Ho,Urban,Primary,Low,43.1,149,,Yes,No,1,No,Yes,HO-2,138,7.7,Yes,Poor,Pharmacy\nGH-0158,47,Female,Accra,Urban,Primary,Low,21.1,141,113,Yes,No,7,No,No,ACC-1,123,17.6,No,Good,CHPS compound\nGH-0159,63,Female,Tamale,Rural,Primary,Low,15.4,143,138,Yes,No,2,Yes,Yes,TAM-2,128,12.3,No,Good,CHPS compound\nGH-0160,41,Male,Ho,Rural,Secondary,Moderate,31.2,144,138,Yes,Yes,1,Yes,No,HO-4,131,10.3,No,Good,CHPS compound\nGH-0161,66,Male,Kumasi,Rural,None,High,30.7,149,138,Yes,No,2,No,No,KUM-1,136,7.0,Yes,Fair,Hospital\nGH-0162,56,Female,Accra,Urban,Secondary,Low,,159,165,Yes,No,2,No,No,ACC-4,149,7.9,No,Very good,Hospital\nGH-0163,19,Female,Accra,Rural,Secondary,Moderate,30.2,107,108,No,No,3,No,No,ACC-4,111,31.7,No,Good,CHPS compound\nGH-0164,33,Male,Ho,Rural,Primary,High,24.3,126,115,No,No,1,No,No,HO-4,142,31.4,No,Fair,Pharmacy\nGH-0165,54,Female,Accra,Rural,None,Moderate,19.9,122,115,No,No,0,No,No,ACC-4,116,33.1,No,Good,CHPS compound\nGH-0166,27,Male,Tamale,Rural,Secondary,Moderate,35.5,121,121,No,No,0,Yes,Yes,TAM-1,114,7.5,Yes,Good,CHPS compound\nGH-0167,52,Female,Cape Coast,Rural,Secondary,Moderate,7.4,103,,No,No,2,Yes,Yes,CAP-4,111,20.0,No,Fair,Hospital\nGH-0168,64,Male,Tamale,Urban,Secondary,Moderate,12.5,132,139,No,No,2,No,No,TAM-3,109,19.2,Yes,Very good,Pharmacy\nGH-0169,43,Female,Tamale,Rural,None,High,27.0,109,108,No,No,2,Yes,Yes,TAM-4,109,12.4,No,Poor,Hospital\nGH-0170,64,Female,Kumasi,Urban,Secondary,Low,36.1,117,103,No,Yes,2,No,No,KUM-3,102,15.0,No,Poor,Hospital\nGH-0171,63,Female,Tamale,Urban,Primary,Low,24.4,144,145,Yes,No,10,Yes,Yes,TAM-1,132,16.1,No,Good,Hospital\nGH-0172,45,Male,Accra,Urban,None,High,27.7,139,141,No,Yes,0,No,No,ACC-1,115,21.5,No,Poor,CHPS compound\nGH-0173,46,Male,Kumasi,Urban,Secondary,Moderate,28.6,142,142,Yes,Yes,1,No,Yes,KUM-2,151,6.2,Yes,Fair,Hospital\nGH-0174,48,Female,Cape Coast,Rural,Secondary,Moderate,37.3,136,142,No,No,1,Yes,Yes,CAP-4,122,34.2,No,Fair,Hospital\nGH-0175,52,Female,Cape Coast,Urban,Secondary,High,34.0,146,136,Yes,No,0,Yes,Yes,CAP-3,143,17.0,Yes,Poor,Pharmacy\nGH-0176,26,Female,Tamale,Rural,Primary,High,29.3,115,125,No,No,3,No,No,TAM-1,114,12.8,Yes,Good,CHPS compound\nGH-0177,44,Male,Cape Coast,Urban,Primary,High,19.7,119,123,No,No,0,No,Yes,CAP-1,103,33.0,No,Very good,Hospital\nGH-0178,24,Female,Tamale,Urban,Primary,Low,34.6,105,98,No,No,0,Yes,No,TAM-3,104,16.4,No,Fair,Pharmacy\nGH-0179,42,Male,Ho,Urban,Primary,Moderate,33.8,157,171,Yes,No,0,No,No,HO-3,146,21.9,No,Good,Hospital\nGH-0180,43,Female,Accra,Urban,Primary,High,26.7,116,98,No,No,2,No,Yes,ACC-2,107,31.1,Yes,Very good,Pharmacy\nGH-0181,22,Female,Ho,Rural,None,Moderate,32.8,139,123,No,No,2,Yes,Yes,HO-3,127,24.5,Yes,Very good,CHPS compound\nGH-0182,30,Male,Ho,Rural,Tertiary,Moderate,16.2,96,91,No,No,1,No,No,HO-4,95,17.4,No,Poor,Hospital\nGH-0183,41,Female,Cape Coast,Rural,Secondary,High,,116,100,No,No,0,No,Yes,CAP-3,117,11.7,Yes,Very good,Hospital\nGH-0184,74,Female,Cape Coast,Rural,Secondary,Moderate,33.1,163,156,Yes,No,4,No,No,CAP-3,163,15.8,No,Very good,CHPS compound\nGH-0185,28,Male,Accra,Urban,Primary,Low,,127,117,No,Yes,0,Yes,Yes,ACC-4,107,19.6,No,Good,Hospital\nGH-0186,48,Female,Kumasi,Rural,Primary,Moderate,39.4,144,129,Yes,No,1,No,No,KUM-4,125,35.5,No,Very good,Pharmacy\nGH-0187,34,Male,Ho,Rural,Secondary,High,18.0,107,82,No,No,2,No,No,HO-4,105,1.6,Yes,Very good,CHPS compound\nGH-0188,41,Female,Tamale,Rural,Tertiary,Moderate,23.1,133,126,No,No,3,Yes,Yes,TAM-1,136,35.7,No,Very good,CHPS compound\nGH-0189,59,Male,Kumasi,Urban,Secondary,Moderate,37.4,135,109,No,No,1,No,No,KUM-3,141,24.5,No,Very good,CHPS compound\nGH-0190,50,Male,Accra,Urban,None,Moderate,31.4,140,125,Yes,No,4,No,No,ACC-3,133,12.0,No,Good,Hospital\nGH-0191,52,Male,Tamale,Urban,Secondary,Low,19.2,114,95,No,No,2,No,No,TAM-2,102,0.6,Yes,Fair,CHPS compound\nGH-0192,52,Female,Kumasi,Urban,Primary,Moderate,22.0,146,141,Yes,No,0,No,No,KUM-2,138,5.0,Yes,Poor,CHPS compound\nGH-0193,53,Male,Kumasi,Rural,None,Moderate,8.8,122,114,No,No,8,No,No,KUM-1,132,12.9,No,Fair,CHPS compound\nGH-0194,18,Male,Kumasi,Urban,,High,24.3,133,108,No,No,3,No,Yes,KUM-3,129,11.1,No,Good,Hospital\nGH-0195,51,Male,Kumasi,Rural,Tertiary,Moderate,39.9,142,143,Yes,No,2,No,No,KUM-2,132,0.5,Yes,Fair,CHPS compound\nGH-0196,45,Male,Kumasi,Urban,Primary,High,34.9,131,137,No,No,0,No,No,KUM-2,134,13.4,No,Very good,CHPS compound\nGH-0197,51,Female,Cape Coast,Urban,Secondary,Low,13.1,97,71,No,No,0,No,Yes,CAP-2,93,3.2,Yes,Very good,Hospital\nGH-0198,57,Male,Tamale,Rural,Primary,Low,22.0,132,146,No,No,2,No,No,TAM-2,130,9.4,No,Fair,CHPS compound\nGH-0199,51,Female,Accra,Urban,None,High,17.1,139,126,No,No,2,No,No,ACC-4,123,6.3,Yes,Very good,Hospital\nGH-0200,55,Female,Accra,Urban,Secondary,Moderate,38.1,149,151,Yes,No,2,Yes,Yes,ACC-2,151,9.3,Yes,Good,Hospital\nGH-0201,31,Male,Ho,Urban,Primary,High,19.4,109,106,No,Yes,1,Yes,Yes,HO-4,112,15.9,No,Very good,Hospital\nGH-0202,52,Male,Kumasi,Rural,None,Moderate,52.3,150,,Yes,No,2,No,No,KUM-2,146,23.7,No,Good,CHPS compound\nGH-0203,45,Male,Tamale,Urban,Secondary,Low,16.0,111,108,No,No,6,No,Yes,TAM-3,99,24.5,No,Very good,Hospital\nGH-0204,45,Female,Ho,Urban,Secondary,Moderate,18.5,118,126,No,No,4,No,No,HO-3,125,35.9,No,Fair,Hospital\nGH-0205,63,Male,Ho,Urban,Primary,Low,18.8,138,134,No,No,1,No,Yes,HO-2,128,15.6,No,Poor,CHPS compound\nGH-0206,40,Male,Ho,Rural,None,Low,29.4,128,120,No,No,0,Yes,Yes,HO-4,117,24.1,No,Fair,Hospital\nGH-0207,37,Female,Tamale,Urban,Secondary,Moderate,32.6,136,145,No,No,0,No,Yes,TAM-2,125,15.1,No,Very good,CHPS compound\nGH-0208,29,Male,Tamale,Rural,Secondary,High,14.8,119,106,No,Yes,1,Yes,No,TAM-1,103,3.3,Yes,Fair,CHPS compound\nGH-0209,42,Male,Cape Coast,Urban,None,Moderate,39.0,125,112,No,No,0,No,Yes,CAP-4,133,22.7,No,Good,CHPS compound\nGH-0210,60,Female,Cape Coast,Urban,Secondary,Moderate,26.8,135,150,No,No,0,No,Yes,CAP-4,117,14.9,No,Good,Pharmacy\nGH-0211,24,Female,Tamale,Urban,Primary,Moderate,28.8,103,103,No,Yes,3,Yes,Yes,TAM-1,104,24.4,Yes,Very good,Pharmacy\nGH-0212,49,Male,Kumasi,Urban,Secondary,High,27.3,136,139,No,No,3,No,No,KUM-4,109,26.8,No,Poor,CHPS compound\nGH-0213,18,Male,Accra,Urban,Tertiary,High,,112,75,No,No,1,No,No,ACC-3,101,25.2,No,Fair,Hospital\nGH-0214,37,Male,Accra,Rural,None,Moderate,23.0,111,107,No,No,0,No,No,ACC-3,103,2.7,Yes,Poor,CHPS compound\nGH-0215,52,Male,Tamale,Rural,Primary,Moderate,9.7,100,109,No,No,1,Yes,Yes,TAM-1,106,12.8,Yes,Fair,Pharmacy\nGH-0216,39,Male,Accra,Rural,Secondary,Low,26.9,126,120,No,No,0,No,No,ACC-4,122,34.6,No,Very good,CHPS compound\nGH-0217,30,Male,Accra,Urban,Tertiary,High,27.4,123,131,No,No,2,No,Yes,ACC-1,118,25.2,No,Very good,CHPS compound\nGH-0218,41,Male,Cape Coast,Rural,Secondary,Moderate,30.1,145,148,Yes,No,3,Yes,Yes,CAP-1,135,10.6,Yes,Good,CHPS compound\nGH-0219,59,Female,Kumasi,Urban,Secondary,Low,28.4,124,108,No,No,2,Yes,Yes,KUM-3,110,12.7,Yes,Fair,CHPS compound\nGH-0220,65,Female,Tamale,Rural,,Low,41.7,169,165,Yes,No,2,No,No,TAM-4,168,11.1,No,Poor,CHPS compound\nGH-0221,66,Male,Accra,Urban,None,Low,24.8,145,140,Yes,No,1,No,Yes,ACC-1,118,28.9,No,Fair,Hospital\nGH-0222,55,Female,Kumasi,Urban,Secondary,Moderate,27.2,112,98,No,No,1,Yes,Yes,KUM-2,116,7.6,Yes,Good,Hospital\nGH-0223,51,Female,Kumasi,Urban,Secondary,Moderate,43.3,135,113,No,No,5,Yes,Yes,KUM-4,124,9.5,No,Fair,CHPS compound\nGH-0224,59,Female,Kumasi,Urban,Primary,Low,30.1,135,,No,No,2,Yes,No,KUM-1,131,11.8,No,Fair,Hospital\nGH-0225,31,Female,Accra,Rural,Secondary,High,26.7,120,126,No,No,4,No,No,ACC-3,117,11.1,No,Good,CHPS compound\nGH-0226,33,Male,Accra,Urban,Primary,Low,35.8,117,113,No,Yes,4,No,No,ACC-2,110,20.1,Yes,Good,Hospital\nGH-0227,63,Male,Kumasi,Urban,None,Low,19.8,107,100,No,No,1,Yes,Yes,KUM-1,107,8.4,No,Fair,CHPS compound\nGH-0228,57,Female,Cape Coast,Rural,Primary,High,20.3,126,123,No,No,1,Yes,Yes,CAP-4,131,11.6,No,Fair,CHPS compound\nGH-0229,57,Male,Kumasi,Urban,None,Moderate,24.4,131,,No,No,4,Yes,Yes,KUM-3,116,13.9,No,Fair,Hospital\nGH-0230,49,Female,Cape Coast,Rural,Secondary,High,27.8,140,142,Yes,No,4,No,Yes,CAP-3,140,3.4,Yes,Fair,CHPS compound\nGH-0231,35,Male,Tamale,Rural,Tertiary,High,29.8,132,111,No,No,1,No,Yes,TAM-2,125,12.4,No,Very good,CHPS compound\nGH-0232,32,Male,Tamale,Urban,None,High,23.8,109,100,No,Yes,4,No,Yes,TAM-4,106,33.8,No,Fair,Hospital\nGH-0233,58,Female,Cape Coast,Rural,Tertiary,Moderate,39.6,155,166,Yes,No,4,No,No,CAP-3,146,3.2,Yes,Very good,CHPS compound\nGH-0234,60,Female,Kumasi,Urban,Primary,Moderate,17.5,118,114,No,No,0,No,No,KUM-3,117,25.1,No,Fair,Hospital\nGH-0235,31,Male,Kumasi,Urban,None,Low,25.7,130,119,No,No,4,No,Yes,KUM-2,131,20.5,No,Fair,Hospital\nGH-0236,36,Female,Accra,Urban,Primary,Moderate,14.7,104,96,No,No,1,No,Yes,ACC-3,89,10.3,No,Good,Hospital\nGH-0237,45,Male,Accra,Rural,Primary,Low,16.5,110,91,No,No,1,Yes,Yes,ACC-1,85,16.0,No,Fair,Pharmacy\nGH-0238,57,Male,Kumasi,Urban,Secondary,Low,16.8,124,129,No,No,0,Yes,Yes,KUM-4,102,26.9,No,Very good,Hospital\nGH-0239,42,Male,Accra,Urban,Secondary,Moderate,,138,148,No,No,1,Yes,Yes,ACC-4,120,23.3,No,Poor,Hospital\nGH-0240,35,Male,Accra,Urban,None,Low,32.5,130,,No,No,0,No,Yes,ACC-3,118,13.8,No,Poor,Hospital\nGH-0241,59,Female,Ho,Rural,Primary,Moderate,29.1,119,,No,No,1,No,No,HO-4,122,20.8,No,Good,Hospital\nGH-0242,54,Male,Cape Coast,Rural,Secondary,High,31.6,154,153,Yes,No,3,No,Yes,CAP-2,149,3.5,Yes,Good,Pharmacy\nGH-0243,37,Female,Cape Coast,Urban,Primary,Moderate,32.0,118,106,No,No,1,No,Yes,CAP-4,126,10.6,No,Fair,CHPS compound\nGH-0244,37,Female,Kumasi,Rural,Secondary,Moderate,29.8,106,111,No,No,2,No,No,KUM-3,107,5.0,Yes,Fair,CHPS compound\nGH-0245,60,Male,Tamale,Urban,None,Low,18.5,105,113,No,No,1,No,Yes,TAM-3,86,13.7,No,Poor,Hospital\nGH-0246,29,Female,Tamale,Rural,Secondary,Moderate,31.7,138,144,No,No,1,No,Yes,TAM-4,130,16.2,No,Good,Hospital\nGH-0247,42,Female,Kumasi,Urban,Primary,Low,13.5,130,119,No,No,1,No,Yes,KUM-2,137,28.5,No,Fair,Hospital\nGH-0248,52,Male,Tamale,Urban,None,Low,39.9,135,132,No,No,4,No,No,TAM-2,117,6.6,No,Good,Pharmacy\nGH-0249,48,Male,Tamale,Rural,Primary,Low,55.4,150,153,Yes,No,2,Yes,Yes,TAM-2,142,25.5,No,Very good,Hospital\nGH-0250,58,Female,Tamale,Rural,Secondary,Moderate,15.8,123,138,No,No,1,No,No,TAM-2,99,9.1,No,Poor,CHPS compound\nGH-0251,24,Male,Cape Coast,Rural,Secondary,High,41.1,118,108,No,No,1,No,No,CAP-4,118,4.5,Yes,Good,Hospital\nGH-0252,46,Male,Tamale,Urban,Primary,Moderate,27.1,132,138,No,No,0,Yes,Yes,TAM-1,116,8.9,No,Poor,Hospital\nGH-0253,50,Male,Tamale,Urban,Primary,High,30.6,146,136,Yes,No,2,Yes,Yes,TAM-2,139,3.4,Yes,Fair,Pharmacy\nGH-0254,50,Male,Cape Coast,Urban,Primary,High,,154,154,Yes,Yes,2,No,No,CAP-1,145,19.5,Yes,Very good,Hospital\nGH-0255,53,Female,Kumasi,Urban,Primary,Moderate,24.0,131,124,No,No,4,No,No,KUM-4,125,3.9,Yes,Poor,Hospital\nGH-0256,31,Female,Ho,Urban,,High,34.2,123,125,No,No,0,No,No,HO-1,123,16.0,No,Very good,CHPS compound\nGH-0257,65,Female,Accra,Urban,Secondary,Moderate,27.8,151,144,Yes,No,4,No,Yes,ACC-1,134,7.8,Yes,Good,CHPS compound\nGH-0258,52,Female,Cape Coast,Urban,Primary,Low,40.4,128,143,No,No,1,Yes,Yes,CAP-1,113,31.1,No,Fair,Hospital\nGH-0259,53,Female,Cape Coast,Urban,None,High,22.5,124,127,No,No,1,No,No,CAP-1,99,19.0,No,Fair,Hospital\nGH-0260,39,Female,Kumasi,Rural,Primary,Moderate,34.9,148,139,Yes,No,0,Yes,Yes,KUM-3,139,11.2,Yes,Poor,CHPS compound\nGH-0261,48,Female,Cape Coast,Rural,Secondary,High,25.6,132,114,No,No,2,No,No,CAP-1,127,27.3,No,Good,CHPS compound\nGH-0262,63,Female,Kumasi,Rural,Primary,,38.5,175,167,Yes,No,6,No,No,KUM-3,171,6.8,No,Fair,Hospital\nGH-0263,46,Male,Accra,Urban,Secondary,High,16.1,133,135,No,No,0,Yes,Yes,ACC-2,146,12.8,Yes,Very good,CHPS compound\nGH-0264,49,Male,Ho,Urban,None,Low,29.9,147,149,Yes,No,5,No,No,HO-2,127,9.7,Yes,Good,Hospital\nGH-0265,50,Female,Kumasi,Urban,None,Low,29.1,137,132,No,No,3,No,No,KUM-1,139,20.4,No,Fair,Hospital\nGH-0266,25,Male,Tamale,Rural,Secondary,Moderate,41.4,121,128,No,Yes,1,No,Yes,TAM-3,105,17.5,No,Fair,CHPS compound\nGH-0267,44,Male,Kumasi,Rural,Secondary,Moderate,26.8,141,127,Yes,No,0,No,No,KUM-3,140,15.8,No,Poor,CHPS compound\nGH-0268,55,Female,Tamale,Rural,None,Moderate,33.7,125,102,No,No,1,Yes,Yes,TAM-4,122,15.6,Yes,Fair,CHPS compound\nGH-0269,40,Male,Cape Coast,Rural,Tertiary,Moderate,15.3,95,96,No,No,0,No,No,CAP-4,108,20.3,No,Fair,Hospital\nGH-0270,45,Male,Kumasi,Urban,Tertiary,Low,26.5,131,148,No,No,0,Yes,Yes,KUM-1,130,31.8,No,Very good,Hospital\nGH-0271,46,Female,Tamale,Urban,Secondary,High,21.0,138,141,No,No,2,Yes,Yes,TAM-2,124,1.1,Yes,Very good,CHPS compound\nGH-0272,25,Male,Cape Coast,Rural,Primary,High,24.6,136,134,No,Yes,0,Yes,Yes,CAP-3,129,19.2,No,Fair,Hospital\nGH-0273,56,Male,Cape Coast,Urban,Primary,Low,17.6,130,131,No,No,1,No,No,CAP-2,114,8.0,No,Good,Pharmacy\nGH-0274,28,Female,Ho,Urban,Primary,Moderate,30.8,119,104,No,No,1,No,Yes,HO-1,108,6.7,No,Poor,CHPS compound\nGH-0275,51,Male,Accra,Urban,Secondary,High,33.0,140,127,Yes,No,4,No,No,ACC-3,110,9.1,No,Poor,CHPS compound\nGH-0276,44,Male,Kumasi,Urban,Primary,Moderate,26.2,135,139,No,No,0,No,No,KUM-2,153,23.1,Yes,Fair,Hospital\nGH-0277,42,Male,Tamale,Urban,Secondary,Moderate,23.7,138,123,No,No,2,No,No,TAM-1,119,14.1,No,Poor,Hospital\nGH-0278,47,Female,Tamale,Urban,Tertiary,High,26.7,131,111,No,No,0,Yes,Yes,TAM-4,120,12.2,Yes,Very good,Hospital\nGH-0279,56,Male,Accra,Urban,Secondary,High,33.5,153,164,Yes,No,0,No,No,ACC-1,131,20.3,Yes,Very good,CHPS compound\nGH-0280,57,Male,Accra,Urban,None,Moderate,29.9,153,151,Yes,Yes,2,No,No,ACC-2,155,17.4,No,Very good,Pharmacy\nGH-0281,40,Female,Ho,Rural,Primary,High,25.8,141,136,Yes,No,4,No,Yes,HO-3,136,22.8,No,Good,Hospital\nGH-0282,44,Female,Tamale,Urban,None,Low,17.9,130,113,No,No,2,No,No,TAM-2,109,30.7,No,Poor,Hospital\nGH-0283,35,Male,Tamale,Rural,Secondary,High,33.9,144,154,Yes,No,1,Yes,Yes,TAM-2,124,24.8,No,Very good,Hospital\nGH-0284,27,Female,Ho,Rural,Primary,Low,30.8,140,,Yes,No,1,No,No,HO-2,141,26.0,No,Very good,CHPS compound\nGH-0285,58,Female,Cape Coast,Rural,Primary,Moderate,22.6,118,120,No,No,1,Yes,Yes,CAP-4,123,18.1,Yes,Good,Hospital\nGH-0286,32,Female,Cape Coast,Urban,None,Moderate,32.9,146,150,Yes,No,1,Yes,Yes,CAP-4,152,23.7,No,Fair,Hospital\nGH-0287,36,Female,Ho,Urban,Secondary,Moderate,17.2,113,111,No,No,1,Yes,Yes,HO-1,119,15.2,No,Fair,Hospital\nGH-0288,51,Male,Cape Coast,Urban,Secondary,High,20.0,110,99,No,Yes,1,No,No,CAP-3,116,6.4,No,Good,Hospital\nGH-0289,28,Female,Kumasi,Urban,,Moderate,43.9,134,124,No,No,0,Yes,Yes,KUM-4,131,14.1,No,Very good,Hospital\nGH-0290,52,Female,Ho,Urban,None,High,36.2,146,130,Yes,No,2,No,Yes,HO-3,132,21.7,Yes,Good,Hospital\nGH-0291,35,Female,Accra,Urban,Secondary,Low,20.8,95,96,No,No,0,No,Yes,ACC-3,79,32.5,No,Fair,CHPS compound\nGH-0292,31,Male,Accra,Urban,Tertiary,Moderate,38.1,132,127,No,No,1,No,No,ACC-4,127,8.6,No,Good,Pharmacy\nGH-0293,38,Female,Cape Coast,Urban,Tertiary,Low,26.0,90,81,No,No,2,No,No,CAP-1,81,10.5,No,Fair,CHPS compound\nGH-0294,35,Male,Accra,Urban,Tertiary,Low,,138,121,No,No,1,No,Yes,ACC-3,141,15.5,Yes,Very good,CHPS compound\nGH-0295,49,Female,Ho,Urban,Tertiary,Moderate,50.4,146,,Yes,No,3,No,No,HO-4,144,15.1,No,Very good,Hospital\nGH-0296,64,Female,Kumasi,Rural,Secondary,High,17.2,131,127,No,No,2,No,No,KUM-3,143,16.6,No,Fair,Pharmacy\nGH-0297,55,Male,Tamale,Urban,None,Moderate,31.6,151,156,Yes,No,3,No,No,TAM-1,145,4.8,Yes,Good,CHPS compound\nGH-0298,52,Male,Tamale,Urban,Tertiary,Moderate,17.5,125,117,No,Yes,3,Yes,Yes,TAM-3,112,18.3,Yes,Fair,Hospital\nGH-0299,38,Male,Kumasi,Urban,Primary,High,20.0,106,116,No,No,1,No,No,KUM-4,103,11.4,Yes,Good,CHPS compound\nGH-0300,43,Female,Ho,Urban,Primary,Low,33.9,104,111,No,No,0,No,Yes,HO-1,117,14.4,Yes,Good,CHPS compound\nGH-0301,47,Male,Ho,Rural,Tertiary,High,16.0,132,128,No,No,3,Yes,No,HO-3,134,22.2,No,Poor,CHPS compound\nGH-0302,45,Male,Accra,Urban,Tertiary,High,25.4,142,148,Yes,No,9,Yes,Yes,ACC-4,126,7.8,No,Very good,Hospital\nGH-0303,48,Male,Tamale,Rural,Primary,Moderate,11.9,112,105,No,No,0,No,No,TAM-3,94,32.7,No,Poor,Hospital\nGH-0304,30,Male,Kumasi,Urban,Tertiary,Moderate,34.1,101,105,No,No,2,Yes,Yes,KUM-1,97,17.4,No,Good,Pharmacy\nGH-0305,69,Male,Tamale,Urban,None,High,29.2,152,169,Yes,Yes,7,No,No,TAM-3,137,22.6,Yes,Poor,Hospital\nGH-0306,28,Male,Accra,Urban,Primary,High,15.9,113,116,No,No,1,Yes,Yes,ACC-4,103,26.0,No,Poor,Hospital\nGH-0307,44,Male,Kumasi,Urban,None,Moderate,23.8,134,120,No,No,2,No,No,KUM-2,132,12.7,Yes,Fair,CHPS compound\nGH-0308,56,Male,Accra,Rural,Secondary,Moderate,20.1,119,111,No,No,2,Yes,Yes,ACC-1,113,20.9,Yes,Poor,CHPS compound\nGH-0309,36,Male,Accra,Rural,None,Moderate,19.0,90,,No,No,0,No,Yes,ACC-1,72,8.0,No,Very good,Hospital\nGH-0310,59,Female,Tamale,Urban,Secondary,Moderate,27.3,127,148,No,No,2,No,Yes,TAM-2,113,12.5,No,Good,Hospital\nGH-0311,27,Female,Tamale,Rural,Primary,Low,20.3,118,119,No,Yes,1,Yes,Yes,TAM-1,113,33.1,No,Poor,CHPS compound\nGH-0312,36,Male,Cape Coast,Rural,Primary,High,37.0,150,141,Yes,No,4,Yes,Yes,CAP-1,139,9.0,No,Very good,CHPS compound\nGH-0313,18,Male,Kumasi,Urban,Secondary,Moderate,26.4,109,107,No,No,0,Yes,Yes,KUM-2,114,6.9,Yes,Very good,Hospital\nGH-0314,59,Female,Cape Coast,Rural,Secondary,Moderate,14.8,116,105,No,No,1,No,No,CAP-4,113,1.3,Yes,Poor,CHPS compound\nGH-0315,30,Female,Cape Coast,Urban,Primary,Moderate,13.6,82,73,No,No,0,Yes,Yes,CAP-1,61,19.6,No,Good,CHPS compound\nGH-0316,36,Female,Tamale,Urban,Primary,Moderate,16.9,128,,No,No,2,No,No,TAM-3,122,7.5,No,Fair,Hospital\nGH-0317,55,Female,Accra,Urban,Secondary,Moderate,27.8,126,113,No,No,2,Yes,Yes,ACC-4,114,31.7,No,Good,Hospital\nGH-0318,50,Male,Kumasi,Urban,Secondary,Moderate,39.1,133,115,No,No,4,No,No,KUM-4,125,22.2,Yes,Good,Hospital\nGH-0319,59,Male,Tamale,Rural,Tertiary,Low,20.9,117,106,No,No,2,No,No,TAM-1,122,22.5,No,Very good,CHPS compound\nGH-0320,54,Female,Tamale,Rural,None,Moderate,13.3,114,104,No,No,3,Yes,Yes,TAM-2,96,11.0,No,Poor,CHPS compound\nGH-0321,57,Male,Cape Coast,Rural,Tertiary,High,10.8,130,137,No,No,1,Yes,Yes,CAP-3,131,19.9,Yes,Good,Hospital\nGH-0322,39,Female,Ho,Rural,Tertiary,Low,30.0,118,113,No,No,1,No,Yes,HO-3,122,29.0,No,Very good,CHPS compound\nGH-0323,44,Male,Accra,Urban,Tertiary,Moderate,22.9,142,135,Yes,No,1,Yes,Yes,ACC-3,120,12.9,No,Good,Hospital\nGH-0324,55,Female,Ho,Rural,Primary,Moderate,26.0,139,144,No,No,2,Yes,Yes,HO-1,137,2.0,Yes,Fair,Hospital\nGH-0325,55,Male,Kumasi,Urban,Secondary,Moderate,22.2,135,134,No,Yes,2,No,No,KUM-3,139,28.8,Yes,Good,Hospital\nGH-0326,57,Female,Ho,Urban,Primary,Low,32.8,152,,Yes,No,2,Yes,Yes,HO-2,139,33.0,No,Very good,CHPS compound\nGH-0327,35,Male,Accra,Rural,Secondary,High,33.0,117,102,No,Yes,2,No,Yes,ACC-2,116,21.1,No,Fair,Hospital\nGH-0328,31,Female,Kumasi,Urban,Primary,Low,27.2,118,123,No,No,4,Yes,Yes,KUM-3,111,10.3,No,Fair,CHPS compound\nGH-0329,29,Female,Cape Coast,Rural,Secondary,Low,27.8,126,120,No,Yes,0,No,No,CAP-3,121,25.3,No,Fair,CHPS compound\nGH-0330,39,Male,Ho,Urban,None,Low,21.4,129,125,No,No,1,No,No,HO-4,136,20.3,No,Fair,Hospital\nGH-0331,44,Male,Accra,Urban,Primary,Moderate,37.2,132,123,No,Yes,0,Yes,Yes,ACC-1,127,2.7,Yes,Fair,CHPS compound\nGH-0332,30,Female,Cape Coast,Rural,Primary,Moderate,21.8,98,94,No,No,0,No,Yes,CAP-1,82,18.3,No,Fair,CHPS compound\nGH-0333,43,Male,Accra,Urban,None,Moderate,17.8,96,100,No,No,1,Yes,Yes,ACC-4,99,7.8,No,Very good,CHPS compound\nGH-0334,53,Female,Ho,Rural,Primary,,39.1,151,145,Yes,No,0,No,No,HO-2,149,17.5,No,Poor,CHPS compound\nGH-0335,58,Female,Ho,Urban,Secondary,Moderate,34.8,139,117,No,No,1,No,Yes,HO-2,147,15.3,No,Good,CHPS compound\nGH-0336,38,Female,Accra,Urban,Secondary,Low,25.4,116,108,No,No,2,No,No,ACC-1,103,29.3,No,Good,Hospital\nGH-0337,53,Male,Ho,Rural,Secondary,Low,28.8,135,122,No,No,0,No,No,HO-1,136,33.9,No,Very good,CHPS compound\nGH-0338,29,Male,Kumasi,Rural,Tertiary,Low,9.4,101,112,No,No,1,No,No,KUM-2,106,17.0,No,Very good,Pharmacy\nGH-0339,37,Male,Cape Coast,Urban,Primary,Moderate,28.6,120,100,No,Yes,1,No,No,CAP-3,112,21.4,Yes,Fair,Pharmacy\nGH-0340,47,Male,Cape Coast,Urban,Secondary,High,32.5,148,133,Yes,No,1,Yes,Yes,CAP-1,132,13.4,No,Very good,Pharmacy\nGH-0341,37,Male,Cape Coast,Rural,None,Moderate,33.8,113,110,No,Yes,1,No,No,CAP-4,108,9.8,No,Fair,CHPS compound\nGH-0342,35,Male,Cape Coast,Urban,Secondary,Low,24.7,105,95,No,No,3,Yes,No,CAP-1,87,7.1,No,Poor,CHPS compound\nGH-0343,49,Female,Tamale,Rural,None,Moderate,21.5,106,86,No,No,3,No,No,TAM-2,85,21.7,No,Good,CHPS compound\nGH-0344,32,Male,Kumasi,Urban,None,Moderate,31.7,122,119,No,No,0,Yes,Yes,KUM-4,116,31.7,No,Very good,Hospital\nGH-0345,47,Male,Ho,Rural,,Low,22.1,131,137,No,No,1,No,No,HO-4,122,15.1,No,Very good,Hospital\nGH-0346,44,Female,Kumasi,Urban,Secondary,Moderate,26.1,115,104,No,No,1,Yes,Yes,KUM-1,112,18.5,Yes,Poor,CHPS compound\nGH-0347,52,Female,Kumasi,Urban,Primary,Low,14.3,118,116,No,No,3,No,No,KUM-3,113,27.0,No,Very good,Hospital\nGH-0348,55,Male,Tamale,Rural,Secondary,High,36.4,159,150,Yes,No,2,No,No,TAM-1,157,24.7,No,Fair,Hospital\nGH-0349,47,Female,Tamale,Rural,Primary,Moderate,26.3,150,149,Yes,No,0,No,Yes,TAM-3,140,24.1,No,Good,Hospital\nGH-0350,39,Male,Ho,Urban,Secondary,Moderate,28.1,138,148,No,No,2,No,No,HO-2,128,13.2,No,Good,Hospital\nGH-0351,58,Male,Tamale,Rural,Secondary,Moderate,35.4,137,137,No,No,4,Yes,Yes,TAM-2,134,32.5,No,Good,CHPS compound\nGH-0352,48,Male,Ho,Rural,None,Moderate,30.7,115,106,No,Yes,2,No,Yes,HO-4,121,7.5,Yes,Poor,CHPS compound\nGH-0353,34,Male,Tamale,Rural,Primary,High,20.7,89,95,No,Yes,0,No,No,TAM-4,96,36.0,No,Fair,CHPS compound\nGH-0354,47,Female,Accra,Urban,Primary,Low,15.0,106,,No,No,3,No,Yes,ACC-2,102,25.4,No,Poor,Hospital\nGH-0355,37,Female,Tamale,Rural,Secondary,Low,32.6,126,140,No,Yes,1,No,No,TAM-4,123,10.6,No,Very good,Pharmacy\nGH-0356,18,Male,Kumasi,Rural,Primary,Moderate,,98,102,No,No,0,Yes,Yes,KUM-4,85,25.3,No,Poor,Hospital\nGH-0357,33,Male,Accra,Urban,Secondary,Low,27.1,146,140,Yes,No,1,No,No,ACC-1,115,9.1,Yes,Good,Pharmacy\nGH-0358,40,Male,Kumasi,Urban,Secondary,Moderate,26.8,110,104,No,No,0,Yes,No,KUM-3,101,18.9,No,Poor,CHPS compound\nGH-0359,39,Male,Accra,Rural,Primary,Low,18.7,108,97,No,No,2,No,No,ACC-1,108,19.2,No,Good,CHPS compound\nGH-0360,25,Male,Kumasi,Urban,Primary,Moderate,33.2,94,89,No,No,1,No,No,KUM-2,106,6.1,Yes,Very good,Hospital\nGH-0361,41,Male,Kumasi,Urban,None,Moderate,32.2,147,128,Yes,No,3,No,Yes,KUM-1,136,29.9,Yes,Good,CHPS compound\nGH-0362,51,Male,Tamale,Rural,Secondary,Low,32.5,127,127,No,Yes,0,No,No,TAM-4,121,10.1,No,Poor,CHPS compound\nGH-0363,30,Male,Ho,Rural,Secondary,Low,18.6,87,78,No,No,1,No,No,HO-3,74,28.5,No,Fair,Pharmacy\nGH-0364,48,Male,Ho,Urban,,Moderate,29.9,131,133,No,No,1,No,Yes,HO-2,110,21.6,No,Very good,CHPS compound\nGH-0365,19,Female,Cape Coast,Rural,Secondary,Moderate,25.8,130,115,No,No,0,Yes,Yes,CAP-1,121,27.4,No,Very good,CHPS compound\nGH-0366,52,Male,Accra,Rural,Tertiary,Low,26.0,155,153,Yes,No,5,Yes,No,ACC-1,119,20.0,Yes,Good,CHPS compound\nGH-0367,38,Male,Cape Coast,Rural,Secondary,Moderate,27.0,122,121,No,No,0,Yes,Yes,CAP-2,117,10.4,No,Poor,Hospital\nGH-0368,28,Male,Tamale,Urban,None,Low,18.4,118,114,No,No,1,No,No,TAM-2,103,24.5,No,Good,Hospital\nGH-0369,60,Female,Cape Coast,Urban,Secondary,Moderate,27.1,129,133,No,No,0,Yes,Yes,CAP-1,107,23.8,No,Poor,Hospital\nGH-0370,51,Female,Tamale,Urban,Secondary,Moderate,12.4,113,112,No,No,1,Yes,Yes,TAM-1,83,17.3,No,Poor,Pharmacy\nGH-0371,57,Female,Accra,Urban,Primary,High,17.7,121,131,No,No,1,Yes,Yes,ACC-2,122,13.0,No,Good,Hospital\nGH-0372,49,Female,Tamale,Urban,Primary,High,24.1,150,,Yes,No,5,No,Yes,TAM-1,135,12.0,Yes,Fair,Pharmacy\nGH-0373,50,Female,Ho,Urban,None,Moderate,19.7,113,,No,No,1,No,Yes,HO-3,100,8.1,No,Poor,Hospital\nGH-0374,40,Female,Kumasi,Rural,Secondary,Moderate,30.9,101,91,No,No,2,Yes,Yes,KUM-1,97,18.2,No,Fair,Hospital\nGH-0375,51,Male,Cape Coast,Rural,None,Moderate,19.1,111,111,No,No,1,Yes,Yes,CAP-3,103,9.1,Yes,Good,CHPS compound\nGH-0376,44,Male,Tamale,Urban,Secondary,High,,142,150,Yes,No,6,No,Yes,TAM-1,147,24.4,No,Very good,CHPS compound\nGH-0377,37,Female,Cape Coast,Rural,Tertiary,Moderate,27.4,124,120,No,No,0,No,Yes,CAP-2,120,7.5,No,Good,CHPS compound\nGH-0378,43,Male,Accra,Urban,Secondary,Moderate,24.4,118,119,No,No,0,No,Yes,ACC-2,130,22.0,No,Fair,CHPS compound\nGH-0379,29,Female,Tamale,Urban,None,Low,21.2,112,116,No,No,0,Yes,Yes,TAM-3,93,27.1,Yes,Fair,Pharmacy\nGH-0380,45,Female,Kumasi,Urban,Secondary,Moderate,28.0,130,110,No,No,1,Yes,Yes,KUM-3,135,24.5,No,Very good,CHPS compound\nGH-0381,51,Male,Tamale,Urban,Secondary,Moderate,21.6,123,,No,Yes,3,Yes,Yes,TAM-3,111,14.6,No,Very good,Pharmacy\nGH-0382,47,Male,Kumasi,Urban,Secondary,Low,32.6,141,111,Yes,No,1,No,No,KUM-3,150,10.1,No,Very good,Pharmacy\nGH-0383,52,Male,Kumasi,Rural,Primary,Low,23.6,115,114,No,No,3,Yes,Yes,KUM-1,119,17.5,Yes,Good,Pharmacy\nGH-0384,41,Female,Tamale,Urban,Tertiary,Low,37.2,117,123,No,Yes,1,Yes,Yes,TAM-4,124,9.5,No,Good,CHPS compound\nGH-0385,45,Female,Accra,Urban,Secondary,Low,14.4,124,122,No,No,1,No,No,ACC-2,125,9.6,No,Fair,Hospital\nGH-0386,45,Female,Accra,Urban,Primary,Low,18.4,117,120,No,No,2,No,No,ACC-1,106,15.0,No,Good,CHPS compound\nGH-0387,70,Male,Accra,Rural,Primary,Low,22.5,99,88,No,No,3,Yes,Yes,ACC-4,93,10.1,Yes,Poor,Pharmacy\nGH-0388,28,Male,Cape Coast,Urban,None,High,34.3,124,125,No,No,0,No,No,CAP-1,99,32.8,Yes,Fair,Hospital\nGH-0389,71,Male,Kumasi,Rural,None,Low,19.2,141,137,Yes,No,1,No,No,KUM-3,126,13.8,Yes,Fair,CHPS compound\nGH-0390,36,Male,Tamale,Rural,None,Moderate,46.2,160,149,Yes,Yes,1,No,No,TAM-2,158,17.0,No,Good,CHPS compound\nGH-0391,41,Female,Ho,Rural,None,High,24.5,134,131,No,Yes,0,No,No,HO-1,147,34.1,No,Very good,Hospital\nGH-0392,49,Male,Accra,Urban,Tertiary,Moderate,39.6,147,141,Yes,Yes,1,No,No,ACC-4,134,6.3,Yes,Very good,Hospital\nGH-0393,53,Male,Kumasi,Urban,Secondary,High,37.1,159,160,Yes,Yes,2,Yes,Yes,KUM-1,154,8.9,Yes,Very good,Pharmacy\nGH-0394,45,Female,Accra,Urban,Primary,Low,34.9,144,131,Yes,No,0,Yes,Yes,ACC-2,141,28.5,No,Very good,CHPS compound\nGH-0395,48,Female,Kumasi,Rural,Secondary,High,19.0,139,120,No,No,2,Yes,Yes,KUM-2,138,4.5,Yes,Fair,Hospital\nGH-0396,63,Female,Ho,Urban,Tertiary,Moderate,55.4,147,145,Yes,No,2,Yes,Yes,HO-2,132,17.7,Yes,Very good,Pharmacy\nGH-0397,79,Male,Accra,Urban,Primary,Moderate,36.7,155,160,Yes,Yes,4,No,No,ACC-1,135,12.1,Yes,Very good,Pharmacy\nGH-0398,50,Male,Ho,Rural,Primary,Moderate,14.2,101,96,No,Yes,0,No,No,HO-1,99,0.5,Yes,Very good,Hospital\nGH-0399,54,Female,Kumasi,Rural,None,Low,17.6,134,114,No,No,1,No,Yes,KUM-1,131,6.5,No,Good,CHPS compound\nGH-0400,44,Male,Kumasi,Urban,None,Low,47.6,136,134,No,Yes,2,No,No,KUM-4,135,24.4,Yes,Good,Pharmacy\nGH-0401,67,Female,Tamale,Urban,None,Low,30.0,125,107,No,No,0,No,No,TAM-3,111,27.8,Yes,Very good,Hospital\nGH-0402,44,Male,Accra,Urban,Secondary,Moderate,29.7,154,158,Yes,Yes,5,Yes,No,ACC-3,140,6.8,Yes,Very good,Hospital\nGH-0403,52,Female,Accra,Urban,Secondary,,28.5,117,114,No,No,6,Yes,Yes,ACC-4,119,15.4,Yes,Good,CHPS compound\nGH-0404,32,Male,Tamale,Rural,Secondary,Moderate,41.1,129,115,No,No,0,Yes,Yes,TAM-3,102,12.7,No,Good,CHPS compound\nGH-0405,18,Female,Tamale,Urban,Secondary,Moderate,39.9,123,116,No,No,0,No,Yes,TAM-2,99,30.0,No,Very good,Pharmacy\nGH-0406,74,Male,Kumasi,Urban,None,Low,35.0,147,147,Yes,No,1,Yes,Yes,KUM-2,165,2.8,Yes,Fair,Hospital\nGH-0407,59,Female,Accra,Urban,Secondary,Moderate,18.2,140,120,Yes,No,3,No,No,ACC-3,128,20.1,No,Poor,Hospital\nGH-0408,57,Male,Ho,Urban,Tertiary,Low,57.9,169,165,Yes,No,3,Yes,Yes,HO-4,170,28.5,No,Very good,Hospital\nGH-0409,33,Male,Tamale,Rural,Secondary,High,15.5,103,98,No,Yes,2,No,No,TAM-1,89,12.1,Yes,Fair,Hospital\nGH-0410,24,Female,Ho,Rural,Secondary,High,10.6,113,111,No,No,0,Yes,Yes,HO-4,102,7.6,No,Very good,Pharmacy\nGH-0411,42,Male,Tamale,Urban,Secondary,Moderate,45.3,140,143,Yes,No,0,Yes,Yes,TAM-2,116,32.8,No,Very good,CHPS compound\nGH-0412,47,Female,Cape Coast,Rural,None,Low,17.9,109,103,No,No,3,No,No,CAP-3,103,15.8,No,Poor,CHPS compound\nGH-0413,54,Female,Cape Coast,Urban,Tertiary,Moderate,21.7,134,118,No,No,2,No,No,CAP-4,128,6.7,No,Fair,CHPS compound\nGH-0414,45,Female,Ho,Urban,Primary,Low,31.3,133,130,No,No,1,No,No,HO-3,120,27.4,No,Fair,Pharmacy\nGH-0415,62,Female,Accra,Urban,None,High,21.7,136,120,No,No,3,Yes,No,ACC-1,130,22.8,Yes,Fair,CHPS compound\nGH-0416,50,Female,Tamale,Urban,Secondary,Moderate,21.2,112,115,No,No,3,Yes,Yes,TAM-4,104,28.1,No,Good,Hospital\nGH-0417,49,Male,Ho,Urban,None,Moderate,24.3,130,124,No,No,2,Yes,Yes,HO-3,126,9.3,Yes,Very good,Pharmacy\nGH-0418,60,Female,Kumasi,Rural,,Low,30.8,119,109,No,No,3,No,Yes,KUM-4,109,28.5,No,Fair,Hospital\nGH-0419,37,Male,Tamale,Rural,Secondary,Moderate,24.2,132,123,No,No,3,Yes,No,TAM-1,123,17.8,No,Fair,CHPS compound\nGH-0420,50,Male,Kumasi,Rural,None,Low,37.2,134,131,No,No,0,Yes,Yes,KUM-1,134,21.7,No,Fair,CHPS compound\n";
/* =====================================================================
   UI core: state, persistence, helpers, charts, code blocks, result card
   ===================================================================== */
const esc = s => String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const $ = (s, el) => (el || document).querySelector(s);
const F = Analysis.fmt;
const TYPE_LABEL = { continuous: "continuous", count: "count", binary: "binary", categorical: "categorical", ordinal: "ordinal", id: "ID / ignore" };
const TYPE_SHORT = { continuous: "cont", count: "count", binary: "bin", categorical: "cat", ordinal: "ord", id: "id" };

const store = {
  get(k, d) { try { const v = localStorage.getItem("quantai:" + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("quantai:" + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
};

const S = {
  section: store.get("section", "chat"),
  tool: store.get("tool", { method: "question", data: "compare" }),
  codeLang: store.get("codeLang", "stata"),
  alpha: store.get("alpha", 0.05),
  raw: null, ds: null, dsSource: "", isExample: true, overrides: {}, missingCodes: "",
  log: [], logSeq: 0, cur: { describe: null, compare: null, regression: null }, errors: {},
  form: {
    describe: { vars: [], group: "" },
    compare: { outcome: "", exposure: "", paired: false, force: "" },
    regression: { outcome: "", preds: [], model: "auto" },
  },
  m: Object.assign({
    question: { fw: "PECO", v: {}, setting: "", finer: {} },
    design: { a: { aim: "association", rare: "common", time: "once", constraint: "time" } },
    sample: { id: "prev", v: {}, adj: { deff: 1, N: "", nonResponse: 0.1, icc: "", m: "" } },
    test: { a: { goal: "compare", outcome: "cont", groups: "3", paired: "ind", normal: "unsure" } },
    checklist: { id: "STROBE", done: {} },
    ai: { mode: "orient", inputs: {} },
  }, store.get("m", {})),
  ai: { out: {}, loading: {}, error: {} },
};
function persist() { store.set("section", S.section); store.set("tool", S.tool); store.set("codeLang", S.codeLang); store.set("alpha", S.alpha); store.set("m", S.m); }

const CAP = { sample: null, downloads: null, ready: true };

/* ---------- small helpers ---------- */
let toastTimer;
function toast(msg) {
  let t = $(".qa-toast"); if (!t) { t = document.createElement("div"); t.className = "qa-toast"; t.setAttribute("role", "status"); document.body.appendChild(t); }
  t.textContent = msg; t.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 2200);
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch (e) {
    const ta = document.createElement("textarea"); ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); toast("Copied"); } catch (e2) { toast("Select the text and copy it manually"); }
    ta.remove();
  }
}
async function saveFile(filename, data) {
  try {
    const blob = data instanceof Blob ? data : new Blob([data], { type: filename.endsWith(".html") ? "text/html" : filename.endsWith(".csv") ? "text/csv" : "text/plain" });
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    toast("Downloaded " + filename); return true;
  } catch (e) { toast("The download failed. Use the copy buttons instead."); return false; }
}
function qaTop() { const m = document.getElementById("qa-main"); if (m) m.scrollTop = 0; }
const getPath = (o, p) => p.split(".").reduce((a, k) => a == null ? a : a[k], o);
function setPath(o, p, v) { const ks = p.split("."); let a = o; ks.slice(0, -1).forEach(k => { if (a[k] == null || typeof a[k] !== "object") a[k] = {}; a = a[k]; }); a[ks[ks.length - 1]] = v; }
const varsUsable = () => S.ds ? S.ds.vars.filter(v => v.type !== "id") : [];
const V = name => S.ds && S.ds.vars.find(v => v.name === name);
const vlabel = v => v.label && v.label !== v.name ? v.label : v.name;
function varOptions(selected, filter, placeholder) {
  const vs = varsUsable().filter(filter || (() => true));
  return `<option value="">${esc(placeholder || "Choose a variable")}</option>` + vs.map(v => `<option value="${esc(v.name)}"${v.name === selected ? " selected" : ""}>${esc(vlabel(v))} (${TYPE_LABEL[v.type]})</option>`).join("");
}

/* ---------- code block ---------- */
const CODE_LANGS = [["stata", "Stata"], ["python", "Python"], ["r", "R"], ["spss", "SPSS"]];
function highlight(lang, src) {
  return src.split("\n").map(line => {
    const t = line.trim(), e = esc(line);
    const isC = (lang === "python" || lang === "r") ? t.startsWith("#") : (lang === "stata" ? t.startsWith("*") || t.startsWith("//") : t.startsWith("*"));
    if (isC) return `<span class="cm">${e}</span>`;
    if (lang === "stata" && line.includes("//")) { const i = line.indexOf("//"); return esc(line.slice(0, i)) + `<span class="cm">${esc(line.slice(i))}</span>`; }
    if ((lang === "python" || lang === "r") && /\s#\s/.test(line) && !/["'][^"']*#/.test(line)) { const i = line.search(/\s#\s/); return esc(line.slice(0, i)) + `<span class="cm">${esc(line.slice(i))}</span>`; }
    return e;
  }).join("\n");
}
/** scripts: {stata, python, r, spss} or a function lang => string. */
const codeRegistry = new Map(); let codeSeq = 0;
function codeBlock(scripts, opts) {
  opts = opts || {};
  const id = "cb" + (++codeSeq);
  codeRegistry.set(id, scripts);
  const get = l => typeof scripts === "function" ? scripts(l) : scripts[l];
  const lang = S.codeLang;
  return `<div class="code" data-code="${id}">
    <div class="code-tabs" role="tablist">${CODE_LANGS.map(([l, n]) => `<button role="tab" aria-selected="${l === lang}" data-act="lang" data-v="${l}">${n}</button>`).join("")}
      <span class="sp"><button class="btn quiet sm" data-act="copy-code" data-id="${id}">Copy</button></span></div>
    <pre class="src" tabindex="0" style="${opts.maxh ? `max-height:${opts.maxh}px` : ""}">${highlight(lang, get(lang) || "")}</pre></div>`;
}
function codeText(id) { const s = codeRegistry.get(id); return typeof s === "function" ? s(S.codeLang) : s[S.codeLang]; }

/* ---------- tables ---------- */
function tableHTML(t) {
  const leftCols = new Set([0]);
  return `<div class="tw"><table class="t"><caption>${esc(t.title)}</caption><thead><tr>${t.columns.map((c, i) => `<th class="${leftCols.has(i) ? "l" : ""}">${esc(c)}</th>`).join("")}</tr></thead>
  <tbody>${t.rows.map(r => `<tr>${r.map((c, i) => `<td class="${leftCols.has(i) ? "l" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>${t.note ? `<div class="tnote">${esc(t.note)}</div>` : ""}</div>`;
}

/* ---------- charts (SVG, theme tokens) ---------- */
function niceTicks(lo, hi, n) {
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo, step0 = span / (n || 5), mag = Math.pow(10, Math.floor(Math.log10(step0))), err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = []; for (let v = Math.floor(lo / step + 1e-9) * step; v < hi + step - step * 1e-9; v += step) out.push(+v.toFixed(10));
  if (out[out.length - 1] < hi - 1e-12) out.push(+(out[out.length - 1] + step).toFixed(10));
  return out;
}
const fmtTick = v => Math.abs(v) >= 1000 ? v.toLocaleString() : (+v.toFixed(3)).toString();
const COLORS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];
function axisY(ticks, y, x0, x1) { return ticks.map(t => `<line class="grid" x1="${x0}" x2="${x1}" y1="${y(t)}" y2="${y(t)}"/><text x="${x0 - 6}" y="${y(t) + 4}" text-anchor="end">${fmtTick(t)}</text>`).join(""); }
function chartHTML(c) {
  if (!c) return "";
  const W = 560, H = 300, m = { l: 54, r: 16, t: 14, b: 46 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  if (c.type === "box") {
    const all = c.groups.flatMap(g => [g.lo, g.hi, ...g.outliers]);
    const ticks = niceTicks(Math.min(...all), Math.max(...all), 5), lo = ticks[0], hi = ticks[ticks.length - 1];
    const y = v => m.t + ih - (v - lo) / (hi - lo) * ih, bw = Math.min(56, iw / c.groups.length * .5);
    const rot = c.groups.length > 4 || c.groups.some(g => String(g.label).length > 10);
    const body = c.groups.map((g, i) => {
      const cx = m.l + iw * (i + .5) / c.groups.length, col = COLORS[i % COLORS.length];
      return `<line x1="${cx}" x2="${cx}" y1="${y(g.lo)}" y2="${y(g.q1)}" stroke="var(--ink-3)"/><line x1="${cx}" x2="${cx}" y1="${y(g.q3)}" y2="${y(g.hi)}" stroke="var(--ink-3)"/>
        <line x1="${cx - bw / 4}" x2="${cx + bw / 4}" y1="${y(g.lo)}" y2="${y(g.lo)}" stroke="var(--ink-3)"/><line x1="${cx - bw / 4}" x2="${cx + bw / 4}" y1="${y(g.hi)}" y2="${y(g.hi)}" stroke="var(--ink-3)"/>
        <rect x="${cx - bw / 2}" y="${y(g.q3)}" width="${bw}" height="${Math.max(1, y(g.q1) - y(g.q3))}" fill="${col}" fill-opacity=".18" stroke="${col}" rx="3"/>
        <line x1="${cx - bw / 2}" x2="${cx + bw / 2}" y1="${y(g.median)}" y2="${y(g.median)}" stroke="${col}" stroke-width="2.5"/>
        <circle cx="${cx}" cy="${y(g.mean)}" r="3" fill="var(--surface)" stroke="${col}" stroke-width="1.5"/>
        ${g.outliers.map(o => `<circle cx="${cx}" cy="${y(o)}" r="2.2" fill="${col}" fill-opacity=".55"/>`).join("")}
        <text x="${cx}" y="${H - m.b + 16}" text-anchor="${rot ? "end" : "middle"}" ${rot ? `transform="rotate(-25 ${cx} ${H - m.b + 16})"` : ""}>${esc(String(g.label).slice(0, 18))}</text>`;
    }).join("");
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Box plot of ${esc(c.ylabel)} by group">${axisY(ticks, y, m.l, W - m.r)}${body}
      <text x="14" y="${m.t + ih / 2}" transform="rotate(-90 14 ${m.t + ih / 2})" text-anchor="middle">${esc(c.ylabel)}</text></svg>
      <p class="sub muted" style="font-size:.76rem">Box = IQR, thick line = median, circle = mean, whiskers = 1.5 × IQR, dots = outliers.</p>`;
  }
  if (c.type === "scatter" || c.type === "residuals") {
    const pts = c.points, xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const xt = niceTicks(Math.min(...xs), Math.max(...xs), 6), yt = niceTicks(Math.min(...ys), Math.max(...ys), 5);
    const x = v => m.l + (v - xt[0]) / (xt[xt.length - 1] - xt[0]) * iw, y = v => m.t + ih - (v - yt[0]) / (yt[yt.length - 1] - yt[0]) * ih;
    let extra = "";
    if (c.type === "scatter" && pts.length > 2) {
      const mx = Stats.mean(xs), my = Stats.mean(ys); let sxy = 0, sxx = 0; pts.forEach(p => { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) ** 2; });
      const b = sxy / sxx, a = my - b * mx, x0 = xt[0], x1 = xt[xt.length - 1];
      extra = `<line x1="${x(x0)}" x2="${x(x1)}" y1="${y(a + b * x0)}" y2="${y(a + b * x1)}" stroke="var(--c2)" stroke-width="2"/>`;
    }
    if (c.type === "residuals" && yt[0] < 0 && yt[yt.length - 1] > 0) extra = `<line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--c2)" stroke-width="1.5"/>`;
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${c.type === "scatter" ? "Scatter plot" : "Residuals versus fitted values"}">${axisY(yt, y, m.l, W - m.r)}
      ${xt.map(t => `<text x="${x(t)}" y="${H - m.b + 16}" text-anchor="middle">${fmtTick(t)}</text>`).join("")}
      ${pts.map(p => `<circle cx="${x(p[0]).toFixed(1)}" cy="${y(p[1]).toFixed(1)}" r="2.6" fill="var(--c1)" fill-opacity=".45"/>`).join("")}${extra}
      <text x="${m.l + iw / 2}" y="${H - 8}" text-anchor="middle">${esc(c.type === "scatter" ? c.xlabel : "Fitted values")}</text>
      <text x="14" y="${m.t + ih / 2}" transform="rotate(-90 14 ${m.t + ih / 2})" text-anchor="middle">${esc(c.type === "scatter" ? c.ylabel : "Residuals")}</text></svg>
      ${c.type === "residuals" ? `<p class="sub muted" style="font-size:.76rem">Look for an even band around zero. A funnel shape means unequal variance; a curve means a non-linear relationship.</p>` : ""}`;
  }
  if (c.type === "roc") {
    const s = 250, p = 34, x = v => p + v * (s - p - 10), y = v => s - p - v * (s - p - 10);
    return `<svg class="chart" style="max-width:300px" viewBox="0 0 ${s} ${s}" role="img" aria-label="ROC curve, AUC ${F.f(c.auc, 3)}">
      ${[0, .25, .5, .75, 1].map(t => `<line class="grid" x1="${x(0)}" x2="${x(1)}" y1="${y(t)}" y2="${y(t)}"/><text x="${p - 5}" y="${y(t) + 4}" text-anchor="end">${t}</text><text x="${x(t)}" y="${s - p + 15}" text-anchor="middle">${t}</text>`).join("")}
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--ink-3)" stroke-dasharray="4 4"/>
      <path d="${c.points.map((q, i) => `${i ? "L" : "M"}${x(q.fpr).toFixed(1)},${y(q.tpr).toFixed(1)}`).join("")}" fill="none" stroke="var(--c1)" stroke-width="2.5"/>
      <text x="${x(.55)}" y="${y(.12)}" style="font-weight:600">AUC = ${F.f(c.auc, 3)}</text>
      <text x="${x(.5)}" y="${s - 4}" text-anchor="middle">1 − specificity</text><text x="10" y="${y(.5)}" transform="rotate(-90 10 ${y(.5)})" text-anchor="middle">Sensitivity</text></svg>`;
  }
  if (c.type === "stack") {
    const rowH = 26, Hs = m.t + c.rowLabels.length * (rowH + 8) + 40, lw = 110, iw2 = W - lw - 20;
    const rows = c.table.map((r, i) => {
      const tot = r.reduce((a, b) => a + b, 0); let acc = 0; const yy = m.t + i * (rowH + 8);
      return `<text x="${lw - 8}" y="${yy + rowH / 2 + 4}" text-anchor="end">${esc(String(c.rowLabels[i]).slice(0, 16))}</text>` + r.map((v, j) => {
        const w = tot ? v / tot * iw2 : 0, xx = lw + acc; acc += w;
        return `<rect x="${xx}" y="${yy}" width="${Math.max(0, w)}" height="${rowH}" fill="${COLORS[j % COLORS.length]}" fill-opacity="${.25 + .6 * (j / Math.max(1, r.length - 1))}"/>${w > 34 ? `<text x="${xx + w / 2}" y="${yy + rowH / 2 + 4}" text-anchor="middle" style="fill:var(--ink)">${Math.round(100 * v / tot)}%</text>` : ""}`;
      }).join("");
    }).join("");
    const ly = m.t + c.rowLabels.length * (rowH + 8) + 12;
    return `<svg class="chart" viewBox="0 0 ${W} ${Hs}" role="img" aria-label="Row percentages">${rows}${c.colLabels.map((l, j) => `<rect x="${lw + j * 110}" y="${ly}" width="12" height="12" fill="${COLORS[j % COLORS.length]}" fill-opacity="${.25 + .6 * (j / Math.max(1, c.colLabels.length - 1))}"/><text x="${lw + j * 110 + 17}" y="${ly + 10}">${esc(String(l).slice(0, 14))}</text>`).join("")}</svg>`;
  }
  if (c.type === "km") {
    const tmax = Math.max(...c.curves.flatMap(cv => cv.points.map(q => q[0]))), xt = niceTicks(0, tmax, 6), x1 = xt[xt.length - 1];
    const x = v => m.l + v / x1 * iw, y = v => m.t + ih - v * ih;
    const lines = c.curves.map((cv, i) => { let d = `M${x(0).toFixed(1)},${y(1).toFixed(1)}`; let prev = 1; cv.points.slice(1).forEach(([t, sv]) => { d += ` H${x(t).toFixed(1)} V${y(sv).toFixed(1)}`; prev = sv; }); d += ` H${x(tmax).toFixed(1)}`; return `<path d="${d}" fill="none" stroke="${COLORS[i % COLORS.length]}" stroke-width="2.2"/>`; }).join("");
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Kaplan–Meier survival curves">${axisY([0, .25, .5, .75, 1], y, m.l, W - m.r)}
      ${xt.map(t => `<text x="${x(t)}" y="${H - m.b + 16}" text-anchor="middle">${fmtTick(t)}</text>`).join("")}${lines}
      <line x1="${m.l}" x2="${W - m.r}" y1="${y(.5)}" y2="${y(.5)}" stroke="var(--ink-3)" stroke-dasharray="3 4"/>
      ${c.curves.length > 1 ? c.curves.map((cv, i) => `<rect x="${m.l + 10 + i * 120}" y="${m.t + ih - 20}" width="12" height="3" fill="${COLORS[i % COLORS.length]}"/><text x="${m.l + 26 + i * 120}" y="${m.t + ih - 16}">${esc(String(cv.label).slice(0, 14))}</text>`).join("") : ""}
      <text x="${m.l + iw / 2}" y="${H - 8}" text-anchor="middle">${esc(c.xlabel)}</text><text x="14" y="${m.t + ih / 2}" transform="rotate(-90 14 ${m.t + ih / 2})" text-anchor="middle">Proportion without the event</text></svg>
      <p class="sub muted" style="font-size:.76rem">Each step down is an event; the dashed line marks 50% (median survival).</p>`;
  }
  if (c.type === "forest") {
    const rows = c.rows.filter(r => isFinite(r.lo) && isFinite(r.hi) && r.lo > 0);
    if (!rows.length) return "";
    const lw = 190, Hs = m.t + rows.length * 26 + 40, iw2 = W - lw - 20;
    const lo = Math.min(...rows.map(r => r.lo), 1), hi = Math.max(...rows.map(r => r.hi), 1), L0 = Math.log(lo / 1.1), L1 = Math.log(hi * 1.1);
    const x = v => lw + (Math.log(v) - L0) / (L1 - L0) * iw2;
    const ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20].filter(t => Math.log(t) >= L0 && Math.log(t) <= L1);
    return `<svg class="chart" viewBox="0 0 ${W} ${Hs}" role="img" aria-label="Forest plot of ${esc(c.label)}">
      ${ticks.map(t => `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${Hs - 30}"/><text x="${x(t)}" y="${Hs - 14}" text-anchor="middle">${t}</text>`).join("")}
      <line x1="${x(1)}" x2="${x(1)}" y1="${m.t}" y2="${Hs - 30}" stroke="var(--ink-3)"/>
      ${rows.map((r, i) => { const yy = m.t + i * 26 + 12; return `<text x="${lw - 8}" y="${yy + 4}" text-anchor="end">${esc(r.name.slice(0, 30))}</text><line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${yy}" y2="${yy}" stroke="var(--c1)" stroke-width="2"/><rect x="${x(r.est) - 4}" y="${yy - 4}" width="8" height="8" fill="var(--c1)"/>`; }).join("")}
      <text x="${lw + iw2 / 2}" y="${Hs - 1}" text-anchor="middle">${esc(c.label)} (log scale)</text></svg>`;
  }
  return "";
}

/* ---------- result card ---------- */
function auditHTML(decision) {
  return `<ol class="audit">${decision.map(d => {
    const cls = d.final ? "fin" : d.ok === true ? "ok" : d.ok === false ? "no" : "";
    const ic = d.final ? "→" : d.ok === true ? "✓" : d.ok === false ? "!" : "•";
    return `<li class="${d.final ? "final" : ""}"><span class="ic ${cls}" aria-hidden="true">${ic}</span><span><b>${esc(d.rule)}.</b> ${esc(d.detail)}</span></li>`;
  }).join("")}</ol>`;
}
function scriptsForEntries(entries) {
  return lang => Codegen.script(lang, S.ds, entries, { date: new Date().toISOString().slice(0, 10), alpha: S.alpha, source: S.dsSource });
}
function resultCard(entry) {
  const r = entry.result;
  const ai = S.ai.out["explain" + entry.id], aiLoading = S.ai.loading["explain" + entry.id], aiErr = S.ai.error["explain" + entry.id];
  return `<article class="card stack" id="res-${entry.id}">
    <div class="res-head"><div class="stack" style="gap:.3rem"><span class="sect-t">Result · analysis ${entry.k} in your log</span><h2>${esc(r.title)}</h2></div>
      <div class="meta"><span class="pill acc">${esc(r.method)}</span>${r.n ? `<span class="pill num">n = ${r.n}</span>` : ""}</div></div>
    ${r.warnings.map(w => `<div class="notice warn">${esc(w)}</div>`).join("")}
    <div class="grid2">
      <section><div class="sect-t">Rules applied</div>${auditHTML(r.decision)}</section>
      <section>${r.chart ? `<div class="sect-t">Chart</div>${chartHTML(r.chart)}` : `<div class="sect-t">How to read this</div><p class="sub">Each line on the left is a rule the app checked. The arrow shows the method the rules selected. Change a variable's type in Data &amp; variables if the app has misread it.</p>`}</section>
    </div>
    <div class="stack">${r.tables.map(tableHTML).join("")}</div>
    ${r.writeup ? `<section class="stack" style="gap:.5rem"><div class="row" style="justify-content:space-between;align-items:center"><div class="sect-t" style="margin:0">Write-up (APA 7)</div><button class="btn quiet sm" data-act="copy-apa" data-id="${entry.id}">Copy text</button></div><p class="apa">${esc(r.writeup)}</p></section>` : ""}
    ${aiExplainHTML(entry)}
    <section class="stack" style="gap:.5rem"><div class="sect-t" style="margin:0">Code to reproduce this analysis by hand</div>
      <p class="sub">A complete script: loads <code class="n">analysis_data.csv</code> (Log &amp; export), codes variables exactly as here, runs the assumption checks, then the chosen test.</p>
      ${codeBlock(scriptsForEntries([{ spec: r.spec, title: r.title }]))}</section>
  </article>`;
}

/* =====================================================================
   Methodology section pages
   ===================================================================== */
const METHOD_TOOLS = [
  { id: "question", k: "01", label: "Research question", sub: "Framework, objectives, hypotheses" },
  { id: "design", k: "02", label: "Study design", sub: "Rule-based design advisor" },
  { id: "sample", k: "03", label: "Sample size", sub: "Formulas with adjustments" },
  { id: "test", k: "04", label: "Choose a test", sub: "Test selector with code" },
  { id: "checklist", k: "05", label: "Reporting checklist", sub: "STROBE, CONSORT, STARD…" },
];

function pageHead(eyebrow, title, intro) { return `<header class="ph"><span class="eyebrow">${esc(eyebrow)}</span><h2 class="ph-t">${esc(title)}</h2>${intro ? `<p>${intro}</p>` : ""}</header>`; }
function optQuestion(q, answers, act) {
  return `<div class="q"><div class="qt">${esc(q.q)}</div><div class="opts">${q.options.map(([v, l]) => `<button class="opt" aria-pressed="${answers[q.id] === v}" data-act="${act}" data-q="${q.id}" data-v="${esc(v)}">${esc(l)}</button>`).join("")}</div></div>`;
}

/* ---------- 01 research question ---------- */
function pageQuestion() {
  const st = S.m.question, fw = Method.FRAMEWORKS[st.fw];
  const vals = {}; fw.fields.forEach(([k, , ex]) => vals[k] = (getPath(st.v, st.fw + "." + k) || "").trim() || ex);
  return pageHead("Methodology · 01", "Research question", "Fill in the framework that matches your study. The question, objectives and hypotheses update as you type. Grey text shows an example you can overwrite.") + `
  <div class="grid2">
    <div class="card stack">
      <div class="field"><label class="label" for="fw">Framework</label><select id="fw" data-act="set" data-path="m.question.fw">${Object.entries(Method.FRAMEWORKS).map(([k, f]) => `<option value="${k}"${k === st.fw ? " selected" : ""}>${esc(f.name)}</option>`).join("")}</select></div>
      ${fw.fields.map(([k, label, ex]) => `<div class="field"><label class="label" for="q-${k}">${esc(k)} — ${esc(label)}</label><input type="text" id="q-${k}" placeholder="${esc(ex)}" value="${esc(getPath(st.v, st.fw + "." + k) || "")}" data-live="question" data-path="m.question.v.${st.fw}.${k}"></div>`).join("")}
      ${st.fw === "PECO" || st.fw === "PICO" ? `<div class="field"><label class="label" for="q-setting">Setting (optional)</label><input type="text" id="q-setting" placeholder="Northern Region, Ghana" value="${esc(st.setting || "")}" data-live="question" data-path="m.question.setting"></div>` : ""}
    </div>
    <div class="card stack" id="live-question">${questionOut(st.fw, vals, st.setting)}</div>
  </div>
  <div class="card stack" style="margin-top:1rem"><h2>FINER check</h2><p class="sub">A good question passes all five. Tick the ones you can defend to your supervisor or ethics committee.</p>
    ${Method.FINER.map(([k, d]) => `<label class="check-item ${st.finer[k] ? "done" : ""}"><input type="checkbox" data-act="finer" data-k="${k}" ${st.finer[k] ? "checked" : ""}><span><b>${k}</b> — ${esc(d)}</span></label>`).join("")}
    <div class="prog" aria-label="FINER progress"><i style="width:${Object.values(st.finer).filter(Boolean).length / 5 * 100}%"></i></div>
  </div>`;
}
function questionOut(fw, vals, setting) {
  const q = Method.buildQuestion(fw, vals, setting);
  const text = [`Research question: ${q.question}`, `General objective: ${q.general}`, "Specific objectives:", ...q.specific.map((s, i) => `${i + 1}. ${s}`), ...(q.h0 ? [`H0: ${q.h0}`, `H1: ${q.h1}`] : []), ...(q.note ? [q.note] : [])].join("\n");
  return `<div class="row" style="justify-content:space-between;align-items:center"><h2>Draft</h2><button class="btn quiet sm" data-act="copy" data-text="${esc(text)}">Copy all</button></div>
    <div><div class="sect-t">Research question</div><p class="apa">${esc(q.question)}</p></div>
    <div><div class="sect-t">General objective</div><p>${esc(q.general)}</p></div>
    <div><div class="sect-t">Specific objectives</div><ol class="clean">${q.specific.map(s => `<li>${esc(s)}</li>`).join("")}</ol></div>
    ${q.h0 ? `<div><div class="sect-t">Hypotheses</div><dl class="kv"><dt>H₀</dt><dd>${esc(q.h0)}</dd><dt>H₁</dt><dd>${esc(q.h1)}</dd></dl></div>` : ""}
    ${q.note ? `<div class="notice info">${esc(q.note)}</div>` : ""}
    <p class="sub muted">Edit the wording to fit your discipline. Specific objectives should each map to one analysis.</p>`;
}

/* ---------- 02 study design ---------- */
function pageDesign() {
  const a = S.m.design.a, qs = Method.DESIGN_QUESTIONS.filter(q => !q.show || q.show(a));
  const complete = qs.every(q => a[q.id]);
  const res = a.aim ? Method.decideDesign(a) : null;
  return pageHead("Methodology · 02", "Study design", "Answer each question. The recommendation follows fixed epidemiological rules, so the same answers always give the same design, with its biases, analysis and reporting guideline.") + `
  <div class="card stack">${qs.map(q => optQuestion(q, a, "design-opt")).join('<hr class="sep">')}
    <div class="row"><button class="btn quiet sm" data-act="design-reset">Start again</button></div></div>
  ${res && complete ? designResult(res) : res ? `<div class="notice info" style="margin-top:1rem">Answer the remaining questions to see the recommendation.</div>` : ""}`;
}
function designResult(res) {
  const d = res.design;
  const ssMap = { cs_desc: "prev", cs_anal: "prop2", cohort_pro: "prop2", cohort_retro: "prop2", case_control: "cc", rct: "mean2", crct: "prop2", quasi: "prop2", diag: "prev" };
  const ckMap = { STROBE: "STROBE", CONSORT: "CONSORT", STARD: "STARD", COREQ: "COREQ", PRISMA: "PRISMA" };
  const ck = Object.keys(ckMap).find(k => d.guideline.includes(k));
  return `<article class="card stack" style="margin-top:1rem">
    <div class="res-head"><div class="stack" style="gap:.25rem"><span class="sect-t">Recommended design</span><h2>${esc(d.name)}</h2></div><div class="meta"><span class="pill acc">${esc(d.family)}</span>${d.evidence ? `<span class="pill">Evidence level ${d.evidence} of 5</span>` : ""}</div></div>
    <p>${esc(d.what)}</p>
    <div><div class="sect-t">Why this design</div><ol class="audit">${res.why.map(w => `<li><span class="ic ok">✓</span><span>${esc(w)}</span></li>`).join("")}</ol></div>
    <div class="grid2"><div><div class="sect-t">Strengths</div><ul class="clean">${d.strengths.map(s => `<li>${esc(s)}</li>`).join("")}</ul></div><div><div class="sect-t">Limitations</div><ul class="clean">${d.limitations.map(s => `<li>${esc(s)}</li>`).join("")}</ul></div></div>
    <div><div class="sect-t">Main biases and how to reduce them</div><ul class="clean">${d.biases.map(s => `<li>${esc(s)}</li>`).join("")}</ul></div>
    <dl class="kv"><dt>Measure</dt><dd>${esc(d.measure)}</dd><dt>Analysis</dt><dd>${esc(d.analysis)}</dd><dt>Sampling</dt><dd>${esc(d.sampling)}</dd><dt>Report with</dt><dd>${esc(d.guideline)}</dd></dl>
    <div class="row">${ssMap[res.id] ? `<button class="btn" data-act="goto-sample" data-v="${ssMap[res.id]}">Calculate the sample size</button>` : ""}${ck ? `<button class="btn ghost" data-act="goto-check" data-v="${ck}">Open the ${ck} checklist</button>` : ""}</div>
    ${res.alternatives.length ? `<hr class="sep"><div><div class="sect-t">Alternatives</div><div class="stack" style="gap:.5rem">${res.alternatives.map(x => `<div><b>${esc(x.design.name)}</b> <span class="sub">— ${esc(x.when)}.</span></div>`).join("")}</div></div>` : ""}
  </article>`;
}

/* ---------- 03 sample size ---------- */
function pageSample() {
  const st = S.m.sample, c = Method.SAMPLE.find(x => x.id === st.id);
  const vals = sampleVals(st, c);
  return pageHead("Methodology · 03", "Sample size", "Pick the calculation that matches your primary objective. Every result shows the formula, the adjustment steps, a methods paragraph and code to check it.") + `
  <div class="grid2">
    <div class="card stack">
      <div class="field"><label class="label" for="ss-id">Calculation</label><select id="ss-id" data-act="set" data-path="m.sample.id">${Method.SAMPLE.map(x => `<option value="${x.id}"${x.id === st.id ? " selected" : ""}>${esc(x.name)}</option>`).join("")}</select><small>${esc(c.use)}</small></div>
      ${c.fields.map(([k, label, def, hint]) => `<div class="field"><label class="label" for="ss-${k}">${esc(label)}</label><input type="number" step="any" id="ss-${k}" value="${esc(vals[k])}" data-live="sample" data-path="m.sample.v.${st.id}.${k}" data-num="1">${hint ? `<small>${esc(hint)}</small>` : ""}</div>`).join("")}
      <hr class="sep"><h3>Adjustments</h3>
      <div class="row"><div class="field"><label class="label" for="ss-m">Cluster size m (optional)</label><input type="number" id="ss-m" value="${esc(st.adj.m)}" data-live="sample" data-path="m.sample.adj.m" data-num="1"></div>
        <div class="field"><label class="label" for="ss-icc">ICC (optional)</label><input type="number" step="any" id="ss-icc" value="${esc(st.adj.icc)}" data-live="sample" data-path="m.sample.adj.icc" data-num="1"></div></div>
      <div class="field"><label class="label" for="ss-deff">Design effect</label><input type="number" step="any" id="ss-deff" value="${esc(st.adj.deff)}" data-live="sample" data-path="m.sample.adj.deff" data-num="1"><small>1 for simple random sampling; about 1.5–2 for multistage cluster surveys. Filled automatically from m and ICC: 1 + (m − 1) × ICC.</small></div>
      <div class="row"><div class="field"><label class="label" for="ss-N">Population size N (optional)</label><input type="number" id="ss-N" value="${esc(st.adj.N)}" data-live="sample" data-path="m.sample.adj.N" data-num="1"><small>Applies the finite population correction.</small></div>
        <div class="field"><label class="label" for="ss-nr">Expected non-response (0–1)</label><input type="number" step="any" id="ss-nr" value="${esc(st.adj.nonResponse)}" data-live="sample" data-path="m.sample.adj.nonResponse" data-num="1"></div></div>
    </div>
    <div class="stack" id="live-sample">${sampleOut()}</div>
  </div>`;
}
function sampleVals(st, c) { const o = {}; c.fields.forEach(([k, , def]) => { const v = getPath(st.v, st.id + "." + k); o[k] = (v === undefined || v === "" || v === null) ? def : v; }); return o; }
function sampleOut() {
  const st = S.m.sample, c = Method.SAMPLE.find(x => x.id === st.id), vals = sampleVals(st, c);
  const num = {}; for (const k in vals) num[k] = Number(vals[k]);
  const bad = Object.entries(num).find(([k, v]) => !isFinite(v) || ((["p", "d", "conf", "alpha", "power", "p1", "p2", "p0"].includes(k)) && (v <= 0 || v >= 1)) || v <= 0);
  if (bad) return `<div class="notice bad">Check "${esc(c.fields.find(f => f[0] === bad[0])[1])}": it must be a positive number${["p", "d", "conf", "alpha", "power", "p1", "p2", "p0"].includes(bad[0]) ? " between 0 and 1" : ""}.</div>`;
  if (c.id === "prop2" && num.p1 === num.p2) return `<div class="notice bad">The two proportions must differ.</div>`;
  if (c.id === "cc" && num.or === 1) return `<div class="notice bad">An odds ratio of 1 means no effect; choose the smallest OR worth detecting.</div>`;
  let deff = Number(st.adj.deff) || 1;
  const m = Number(st.adj.m), icc = Number(st.adj.icc);
  if (m > 1 && icc > 0) deff = 1 + (m - 1) * icc;
  const r = Method.sampleSize(c.id, num, { deff, N: Number(st.adj.N) || 0, nonResponse: Math.min(.9, Math.max(0, Number(st.adj.nonResponse) || 0)) });
  const g = c.groupNames || ["group 1", "group 2"];
  return `<div class="card stack">
    <span class="sect-t">Required sample</span>
    <div class="row" style="align-items:baseline;gap:.6rem"><span class="big num">${r.total.toLocaleString()}</span><span class="sub">${c.perGroup ? `total: ${r.final1.toLocaleString()} ${g[0]} + ${r.final2.toLocaleString()} ${g[1]}` : "participants"}</span></div>
    <dl class="kv num"><dt>Formula</dt><dd>${esc(c.formula)}</dd>${r.parts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}
      <dt>Before adjustment</dt><dd>${c.perGroup ? `${r.base1} + ${r.base2}` : r.base}</dd>${r.steps.map(s => `<dt>${esc(s.label)}</dt><dd>${esc(s.value)}</dd>`).join("")}</dl>
    ${deff !== (Number(st.adj.deff) || 1) ? `<div class="notice info">Design effect computed from m and ICC: ${F.f(deff, 3)}.</div>` : ""}
    <p class="sub muted">Rounded up at the end. Each group is rounded separately.</p></div>
  <div class="card stack"><div class="row" style="justify-content:space-between;align-items:center"><h3>Methods paragraph</h3><button class="btn quiet sm" data-act="copy" data-text="${esc(r.methodsText)}">Copy</button></div><p class="apa">${esc(r.methodsText)}</p></div>
  <div class="card stack"><h3>Check it in your software</h3>${codeBlock(r.code)}</div>`;
}

/* ---------- 04 choose a test ---------- */
function pageTest() {
  const a = S.m.test.a, qs = Method.TEST_QUESTIONS.filter(q => !q.show || q.show(a));
  const complete = qs.every(q => a[q.id]);
  const res = complete ? Method.selectTest(a) : null;
  return pageHead("Methodology · 04", "Choose a statistical test", "Answer by the type of your variables. You get the test, its assumptions, the alternative if assumptions fail, and code for Stata, Python, R and SPSS. To run tests on real data, use Data analysis.") + `
  <div class="card stack">${qs.map(q => optQuestion(q, a, "test-opt")).join('<hr class="sep">')}<div class="row"><button class="btn quiet sm" data-act="test-reset">Start again</button></div></div>
  ${res && res.id ? testResult(res) : res && res.why.length ? `<div class="notice info" style="margin-top:1rem">${esc(res.why.join(" "))}</div>` : ""}
  <div class="card stack" style="margin-top:1rem"><details class="fold"><summary>All tests in the catalogue (${Object.keys(Method.TESTS).length})</summary>
    <div class="tw" style="margin-top:.6rem"><table class="t"><thead><tr><th class="l">Test</th><th class="l">Use it to</th><th class="l">Effect size</th></tr></thead><tbody>${Object.values(Method.TESTS).map(t => `<tr><td class="l"><b>${esc(t.name)}</b></td><td class="l">${esc(t.when)}</td><td class="l">${esc(t.effect)}</td></tr>`).join("")}</tbody></table></div></details></div>`;
}
function testCard(t, title, note) {
  return `<div class="stack" style="gap:.6rem"><div class="res-head"><div class="stack" style="gap:.2rem"><span class="sect-t">${esc(title)}</span><h2>${esc(t.name)}</h2></div></div>
    ${note ? `<p class="sub">${esc(note)}</p>` : ""}<p>${esc(t.when)}</p>
    <div class="grid2"><div><div class="sect-t">Check before trusting it</div><ul class="clean">${t.assumptions.map(s => `<li>${esc(s)}</li>`).join("")}</ul></div><div><div class="sect-t">Report this effect size</div><p>${esc(t.effect)}</p></div></div>
    ${codeBlock(t.code, { maxh: 220 })}</div>`;
}
function testResult(res) {
  const runnable = ["student", "welch", "mwu", "paired_t", "wilcoxon", "anova", "welch_anova", "kruskal", "pearson", "spearman", "chi2", "fisher", "mcnemar", "linear", "logistic", "modpoisson", "poisson", "ordinal", "multinomial", "km", "cox"].includes(res.id);
  return `<article class="card stack" style="margin-top:1rem">${res.why.map(w => `<div class="notice info">${esc(w)}</div>`).join("")}
    ${testCard(res.test, "Recommended test")}
    ${res.variant ? `<hr class="sep">${testCard(res.variant.test, "If the variance or dispersion check fails", "Use this version when the equal-variance (or dispersion) assumption is not met.")}` : ""}
    ${res.alternative ? `<hr class="sep">${testCard(res.alternative.test, "If the assumptions fail", "Non-parametric or exact alternative.")}` : ""}
    <p class="sub">Code uses placeholder names (<code class="n">y</code>, <code class="n">group</code>, <code class="n">x</code>). ${runnable ? `This test runs automatically, with every check, in <a href="#" data-act="goto-data">Data analysis</a>.` : "Data analysis in this app does not run this test; use the code above."}</p></article>`;
}

/* ---------- 05 checklists ---------- */
function pageChecklist() {
  const st = S.m.checklist, ck = Method.CHECKLISTS[st.id];
  const items = ck.sections.flatMap(([s, its]) => its.map((t, i) => s + "|" + i));
  const done = st.done[st.id] || {}, nDone = items.filter(k => done[k]).length;
  const txt = ck.sections.map(([s, its]) => `${s}\n` + its.map((t, i) => `[${done[s + "|" + i] ? "x" : " "}] ${t}`).join("\n")).join("\n\n");
  return pageHead("Methodology · 05", "Reporting checklist", "Tick items as your manuscript or thesis covers them. Ticks stay in this browser. Items are paraphrased; always cite and check the official statement.") + `
  <div class="card stack"><div class="row" style="justify-content:space-between"><div class="field" style="max-width:360px"><label class="label" for="ck">Guideline</label><select id="ck" data-act="set" data-path="m.checklist.id">${Object.entries(Method.CHECKLISTS).map(([k, c]) => `<option value="${k}"${k === st.id ? " selected" : ""}>${esc(c.name)}</option>`).join("")}</select></div>
    <div class="row"><button class="btn quiet sm" data-act="copy" data-text="${esc(ck.name + "\n\n" + txt)}">Copy as text</button><a class="btn ghost sm" href="${esc(ck.url)}" target="_blank" rel="noopener">Official statement</a></div></div>
    <div><div class="row" style="justify-content:space-between"><span class="sub num">${nDone} of ${items.length} covered</span></div><div class="prog"><i style="width:${nDone / items.length * 100}%"></i></div></div>
    ${ck.sections.map(([s, its]) => `<section><h3 style="margin:.6rem 0 .2rem">${esc(s)}</h3>${its.map((t, i) => { const k = s + "|" + i; return `<label class="check-item ${done[k] ? "done" : ""}"><input type="checkbox" data-act="check" data-k="${esc(k)}" ${done[k] ? "checked" : ""}><span>${esc(t)}</span></label>`; }).join("")}</section>`).join("")}
  </div>`;
}


/* =====================================================================
   Data analysis section pages, rendering, events, start-up
   ===================================================================== */
const DATA_TOOLS = [
  { id: "data", k: "01", label: "Data & variables", sub: "Upload, types, coding" },
  { id: "describe", k: "02", label: "Describe (Table 1)", sub: "Summary table by group" },
  { id: "compare", k: "03", label: "Compare & relate", sub: "Automatic test selection" },
  { id: "regression", k: "04", label: "Regression", sub: "Linear, logistic, Poisson, ordinal, mixed" },
  { id: "survival", k: "05", label: "Survival", sub: "Kaplan–Meier, log-rank, Cox" },
  { id: "export", k: "05", label: "Log & export", sub: "Scripts, data, report" },
  { id: "rules", k: "§", label: "Analysis rules", sub: "Every rule the app applies" },
];

/* ---------- dataset loading ---------- */
function loadDataset(headers, rows, source, isExample, notes) {
  S.raw = { headers, rows }; S.readNotes = notes || []; S.dsSource = source; S.isExample = !!isExample; S.overrides = isExample ? exampleOverrides() : {};
  S.log = []; S.logSeq = 0; S.cur = { describe: null, compare: null, regression: null, survival: null }; S.errors = {};
  S.form = { describe: { vars: [], group: "" }, compare: { outcome: "", exposure: "", paired: false, force: "" }, regression: { outcome: "", preds: [], model: "auto", cluster: "" }, survival: { time: "", event: "", group: "", covs: [] } };
  rebuild();
}
function exampleOverrides() { return { education_level: { type: "ordinal", order: ["None", "Primary", "Secondary", "Tertiary"] }, salt_intake: { type: "ordinal", order: ["Low", "Moderate", "High"] }, self_rated_health: { type: "ordinal", order: ["Poor", "Fair", "Good", "Very good"] }, usual_source_of_care: { ref: "Hospital" } }; }
function rebuild() {
  const codes = S.missingCodes.split(",").map(s => s.trim()).filter(Boolean);
  const varMissing = {}; Object.entries(S.overrides || {}).forEach(([k, o]) => { if (o.missing && o.missing.length) varMissing[k] = o.missing; });
  S.ds = Data.build(S.raw.headers, S.raw.rows, { missingCodes: codes, varMissing });
  S.ds.vars.forEach(applyOverride);
  S.quality = Data.quality(S.ds, { notes: S.readNotes || [] });
}
/** Apply one fix offered by the data check. */
function applyQualityFix(fix) {
  if (!fix) return "";
  if (fix.kind === "missing") { const o = S.overrides[fix.var] = S.overrides[fix.var] || {}; o.missing = [...new Set([...(o.missing || []), ...fix.codes])]; delete o.order; rebuild(); rerunLog(); return `${fix.codes.join(", ")} now count${fix.codes.length > 1 ? "" : "s"} as missing in ${fix.var}.`; }
  if (fix.kind === "dedupe") { const seen = new Set(), before = S.raw.rows.length; S.raw.rows = S.raw.rows.filter(r => { const k = r.join("\u0001"); if (seen.has(k)) return false; seen.add(k); return true; }); rebuild(); rerunLog(); return `Removed ${before - S.raw.rows.length} duplicate row${before - S.raw.rows.length > 1 ? "s" : ""}.`; }
  return "";
}
function applyAllFixes() { const done = []; let guard = 0; while (guard++ < 30) { const q = (S.quality || []).find(x => x.fix); if (!q) break; done.push(applyQualityFix(q.fix)); } return done; }
function qualityHTML(compact) {
  const q = S.quality || [], warn = q.filter(x => x.sev === "warn"), info = q.filter(x => x.sev !== "warn");
  if (!q.length) return `<div class="notice ok">No problems found: no duplicates, missing-value codes, impossible values or layout issues.</div>`;
  const item = (x, i) => `<li class="dq-${x.sev}"><span class="dq-dot" aria-hidden="true"></span><span>${esc(x.msg)}${x.fix ? ` <button class="btn quiet sm" data-act="dq-fix" data-i="${i}">${esc(x.fixLabel || "Fix")}</button>` : ""}</span></li>`;
  const list = arr => arr.map(x => item(x, q.indexOf(x))).join("");
  return `<ul class="dq">${list(warn)}${compact && info.length > 4 ? list(info.slice(0, 4)) + `<li class="dq-info"><span class="dq-dot"></span><span>${info.length - 4} more note${info.length - 4 > 1 ? "s" : ""} in Data & variables.</span></li>` : list(info)}</ul>${q.filter(x => x.fix).length > 1 ? `<div class="row"><button class="btn sm" data-act="dq-fix-all">Apply all ${q.filter(x => x.fix).length} fixes</button></div>` : ""}`;
}
function applyOverride(v) {
  const o = S.overrides[v.name]; if (!o) return;
  if (o.type) Data.setType(v, o.type);
  if (v.levels && o.order && o.order.length === v.allLevels.length && o.order.every(l => v.allLevels.includes(l))) v.levels = [...o.order];
  else if (v.levels && o.ref) Data.setReference(v, o.ref);
}
function rerunLog() {
  const kept = [], dropped = [];
  S.log.forEach(e => { try { e.result = runAnalysis(e.kind, e.params); kept.push(e); } catch (err) { dropped.push(e); } });
  S.log = kept;
  ["describe", "compare", "regression", "survival"].forEach(k => { if (S.cur[k] && !kept.includes(S.cur[k])) S.cur[k] = null; });
  if (kept.length) toast(`Re-ran ${kept.length} logged analys${kept.length === 1 ? "is" : "es"} with the new settings` + (dropped.length ? `; removed ${dropped.length} that no longer fit` : ""));
}
async function readFile(file) {
  const name = file.name, ext = name.split(".").pop().toLowerCase();
  try {
    if (["xlsx", "xls", "xlsm"].includes(ext)) {
      const XLSX = await loadSheetJs();
      const wb = XLSX.read(await file.arrayBuffer(), { type: "array" });
      // use the sheet holding the most data (the first sheet is often a cover page or notes)
      const grids = wb.SheetNames.map(sn => ({ sn, aoa: XLSX.utils.sheet_to_json(wb.Sheets[sn], { header: 1, raw: false, defval: "", blankrows: false }) }));
      grids.forEach(g => { g.cells = g.aoa.reduce((a, r) => a + r.filter(c => String(c).trim() !== "").length, 0); });
      const best = grids.sort((a, b) => b.cells - a.cells)[0];
      if (!best || !best.cells) throw new Error("The workbook is empty.");
      const t = Data.tidyTable(best.aoa);
      if (!t.headers.length || !t.rows.length) throw new Error("No table found in the workbook.");
      if (wb.SheetNames.length > 1) t.notes.unshift({ kind: "sheet", msg: `Used sheet "${best.sn}", the one with the most data (the workbook has ${wb.SheetNames.length} sheets).` });
      loadDataset(t.headers, t.rows, name + (wb.SheetNames.length > 1 ? ` (sheet "${best.sn}")` : ""), false, t.notes);
    } else {
      const p = Data.parseDelimited(await file.text());
      if (!p.headers.length || !p.rows.length) throw new Error("No rows found. The first row must hold variable names.");
      loadDataset(p.headers, p.rows, name, false, p.notes);
    }
    S.errors.data = null; toast(`Loaded ${S.ds.nRows.toLocaleString()} rows × ${S.ds.vars.length} variables`);
  } catch (e) { S.errors.data = `Couldn't read "${name}": ${e.message || e}. Use a .csv or .xlsx file with variable names in the first row.`; }
  render();
}
let sheetJsP = null;
function loadSheetJs() {
  if (window.XLSX) return Promise.resolve(window.XLSX);
  if (!sheetJsP) sheetJsP = new Promise((res, rej) => { const s = document.createElement("script"); s.src = "assets/vendor/xlsx.full.min.js"; s.onload = () => res(window.XLSX); s.onerror = () => { sheetJsP = null; rej(new Error("the Excel reader could not load; save the sheet as CSV instead")); }; document.head.appendChild(s); });
  return sheetJsP;
}

/* ---------- running analyses ---------- */
function runAnalysis(kind, p) {
  const opts = { alpha: S.alpha, force: p.force || undefined };
  if (kind === "describe") return Analysis.table1(S.ds, p.vars.map(V).filter(Boolean), p.group ? V(p.group) : null, opts);
  if (kind === "compare") { if (!V(p.outcome) || !V(p.exposure)) throw Object.assign(new Error("Choose both variables."), { user: true }); return Analysis.auto(S.ds, V(p.outcome), V(p.exposure), Object.assign(opts, { paired: p.paired })); }
  if (kind === "regression") {
    const y = V(p.outcome); if (!y) throw Object.assign(new Error("Choose an outcome."), { user: true });
    let model = p.model || "auto";
    if (model === "auto") model = y.type === "ordinal" ? "ordinal" : y.type === "categorical" ? "multinomial" : (p.cluster && ["continuous", "count"].includes(y.type) && y.type === "continuous") ? "mixed" : "auto";
    if (["ordinal", "multinomial", "mixed"].includes(model)) { const r = Analysis.regressionExt(S.ds, y, p.preds.map(V).filter(Boolean), Object.assign(opts, { model, cluster: p.cluster || null })); if (p.model === "auto") r.decision.unshift({ rule: "Model from outcome type", detail: `"${vlabel(y)}" is ${y.type}${model === "mixed" ? " with a cluster variable" : ""} → ${r.method.toLowerCase()}.` }); return r; }
    return Analysis.regression(S.ds, y, p.preds.map(V).filter(Boolean), Object.assign(opts, { model }));
  }
  if (kind === "survival") {
    if (!V(p.time) || !V(p.event)) throw Object.assign(new Error("Choose the follow-up time and the event variable."), { user: true });
    return Analysis.survival(S.ds, V(p.time), V(p.event), p.group ? V(p.group) : null, (p.covs || []).map(V).filter(Boolean), opts);
  }
}
function doRun(kind) {
  const params = JSON.parse(JSON.stringify(S.form[kind]));
  if (kind === "describe" && !params.vars.length) { S.errors[kind] = "Choose at least one variable to describe."; render(); return; }
  try {
    const result = runAnalysis(kind, params);
    const entry = { id: ++S.logSeq, k: S.log.length + 1, kind, params, result };
    S.log.push(entry); S.cur[kind] = entry; S.errors[kind] = null;
    renumber(); render();
    requestAnimationFrame(() => { const el = document.getElementById("res-" + entry.id); if (el) el.scrollIntoView({ behavior: "smooth", block: "start" }); });
  } catch (e) { S.errors[kind] = e.user ? e.message : "The calculation failed: " + (e.message || e); console.error(e); render(); }
}
const renumber = () => S.log.forEach((e, i) => e.k = i + 1);

/* ---------- pages ---------- */
function exampleBanner() {
  return S.isExample ? `<div class="notice info" style="margin-bottom:1rem"><b>Example data.</b> ${/malaria/i.test(S.dsSource) ? "Fictional monthly malaria cases (48 months), loaded to show forecasting." : "A fictional survey of 420 adults in five Ghanaian districts, loaded so you can try every tool."} Upload your own file in <a href="#" data-act="tool" data-v="data">Data &amp; variables</a>.</div>` : "";
}
function pageData() {
  const ds = S.ds;
  return pageHead("Data analysis", "Data & variables", "Upload a CSV or Excel file with one row per person or observation and variable names at the top. QuantAI detects each variable's type, skips title lines, and checks the file for common problems. Confirm the types below: the type decides which tests are allowed. Your data stay in this browser tab and are never uploaded.") + exampleBanner() + `
  <div class="grid2">
    <div class="card stack"><label class="drop" id="drop" tabindex="0"><input type="file" id="file" accept=".csv,.txt,.tsv,.xlsx,.xls" hidden><b>Choose a file</b> or drop it here<br><span class="sub">CSV, TSV or Excel (.xlsx) · one row per observation</span></label>
      ${S.errors.data ? `<div class="notice bad">${esc(S.errors.data)}</div>` : ""}
      <div class="row"><button class="btn quiet sm" data-act="load-example">Load the example data</button></div></div>
    <div class="card stack"><dl class="kv num"><dt>Dataset</dt><dd>${esc(S.dsSource)}</dd><dt>Rows</dt><dd>${ds.nRows.toLocaleString()}</dd><dt>Variables</dt><dd>${ds.vars.length} (${ds.vars.filter(v => v.type === "id").length} ignored)</dd><dt>Complete rows</dt><dd>${Data.completeRows(ds, ds.vars.filter(v => v.type !== "id")).length.toLocaleString()}</dd></dl>
      <div class="row"><div class="field"><label class="label" for="miss">Extra missing-value codes</label><input type="text" id="miss" placeholder="e.g. 99, -9, 999" value="${esc(S.missingCodes)}" data-act="missing"><small>Empty cells, NA, N/A, NaN, null and "." are always missing.</small></div>
        <div class="field" style="flex:0 1 150px"><label class="label" for="alpha">Significance level α</label><select id="alpha" data-act="alpha">${[0.05, 0.01, 0.1].map(a => `<option value="${a}"${a === S.alpha ? " selected" : ""}>${a}</option>`).join("")}</select></div></div></div>
  </div>
  <div class="card stack" style="margin-top:1rem"><div class="row" style="justify-content:space-between;align-items:center"><h2>Data check</h2><span class="sub">${(S.quality || []).filter(x => x.sev === "warn").length} to review · ${(S.quality || []).filter(x => x.sev !== "warn").length} notes</span></div>${qualityHTML(false)}</div>
  <div class="card stack" style="margin-top:1rem"><div class="row" style="justify-content:space-between;align-items:center"><h2>Variables</h2><span class="sub">Name in code = the name every exported script uses.</span></div>
    <div class="tw"><table class="t vt"><thead><tr><th class="l">Variable</th><th class="l">Name in code</th><th class="l">Type</th><th class="l">Coding (first = reference / 0)</th><th>Missing</th><th class="l">Summary</th></tr></thead><tbody>
    ${ds.vars.map(v => `<tr><td class="l"><b>${esc(v.label)}</b></td><td class="l"><code>${esc(v.name)}</code></td>
      <td class="l"><select aria-label="Type of ${esc(v.label)}" data-act="vtype" data-var="${esc(v.name)}">${Object.entries(TYPE_LABEL).filter(([t]) => typeAllowed(v, t)).map(([t, l]) => `<option value="${t}"${t === v.type ? " selected" : ""}>${l}</option>`).join("")}</select>${v.type !== v.inferredType ? ` <span class="pill warn" title="Detected as ${v.inferredType}">changed</span>` : ""}${v.why && v.type === v.inferredType ? `<br><small class="sub">${esc(v.why)}</small>` : ""}</td>
      <td class="l">${codingCell(v)}</td>
      <td class="num">${v.nMissing ? `<span title="${(100 * v.nMissing / ds.nRows).toFixed(1)}%">${v.nMissing}</span>` : '<span class="muted">0</span>'}</td>
      <td class="l sub">${esc(summaryOf(v))}</td></tr>`).join("")}</tbody></table></div>
    <p class="sub">Types: <b>continuous</b> measurements; <b>count</b> whole numbers from 0 (visits); <b>binary</b> two categories; <b>categorical</b> unordered groups; <b>ordinal</b> ordered categories (Likert, education); <b>ID / ignore</b> excluded.</p>
  </div>
  <div class="card stack" style="margin-top:1rem"><h2>First rows</h2><div class="tw"><table class="t"><thead><tr>${ds.vars.map(v => `<th class="l">${esc(v.name)}</th>`).join("")}</tr></thead><tbody>${Array.from({ length: Math.min(6, ds.nRows) }, (_, i) => `<tr>${ds.vars.map(v => `<td class="l">${v.values[i] === null ? '<span class="muted">·</span>' : esc(v.values[i])}</td>`).join("")}</tr>`).join("")}</tbody></table></div></div>`;
}
function typeAllowed(v, t) {
  if (t === "id") return true;
  if (t === "continuous" || t === "count") return v.numeric;
  if (t === "binary") return v.nUnique === 2;
  if (t === "categorical" || t === "ordinal") return v.nUnique >= 2 && v.nUnique <= 50;
  return true;
}
function codingCell(v) {
  if (!v.levels) return '<span class="muted">—</span>';
  if (v.type === "ordinal") return `<input type="text" aria-label="Order of ${esc(v.label)}" value="${esc(v.levels.join(", "))}" data-act="vorder" data-var="${esc(v.name)}" title="Lowest to highest, comma-separated">`;
  if (v.type === "binary") return `<select aria-label="Reference of ${esc(v.label)}" data-act="vref" data-var="${esc(v.name)}">${v.allLevels.map(l => `<option value="${esc(l)}"${l === v.levels[0] ? " selected" : ""}>0 = ${esc(l)}, 1 = ${esc(v.allLevels.find(x => x !== l))}</option>`).join("")}</select>`;
  return `<select aria-label="Reference of ${esc(v.label)}" data-act="vref" data-var="${esc(v.name)}">${v.allLevels.map(l => `<option value="${esc(l)}"${l === v.levels[0] ? " selected" : ""}>ref: ${esc(l)}</option>`).join("")}</select> <span class="sub">${v.levels.length} levels</span>`;
}
function summaryOf(v) {
  const vals = v.values.filter(x => x !== null);
  if (!vals.length) return "no values";
  if (v.type === "continuous" || v.type === "count") { const n = vals.map(Number).filter(isFinite), d = Stats.describe(n); return `mean ${F.f(d.mean)}, SD ${F.f(d.sd)}, median ${F.f(d.median)}, range ${F.f(d.min, 1)}–${F.f(d.max, 1)}`; }
  if (v.levels) { const cnt = v.levels.map(l => [l, vals.filter(x => x === l).length]); const small = cnt.filter(c => c[1] < 5); return cnt.slice(0, 5).map(([l, c]) => `${l} ${c}`).join(" · ") + (cnt.length > 5 ? " …" : "") + (small.length ? ` — sparse: ${small.map(s => s[0]).join(", ")}` : ""); }
  return `${v.nUnique} distinct values`;
}

function pageDescribe() {
  const f = S.form.describe, vs = varsUsable();
  return pageHead("Data analysis", "Describe the sample (Table 1)", "Pick the variables for your descriptive table, and optionally a grouping variable. Each continuous variable is summarised as mean (SD) or median (IQR) depending on its distribution, with the matching test.") + exampleBanner() + `
  <div class="card stack">
    <div><span class="label">Variables to describe</span><div class="chips">${vs.map(v => `<button class="chip" aria-pressed="${f.vars.includes(v.name)}" data-act="desc-var" data-v="${esc(v.name)}">${esc(vlabel(v))}<span class="t">${TYPE_SHORT[v.type]}</span></button>`).join("")}</div>
      <div class="row" style="margin-top:.4rem"><button class="btn quiet sm" data-act="desc-all">Select all</button><button class="btn quiet sm" data-act="desc-none">Clear</button></div></div>
    <div class="row"><div class="field" style="max-width:420px"><label class="label" for="d-group">Compare by (optional)</label><select id="d-group" data-act="form" data-path="describe.group">${varOptions(f.group, v => v.type === "binary" || v.type === "categorical", "No grouping: overall only")}</select></div>
      <button class="btn" data-act="run" data-v="describe">Build Table 1</button></div>
    ${S.errors.describe ? `<div class="notice bad">${esc(S.errors.describe)}</div>` : ""}
  </div>
  <div style="margin-top:1rem">${S.cur.describe ? resultCard(S.cur.describe) : ""}</div>`;
}

function comparePlan() {
  const f = S.form.compare, o = V(f.outcome), e = V(f.exposure);
  if (!o || !e) return "Choose two variables to see which family of tests the rules will consider.";
  if (o.name === e.name) return "Pick two different variables.";
  if (f.paired) return (o.type === "binary" && e.type === "binary") ? "Paired binary → McNemar's test (exact version if fewer than 25 discordant pairs)." : "Paired numeric → paired t-test if the differences are normal, otherwise Wilcoxon signed-rank.";
  const num = t => ["continuous", "count", "ordinal"].includes(t), cat = t => ["binary", "categorical"].includes(t);
  if (["continuous", "count"].includes(o.type) && e.type === "ordinal") return `Numeric outcome × ordinal exposure with ${e.levels.length} categories → compared as groups (ANOVA, Welch's ANOVA or Kruskal–Wallis by the checks${e.levels.length === 2 ? "; with two categories: t-test family" : ""}).`;
  if (num(o.type) && cat(e.type)) { const k = e.levels.length; return k === 2 ? `Numeric outcome × 2 groups → Student t-test, Welch t-test or Mann–Whitney U, chosen by the normality and equal-variance checks${o.type === "ordinal" ? " (ordinal → Mann–Whitney)" : ""}.` : `Numeric outcome × ${k} groups → one-way ANOVA, Welch's ANOVA or Kruskal–Wallis, with Bonferroni post-hoc tests if significant.`; }
  if (cat(o.type) && num(e.type) && e.type !== "ordinal") return `Categorical outcome × numeric exposure → the exposure is compared across the outcome groups (t-test/ANOVA family). For adjusted odds ratios use Regression.`;
  if (num(o.type) && num(e.type)) return "Two numeric variables → Pearson's r if both are normal, otherwise Spearman's rho.";
  return "Two categorical variables → chi-square test, or Fisher's exact test if expected counts are small. 2×2 tables also give the odds ratio and risk ratio.";
}
function pageCompare() {
  const f = S.form.compare;
  return pageHead("Data analysis", "Compare & relate", "Choose an outcome and an exposure (or a second variable). The app checks the assumptions, picks the correct test, and records every rule it applied.") + exampleBanner() + `
  <div class="card stack">
    <div class="row">
      <div class="field"><label class="label" for="c-out">${f.paired ? "First measurement" : "Outcome (dependent variable)"}</label><select id="c-out" data-act="form" data-path="compare.outcome">${varOptions(f.outcome)}</select></div>
      <div class="field"><label class="label" for="c-exp">${f.paired ? "Second measurement" : "Exposure / grouping variable"}</label><select id="c-exp" data-act="form" data-path="compare.exposure">${varOptions(f.exposure)}</select></div>
    </div>
    <div class="row" style="align-items:center">
      <label class="row" style="gap:.4rem;align-items:center;cursor:pointer;flex-wrap:nowrap"><input type="checkbox" data-act="form-check" data-path="compare.paired" ${f.paired ? "checked" : ""}><span>Paired data (same people measured twice)</span></label>
      <div class="field" style="max-width:300px"><label class="label" for="c-force">Test choice</label><select id="c-force" data-act="form" data-path="compare.force">
        ${[["", "Automatic (follow the rules)"], ["parametric", "Force parametric"], ["nonparametric", "Force non-parametric"], ["chi2", "Force chi-square"], ["fisher", "Force Fisher's exact"]].map(([v, l]) => `<option value="${v}"${v === f.force ? " selected" : ""}>${l}</option>`).join("")}</select></div>
      <button class="btn" data-act="run" data-v="compare">Run the analysis</button>
    </div>
    <div class="notice info"><b>Plan.</b> ${esc(comparePlan())}</div>
    ${S.errors.compare ? `<div class="notice bad">${esc(S.errors.compare)}</div>` : ""}
  </div>
  <div style="margin-top:1rem">${S.cur.compare ? resultCard(S.cur.compare) : ""}</div>`;
}

function pageRegression() {
  const f = S.form.regression, vs = varsUsable().filter(v => v.name !== f.outcome);
  const o = V(f.outcome);
  const models = [["auto", "Automatic (from the outcome type)"], ["linear", "Linear (continuous outcome)"], ["logistic", "Logistic (binary outcome → odds ratios)"], ["modpoisson", "Modified Poisson (binary outcome → prevalence/risk ratios)"], ["poisson", "Poisson (count outcome → rate ratios)"], ["ordinal", "Ordinal logistic (ordered categories)"], ["multinomial", "Multinomial logistic (3+ unordered categories)"], ["mixed", "Linear mixed model (clustered or repeated data)"]];
  return pageHead("Data analysis", "Regression", "Model an outcome with several predictors. You get crude and adjusted estimates side by side, and the model's diagnostics decide whether robust standard errors are needed.") + exampleBanner() + `
  <div class="card stack">
    <div class="row"><div class="field"><label class="label" for="r-out">Outcome</label><select id="r-out" data-act="form" data-path="regression.outcome">${varOptions(f.outcome, v => ["continuous", "count", "binary", "ordinal", "categorical"].includes(v.type))}</select></div>
      <div class="field"><label class="label" for="r-model">Model</label><select id="r-model" data-act="form" data-path="regression.model">${models.map(([v, l]) => `<option value="${v}"${v === f.model ? " selected" : ""}>${l}</option>`).join("")}</select></div></div>
    ${f.model === "mixed" ? `<div class="row"><div class="field" style="max-width:420px"><label class="label" for="r-cluster">Cluster variable (random intercept)</label><select id="r-cluster" data-act="form" data-path="regression.cluster">${varOptions(f.cluster, v => v.name !== f.outcome && !f.preds.includes(v.name), "Choose the cluster (clinic, school, village, person ID)")}</select><small>Observations in the same cluster are allowed to be correlated.</small></div></div>` : ""}
    <div><span class="label">Predictors (exposure first, then confounders)</span><div class="chips">${vs.map(v => `<button class="chip" aria-pressed="${f.preds.includes(v.name)}" data-act="reg-pred" data-v="${esc(v.name)}">${esc(vlabel(v))}<span class="t">${TYPE_SHORT[v.type]}</span></button>`).join("")}</div></div>
    ${o && o.type === "binary" && f.model === "auto" ? (() => { const vals = o.values.filter(x => x !== null), prev = vals.filter(x => x === o.levels[1]).length / vals.length; return prev > .1 ? `<div class="notice warn"><b>Common outcome (${F.f(100 * prev, 1)}%).</b> Odds ratios will overstate prevalence or risk ratios. For cross-sectional or cohort data, choose Modified Poisson.</div>` : ""; })() : ""}
    <div class="row"><button class="btn" data-act="run" data-v="regression">Fit the model</button><span class="sub">${f.preds.length} predictor${f.preds.length === 1 ? "" : "s"} selected</span></div>
    ${S.errors.regression ? `<div class="notice bad">${esc(S.errors.regression)}</div>` : ""}
  </div>
  <div style="margin-top:1rem">${S.cur.regression ? resultCard(S.cur.regression) : ""}</div>`;
}

function pageSurvival() {
  const f = S.form.survival;
  return pageHead("Data analysis", "Survival analysis", "Analyse time until an event (death, default from care, recovery, relapse). Kaplan–Meier curves and median survival for each group, the log-rank test, and Cox regression for adjusted hazard ratios.") + exampleBanner() + `
  <div class="card stack">
    <div class="row">
      <div class="field"><label class="label" for="s-time">Follow-up time</label><select id="s-time" data-act="form" data-path="survival.time">${varOptions(f.time, v => ["continuous", "count"].includes(v.type), "Time from start to event or censoring")}</select></div>
      <div class="field"><label class="label" for="s-event">Event</label><select id="s-event" data-act="form" data-path="survival.event">${varOptions(f.event, v => v.type === "binary", "Binary: event vs censored")}</select><small>${V(f.event) ? `Event = ${esc(V(f.event).levels[1])}; ${esc(V(f.event).levels[0])} = censored. Change the coding in Data & variables.` : "The second category of a binary variable is the event."}</small></div>
      <div class="field"><label class="label" for="s-group">Compare groups (optional)</label><select id="s-group" data-act="form" data-path="survival.group">${varOptions(f.group, v => v.type === "binary" || v.type === "categorical", "No grouping")}</select></div>
    </div>
    <div><span class="label">Adjust for (optional, Cox regression)</span><div class="chips">${varsUsable().filter(v => ![f.time, f.event, f.group].includes(v.name)).map(v => `<button class="chip" aria-pressed="${f.covs.includes(v.name)}" data-act="surv-cov" data-v="${esc(v.name)}">${esc(vlabel(v))}<span class="t">${TYPE_SHORT[v.type]}</span></button>`).join("")}</div></div>
    <div class="row"><button class="btn" data-act="run" data-v="survival">Run survival analysis</button></div>
    ${S.errors.survival ? `<div class="notice bad">${esc(S.errors.survival)}</div>` : ""}
  </div>
  <div style="margin-top:1rem">${S.cur.survival ? resultCard(S.cur.survival) : ""}</div>`;
}

function pageExport() {
  const entries = S.log.map(e => ({ spec: e.result.spec, title: e.result.title }));
  return pageHead("Data analysis", "Analysis log & export", "Every analysis you run is logged here. Export one script per language that reproduces all of them from the cleaned data file, plus a results report.") + `
  <div class="card stack">
    ${S.log.length ? `<div>${S.log.map(e => `<div class="log-item"><span class="k">${String(e.k).padStart(2, "0")}</span><div style="min-width:0"><b>${esc(e.result.title)}</b><div class="sub">${esc(e.result.method)} · n = ${e.result.n}</div></div>
      <div class="row" style="gap:.3rem"><button class="btn quiet sm" data-act="log-view" data-id="${e.id}">View</button><button class="btn quiet sm" data-act="log-remove" data-id="${e.id}" aria-label="Remove analysis ${e.k}">Remove</button></div></div>`).join("")}</div>`
      : `<p class="sub">No analyses yet. Run something in Describe, Compare &amp; relate, or Regression and it appears here.</p>`}
  </div>
  ${S.log.length ? `<div class="card stack" style="margin-top:1rem"><h2>Download</h2>
    <p class="sub">The zip holds <code class="n">analysis.do</code>, <code class="n">analysis.py</code>, <code class="n">analysis.R</code>, <code class="n">analysis.sps</code>, the cleaned <code class="n">analysis_data.csv</code> they all read, a results report, and a README with the variable-name map. Put them in one folder and run any script.</p>
    <div class="row"><button class="btn" data-act="dl-zip">Download everything (.zip)</button><button class="btn ghost" data-act="dl-report">Results report (.html)</button><button class="btn ghost" data-act="dl-csv">Cleaned data (.csv)</button></div>
    ${true ? `<div class="notice info">If downloads are blocked in your browser, copy each script below and the cleaned CSV.</div><div class="row"><button class="btn quiet sm" data-act="copy-csv">Copy cleaned CSV</button></div>` : ""}</div>
  <div class="card stack" style="margin-top:1rem"><h2>Full scripts</h2>${codeBlock(scriptsForEntries(entries), { maxh: 560 })}</div>` : ""}`;
}

function pageRules() {
  const R = Analysis.RULES;
  const rows = [
    ["Significance level", `α = ${S.alpha} (change it in Data & variables). All tests are two-sided.`, ""],
    ["Normality", R.normality, "Shapiro & Wilk (1965); Royston (1995) algorithm"],
    ["Equal variances", R.variance, "Brown & Forsythe (1974)"],
    ["Two groups", "Normal + equal variances → Student t-test; normal + unequal → Welch's t-test; not normal or ordinal → Mann–Whitney U (normal approximation with tie and continuity correction).", "Welch (1947); Mann & Whitney (1947)"],
    ["Three or more groups", "Normal + equal variances → one-way ANOVA; normal + unequal → Welch's ANOVA; otherwise Kruskal–Wallis. " + R.posthoc, ""],
    ["Two numeric variables", "Both normal → Pearson's r with Fisher-z confidence interval; otherwise (or ordinal) → Spearman's rho.", ""],
    ["Two categorical variables", R.expected + " 2×2 tables add odds ratio and risk ratio with Woolf (log) CIs; zero cells get +0.5 (Haldane).", "Cochran (1954)"],
    ["Paired data", "Numeric: paired t-test if the differences are normal, otherwise Wilcoxon signed-rank (zeros dropped). Binary: McNemar's test, exact binomial p when fewer than 25 discordant pairs.", "McNemar (1947)"],
    ["Regression sample size", R.epv, "Peduzzi et al. (1996)"],
    ["Multicollinearity", R.vif, ""],
    ["Linear regression", R.hetero + " Residual normality (Shapiro–Wilk) and Cook's distance (> 4/n) are reported.", "Koenker (1981); White (1980)"],
    ["Logistic regression", "Wald 95% CIs for odds ratios; Hosmer–Lemeshow with 10 risk groups; ROC AUC; separation is flagged when the model fails to converge or predicts 0/1. " + R.commonOutcome, "Hosmer & Lemeshow (2013)"],
    ["Prevalence ratios", "Modified Poisson: log link with robust (HC0) sandwich standard errors.", "Zou (2004)"],
    ["Ordinal outcome", "Ordinal logistic regression (proportional odds), cumulative odds ratios with Wald CIs. The proportional-odds assumption is checked in the exported code (Brant test).", "McCullagh (1980)"],
    ["Nominal outcome (3+ categories)", "Multinomial logistic regression against the first category; relative risk ratios; events per variable checked in the smallest category.", ""],
    ["Clustered or repeated data", "Linear mixed model with a random intercept (REML); fixed-effect SEs from (X′V⁻¹X)⁻¹ as Stata and R; the ICC shows how much variance lies between clusters.", "Laird & Ware (1982)"],
    ["Time to event", "Kaplan–Meier with Greenwood SEs and log(−log) CIs; log-rank test for groups; Cox regression with Efron ties. Proportional hazards are checked in the exported code (Schoenfeld residuals).", "Kaplan & Meier (1958); Cox (1972)"],
    ["Count outcomes", R.dispersion + " Excess zeros are flagged.", ""],
    ["Missing data", "Each analysis uses complete cases for the variables involved and reports how many rows were excluded. Crude regression estimates use all rows with that predictor and the outcome.", ""],
    ["Descriptive summaries", "Mean (SD) when normal by the normality rule, otherwise median (IQR, type-7 quantiles as R and Python). Categorical: n (column %).", ""],
    ["Reference categories", "The first category in Data & variables is the reference (binary: coded 0). Exported scripts recode explicitly so Stata, Python, R and SPSS use the same reference.", ""],
  ];
  return pageHead("Data analysis · rules", "Analysis rules", "Every decision the app makes follows one of these rules. They are fixed, so the same data always give the same test, and each result shows which rules fired.") + `
  <div class="card"><div class="tw"><table class="t"><thead><tr><th class="l">Situation</th><th class="l">Rule</th><th class="l">Source</th></tr></thead><tbody>${rows.map(r => `<tr><td class="l"><b>${esc(r[0])}</b></td><td class="l">${esc(r[1])}</td><td class="l sub">${esc(r[2])}</td></tr>`).join("")}</tbody></table></div>
  <p class="sub" style="margin-top:.8rem">Verification: every statistic in this app was checked against SciPy and statsmodels to at least six decimal places. You can override any automatic choice with "Test choice" in Compare &amp; relate; the override is recorded in the rules list.</p></div>`;
}

/* ---------- export builders ---------- */
function reportHTML() {
  const date = new Date().toISOString().slice(0, 10);
  const css = "body{font:15px/1.55 Georgia,serif;color:#12202E;max-width:900px;margin:2rem auto;padding:0 16px}h1{font-size:1.7rem}h2{font-size:1.2rem;margin-top:2.2rem;border-top:1px solid #D6DEE6;padding-top:1rem}table{border-collapse:collapse;width:100%;font:13px/1.4 Arial,sans-serif;margin:.6rem 0}caption{text-align:left;font-weight:bold;padding:.3rem 0}th,td{border-bottom:1px solid #D6DEE6;padding:.35rem .5rem;text-align:right}th:first-child,td:first-child{text-align:left}.apa{background:#EEF5F7;border-left:3px solid #0E5F74;padding:.6rem .9rem}ol{font:13px/1.5 Arial,sans-serif}.note{font:12px Arial,sans-serif;color:#555}";
  const body = S.log.map(e => { const r = e.result; return `<h2>${e.k}. ${esc(r.title)}</h2><p><b>Method:</b> ${esc(r.method)} · n = ${r.n}</p>${r.warnings.map(w => `<p class="note">${esc(w)}</p>`).join("")}<p><b>Rules applied</b></p><ol>${r.decision.map(d => `<li><b>${esc(d.rule)}.</b> ${esc(d.detail)}</li>`).join("")}</ol>${r.tables.map(t => `<table><caption>${esc(t.title)}</caption><tr>${t.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr>${t.rows.map(row => `<tr>${row.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</table>${t.note ? `<p class="note">${esc(t.note)}</p>` : ""}`).join("")}${r.writeup ? `<p class="apa">${esc(r.writeup)}</p>` : ""}`; }).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analysis results</title><style>${css}</style></head><body><h1>Analysis results</h1><p class="note">Dataset: ${esc(S.dsSource)} · ${S.ds.nRows} rows · α = ${S.alpha} · generated ${date} with QuantAI (modelanalysishub.com). Every number was computed by deterministic code; scripts that reproduce it are in the same folder.</p>${body}</body></html>`;
}
function readme() {
  const date = new Date().toISOString().slice(0, 10);
  return `QuantAI by Model Analysis Hub — analysis export (${date})
Dataset: ${S.dsSource} (${S.ds.nRows} rows)

FILES
  analysis_data.csv   cleaned data: short variable names, missing values left empty
  analysis.do         Stata 14+      →  do analysis.do
  analysis.py         Python 3       →  pip install pandas numpy scipy statsmodels ; python analysis.py
  analysis.R          R 4.0+, base R →  source("analysis.R")
  analysis.sps        SPSS 25+       →  File › Open › Syntax, then Run › All
  results_report.html the results as shown in the app

Keep all files in the same folder and set it as the working directory.

VARIABLE NAMES (original → name in code, type, coding)
${S.ds.vars.map(v => `  ${v.label} → ${v.name}  [${v.type}]${v.levels ? "  " + v.levels.map((l, i) => `${v.type === "binary" ? i : i + 1}=${l}`).join(", ") : ""}`).join("\n")}

NOTES
- Scripts recode categories explicitly, so the reference group is identical in every language.
- Known small differences: Stata's ranksum/signrank and SPSS's Mann–Whitney omit the continuity
  correction the app, R and Python use; Stata/SPSS percentiles use a different definition (medians agree);
  Stata's robust SEs after poisson include an n/(n-1) factor.
`;
}
async function downloadZip() {
  if (!window.JSZip) { toast("The zip library didn't load. Use the individual downloads or copy buttons."); return; }
  const entries = S.log.map(e => ({ spec: e.result.spec, title: e.result.title })), sc = scriptsForEntries(entries);
  const zip = new JSZip(), folder = zip.folder("quantai-analysis");
  folder.file("analysis_data.csv", Data.toCSV(S.ds));
  Codegen.LANGS.forEach(l => folder.file("analysis." + l.ext, sc(l.id)));
  folder.file("results_report.html", reportHTML());
  folder.file("README.txt", readme());
  const blob = await zip.generateAsync({ type: "blob" });
  await saveFile("quantai-analysis.zip", blob);
}

/* ---------- rendering ---------- */
function render() {
  codeRegistry.clear();
  document.querySelectorAll(".qa-sections button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === S.section)));
  const tools = S.section === "method" ? METHOD_TOOLS : DATA_TOOLS, cur = S.tool[S.section];
  ($("#qa-rail") || {}).innerHTML = `<div class="rail-group"><h4>${S.section === "method" ? "Methodology" : "Data analysis"}</h4><nav aria-label="Tools">${tools.map(t => `<button data-act="tool" data-v="${t.id}" ${t.id === cur ? 'aria-current="page"' : ""}><span class="k">${t.k}</span><span>${esc(t.label)}</span><small>${esc(t.sub)}</small></button>`).join("")}</nav></div>
    <p class="rail-note">${S.section === "method" ? "Rule-based: the same answers always give the same advice." : `${S.log.length} analys${S.log.length === 1 ? "is" : "es"} in the log. Data never leave this tab.`}</p>`;
  const pages = { question: pageQuestion, design: pageDesign, sample: pageSample, test: pageTest, checklist: pageChecklist, forecast: pageForecast, data: pageData, describe: pageDescribe, compare: pageCompare, regression: pageRegression, survival: pageSurvival, export: pageExport, rules: pageRules };
  $("#qa-main").innerHTML = (pages[cur] || pages.question)();
  persist();
}

/* ---------- events ---------- */
function onClick(e) {
  const el = e.target.closest("[data-act]"); if (!el) return;
  const act = el.dataset.act, v = el.dataset.v;
  if (el.tagName === "SELECT" || (el.tagName === "INPUT" && el.type !== "checkbox" && el.type !== "button")) return;
  if (el.tagName === "A" || el.tagName === "BUTTON") e.preventDefault();
  switch (act) {
    case "sec": S.section = v; render(); qaTop(); break;
    case "tool": S.tool[S.section] = v; render(); qaTop(); break;
    case "goto-data": S.section = "data"; S.tool.data = "compare"; render(); qaTop(); break;
    case "goto-sample": S.m.sample.id = v; S.tool.method = "sample"; render(); qaTop(); break;
    case "goto-check": S.m.checklist.id = v; S.tool.method = "checklist"; render(); qaTop(); break;
    case "set": setPath(S, el.dataset.path, v); render(); break;
    case "design-opt": S.m.design.a[el.dataset.q] = v; pruneAnswers(S.m.design.a, Method.DESIGN_QUESTIONS); render(); break;
    case "design-reset": S.m.design.a = {}; render(); break;
    case "test-opt": S.m.test.a[el.dataset.q] = v; pruneAnswers(S.m.test.a, Method.TEST_QUESTIONS); render(); break;
    case "test-reset": S.m.test.a = {}; render(); break;
    case "finer": break;
    case "check": break;
    case "lang": S.codeLang = v; refreshCode(); persist(); break;
    case "copy-code": copyText(codeText(el.dataset.id)); break;
    case "copy": copyText(el.dataset.text); break;
    case "copy-apa": { const en = S.log.find(x => x.id === +el.dataset.id); if (en) copyText(en.result.writeup); break; }
    case "load-example": loadExample(); render(); toast("Example data loaded"); break;
    case "dq-fix": { const q = (S.quality || [])[+el.dataset.i]; if (q && q.fix) { const m = applyQualityFix(q.fix); render(); toast(m); } break; }
    case "dq-fix-all": { const d = applyAllFixes(); render(); toast(d.length + " fix" + (d.length === 1 ? "" : "es") + " applied"); break; }
    case "desc-var": toggle(S.form.describe.vars, v); render(); break;
    case "desc-all": S.form.describe.vars = varsUsable().map(x => x.name).filter(n => n !== S.form.describe.group); render(); break;
    case "desc-none": S.form.describe.vars = []; render(); break;
    case "reg-pred": toggle(S.form.regression.preds, v); render(); break;
    case "surv-cov": toggle(S.form.survival.covs, v); render(); break;
    case "run": doRun(v); break;
    case "log-remove": S.log = S.log.filter(x => x.id !== +el.dataset.id); ["describe", "compare", "regression", "survival"].forEach(k => { if (S.cur[k] && S.cur[k].id === +el.dataset.id) S.cur[k] = null; }); renumber(); render(); break;
    case "log-view": { const en = S.log.find(x => x.id === +el.dataset.id); if (en) { S.cur[en.kind] = en; S.form[en.kind] = JSON.parse(JSON.stringify(en.params)); S.tool.data = en.kind; render(); qaTop(); } break; }
    case "dl-zip": downloadZip(); break;
    case "dl-report": saveFile("results_report.html", reportHTML()); break;
    case "dl-csv": saveFile("analysis_data.csv", Data.toCSV(S.ds)); break;
    case "copy-csv": copyText(Data.toCSV(S.ds)); break;
  }
}
function refreshCode() {
  document.querySelectorAll("[data-code]").forEach(box => {
    const id = box.dataset.code, s = codeRegistry.get(id); if (!s) return;
    box.querySelectorAll("[data-act=lang]").forEach(b => b.setAttribute("aria-selected", String(b.dataset.v === S.codeLang)));
    box.querySelector("pre").innerHTML = highlight(S.codeLang, (typeof s === "function" ? s(S.codeLang) : s[S.codeLang]) || "");
  });
}
function pruneAnswers(a, qs) { let changed = true; while (changed) { changed = false; qs.forEach(q => { if (q.show && !q.show(a) && a[q.id] !== undefined) { delete a[q.id]; changed = true; } }); } }
const toggle = (arr, x) => { const i = arr.indexOf(x); if (i >= 0) arr.splice(i, 1); else arr.push(x); };
function onChange(e) {
  const el = e.target, act = el.dataset.act;
  if (el.id === "file" && el.files && el.files[0]) { readFile(el.files[0]); el.value = ""; return; }
  if (act === "set") { setPath(S, el.dataset.path, el.value); render(); return; }
  if (act === "finer") { S.m.question.finer[el.dataset.k] = el.checked; render(); return; }
  if (act === "check") { const id = S.m.checklist.id; S.m.checklist.done[id] = S.m.checklist.done[id] || {}; S.m.checklist.done[id][el.dataset.k] = el.checked; render(); return; }
  if (act === "form") { setPath(S.form, el.dataset.path, el.value); if (el.dataset.path === "regression.outcome") S.form.regression.preds = S.form.regression.preds.filter(p => p !== el.value); render(); return; }
  if (act === "form-check") { setPath(S.form, el.dataset.path, el.checked); render(); return; }
  if (act === "alpha") { S.alpha = Number(el.value); rerunLog(); render(); return; }
  if (act === "missing") { S.missingCodes = el.value; rebuild(); rerunLog(); render(); return; }
  if (act === "vtype") { const o = S.overrides[el.dataset.var] = S.overrides[el.dataset.var] || {}; o.type = el.value; delete o.order; delete o.ref; rebuild(); rerunLog(); render(); return; }
  if (act === "vref") { const o = S.overrides[el.dataset.var] = S.overrides[el.dataset.var] || {}; o.ref = el.value; delete o.order; rebuild(); rerunLog(); render(); return; }
  if (act === "vorder") {
    const v = V(el.dataset.var), arr = el.value.split(",").map(s => s.trim()).filter(Boolean);
    if (arr.length !== v.allLevels.length || !arr.every(l => v.allLevels.includes(l))) { toast(`List all ${v.allLevels.length} categories exactly: ${v.allLevels.join(", ")}`); el.value = v.levels.join(", "); return; }
    const o = S.overrides[v.name] = S.overrides[v.name] || {}; o.order = arr; rebuild(); rerunLog(); render(); return;
  }
  if (el.dataset.path && el.dataset.path.startsWith("m.")) { setPath(S, el.dataset.path, el.dataset.num ? (el.value === "" ? "" : Number(el.value)) : el.value); persist(); }
}
function onInput(e) {
  const el = e.target;
  if (!el.dataset.path || !el.dataset.path.startsWith("m.") || el.dataset.act) return;
  setPath(S, el.dataset.path, el.dataset.num ? (el.value === "" ? "" : el.value) : el.value);
  if (el.dataset.live === "question") { const st = S.m.question, fw = Method.FRAMEWORKS[st.fw], vals = {}; fw.fields.forEach(([k, , ex]) => vals[k] = (getPath(st.v, st.fw + "." + k) || "").trim() || ex); $("#live-question").innerHTML = questionOut(st.fw, vals, st.setting); }
  if (el.dataset.live === "sample") { codeRegistry.clear(); $("#live-sample").innerHTML = sampleOut(); }
  persist();
}
function loadExample() { const p = Data.parseDelimited(EXAMPLE_CSV); loadDataset(p.headers, p.rows, "Example: Ghana hypertension survey (fictional)", true); seedExampleForms(); }
function seedExampleForms() {
  S.form.describe = { vars: ["age_years", "sex", "residence", "education_level", "bmi", "systolic_bp"], group: "hypertension" };
  S.form.compare = { outcome: "systolic_bp", exposure: "salt_intake", paired: false, force: "" };
  S.form.regression = { outcome: "hypertension", preds: ["age_years", "bmi", "residence", "sex"], model: "modpoisson", cluster: "" };
  S.form.survival = { time: "follow_up_months", event: "lost_to_follow_up", group: "hypertension", covs: ["age_years", "smoker"] };
}

/* =====================================================================
   Forecast tool (website build): Holt-Winters from quantai-engine.js
   ===================================================================== */
DATA_TOOLS.splice(DATA_TOOLS.findIndex(t => t.id === "export"), 0, { id: "forecast", k: "06", label: "Forecast", sub: "Holt-Winters time series" });
S.fc = { date: "", value: "", h: "", res: null, err: null };

function fcDateVars() {
  if (!S.ds || !window.QuantAI) return [];
  return S.ds.vars.filter(v => {
    const vals = v.values.filter(x => x !== null);
    if (vals.length < 6) return false;
    const ok = vals.filter(x => window.QuantAI.toDate(x)).length;
    return ok >= 0.8 * vals.length && !(v.numeric && vals.every(x => /^\d{1,3}$/.test(x)));
  });
}
function fcStep(ms) {
  if (ms.length < 3) return null;
  const d = []; for (let i = 1; i < ms.length; i++) d.push((ms[i] - ms[i - 1]) / 86400000);
  const med = Stats.median(d);
  if (med >= 27 && med <= 32) return { name: "month", period: 12 };
  if (med >= 6 && med <= 8) return { name: "week", period: 52 };
  if (med >= 0.9 && med <= 1.1) return { name: "day", period: 7 };
  if (med >= 88 && med <= 93) return { name: "quarter", period: 4 };
  if (med >= 360 && med <= 370) return { name: "year", period: 1 };
  return { name: "step", period: 1 };
}
function fcSeries(dv, yv) {
  const Q = window.QuantAI, pts = [];
  for (let i = 0; i < S.ds.nRows; i++) {
    const y = yv.values[i]; if (y === null || !Data.isNum(y)) continue;
    if (dv) { const d = Q.toDate(dv.values[i]); if (!d) continue; pts.push({ d, v: Number(y) }); } else pts.push({ d: i, v: Number(y) });
  }
  if (dv) {
    pts.sort((a, b) => a.d - b.d);
    const m = []; pts.forEach(p => { const l = m[m.length - 1]; if (l && +l.d === +p.d) { l.v += p.v; l.k++; } else m.push({ d: p.d, v: p.v, k: 1 }); });
    return { x: m.map(p => p.d), y: m.map(p => p.v), merged: m.some(p => p.k > 1) };
  }
  return { x: pts.map(p => p.d), y: pts.map(p => p.v), merged: false };
}
function runForecast() {
  const f = S.fc, Q = window.QuantAI;
  f.err = null; f.res = null;
  if (!Q) { f.err = "The forecasting engine did not load. Refresh the page."; render(); return; }
  const dv = f.date && f.date !== "__row" ? V(f.date) : null, yv = V(f.value);
  if (!yv) { f.err = "Choose the value to forecast."; render(); return; }
  const ts = fcSeries(dv, yv);
  if (ts.y.length < 6) { f.err = `A forecast needs at least 6 time points; "${vlabel(yv)}" has ${ts.y.length}.`; render(); return; }
  const step = dv ? fcStep(ts.x.map(d => +d)) : null, period = step ? step.period : 1;
  const h = Math.max(1, Math.min(60, parseInt(f.h, 10) || (period > 1 ? Math.min(12, period) : 6)));
  const hw = Q.holtWinters(ts.y, period, h);
  if (!hw) { f.err = "The series could not be fitted. Check that it has no gaps or constant values."; render(); return; }
  const lab = d => dv ? Q.fmtDate(d, step) : String(d + 1);
  const xs = ts.x.map(lab), fx = []; for (let i = 1; i <= h; i++) fx.push(dv ? Q.fmtDate(Q.addSteps(ts.x[ts.x.length - 1], step, i), step) : String(ts.y.length + i));
  f.res = { dv: dv ? dv.name : null, yv: yv.name, ts, step, period, h, hw, xs, fx, merged: ts.merged };
  render();
}
function fcChart(r) {
  const W = 640, H = 300, m = { l: 56, r: 16, t: 16, b: 46 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const n = r.ts.y.length, N = n + r.h, all = [...r.ts.y, ...r.hw.lower, ...r.hw.upper];
  const yt = niceTicks(Math.min(...all), Math.max(...all), 5), y0 = yt[0], y1 = yt[yt.length - 1];
  const x = i => m.l + i / (N - 1) * iw, y = v => m.t + ih - (v - y0) / (y1 - y0) * ih;
  const path = (arr, off) => arr.map((v, i) => isFinite(v) ? `${x(i + off).toFixed(1)},${y(v).toFixed(1)}` : null).filter(Boolean).join(" ");
  const band = [...r.hw.upper.map((v, i) => `${x(n + i).toFixed(1)},${y(v).toFixed(1)}`), ...r.hw.lower.map((v, i) => `${x(n + i).toFixed(1)},${y(v).toFixed(1)}`).reverse()].join(" ");
  const labs = [...r.xs, ...r.fx], every = Math.max(1, Math.ceil(N / 8));
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Observed series with forecast">${axisY(yt, y, m.l, W - m.r)}
    <rect x="${x(n - .5)}" y="${m.t}" width="${W - m.r - x(n - .5)}" height="${ih}" fill="var(--surface-2)" opacity=".6"/>
    <polygon points="${band}" fill="var(--c5)" fill-opacity=".14"/>
    <polyline points="${path(r.ts.y, 0)}" fill="none" stroke="var(--c1)" stroke-width="2"/>
    <polyline points="${path(r.hw.fitted, 0)}" fill="none" stroke="var(--ink-3)" stroke-width="1.3" stroke-dasharray="4 3"/>
    <polyline points="${path([r.ts.y[n - 1], ...r.hw.forecast], n - 1)}" fill="none" stroke="var(--c5)" stroke-width="2.5"/>
    ${labs.map((l, i) => (i % every === 0 && (N - 1 - i >= every * .6 || i === N - 1)) || i === N - 1 ? `<text x="${x(i)}" y="${H - m.b + 16}" text-anchor="middle">${esc(String(l))}</text>` : "").join("")}
    <text x="${x(n) + 4}" y="${m.t + 12}">forecast</text></svg>
    <p class="sub muted" style="font-size:.76rem">Solid blue = observed, dashed grey = model fit, red = forecast with its 80% range (shaded band).</p>`;
}
function fcCode(r) {
  const hw = r.hw, dv = r.dv, y = r.yv, a = hw.alpha, b = hw.beta, g = hw.gamma, p = r.period, h = r.h, seas = hw.seasonal;
  const note = "The smoothing parameters are fixed to QuantAI's estimates; each program initialises level, trend and season slightly differently, so forecasts will be close but not identical.";
  return {
    python: `# Requires: pip install pandas statsmodels\nimport pandas as pd\nfrom statsmodels.tsa.holtwinters import ExponentialSmoothing\n# ${note}\n\ndf = pd.read_csv("analysis_data.csv", keep_default_na=False, na_values=[""])\n` +
      (dv ? `df["_date"] = pd.to_datetime(df["${dv}"], dayfirst=True)\ns = df.dropna(subset=["${y}"]).groupby("_date")["${y}"].sum().sort_index()  # rows sharing a date are summed\n` : `s = df["${y}"].dropna().reset_index(drop=True)\n`) +
      `m = ExponentialSmoothing(s.to_numpy(dtype=float), trend="add", seasonal=${seas ? `"add", seasonal_periods=${p}` : "None"}, initialization_method="legacy-heuristic")\nfit = m.fit(smoothing_level=${a}, smoothing_trend=${b}${seas ? `, smoothing_seasonal=${g}` : ""}, optimized=False)\nprint(fit.forecast(${h}))`,
    r: `# Base R. ${note}\ndat <- read.csv("analysis_data.csv", na.strings = "", stringsAsFactors = FALSE)\n` +
      (dv ? `dat <- dat[!is.na(dat$${y}), ]\nagg <- aggregate(${y} ~ ${dv}, data = dat, FUN = sum)  # check that ${dv} sorts in time order\nx <- ts(agg$${y}, frequency = ${p})\n` : `x <- ts(na.omit(dat$${y}), frequency = ${p})\n`) +
      `fit <- HoltWinters(x, alpha = ${a}, beta = ${b}, gamma = ${seas ? g : "FALSE"})\npredict(fit, n.ahead = ${h}, prediction.interval = TRUE, level = 0.80)`,
    stata: `* ${note}\nimport delimited using "analysis_data.csv", clear varnames(1) case(preserve)\n` +
      (dv ? `* Convert ${dv} to a Stata date first, e.g. for "2024-01": generate t = monthly(${dv}, "YM") ; format t %tm\n* (daily dates: generate t = date(${dv}, "DMY") ; format t %td)\ncollapse (sum) ${y}, by(t)\ntsset t\n` : `generate t = _n\ntsset t\n`) +
      (seas ? `tssmooth shwinters fc_${y} = ${y}, parms(${a} ${b} ${g}) period(${p}) forecast(${h}) additive` : `tssmooth hwinters fc_${y} = ${y}, parms(${a} ${b}) forecast(${h})`),
    spss: `* ${note}.\n* Define dates first (Data > Define Date and Time), then:.\nTSMODEL\n  /MODELSUMMARY PRINT=[MODELFIT]\n  /MODELDETAILS PRINT=[PARAMETERS FORECASTS]\n  /OUTPUTFILTER DISPLAY=ALLMODELS\n  /MODEL DEPENDENT=${y}\n  /EXSMOOTH TYPE=${seas ? "WINTERSADDITIVE" : "HOLT"} TRANSFORM=NONE.\nPREDICT THRU END+${h}.`,
  };
}
function pageForecast() {
  const f = S.fc, dvs = fcDateVars(), nums = varsUsable().filter(v => v.type === "continuous" || v.type === "count");
  if (!f.value || !V(f.value)) f.value = nums.length ? nums[nums.length - 1].name : "";
  if (!f.date || (f.date !== "__row" && !V(f.date))) f.date = dvs.length ? dvs[0].name : "__row";
  const r = f.res, Q = window.QuantAI;
  let out = "";
  if (r) {
    const hw = r.hw, y = V(r.yv), unit = r.step ? r.step.name : "step", last = r.ts.y[r.ts.y.length - 1], end = hw.forecast[r.h - 1];
    const change = (end - last) / Math.abs(last || 1);
    const writeup = `Holt-Winters exponential smoothing (${hw.seasonal ? `additive ${unit}ly seasonality, period ${r.period}` : "Holt's linear trend, no seasonal pattern detected"}) was fitted to ${r.ts.y.length} ${unit === "step" ? "time points" : unit + "s"} of ${vlabel(y)} (α = ${F.noLead(F.f(hw.alpha))}, β = ${F.noLead(F.f(hw.beta))}${hw.seasonal ? `, γ = ${F.noLead(F.f(hw.gamma))}` : ""}). The in-sample mean absolute percentage error was ${isFinite(hw.mape) ? F.f(100 * hw.mape, 1) + "%" : "not defined (the series contains zeros)"}. The forecast for ${r.fx[r.h - 1]} is ${Q.fmt(end)} (80% range ${Q.fmt(hw.lower[r.h - 1])} to ${Q.fmt(hw.upper[r.h - 1])}), ${Math.abs(change) < 0.02 ? "about the same as" : (change > 0 ? `${F.f(100 * change, 0)}% higher than` : `${F.f(-100 * change, 0)}% lower than`)} the last observed value of ${Q.fmt(last)}.`;
    out = `<article class="card stack" style="margin-top:1rem">
      <div class="res-head"><div class="stack" style="gap:.3rem"><span class="sect-t">Result</span><h2>Forecast of ${esc(vlabel(y))}</h2></div><div class="meta"><span class="pill acc">${hw.seasonal ? "Holt-Winters (seasonal)" : "Holt's linear trend"}</span><span class="pill num">${r.ts.y.length} points · ${r.h} ahead</span></div></div>
      ${r.merged ? `<div class="notice warn">Some rows share the same date, so their values were added together.</div>` : ""}
      ${r.ts.y.length < 2 * r.period + 2 && r.period > 1 ? `<div class="notice warn">There are fewer than two full ${unit === "month" ? "years" : "cycles"} of data, so seasonality could not be estimated; the forecast uses trend only.</div>` : ""}
      <div class="grid2"><section><div class="sect-t">How it was fitted</div><ol class="audit">
        <li><span class="ic">•</span><span><b>Time step.</b> ${r.step ? `Median gap between dates → ${esc(unit)}ly data${r.period > 1 ? `, seasonal period ${r.period}` : ""}.` : "No date column: rows are taken in file order."}</span></li>
        <li><span class="ic ${hw.seasonal ? "ok" : ""}">${hw.seasonal ? "✓" : "•"}</span><span><b>Seasonality.</b> ${hw.seasonal ? "At least two full cycles, so a repeating pattern is estimated." : "Not estimated (needs at least two full cycles plus two points)."}</span></li>
        <li><span class="ic ok">✓</span><span><b>Parameters.</b> Grid search for the smallest one-step-ahead squared error: α = ${F.f(hw.alpha)}, β = ${F.f(hw.beta)}${hw.seasonal ? `, γ = ${F.f(hw.gamma)}` : ""}.</span></li>
        <li><span class="ic">•</span><span><b>Accuracy in the data.</b> RMSE = ${Q.fmt(hw.rmse)}${isFinite(hw.mape) ? `, MAPE = ${F.f(100 * hw.mape, 1)}%` : ""}.</span></li>
        <li class="final"><span class="ic fin">→</span><span><b>Range.</b> 80% prediction band that widens with the horizon.</span></li></ol></section>
        <section><div class="sect-t">Chart</div>${fcChart(r)}</section></div>
      ${tableHTML({ title: "Forecast", columns: ["Period", "Forecast", "80% low", "80% high"], rows: hw.forecast.map((v, i) => [r.fx[i], Q.fmt(v), Q.fmt(hw.lower[i]), Q.fmt(hw.upper[i])]) })}
      <section class="stack" style="gap:.5rem"><div class="row" style="justify-content:space-between;align-items:center"><div class="sect-t" style="margin:0">Write-up</div><button class="btn quiet sm" data-act="copy" data-text="${esc(writeup)}">Copy text</button></div><p class="apa">${esc(writeup)}</p></section>
      <div class="notice info">A forecast assumes the past pattern continues. Shocks such as outbreaks, policy changes or stock-outs are not in the model.</div>
      <section class="stack" style="gap:.5rem"><div class="sect-t" style="margin:0">Code to reproduce it</div>${codeBlock(fcCode(r), { maxh: 300 })}</section>
    </article>`;
  }
  return pageHead("Data analysis", "Forecast", "Forecast a value over time with Holt-Winters exponential smoothing. QuantAI detects the time step, adds a seasonal pattern when there is enough history, and shows an 80% range.") + exampleBanner() + `
  <div class="card stack">
    ${!dvs.length ? `<div class="notice info">No date column was found in this data. You can forecast in row order, or <a href="#" data-act="fc-example">load the monthly malaria example</a> (48 months of clinic data).</div>` : ""}
    <div class="row">
      <div class="field"><label class="label" for="fc-date">Time</label><select id="fc-date" data-act="fc" data-k="date">${dvs.map(v => `<option value="${esc(v.name)}"${v.name === f.date ? " selected" : ""}>${esc(vlabel(v))}</option>`).join("")}<option value="__row"${f.date === "__row" ? " selected" : ""}>Row order (no date column)</option></select></div>
      <div class="field"><label class="label" for="fc-y">Value to forecast</label><select id="fc-y" data-act="fc" data-k="value">${nums.map(v => `<option value="${esc(v.name)}"${v.name === f.value ? " selected" : ""}>${esc(vlabel(v))}</option>`).join("")}</select></div>
      <div class="field" style="flex:0 1 150px"><label class="label" for="fc-h">Periods ahead</label><input type="number" min="1" max="60" id="fc-h" value="${esc(f.h)}" placeholder="auto" data-act="fc" data-k="h"></div>
      <button class="btn" data-act="fc-run">Forecast</button>
    </div>
    ${f.err ? `<div class="notice bad">${esc(f.err)}</div>` : ""}
  </div>${out}`;
}
async function loadForecastExample() {
  try {
    const txt = await (await fetch("assets/data/sample-health-monthly.csv")).text();
    const p = Data.parseDelimited(txt);
    loadDataset(p.headers, p.rows, "Example: monthly malaria cases (fictional)", true);
    S.overrides = {}; rebuild();
    S.fc = { date: "month", value: "malaria_cases", h: "12", res: null, err: null };
    runForecast();
  } catch (e) { S.fc.err = "The example could not be loaded. Upload a file with a date column instead."; render(); }
}
document.addEventListener("click", e => {
  const el = e.target.closest("[data-act]"); if (!el) return;
  if (el.dataset.act === "fc-run") { e.preventDefault(); runForecast(); }
  if (el.dataset.act === "fc-example") { e.preventDefault(); loadForecastExample(); }
});
document.addEventListener("change", e => {
  const el = e.target; if (el.dataset.act !== "fc") return;
  S.fc[el.dataset.k] = el.value; if (el.dataset.k !== "h") S.fc.res = null; render();
});

const _loadDatasetBase = loadDataset;
loadDataset = function () { S.fc = { date: "", value: "", h: "", res: null, err: null }; return _loadDatasetBase.apply(this, arguments); };

/* =====================================================================
   AI assistant (website build): answers through the Model Analysis Hub
   Cloudflare worker (Groq). Only statistical summaries and results are
   sent — never the uploaded file. Every number still comes from the
   deterministic engine; the assistant only explains.
   ===================================================================== */
const AI_URL = "https://mah-assistant.kamalamadu8.workers.dev";
const AI_PRIVACY = "Your question and a statistical summary (never your file) are sent to the Model Analysis Hub assistant. Numbers come from QuantAI's own calculations; the assistant only explains them, so check anything important.";
S.explain = {};

const aiErrText = code => ({ rate_limited: "You've asked a lot of questions in a short time. Wait a few minutes and try again.", busy: "The assistant is busy right now. Try again in a minute.", not_configured: "The assistant isn't set up yet.", network: "The assistant couldn't be reached. Check your connection; all other tools still work." })[code] || "The assistant couldn't answer right now. All other tools still work.";
async function askAI(mode, context, messages) {
  let res;
  try {
    res = await fetch(AI_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode, context: context.slice(0, 11900), messages: messages.slice(-10).map(m => ({ role: m.role, content: m.content.slice(0, 590) })) }) });
  } catch (e) { throw Object.assign(new Error("network"), { code: "network" }); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.reply) throw Object.assign(new Error(data.error || "failed"), { code: data.error || "failed" });
  return data.reply;
}
/** Minimal, safe formatting: escape, then **bold**, paragraphs and simple list lines. */
function aiFormat(text) {
  return esc(text).split(/\n{2,}/).map(par => {
    const lines = par.split("\n");
    if (lines.every(l => /^\s*([-*•]|\d+[.)])\s+/.test(l))) return `<ul class="clean">${lines.map(l => `<li>${l.replace(/^\s*([-*•]|\d+[.)])\s+/, "")}</li>`).join("")}</ul>`;
    return `<p>${lines.join("<br>")}</p>`;
  }).join("").replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
}

/* ---------- context builders (summaries only) ---------- */
function dataContext(extra) {
  const ds = S.ds, L = [];
  L.push(`DATASET: ${S.dsSource}; ${ds.nRows} rows; significance level ${S.alpha}.`);
  L.push("VARIABLES (name in code | original name | type | missing | summary):");
  ds.vars.filter(v => v.type !== "id").forEach(v => L.push(`- ${v.name} | ${v.label} | ${v.type} | ${v.nMissing} missing | ${summaryOf(v)}`));
  const q = S.quality || [];
  if (q.length) { L.push("", "DATA CHECK (QuantAI's automatic checks of this file; mention problems that affect the requested analysis):"); q.slice(0, 14).forEach(x => L.push(`- [${x.sev === "warn" ? "REVIEW" : "note"}] ${x.msg}`)); }
  const log = S.log.slice(-6);
  if (log.length) {
    L.push("", "ANALYSES ALREADY RUN IN QUANTAI (results are exact):");
    log.forEach(e => { const r = e.result; L.push(`${e.k}. ${r.title} — ${r.method}; n = ${r.n}. ${r.writeup || ""}`); r.decision.filter(d => d.ok === false).forEach(d => L.push(`   warning: ${d.rule}: ${d.detail}`)); });
  }
  if (S.fc && S.fc.res) { const r = S.fc.res, hw = r.hw; L.push("", `FORECAST: ${r.yv}, ${r.ts.y.length} points, ${hw.seasonal ? "seasonal Holt-Winters" : "Holt linear trend"}, next values ${hw.forecast.slice(0, 6).map((v, i) => `${r.fx[i]}: ${window.QuantAI.fmt(v)} (80% ${window.QuantAI.fmt(hw.lower[i])}–${window.QuantAI.fmt(hw.upper[i])})`).join("; ")}.`); }
  L.push("", "QUANTAI TOOLS: Data & variables (types, coding, data check), Describe (Table 1), Compare & relate (tests chosen by assumption checks), Regression (linear, logistic, Poisson, modified Poisson, ordinal, multinomial, mixed), Survival (Kaplan-Meier, log-rank, Cox), Forecast (Holt-Winters), Log & export (Stata, Python, R, SPSS code, HTML report).");
  if (extra) L.push("", extra);
  return L.join("\n");
}
function resultContext(r) {
  return [`RESULT TO EXPLAIN: ${r.title}`, `Method chosen: ${r.method}; n = ${r.n}`, "Rules QuantAI applied:", ...r.decision.map(d => `- ${d.rule}: ${d.detail}`),
    ...r.tables.map(t => `${t.title}\n${t.columns.join(" | ")}\n${t.rows.slice(0, 14).map(x => x.join(" | ")).join("\n")}`), `APA summary: ${r.writeup}`, ...r.warnings.map(w => `Note: ${w}`)].join("\n");
}
function methodsContext() {
  const m = S.m, L = ["The visitor is planning a study in QuantAI's methodology tools. Their current inputs:"];
  const q = m.question, fw = Method.FRAMEWORKS[q.fw]; L.push(`Research question framework ${q.fw}: ` + fw.fields.map(([k, label, ex]) => `${label} = ${getPath(q.v, q.fw + "." + k) || ex + " (example)"}`).join("; "));
  const d = Method.DESIGN_QUESTIONS.filter(x => m.design.a[x.id]).map(x => `${x.q} → ${(x.options.find(o => o[0] === m.design.a[x.id]) || [])[1]}`);
  if (d.length) { L.push("Design answers: " + d.join("; ")); const r = Method.decideDesign(m.design.a); if (r) L.push(`Rule-based design recommendation: ${r.design.name}. Alternatives: ${r.alternatives.map(a => a.design.name).join(", ") || "none"}.`); }
  const sc = Method.SAMPLE.find(x => x.id === m.sample.id); if (sc) L.push(`Sample size calculator selected: ${sc.name}; inputs ${JSON.stringify(sampleVals(m.sample, sc))}; adjustments ${JSON.stringify(m.sample.adj)}.`);
  const t = Method.selectTest(m.test.a); if (t && t.id) L.push(`Test selector result: ${t.test.name}${t.alternative ? `; alternative ${t.alternative.test.name}` : ""}.`);
  L.push("QuantAI methodology tools: Research question, Study design, Sample size, Choose a test, Reporting checklist (STROBE, CONSORT, STARD, COREQ, PRISMA).");
  return L.join("\n");
}

/* ---------- explain button on result cards ---------- */
function aiExplainHTML(entry) {
  const st = S.explain[entry.id] || {};
  return `<section class="stack" style="gap:.5rem"><div class="row" style="justify-content:space-between;align-items:center"><div class="sect-t" style="margin:0">Plain-language explanation (AI)</div>
    <button class="btn ghost sm" data-act="ai-explain" data-id="${entry.id}" ${st.busy ? "disabled" : ""}>${st.busy ? '<span class="spin"></span> Explaining…' : st.text ? "Explain again" : "Explain in plain language"}</button></div>
    ${st.err ? `<div class="notice bad">${esc(st.err)}</div>` : ""}
    ${st.text ? `<div class="stack" style="background:var(--surface-2);padding:.8rem 1rem;border-radius:10px">${aiFormat(st.text)}</div><p class="sub muted" style="font-size:.76rem">The assistant explains QuantAI's numbers; it did not calculate them.</p>` : ""}</section>`;
}
async function explainResult(id) {
  const en = S.log.find(x => x.id === id); if (!en) return;
  const st = S.explain[id] = { busy: true };
  render();
  try {
    st.text = await askAI("data", dataContext(resultContext(en.result)), [{ role: "user", content: "Explain the RESULT TO EXPLAIN in plain language for a student writing a thesis: what was tested and why this method was chosen (from the rules), what the result means in practice, and two cautions. Use only the numbers given. 120 to 200 words." }]);
  } catch (e) { st.err = aiErrText(e.code); }
  st.busy = false; render();
}

document.addEventListener("click", e => {
  const el = e.target.closest("[data-act]"); if (!el) return;
  const act = el.dataset.act;
  if (act === "ai-explain") { e.preventDefault(); explainResult(+el.dataset.id); }
});

/* =====================================================================
   QuantAI Chat (website build): a conversation interface. The assistant
   (Groq via the Model Analysis Hub worker) reads a summary of the data,
   the analyses run and the methodology inputs, and can ask QuantAI to
   run an analysis. QuantAI's engine computes every result shown.
   ===================================================================== */
const CHAT_KEY = "chats";
S.chats = (() => { const c = store.get(CHAT_KEY, []); return Array.isArray(c) ? c.slice(0, 30) : []; })();
S.chatId = S.chats.length ? S.chats[0].id : null;
S.chatBusy = false;
const saveChats = () => store.set(CHAT_KEY, S.chats.slice(0, 30).map(c => ({ ...c, messages: c.messages.slice(-60).map(m => ({ role: m.role, content: m.content, action: m.action || null, note: m.note || null, suggest: m.suggest || null, tools: m.tools || null })) })));
const curChat = () => S.chats.find(c => c.id === S.chatId) || null;
function newChat() { const c = { id: "c" + Date.now().toString(36), title: "New chat", created: Date.now(), messages: [] }; S.chats.unshift(c); S.chatId = c.id; saveChats(); return c; }

/* ---------- resolving variable names the assistant (or user) gives ---------- */
function resolveVar(name, types) {
  if (!name || !S.ds) return null;
  const n = String(name).trim().toLowerCase(), vs = varsUsable();
  const hit = vs.find(v => v.name === n) || vs.find(v => v.label.toLowerCase() === n) || vs.find(v => v.name.replace(/_/g, " ") === n.replace(/_/g, " ")) ||
    vs.find(v => v.label.toLowerCase().includes(n) || n.includes(v.label.toLowerCase()));
  if (hit && types && !types.includes(hit.type)) return null;
  if (hit) return hit;
  return fuzzyVar(n, types);
}
/** Word-overlap match so "blood pressure", "gender" or "age" find "Systolic BP", "Sex", "Age (years)". */
const VAR_SYN = { "blood pressure": "bp", "bloodpressure": "bp", gender: "sex", male: "sex", female: "sex", men: "sex", women: "sex", "men and women": "sex", "males and females": "sex", years: "age", old: "age", weight: "bmi", "body mass": "bmi", salt: "salt", smoking: "smoker", smokes: "smoker", school: "education", schooling: "education", rural: "residence", urban: "residence", location: "district", followup: "follow" };
const VAR_STOP = new Set(["the", "of", "a", "an", "and", "in", "on", "for", "to", "with", "my", "is", "are", "level", "levels", "between", "by", "among", "people", "who", "have", "has", "rate", "rates", "number", "value", "values", "total", "data", "variable", "variables"]);
function fuzzyVar(n, types) {
  let q = " " + n.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ") + " ";
  Object.keys(VAR_SYN).sort((a, b) => b.length - a.length).forEach(k => { q = q.split(" " + k + " ").join(" " + VAR_SYN[k] + " "); });
  const qw = q.trim().split(" ").filter(w => w.length > 1 && !VAR_STOP.has(w));
  if (!qw.length) return null;
  let best = null, bestScore = 0;
  varsUsable().forEach(v => {
    if (types && !types.includes(v.type)) return;
    const vw = (v.name + " " + v.label).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().split(" ");
    let sc = 0;
    qw.forEach(w => { if (vw.includes(w)) sc += 2; else if (w.length > 3 && vw.some(x => x.length > 3 && (x.startsWith(w) || w.startsWith(x)))) sc += 1; });
    // prefer the plainest variable (e.g. "Systolic BP" over "Systolic BP 3 months")
    sc -= vw.length * 0.01;
    if (sc > bestScore) { bestScore = sc; best = v; }
  });
  return bestScore >= 1 ? best : null;
}
const ACTION_LABEL = { describe: "Describe", compare: "Compare & relate", regression: "Regression", survival: "Survival analysis", forecast: "Forecast", open: "Open tool", check: "Data check", fix: "Data fixes", report: "Report" };
function addEntry(kind, params, result) {
  const entry = { id: ++S.logSeq, k: S.log.length + 1, kind, params, result };
  S.log.push(entry); S.cur[kind] = entry; S.form[kind] = JSON.parse(JSON.stringify(params)); renumber();
  return entry;
}
/** Run an action through the deterministic engine. Returns {entry} | {forecast:true} | {nav} ; throws user errors. */
function runAction(a) {
  const need = (nm, label, types) => { const v = resolveVar(nm, types); if (!v) throw Object.assign(new Error(`I couldn't find a suitable variable for ${label} ("${nm || "none given"}"). Variables: ${varsUsable().map(x => x.name).join(", ")}.`), { user: true }); return v.name; };
  const many = (arr, label) => (Array.isArray(arr) ? arr : []).map(x => need(x, label));
  if (a.type === "describe") { const vars = many(a.vars && a.vars.length ? a.vars : varsUsable().map(v => v.name).slice(0, 8), "the table"); const group = a.group ? need(a.group, "the grouping variable", ["binary", "categorical"]) : ""; return { entry: addEntry("describe", { vars: vars.filter(v => v !== group), group }, runAnalysis("describe", { vars: vars.filter(v => v !== group), group })) }; }
  if (a.type === "compare") { const p = { outcome: need(a.outcome, "the outcome"), exposure: need(a.exposure, "the exposure"), paired: !!a.paired, force: "" }; return { entry: addEntry("compare", p, runAnalysis("compare", p)) }; }
  if (a.type === "regression") { const p = { outcome: need(a.outcome, "the outcome"), preds: many(a.predictors, "the predictors"), model: ["auto", "linear", "logistic", "modpoisson", "poisson", "ordinal", "multinomial", "mixed"].includes(a.model) ? a.model : "auto", cluster: a.cluster ? need(a.cluster, "the cluster variable") : "" }; return { entry: addEntry("regression", p, runAnalysis("regression", p)) }; }
  if (a.type === "survival") { const p = { time: need(a.time, "follow-up time", ["continuous", "count"]), event: need(a.event, "the event", ["binary"]), group: a.group ? need(a.group, "the group", ["binary", "categorical"]) : "", covs: many(a.covariates, "the covariates") }; return { entry: addEntry("survival", p, runAnalysis("survival", p)) }; }
  if (a.type === "forecast") {
    const value = need(a.value, "the value to forecast", ["continuous", "count"]), dv = a.date ? resolveVar(a.date) : null;
    S.fc = { date: dv ? dv.name : (fcDateVars()[0] ? fcDateVars()[0].name : "__row"), value, h: String(Math.max(1, Math.min(60, parseInt(a.horizon, 10) || 12))), res: null, err: null };
    runForecast(); if (S.fc.err) throw Object.assign(new Error(S.fc.err), { user: true }); return { forecast: true };
  }
  if (a.type === "check") return { check: true };
  if (a.type === "fix") { const d = applyAllFixes(); return { check: true, fixed: d }; }
  if (a.type === "report") { if (!S.log.length) throw Object.assign(new Error("There are no analyses to report yet. Ask me to run one first."), { user: true }); saveFile("quantai-report.html", reportHTML()); return { report: true }; }
  if (a.type === "open") { const t = a.tool; if (METHOD_TOOLS.some(x => x.id === t)) return { nav: ["method", t] }; if (DATA_TOOLS.some(x => x.id === t)) return { nav: ["data", t] }; throw Object.assign(new Error(`There is no "${t}" tool.`), { user: true }); }
  throw Object.assign(new Error("I don't know how to run that yet."), { user: true });
}
/* ---------- built-in answers to basic research questions (work without the AI) ---------- */
const KB_GROUP = { "p-value": "stats", "Confidence interval": "stats", "Statistical significance": "stats", "Null and alternative hypotheses": "stats", "Type I and Type II errors": "stats", "Statistical power": "stats", "Sample size": "design", "Confounding": "epi", "Bias": "epi", "Dependent and independent variables": "measure", "Types of variables": "measure", "Mean, median and spread": "stats", "Normal distribution": "stats", "t-test": "stats", "Chi-square test": "stats", "ANOVA": "stats", "Non-parametric tests": "stats", "Correlation": "stats", "Linear regression": "regression", "Logistic regression and odds ratios": "regression", "Prevalence and incidence": "epi", "Prevalence and risk ratios": "epi", "Cross-sectional study": "design", "Cohort study": "design", "Case-control study": "design", "Randomised controlled trial": "design", "Choosing a study design": "design", "Qualitative vs quantitative research": "qual", "Sampling methods": "design", "Validity and reliability": "measure", "Research questions and objectives": "thesis", "Literature and systematic reviews": "review", "Research ethics": "ethics", "Survival analysis": "regression", "Paired vs independent data": "stats", "Pilot study": "measure", "Designing a questionnaire": "measure" };
const RESEARCH_KB = (window.QA_KB_EXTRA || []).concat([
  { k: ["p-value", "p value", "pvalue"], t: "p-value", a: "A **p-value** is the probability of getting a result at least as extreme as yours if there were really no effect (if the null hypothesis were true). A small p-value (usually below 0.05) means your result would be unusual if there were no effect, so you reject the null hypothesis.\n\nIt does **not** tell you how big or important the effect is, and it is not the probability that your hypothesis is true. Always report it with the effect size and its 95% confidence interval." },
  { k: ["confidence interval", "95% ci", "95 ci"], t: "Confidence interval", a: "A **95% confidence interval** is the range of values that is compatible with your data for the true effect. If you repeated the study many times, 95% of the intervals calculated this way would contain the true value.\n\nA narrow interval means a precise estimate. For a difference, if the interval includes 0 the result is not significant; for a ratio (odds ratio, risk ratio), if it includes 1 it is not significant." },
  { k: ["statistical significance", "statistically significant", "significance level", "alpha level"], t: "Statistical significance", a: "A result is **statistically significant** when its p-value is below a level chosen before the analysis, usually **0.05** (the significance level, alpha). It means the result is unlikely to be due to chance alone.\n\nSignificant does not mean important: with a large sample, tiny differences become significant. Judge importance from the size of the effect and its confidence interval." },
  { k: ["null hypothesis", "alternative hypothesis", "hypothesis testing", "h0"], t: "Null and alternative hypotheses", a: "The **null hypothesis (H0)** says there is no effect or no difference, for example \"mean blood pressure is the same in men and women\". The **alternative hypothesis (H1)** says there is an effect or difference.\n\nA statistical test measures how compatible your data are with H0. You either reject H0 (evidence of an effect) or fail to reject it. You never \"prove\" H0." },
  { k: ["type i error", "type 1 error", "type ii error", "type 2 error", "false positive", "false negative"], t: "Type I and Type II errors", a: "A **Type I error** is a false positive: concluding there is an effect when there isn't. Its probability is your significance level (usually 5%).\n\nA **Type II error** is a false negative: missing a real effect. Its probability is beta, and **power = 1 − beta** (usually set at 80%). Larger samples reduce Type II errors." },
  { k: ["power", "statistical power"], t: "Statistical power", a: "**Power** is the probability that your study detects a real effect of a given size. Studies are usually designed for **80% power** at a 5% significance level.\n\nPower rises with a larger sample, a bigger true effect and less variability. Low-powered studies often miss real effects. QuantAI's sample size tool works out the sample needed for the power you want." },
  { k: ["sample size", "how many participants", "how many respondents"], t: "Sample size", a: "The **sample size** is how many participants you need to answer your question with enough precision or power. It depends on the expected proportion or effect size, the precision or power you want, the significance level, and the expected non-response.\n\nFor a survey estimating a proportion, a common formula is Cochran's: n = Z² × p(1 − p) / d². Use QuantAI's sample size tool to calculate it with adjustments.", tool: ["method", "sample", "Work out a sample size"] },
  { k: ["confounder", "confounding", "confounding variable"], t: "Confounding", a: "A **confounder** is a third variable that is linked to both the exposure and the outcome and distorts the relationship between them. For example, age can confound the link between coffee and heart disease, because older people drink more coffee and have more heart disease.\n\nYou deal with confounding by design (randomisation, matching, restriction) or in the analysis (stratification or adjusting for it in a regression model)." },
  { k: ["bias", "selection bias", "information bias", "recall bias", "measurement bias"], t: "Bias", a: "**Bias** is a systematic error that pushes results away from the truth.\n\n- **Selection bias:** the people studied differ from the population you want to describe, e.g. only surveying people who attend clinic.\n- **Information (measurement) bias:** exposures or outcomes are measured wrongly, e.g. **recall bias** when cases remember past exposures better than controls.\n- **Confounding** is sometimes grouped with bias.\n\nBias is reduced by good design; a bigger sample does not fix it." },
  { k: ["independent variable", "dependent variable", "exposure", "outcome variable", "explanatory variable", "predictor"], t: "Dependent and independent variables", a: "The **dependent variable (outcome)** is what you are trying to explain or predict, e.g. hypertension. **Independent variables (exposures or predictors)** are the factors you think influence it, e.g. age, salt intake, BMI.\n\nIn a regression, the outcome sits on the left of the model and the predictors on the right." },
  { k: ["types of data", "type of variable", "types of variables", "continuous variable", "categorical variable", "ordinal variable", "nominal", "binary variable"], t: "Types of variables", a: "- **Continuous:** measured on a scale, e.g. age, BMI, blood pressure.\n- **Categorical (nominal):** groups with no order, e.g. region, religion.\n- **Ordinal:** groups with an order, e.g. education level, Likert scales.\n- **Binary:** two groups, e.g. yes/no, male/female.\n- **Count:** whole numbers of events, e.g. number of clinic visits.\n\nThe type of your outcome decides which test or regression to use." },
  { k: ["mean", "median", "mode", "average", "standard deviation", "interquartile range", "iqr"], t: "Mean, median and spread", a: "The **mean** is the average; the **median** is the middle value; the **mode** is the most common value. For skewed data (e.g. income) the median is a better summary than the mean.\n\nReport spread with the **standard deviation (SD)** alongside a mean, or the **interquartile range (IQR)** alongside a median. QuantAI's Table 1 picks the right one for each variable." },
  { k: ["normal distribution", "normality", "normally distributed", "shapiro", "skewed"], t: "Normal distribution", a: "A **normal distribution** is the symmetric bell-shaped curve where mean, median and mode are equal. Many tests (t-test, ANOVA, linear regression) assume the data or residuals are roughly normal.\n\nCheck with a histogram, a Q-Q plot or the Shapiro–Wilk test. If data are clearly skewed, use a non-parametric test (Mann–Whitney, Kruskal–Wallis) or transform the variable. QuantAI checks this automatically before choosing a test." },
  { k: ["t-test", "t test", "ttest", "student t"], t: "t-test", a: "A **t-test** compares the **means of a continuous variable between two groups**, e.g. blood pressure in men vs women.\n\n- **Independent t-test:** two separate groups. Use **Welch's** version if the group variances differ.\n- **Paired t-test:** the same people measured twice (before/after).\n\nIt assumes roughly normal data; otherwise use Mann–Whitney (independent) or Wilcoxon (paired)." },
  { k: ["chi-square", "chi square", "chi squared", "fisher"], t: "Chi-square test", a: "The **chi-square test** checks whether two **categorical variables** are associated, e.g. smoking (yes/no) and hypertension (yes/no), by comparing observed and expected counts in a table.\n\nIt needs expected counts of at least 5 in most cells; if counts are small, use **Fisher's exact test**. QuantAI checks this and switches automatically." },
  { k: ["anova", "analysis of variance", "kruskal"], t: "ANOVA", a: "**ANOVA (analysis of variance)** compares the **means of a continuous variable across three or more groups**, e.g. BMI across four regions. A significant result means at least one group differs; post-hoc tests show which.\n\nIt assumes roughly normal data and similar variances. If not, use the **Kruskal–Wallis** test." },
  { k: ["mann-whitney", "mann whitney", "wilcoxon", "non-parametric", "nonparametric"], t: "Non-parametric tests", a: "**Non-parametric tests** don't assume a normal distribution; they work on ranks. Use them for skewed data, small samples or ordinal outcomes.\n\n- Mann–Whitney U: two independent groups (instead of the t-test).\n- Wilcoxon signed-rank: paired data (instead of the paired t-test).\n- Kruskal–Wallis: three or more groups (instead of ANOVA).\n\nReport medians and IQRs with them." },
  { k: ["correlation", "pearson", "spearman", "correlation and causation", "correlation vs causation"], t: "Correlation", a: "**Correlation** measures how strongly two continuous variables move together, from −1 (perfect negative) to +1 (perfect positive); 0 means no linear relationship. Use **Pearson** for normal data and **Spearman** for skewed or ordinal data.\n\n**Correlation is not causation:** two things can move together because of a confounder or by chance. Causal claims need a suitable design and adjustment." },
  { k: ["linear regression", "multiple regression", "regression coefficient"], t: "Linear regression", a: "**Linear regression** models a **continuous outcome** (e.g. blood pressure) as a function of one or more predictors. Each coefficient is the expected change in the outcome for a one-unit increase in that predictor, holding the others constant. R² shows how much of the variation the model explains.\n\nCheck the assumptions: linearity, normal residuals, constant variance and no strong multicollinearity." },
  { k: ["logistic regression", "odds ratio", "odds"], t: "Logistic regression and odds ratios", a: "**Logistic regression** models a **binary outcome** (yes/no, e.g. hypertension). Results are **odds ratios (OR)**: OR > 1 means higher odds of the outcome, OR < 1 lower odds, OR = 1 no association. Adjusted ORs control for the other variables in the model.\n\nWhen the outcome is common (above about 10%), odds ratios exaggerate risk; **prevalence or risk ratios** from modified Poisson regression are easier to interpret." },
  { k: ["prevalence", "incidence"], t: "Prevalence and incidence", a: "**Prevalence** is the proportion of a population that **has** a condition at a point in time (existing cases), e.g. 30% of adults have hypertension. **Incidence** is the rate of **new** cases over a period among people at risk, e.g. 5 new cases per 1,000 person-years.\n\nCross-sectional studies measure prevalence; cohort studies measure incidence." },
  { k: ["prevalence ratio", "relative risk", "risk ratio", "modified poisson"], t: "Prevalence and risk ratios", a: "A **prevalence ratio (PR)** or **risk ratio (RR)** compares the proportion with the outcome in one group to another. PR = 1.5 means the outcome is 50% more common in the exposed group.\n\nThey are easier to interpret than odds ratios when the outcome is common. Adjusted PRs come from **modified Poisson regression** with robust standard errors, which QuantAI can run." },
  { k: ["cross-sectional", "cross sectional", "survey design"], t: "Cross-sectional study", a: "A **cross-sectional study** measures exposure and outcome at the **same point in time**, like a snapshot, usually through a survey. It is quick and cheap and good for estimating **prevalence** and finding associations.\n\nIts main weakness: you can't tell which came first, so it can't show cause and effect." },
  { k: ["cohort study", "cohort", "prospective study", "retrospective cohort", "longitudinal"], t: "Cohort study", a: "A **cohort study** follows people **over time**, comparing those exposed and not exposed to see who develops the outcome. It can be **prospective** (from now forward) or **retrospective** (using past records).\n\nIt measures **incidence** and **relative risk** and shows that exposure came before the outcome, but it takes time, costs more, and loses people to follow-up." },
  { k: ["case-control", "case control"], t: "Case-control study", a: "A **case-control study** starts with people who **have** the outcome (cases) and people who **don't** (controls), then looks back at their past exposures. It is efficient for **rare diseases** and gives **odds ratios**.\n\nIt is prone to **recall bias** and to bias in how controls are chosen, and it can't measure prevalence or incidence directly." },
  { k: ["randomised controlled trial", "randomized controlled trial", "rct", "clinical trial", "randomisation", "randomization", "experimental study"], t: "Randomised controlled trial", a: "In a **randomised controlled trial (RCT)**, participants are **randomly allocated** to an intervention or a control group and followed to compare outcomes. Randomisation balances known and unknown confounders, so an RCT gives the strongest evidence of cause and effect.\n\nIt can be expensive, and it isn't ethical or practical for every question (you can't randomise people to smoke). Report it with the CONSORT checklist." },
  { k: ["study design", "research design", "types of study", "which design"], t: "Choosing a study design", a: "The main designs:\n\n- **Cross-sectional:** prevalence and associations at one time point.\n- **Case-control:** rare outcomes; looks back at exposures.\n- **Cohort:** follows people forward; incidence and risk.\n- **RCT:** tests an intervention; strongest for cause and effect.\n- **Qualitative:** experiences, views and reasons.\n\nThe right one depends on your question, the outcome's frequency, time and budget. QuantAI's design advisor recommends one from a few questions.", tool: ["method", "design", "Choose a study design"] },
  { k: ["qualitative", "quantitative", "mixed methods"], t: "Qualitative vs quantitative research", a: "**Quantitative research** measures things with numbers and tests relationships statistically (surveys, experiments). **Qualitative research** explores experiences, meanings and reasons through words (interviews, focus groups), analysed for themes.\n\n**Mixed methods** combines both, e.g. a survey to measure how common a problem is plus interviews to understand why. Report qualitative work with COREQ." },
  { k: ["sampling", "sampling technique", "sampling method", "random sampling", "stratified", "cluster sampling", "convenience sampling", "purposive", "snowball", "systematic sampling"], t: "Sampling methods", a: "**Probability sampling** (everyone has a known chance of selection, so results generalise):\n- Simple random, systematic (every kth person), stratified (sample within groups), cluster (sample whole groups such as schools or communities).\n\n**Non-probability sampling** (quicker, but less generalisable):\n- Convenience, purposive (chosen for a reason, common in qualitative work), snowball (participants recruit others), quota.\n\nCluster samples need a larger size (design effect) and analysis that accounts for clustering." },
  { k: ["validity", "reliability", "internal validity", "external validity"], t: "Validity and reliability", a: "**Reliability** is consistency: does the tool give the same result when repeated? (e.g. test–retest, Cronbach's alpha for questionnaires).\n\n**Validity** is accuracy: does it measure what it claims to? **Internal validity** means your results are free of bias and confounding; **external validity** means they generalise to other people and settings.\n\nA tool can be reliable without being valid, but not valid without being reliable." },
  { k: ["research question", "research objective", "objectives", "pico", "peco", "problem statement"], t: "Research questions and objectives", a: "A good **research question** is specific, answerable and relevant. Frameworks help: **PICO** (Population, Intervention, Comparison, Outcome) for interventions and **PECO** (Exposure instead of Intervention) for observational studies.\n\nExample: \"Among adults in Ho (P), is high salt intake (E) compared with low intake (C) associated with hypertension (O)?\" **Objectives** then break the question into measurable steps. QuantAI's question builder helps you write them.", tool: ["method", "question", "Build a research question"] },
  { k: ["literature review", "systematic review", "meta-analysis", "meta analysis"], t: "Literature and systematic reviews", a: "A **literature review** summarises and critiques existing research on your topic to show what's known, the gaps, and why your study is needed. Organise it by themes, not paper by paper.\n\nA **systematic review** uses a pre-set, reproducible search and selection process (reported with **PRISMA**). A **meta-analysis** statistically combines the results of similar studies." },
  { k: ["ethics", "ethical approval", "informed consent", "consent", "irb", "ethical clearance"], t: "Research ethics", a: "Research with people needs **ethical approval** from an ethics committee (IRB) before data collection. Key principles: **informed consent** (participants understand the study and agree voluntarily), **confidentiality** (protecting identities and data), **minimising harm**, and the right to **withdraw** at any time.\n\nIn Ghana, studies are typically reviewed by a university or institutional ethics committee, or the Ghana Health Service Ethics Review Committee for health research." },
  { k: ["survival analysis", "kaplan-meier", "kaplan meier", "log-rank", "log rank", "cox regression", "hazard ratio", "time to event"], t: "Survival analysis", a: "**Survival analysis** studies the **time until an event** (death, relapse, default from care), allowing for people whose follow-up ends before the event (**censoring**).\n\n- **Kaplan–Meier curves** show the proportion event-free over time.\n- The **log-rank test** compares curves between groups.\n- **Cox regression** gives **hazard ratios** adjusted for other variables (HR > 1 means a higher rate of the event)." },
  { k: ["paired", "independent samples", "related samples", "repeated measures", "before and after"], t: "Paired vs independent data", a: "Data are **independent** when each group contains different people (men vs women). They are **paired (related)** when the same people are measured twice (before and after training) or are matched.\n\nPaired data need paired tests: paired t-test, Wilcoxon signed-rank, or McNemar's test for binary outcomes. Using an independent test on paired data is a common mistake." },
  { k: ["pilot study", "pre-test", "pretest", "pre-testing"], t: "Pilot study", a: "A **pilot study** (or pre-test) is a small trial run before the main study, often with 10–30 people similar to your target group. It checks whether questions are understood, how long data collection takes, and whether recruitment works.\n\nPilot participants are normally **not** included in the main analysis." },
  { k: ["questionnaire", "likert", "survey questions", "data collection tool"], t: "Designing a questionnaire", a: "A good **questionnaire** uses short, clear, neutral questions, one idea per question, in a logical order. Use **closed questions** (yes/no, multiple choice, **Likert scales** such as strongly disagree to strongly agree) for analysis, and a few open questions for detail.\n\nAvoid leading or double-barrelled questions, translate and back-translate if needed (e.g. into Twi or Ewe), and **pilot** it before use." },
]).map(e => Object.assign({ c: KB_GROUP[e.t] }, e, { k: e.k.concat([e.t.toLowerCase().replace(/\s*\(.*?\)/g, "")]) }));
const KB_TRIGGER = /\b(how (to|do i|should i|can i|do you) (write|structure|calculate|choose|interpret|report|analy[sz]e|cite|present|test|check|handle|use|do|get|find|search|select|deal)|steps? (to|for|in)|structure of|format of|tips? (for|on)|what should|chapter|what (is|are|does|do)|what's|whats|define|definition|meaning of|meaning|mean by|explain|difference between|differences between|how (do|does|to|should|can) (i |we |you )?(choose|use|interpret|calculate|report|write|conduct|do|know)|when (should|do|to) (i |we )?use|why (do|is|are)|types of|tell me about|examples? of)\b/i;
/** Up to three other handbook topics from the same group, as follow-up questions. */
function kbRelated(entry) {
  if (!entry.c) return [];
  const same = RESEARCH_KB.filter(e => e.c === entry.c && e.t !== entry.t);
  const start = same.length ? (entry.t.length * 7) % same.length : 0;
  return same.slice(start).concat(same.slice(0, start)).slice(0, 3).map(e => "Explain " + e.t.replace(/\s*\(.*?\)/g, "").toLowerCase());
}
function kbHits(low) {
  const hits = [];
  RESEARCH_KB.forEach(e => {
    let best = 0, pos = 1e9;
    e.k.forEach(k => {
      const at = low.indexOf(" " + k + " ") >= 0 ? low.indexOf(" " + k + " ") : low.indexOf(" " + k + "s ") >= 0 ? low.indexOf(" " + k + "s ") : (k.length > 6 ? low.indexOf(k) : -1);
      if (at < 0) return;
      // longer, more specific phrases score higher; topics named earlier in the question win ties
      const sc = k.length - at * 0.1;
      if (sc > best) { best = sc; pos = at; }
    });
    if (best > 0) hits.push({ e, sc: best, pos });
  });
  return hits.sort((x, y) => y.sc - x.sc);
}
function kbAnswer(text) {
  const low = " " + text.toLowerCase().replace(/[?!.,]/g, " ").replace(/\s+/g, " ") + " ";
  const asked = KB_TRIGGER.test(text) || text.trim().split(/\s+/).length <= 4;
  if (!asked) return null;
  // "difference between A and B" / "A vs B": look up each side on its own
  const cmp = low.match(/(?:difference|differences|compare|comparison|distinguish)\s+(?:between\s+)?(.+?)\s+(?:and|vs|versus|with)\s+(.+?)\s*$/) || low.match(/^\s*(?:what is\s+)?(.+?)\s+(?:vs|versus)\s+(.+?)\s*$/);
  if (cmp) {
    let a = cmp[1].replace(/^(a|an|the)\s+/, "").trim(), b = cmp[2].replace(/^(a|an|the)\s+/, "").trim();
    const tail = b.split(" ").slice(1).join(" ");
    if (a.split(" ").length === 1 && tail) a = a + " " + tail;   // "stratified and cluster sampling" -> "stratified sampling"
    const ha = kbHits(" " + a + " ")[0], hb = kbHits(" " + b + " ")[0];
    if (ha && hb && ha.e !== hb.e) return { t: ha.e.t + " vs " + hb.e.t, a: `**${ha.e.t}**\n\n${ha.e.a}\n\n**${hb.e.t}**\n\n${hb.e.a}`, tool: ha.e.tool || hb.e.tool, c: ha.e.c };
    if (ha && hb) return ha.e;
  }
  const hits = kbHits(low);
  return hits.length ? hits[0].e : null;
}

const DIRECT_REPLY = {
  compare: a => `Comparing **${vlabel(V(a.outcome) || { name: a.outcome })}** by **${vlabel(V(a.exposure) || { name: a.exposure })}**. QuantAI checked the assumptions and chose the test for you. The rules it applied are listed below the result.`,
  regression: a => `Looking at what is associated with **${vlabel(V(a.outcome) || { name: a.outcome })}**, adjusting for ${a.predictors.length} variables at once${a.model === "modpoisson" ? ", reported as prevalence ratios" : a.model === "logistic" ? ", reported as odds ratios" : ""}. Change the variables in the full result if you want a different model.`,
  survival: a => `Survival analysis of **${vlabel(V(a.event) || { name: a.event })}** over **${vlabel(V(a.time) || { name: a.time })}**${a.group ? ` by **${vlabel(V(a.group) || { name: a.group })}**` : ""}: Kaplan–Meier, log-rank test and Cox regression.`,
  forecast: a => `Forecasting **${vlabel(V(a.value) || { name: a.value })}** for the next ${a.horizon} periods.`,
  describe: a => `Here is a summary table of your variables${a.group ? ` by **${vlabel(V(a.group) || { name: a.group })}**` : ""}.`,
  open: a => ({ design: "The study design advisor can help with that: answer a few questions and it recommends a design, with the reasons.", sample: "The sample size calculator works that out for you, with adjustments for non-response.", test: "The test selector can choose that: tell it your outcome and comparison, and it picks the test and gives you the code.", question: "The research question builder turns your topic into a clear question, objectives and hypotheses.", checklist: "The reporting checklists cover STROBE, CONSORT, STARD, COREQ and PRISMA." })[a.tool] || "This tool can help with that.",
  check: () => "Here is QuantAI's check of your data.",
  fix: () => "Applying the suggested fixes.",
  report: () => "Preparing your report.",
};
function OFFLINE_HELP() {
  const vs = varsUsable(), demo = /\b(sex|gender|male|female|age)\b|_id$|\bid\b/i, nums = vs.filter(v => v.type === "continuous" && !demo.test(v.name + " " + v.label)), num = nums[0] || vs.find(v => v.type === "continuous"), cat = vs.find(v => ["binary", "categorical"].includes(v.type) && /\b(sex|gender)\b/i.test(v.name + " " + v.label)) || vs.find(v => ["binary", "categorical"].includes(v.type)), bins = vs.filter(v => v.type === "binary" && !demo.test(v.name + " " + v.label) && v !== cat), bin = mainOutcome() || bins[0];
  const ex = [num && cat ? `compare ${vlabel(num)} by ${vlabel(cat)}` : null, bin ? `risk factors for ${vlabel(bin)}` : null, "describe my data", "check my data for problems"].filter(Boolean);
  return { text: "I couldn't work that one out on my own, and the AI assistant isn't answering right now. Here's what I can do straight away: tap one, or attach your own data with the clip.", suggest: ex, tools: [["method", "question", "Plan a study: research question"], ["method", "design", "Choose a study design"], ["method", "sample", "Work out a sample size"]] };
}
/** Offline fallback for the most common request when the assistant can't be reached. */
function localIntent(text) {
  const t = text.trim(), low = t.toLowerCase();
  const clean = x => String(x || "").replace(/\b(use|using|as|with)\b.*$/i, "").replace(/[?.!,;:]+.*$/, "").replace(/^(the|a|an)\s+/i, "").trim();
  const preferPR = /prevalence ratio|modified poisson/i.test(t), preferOR = /odds ratio|logistic/i.test(t);
  // planning tools
  if (/\b(study design|which design|what design|design (for|of) (a|my|the) study)\b/i.test(t)) return { type: "open", tool: "design" };
  if (/\bsample size|how many (participants|people|respondents|subjects)\b/i.test(t)) return { type: "open", tool: "sample" };
  if (/\b(which|what) (statistical )?test\b|choose a test/i.test(t)) return { type: "open", tool: "test" };
  if (/\b(research question|objectives?|hypothes[ie]s)\b/i.test(t) && !/\b(test|compare|regression)\b/i.test(t)) return { type: "open", tool: "question" };
  if (/\b(checklist|strobe|consort|stard|coreq|prisma)\b/i.test(t)) return { type: "open", tool: "checklist" };
  // survival
  if (/\b(survival|kaplan|log-?rank|cox|hazard|time to|lost to follow|loss to follow|drop ?out)\b/i.test(t)) {
    const f = S.form.survival || {}, by = t.match(/\b(?:by|between|among (?:people|those|patients) with|for (?:people|those|patients) with|with)\s+(.+?)[?.!]*$/i);
    const tv = (f.time && V(f.time)) || varsUsable().find(v => ["continuous", "count"].includes(v.type) && /time|month|day|week|year|follow|duration/i.test(v.name + " " + v.label));
    const ev = (f.event && V(f.event)) || varsUsable().find(v => v.type === "binary" && /event|death|died|dead|lost|relapse|status|default/i.test(v.name + " " + v.label));
    const g = by ? resolveVar(clean(by[1]), ["binary", "categorical"]) : (f.group ? V(f.group) : null);
    if (tv && ev) return { type: "survival", time: tv.name, event: ev.name, group: g ? g.name : "", covariates: f.covs || [] };
  }
  // forecast
  const fm = t.match(/\b(?:forecast|project|predict(?: the)? (?:next|future))\s+(.+?)(?:\s+for\b.*)?[?.!]*$/i);
  if (fm) { const v = resolveVar(clean(fm[1]), ["continuous", "count"]); if (v) return { type: "forecast", value: v.name, horizon: (t.match(/(\d+)\s*(months?|weeks?|periods?|years?|days?)/i) || [])[1] || 12 }; }
  // risk factors / what affects X -> regression
  const rm = t.match(/(?:risk factors?|factors?|predictors?|determinants?|what)\s+(?:are\s+|is\s+)?(?:associated with|related to|linked to|for|of|affecting|that affect|affects?|drives?|influences?|predicts?|explains?)\s+(.+)$/i) || t.match(/^(?:predict|model)\s+(.+)$/i);
  if (rm) {
    const out = resolveVar(clean(rm[1]));
    if (out) {
      const f = S.form.regression || {};
      let preds = f.outcome === out.name && f.preds && f.preds.length ? f.preds.slice()
        : varsUsable().filter(v => v.name !== out.name && ["binary", "categorical", "continuous", "count", "ordinal"].includes(v.type) && !/id$|_id|time|month|follow|date/i.test(v.name) && (v.nMissing || 0) < S.ds.nRows * 0.3).slice(0, 6).map(v => v.name);
      if (preds.length) return { type: "regression", outcome: out.name, predictors: preds, model: preferPR ? "modpoisson" : preferOR ? "logistic" : (f.outcome === out.name && f.model) || "auto" };
    }
  }
  const m = text.match(/(?:compare|difference in|differ(?:ence)?s? in)\s+(.+?)\s+(?:by|between|across|among)\s+(.+?)[?.!]*$/i);
  if (/\b(check|clean|problems?|issues?|quality)\b.*\b(data|file|dataset)\b|\b(data|file)\b.*\b(check|clean|problems?|issues?)\b/i.test(text)) return { type: "check" };
  if (/^\s*(fix|apply)\b.*\b(them|all|fixes|problems?|data|it)\b/i.test(text)) return { type: "fix" };
  if (/\b(download|export|make|create|give)\b.*\breport\b/i.test(text)) return { type: "report" };
  if (/\b(describe|summari[sz]e|table ?1|overview|summary)\b/i.test(text)) { const g = text.match(/\bby\s+(.+?)[?.!]*$/i), gv = g ? resolveVar(clean(g[1]), ["binary", "categorical"]) : null; return { type: "describe", vars: [], group: gv ? gv.name : "" }; }
  // whole-dataset requests: check, Table 1, then the main outcome's associated factors
  if (/\b(analy[sz]e|analysis of|run (an )?analysis on|work on|look (at|into)|go through|examine|explore)\b.*\b(my|the|this|whole|entire|all)?\s*(data|dataset|file|survey|everything)\b|\bfull analysis\b|\banaly[sz]e (it|everything)\b/i.test(t)) {
    const g = mainGroup(), o = mainOutcome(), acts = [{ type: "check" }, { type: "describe", vars: [], group: (o || g) ? (o || g).name : "" }];
    if (o) {
      const preds = varsUsable().filter(v => v.name !== o.name && ["binary", "categorical", "continuous", "count", "ordinal"].includes(v.type) && !/time|date|comment|remark|_id$/i.test(v.name) && (v.nMissing || 0) < S.ds.nRows * 0.3 && !(v.levels && v.levels.length > 8)).slice(0, 7).map(v => v.name);
      if (preds.length) acts.push({ type: "regression", outcome: o.name, predictors: preds, model: "auto" });
    }
    return acts;
  }
  // "is X associated / related / linked with Y", "relationship between X and Y", "does X affect Y"
  const am = t.match(/(?:is|are|does|do)\s+(?:there\s+(?:an?\s+)?(?:association|relationship|link)\s+between\s+)?(.+?)\s+(?:associated|related|linked|correlated|connected)\s+(?:with|to)\s+(.+?)[?.!]*$/i)
    || t.match(/(?:association|relationship|link|correlation)\s+between\s+(.+?)\s+and\s+(.+?)[?.!]*$/i)
    || t.match(/^(?:does|do|did|can)\s+(.+?)\s+(?:affect|influence|predict|determine|impact)\s+(.+?)[?.!]*$/i);
  if (am) {
    let a = resolveVar(clean(am[1])), b = resolveVar(clean(am[2]));
    if (a && b && a.name !== b.name) {
      if (/affect|influence|predict|determine|impact/i.test(t) && /^(does|do|did|can)/i.test(t)) [a, b] = [b, a];   // "does X affect Y": Y is the outcome
      const num = x => ["continuous", "count"].includes(x.type);
      if (!num(a) && num(b)) [a, b] = [b, a];   // a numeric variable is compared across the groups of the other
      return { type: "compare", outcome: a.name, exposure: b.name, paired: false };
    }
  }
  // "is X higher in/among Y", "does X differ by Y", "X vs Y"
  const dm = t.match(/(?:is|are|does|do)\s+(.+?)\s+(?:higher|lower|different|differ|more|less|greater)\s+(?:in|among|for|by|between|across)\s+(.+?)[?.!]*$/i) || t.match(/^(.+?)\s+(?:vs\.?|versus)\s+(.+?)[?.!]*$/i);
  if (dm) { const o = resolveVar(clean(dm[1])), e = resolveVar(clean(dm[2])); if (o && e && o.name !== e.name) return { type: "compare", outcome: o.name, exposure: e.name, paired: false }; }
  if (m) { const o = resolveVar(m[1]), e = resolveVar(m[2].replace(/^(the )?(groups? of |categories of )/i, "").replace(/\b(men and women|males? and females?)\b/i, "sex")); if (o && e) return { type: "compare", outcome: o.name, exposure: e.name, paired: false }; }
  return null;
}
function parseAgent(reply) {
  const t = String(reply || "").trim();
  const tryJ = s => { try { return JSON.parse(s); } catch (e) { return null; } };
  let j = tryJ(t); if (!j) { const i = t.indexOf("{"), k = t.lastIndexOf("}"); if (i >= 0 && k > i) j = tryJ(t.slice(i, k + 1)); }
  if (j && typeof j.reply === "string") return { reply: j.reply, action: j.action && typeof j.action === "object" ? j.action : null };
  return { reply: t, action: null };
}
function agentContext() {
  const parts = [dataContext(), "", "METHODOLOGY INPUTS:", methodsContext()];
  return parts.join("\n").slice(0, 11900);
}
async function sendChatMessage(text) {
  text = (text || "").trim(); if (!text || S.chatBusy) return;
  const c = curChat() || newChat();
  c.messages.push({ role: "user", content: text });
  if (c.title === "New chat") c.title = text.slice(0, 60);
  S.chatBusy = true; saveChats(); render();
  let reply, action, note = null, helpExtra = null;
  const deeper = /^in more depth:\s*/i.test(text);
  const kbFirst = !deeper && KB_TRIGGER.test(text) ? kbAnswer(text) : null;
  const direct = kbFirst || deeper ? null : localIntent(text);
  if (Array.isArray(direct)) { reply = "Here is a full first pass: a check of the file, a summary table, and the factors associated with the main outcome. Ask me to compare anything else, or tap \"Open full result and code\" on any result."; action = direct; }
  else if (kbFirst) {
    reply = kbFirst.a; action = null;
    helpExtra = { suggest: kbRelated(kbFirst).concat(["In more depth: " + text]), tools: kbFirst.tool ? [kbFirst.tool] : [] };
  } else if (direct && !Array.isArray(direct)) {
    reply = DIRECT_REPLY[direct.type] ? DIRECT_REPLY[direct.type](direct) : "Done.";
    action = direct;
  } else try {
    const hist = c.messages.filter(m => m.role === "user" || m.role === "assistant").slice(-10).map(m => ({ role: m.role, content: m.role === "assistant" && m.action ? `${m.content}\n[QuantAI ran: ${JSON.stringify(m.action)}]` : m.content }));
    ({ reply, action } = parseAgent(await askAI("agent", agentContext(), hist)));
  } catch (e) {
    const kbHit = KB_TRIGGER.test(text) ? kbAnswer(text) : null;
    const local = kbHit ? null : localIntent(text);
    if (local) { reply = "I ran this directly with QuantAI's rules."; action = local; }
    else { const kb = kbAnswer(text); if (kb) { reply = kb.a; action = null; helpExtra = { suggest: [], tools: kb.tool ? [kb.tool] : [] }; } else { const h = OFFLINE_HELP(); reply = h.text; action = null; note = "help"; helpExtra = h; } }
  }
  const msg = { role: "assistant", content: reply, action: action || null, note };
  if (helpExtra) { msg.suggest = helpExtra.suggest; msg.tools = helpExtra.tools; }
  const acts = Array.isArray(action) ? action.filter(a => a && typeof a === "object").slice(0, 4) : action ? [action] : [];
  if (acts.length) msg.action = acts.length === 1 ? acts[0] : acts;
  const errs = [];
  acts.forEach(a => {
    try {
      const r = runAction(a);
      if (r.entry) (msg.refs = msg.refs || []).push(r.entry.id);
      if (r.forecast) msg.forecast = true;
      if (r.check) msg.check = true;
      if (r.fixed) msg.content += r.fixed.length ? "\n\n" + r.fixed.map(x => "- " + x).join("\n") : "\n\nThere was nothing that needed an automatic fix.";
      if (r.report) msg.content += "\n\nThe report (every analysis with its tables, rules and write-up) has been downloaded as **quantai-report.html**. Open it in a browser and print to PDF, or copy the tables into Word.";
      if (r.nav) { const list = r.nav[0] === "method" ? METHOD_TOOLS : DATA_TOOLS, t = list.find(x => x.id === r.nav[1]); msg.tools = [[r.nav[0], r.nav[1], t ? "Open " + t.label.toLowerCase() : "Open tool"]]; }
    } catch (err) { errs.push(err.user ? err.message : "The analysis failed: " + err.message); }
  });
  if (msg.refs) msg.ref = msg.refs[0];
  if (errs.length) { msg.content = (msg.content ? msg.content + "\n\n" : "") + errs.join("\n\n"); msg.note = msg.refs || msg.check ? null : "error"; msg.failed = !msg.refs && !msg.check && !msg.forecast; }
  c.messages.push(msg); S.chatBusy = false; saveChats();
  render();
}

/* ---------- rendering ---------- */
function chatResultCard(m) {
  if ((m.refs && m.refs.length > 1) || (m.check && m.ref)) return (m.refs || [m.ref]).map(id => chatResultCard(Object.assign({}, m, { refs: null, ref: id, check: false, forecast: false, action: null }))).join("") + (m.check ? chatResultCard(Object.assign({}, m, { refs: null, ref: null })) : "");
  if (m.check) return `<div class="qa-result"><div class="res-head"><h3>Data check: ${esc(S.dsSource)}</h3><span class="pill num">${S.ds.nRows} rows · ${S.ds.vars.length} variables</span></div>${qualityHTML(true)}<div class="row"><button class="btn quiet sm" data-act="chat-open" data-sec="data" data-tool="data">Review variable types</button></div></div>`;
  if (m.forecast && S.fc.res) {
    const r = S.fc.res, hw = r.hw, Q = window.QuantAI;
    return `<div class="qa-result"><div class="res-head"><h3>Forecast of ${esc(vlabel(V(r.yv)))}</h3><span class="pill acc">${hw.seasonal ? "Holt-Winters (seasonal)" : "Holt's linear trend"}</span></div>
      ${tableHTML({ title: "Next periods", columns: ["Period", "Forecast", "80% range"], rows: hw.forecast.slice(0, 6).map((v, i) => [r.fx[i], Q.fmt(v), `${Q.fmt(hw.lower[i])} to ${Q.fmt(hw.upper[i])}`]) })}
      <div class="row"><button class="btn quiet sm" data-act="chat-open" data-sec="data" data-tool="forecast">Open the full forecast</button></div></div>`;
  }
  const en = m.ref && S.log.find(x => x.id === m.ref);
  if (!en) return m.action && !m.failed && (Array.isArray(m.action) || !["open", "report", "check", "fix"].includes(m.action.type)) ? `<div class="notice info" style="font-size:.84rem">${esc(Array.isArray(m.action) ? m.action.length + " analyses" : ACTION_LABEL[m.action.type] || "Analysis")} from an earlier visit. Results aren't stored, so the data must be loaded again. <button class="btn quiet sm" data-act="chat-rerun" data-i="${esc(JSON.stringify(m.action))}">Run it again</button></div>` : "";
  const r = en.result, main = r.tables.find(t => /coefficient|ratio|Fixed|Hazard|t-test|ANOVA|Kruskal|Mann|chi|Fisher|correlation|McNemar|Wilcoxon|Table 1/i.test(t.title)) || r.tables[0];
  const ex = S.explain[en.id] || {};
  return `<div class="qa-result">
    <div class="res-head"><div class="stack" style="gap:.15rem"><span class="sect-t">Analysis ${en.k}</span><h3>${esc(r.title)}</h3></div><div class="meta"><span class="pill acc">${esc(r.method)}</span>${r.n ? `<span class="pill num">n = ${r.n}</span>` : ""}</div></div>
    <details class="fold"><summary>Rules applied (${r.decision.length})</summary>${auditHTML(r.decision)}</details>
    ${tableHTML(Object.assign({}, main, { rows: main.rows.slice(0, 12) }))}
    ${r.writeup ? `<p class="apa">${esc(r.writeup)}</p>` : ""}
    ${ex.text ? `<div class="stack" style="background:var(--surface-2);padding:.7rem .9rem;border-radius:10px">${aiFormat(ex.text)}</div>` : ""}${ex.err ? `<div class="notice bad">${esc(ex.err)}</div>` : ""}
    <div class="row"><button class="btn quiet sm" data-act="chat-open" data-sec="data" data-tool="${en.kind}" data-id="${en.id}">Open full result and code</button>${r.writeup ? `<button class="btn quiet sm" data-act="copy-apa" data-id="${en.id}">Copy write-up</button>` : ""}<button class="btn ghost sm" data-act="ai-explain" data-id="${en.id}" ${ex.busy ? "disabled" : ""}>${ex.busy ? '<span class="spin"></span> Explaining…' : "Explain in plain language"}</button></div>
  </div>`;
}
document.addEventListener("click", e => {
  const el = e.target.closest("[data-act]"); if (!el) return;
  const act = el.dataset.act;
  if (act === "chat-del") { e.preventDefault(); e.stopPropagation(); S.chats = S.chats.filter(x => x.id !== el.dataset.id); if (S.chatId === el.dataset.id) S.chatId = S.chats.length ? S.chats[0].id : null; saveChats(); render(); return; }
  if (act === "chat-new") { e.preventDefault(); newChat(); render(); const i = $("#qa-chat-input"); if (i) i.focus(); }
  if (act === "chat-pick") { e.preventDefault(); S.chatId = el.dataset.id; render(); }
  if (act === "chat-starter") { e.preventDefault(); sendChatMessage(el.dataset.text); }
  if (act === "chat-clear") { e.preventDefault(); const c = curChat(); if (c) { c.messages = []; c.title = "New chat"; saveChats(); render(); } }
  if (act === "chat-open") { e.preventDefault(); const id = +el.dataset.id; const en = S.log.find(x => x.id === id); if (en) S.cur[en.kind] = en; S.section = el.dataset.sec; S.tool[el.dataset.sec] = el.dataset.tool; render(); const mm = document.getElementById("qa-main"); if (mm) mm.scrollTop = 0; }
  if (act === "chat-rerun") { e.preventDefault(); const a0 = JSON.parse(el.dataset.i), c = curChat(), list = Array.isArray(a0) ? a0 : [a0], refs = []; let fc = false, ck = false; const errs = [];
    list.forEach(a => { try { const r = runAction(a); if (r.entry) refs.push(r.entry.id); fc = fc || !!r.forecast; ck = ck || !!r.check; } catch (err) { errs.push(err.message); } });
    c.messages.push({ role: "assistant", content: (refs.length || fc || ck ? `Ran ${list.length > 1 ? "the " + list.length + " analyses" : ACTION_LABEL[list[0].type] || "the analysis"} again with the current data.` : "") + (errs.length ? "\n\n" + errs.join("\n\n") : ""), action: a0, refs, ref: refs[0] || null, forecast: fc, check: ck, note: refs.length || fc || ck ? null : "error" }); saveChats(); render(); }
});
document.addEventListener("submit", e => {
  if (e.target.id !== "qa-chat-form") return;
  e.preventDefault(); const i = $("#qa-chat-input"); const t = i.value; i.value = ""; sendChatMessage(t);
});
document.addEventListener("keydown", e => {
  if (e.target.id === "qa-chat-input" && e.key === "Enter" && !e.shiftKey) { e.preventDefault(); const t = e.target.value; e.target.value = ""; sendChatMessage(t); }
});
document.addEventListener("change", async e => {
  if (e.target.id !== "qa-chat-file" || !e.target.files[0]) return;
  const f = e.target.files[0]; e.target.value = "";
  await readFile(f);
  pushUploadMessage(f.name);
  render();
});

const OUTCOME_RE = /hypertens|diabet|disease|status|outcome|positive|diagnos|infect|malaria|death|died|pass|default|anaemi|anemi|stunt|wasting|underweight|obes|overweight|complication|adheren|compliance|uptake|utili[sz]|satisf|depress|anxiety|stress|delivery|vaccin|immuni[sz]|test(ed)?|screen|use of|using|practice|knowledge|awareness/i;
const DEMO_RE = /\b(sex|gender|male|female|age|marital|religion|ethnic|tribe|occupation|region|district|residence|education|income|household|name|phone|date|comment|remark|interviewer)\b|_id$|\bid\b|s\/n/i;
/** The variable a study is most likely about: a yes/no question such as "Do you have hypertension?". */
function mainOutcome() {
  const vs = varsUsable(), bins = vs.filter(v => v.type === "binary");
  const txt = v => v.name + " " + v.label;
  return bins.find(v => /^(do|did|have|has|are|is|were|was)\b/i.test(v.label) && OUTCOME_RE.test(txt(v)) && !DEMO_RE.test(txt(v)))
    || bins.find(v => OUTCOME_RE.test(txt(v)) && !DEMO_RE.test(txt(v)) && !/knowledge|q\d/i.test(txt(v)))
    || bins.find(v => /^(do|did|have|has|are|is|were|was)\b/i.test(v.label) && !DEMO_RE.test(txt(v)))
    || bins.find(v => !DEMO_RE.test(txt(v))) || null;
}
const mainGroup = () => varsUsable().find(v => ["binary", "categorical"].includes(v.type) && /\b(sex|gender)\b/i.test(v.name + " " + v.label)) || null;
/** A real time series: one row per date (monthly sales, weekly cases), not survey interview dates. */
function isTimeSeries() {
  const d = fcDateVars()[0]; if (!d) return false;
  const vals = d.values.filter(x => x !== null && x !== undefined), u = new Set(vals.map(String));
  return vals.length >= 8 && u.size >= 0.9 * vals.length;
}
/** Questions that make sense for the loaded file. */
function dataSuggestions(fixable) {
  const vs = varsUsable(), demo = /\b(sex|gender|male|female|age)\b|_id$|\bid\b/i;
  const num = vs.filter(v => v.type === "continuous" && !demo.test(v.name + " " + v.label))[0] || vs.find(v => v.type === "continuous");
  const grp = vs.find(v => ["binary", "categorical"].includes(v.type) && /\b(sex|gender)\b/i.test(v.name + " " + v.label)) || vs.find(v => ["binary", "categorical"].includes(v.type));
  const outc = mainOutcome();
  const out = [];
  if (fixable) out.push("Fix them");
  out.push("Analyse my data");
  out.push("Describe my data" + (grp ? " by " + vlabel(grp) : ""));
  if (outc) out.push(`What factors are associated with ${vlabel(outc)}`);
  else if (num) out.push(`What affects ${vlabel(num)}?`);
  if (outc && grp && outc !== grp) out.push(`Is ${vlabel(outc).replace(/^(do you have|have you|are you)\s+/i, "").replace(/\?$/, "")} associated with ${vlabel(grp)}?`);
  else if (num && grp) out.push(`Compare ${vlabel(num)} by ${vlabel(grp)}`);
  if (isTimeSeries() && num) out.push(`Forecast ${vlabel(num)}`);
  return out.slice(0, 5);
}
/** After an upload: say what was read, what was fixed automatically, and what needs a decision. */
function pushUploadMessage(fname) {
  const c = curChat() || newChat();
  if (S.errors.data) { c.messages.push({ role: "assistant", content: S.errors.data, note: "error" }); saveChats(); return; }
  const q = S.quality || [], warn = q.filter(x => x.sev === "warn"), fixable = q.filter(x => x.fix);
  const types = {}; S.ds.vars.forEach(v => { types[v.type] = (types[v.type] || 0) + 1; });
  const tl = ["continuous", "count", "binary", "categorical", "ordinal", "id"].filter(t => types[t]).map(t => `${types[t]} ${t === "id" ? "ignored (IDs or free text)" : t}`).join(", ");
  if (c.title === "New chat") c.title = fname;
  let text = `Loaded **${fname}**: ${S.ds.nRows.toLocaleString()} rows and ${S.ds.vars.length} variables (${tl}).\n\n`;
  text += warn.length ? `I checked the file and found **${warn.length} thing${warn.length > 1 ? "s" : ""} to review** before analysing${fixable.length ? `; ${fixable.length === 1 ? "one has" : fixable.length + " have"} a one-click fix, or say "fix them"` : ""}.` : "I checked the file and found nothing that needs fixing.";
  text += " Then ask me what you'd like to find out, or tap a suggestion.";
  c.messages.push({ role: "assistant", content: text, check: true, action: { type: "check" }, suggest: dataSuggestions(fixable.length) });
  saveChats();
}

/* =====================================================================
   QuantAI app shell (website build): a full-screen chat layout.
   Sidebar: new chat, tools, chat history, data. Main: the conversation,
   or a tool page with a way back to the chat.
   ===================================================================== */
const ICON = {
  plus: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  menu: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg>',
  x: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  up: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
  clip: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>',
  back: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>',
  book: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5v14Z"/><path d="M20 17v4H6.5A2.5 2.5 0 0 1 4 18.5"/></svg>',
  table: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/></svg>',
  chat: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
  trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>',
  home: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/></svg>',
};
S.sideOpen = false;
S.grpOpen = {};
const grpIsOpen = sec => sec in S.grpOpen ? S.grpOpen[sec] : S.section === sec;
const toolLabel = () => { const list = S.section === "method" ? METHOD_TOOLS : DATA_TOOLS; const t = list.find(x => x.id === S.tool[S.section]); return t ? t.label : ""; };

function sideHTML() {
  const toolBtn = (sec, t) => `<button class="side-item ${S.section === sec && S.tool[sec] === t.id ? "on" : ""}" data-act="tool-open" data-sec="${sec}" data-tool="${t.id}">${esc(t.label)}</button>`;
  return `<div class="side-head">
      <a class="side-brand" href="index.html" title="Model Analysis Hub home"><img src="assets/img/logo-mark.webp" alt="" width="28" height="28"><span>QuantAI</span></a>
      <button class="icon-btn side-x" data-act="side-close" aria-label="Close sidebar">${ICON.x}</button></div>
    <button class="side-new" data-act="chat-new">${ICON.plus}<span>New chat</span></button>
    <div class="side-scroll">
      <div class="side-h">Chats</div>
      ${S.chats.length ? S.chats.map(c => `<div class="side-chat ${S.section === "chat" && c.id === S.chatId ? "on" : ""}"><button class="side-item" data-act="chat-pick" data-id="${c.id}" title="${esc(c.title)}">${esc(c.title)}</button><button class="icon-btn sm" data-act="chat-del" data-id="${c.id}" aria-label="Delete chat">${ICON.trash}</button></div>`).join("") : `<p class="side-empty">Your chats will appear here.</p>`}
      <details class="side-more" ${S.section !== "chat" ? "open" : ""}><summary>More tools</summary>
        <details class="side-grp" data-grp="method" ${grpIsOpen("method") ? "open" : ""}><summary>${ICON.book}<span>Plan a study</span></summary>${METHOD_TOOLS.map(t => toolBtn("method", t)).join("")}</details>
        <details class="side-grp" data-grp="data" ${grpIsOpen("data") ? "open" : ""}><summary>${ICON.table}<span>Analyse data</span></summary>${DATA_TOOLS.map(t => toolBtn("data", t)).join("")}</details>
      </details>
    </div>
    <div class="side-foot">
      <div class="side-data"><span class="side-h" style="padding:0">Data in use</span><b title="${esc(S.dsSource)}">${esc(S.dsSource)}</b><small>${S.ds ? `${S.ds.nRows.toLocaleString()} rows · ${S.ds.vars.length} variables` : ""}</small>
        <div class="side-data-btns"><label class="side-link-btn">Upload data<input type="file" id="qa-side-file" accept=".csv,.txt,.tsv,.xlsx,.xls" hidden></label><button class="side-link-btn" data-act="tool-open" data-sec="data" data-tool="data">Variables</button></div></div>
      <div class="side-links"><a href="index.html">${ICON.home}Model Analysis Hub</a><a href="https://wa.me/233595586430" target="_blank" rel="noopener">Talk to Amadu</a></div>
    </div>`;
}
function composerHTML(big) {
  return `<form id="qa-chat-form" class="composer ${big ? "big" : ""}">
      <label class="icon-btn attach" title="Upload a CSV or Excel file">${ICON.clip}<input type="file" id="qa-chat-file" accept=".csv,.txt,.tsv,.xlsx,.xls" hidden><span class="sr">Upload data</span></label>
      <textarea id="qa-chat-input" rows="1" maxlength="580" placeholder="Ask a question, or attach your data" aria-label="Message QuantAI"></textarea>
      <button type="submit" class="send" aria-label="Send" ${S.chatBusy ? "disabled" : ""}>${ICON.up}</button>
    </form>`;
}
const SUGGEST = () => S.isExample && /hypertension/i.test(S.dsSource) ? [
  ["Compare blood pressure", "between men and women", "Compare systolic BP between men and women"],
  ["Find risk factors", "for hypertension, as prevalence ratios", "Which factors are associated with hypertension? Use prevalence ratios."],
  ["Survival analysis", "loss to follow-up by hypertension", "Is loss to follow-up higher among people with hypertension?"],
  ["Plan a study", "design for salt intake and hypertension", "What study design fits a study of salt intake and hypertension?"],
] : [["Describe my data", "a summary table of every variable", "Describe my data"], ["Find relationships", "with my main outcome", "Which variables are related to my main outcome?"], ["Choose a test", "for my research question", "Which test should I use for my research question?"], ["Report results", "in APA style", "How should I report these results?"]];

function renderChatMain() {
  const c = curChat(), msgs = c ? c.messages : [];
  if (!msgs.length && !S.chatBusy) {
    return `<div class="chat-empty">
      <div class="chat-hello"><img src="assets/img/logo-mark.webp" alt="" width="44" height="44"><h1>What would you like to find out?</h1>
        <p>Type your research topic or question, or attach your data with the clip. QuantAI runs the statistics, checks the assumptions and explains the results.</p></div>
      ${composerHTML(true)}
      <div class="suggest">${SUGGEST().map(([a, b, t]) => `<button type="button" data-act="chat-starter" data-text="${esc(t)}"><b>${esc(a)}</b><span>${esc(b)}</span></button>`).join("")}</div>
      <p class="chat-data">Using <b>${esc(S.dsSource)}</b>. Attach your own CSV or Excel file with the clip, or from the sidebar.</p>
    </div>`;
  }
  return `<div class="chat-thread" id="qa-thread" aria-live="polite">
      ${msgs.map(m => m.role === "user" ? `<div class="msg you"><div class="bubble">${esc(m.content)}</div></div>`
        : `<div class="msg bot"><img class="av" src="assets/img/logo-mark.webp" alt="" width="28" height="28"><div class="body"><div class="text ${m.note === "error" ? "err" : ""}">${aiFormat(m.content)}</div>${chatResultCard(m)}${msgChips(m)}</div></div>`).join("")}
      ${S.chatBusy ? `<div class="msg bot"><img class="av" src="assets/img/logo-mark.webp" alt="" width="28" height="28"><div class="body"><div class="typing" aria-label="QuantAI is thinking"><i></i><i></i><i></i></div></div></div>` : ""}
    </div>
    <div class="chat-dock">${composerHTML(false)}<p class="disclaimer">QuantAI's engine computes every number. Explanations come from an AI assistant and can be wrong, so check anything important.</p></div>`;
}
function msgChips(m) {
  const sug = (m.suggest || []).map(t => /^in more depth:/i.test(t) ? `<button type="button" class="chip tool" data-act="chat-starter" data-text="${esc(t)}">Go deeper with AI →</button>` : `<button type="button" class="chip" data-act="chat-starter" data-text="${esc(t)}">${esc(t)}</button>`).join("");
  const tools = (m.tools || []).map(([sec, id, label]) => `<button type="button" class="chip tool" data-act="chat-open" data-sec="${esc(sec)}" data-tool="${esc(id)}">${esc(label)} →</button>`).join("");
  return sug || tools ? `<div class="chips">${sug}${tools}</div>` : "";
}
function topHTML() {
  if (S.section === "chat") { const c = curChat(); return `<button class="icon-btn only-m" data-act="side-open" aria-label="Open sidebar">${ICON.menu}</button><span class="top-title">${esc(c && c.messages.length ? c.title : "QuantAI")}</span><button class="icon-btn only-m" data-act="chat-new" aria-label="New chat">${ICON.plus}</button>`; }
  return `<button class="icon-btn only-m" data-act="side-open" aria-label="Open sidebar">${ICON.menu}</button><button class="back-btn" data-act="to-chat">${ICON.back}<span>Back to chat</span></button><span class="top-title">${esc(S.section === "method" ? "Plan a study" : "Analyse data")} · ${esc(toolLabel())}</span>`;
}
const _baseRender = render;
function appRender() {
  const side = $("#qa-side"), main = $("#qa-main"), top = $("#qa-top"); if (!side || !main) return _baseRender();
  side.innerHTML = sideHTML(); side.classList.toggle("open", S.sideOpen);
  const scrim = $("#qa-scrim"); if (scrim) scrim.hidden = !S.sideOpen;
  top.innerHTML = topHTML();
  if (S.section === "chat") {
    codeRegistry.clear();
    main.className = "qa-main is-chat"; main.innerHTML = renderChatMain();
    main.scrollTop = main.scrollHeight;
    persist();
  } else {
    main.className = "qa-main is-tool";
    _baseRender();
    const wrap = document.createElement("div"); wrap.className = "tool-wrap";
    while (main.firstChild) wrap.appendChild(main.firstChild);
    main.appendChild(wrap);
  }
  autoGrow();
}
render = appRender;
function autoGrow() { const t = $("#qa-chat-input"); if (!t) return; t.style.height = "auto"; t.style.height = Math.min(220, t.scrollHeight) + "px"; }
function scrollMainTop() { const m = $("#qa-main"); if (m) m.scrollTop = 0; }

document.addEventListener("click", e => {
  const el = e.target.closest("[data-act]"); if (!el) return;
  const act = el.dataset.act;
  if (act === "tool-open") { e.preventDefault(); S.section = el.dataset.sec; S.tool[S.section] = el.dataset.tool; S.sideOpen = false; render(); scrollMainTop(); }
  if (act === "to-chat") { e.preventDefault(); S.section = "chat"; render(); }
  if (act === "side-open") { e.preventDefault(); S.sideOpen = true; render(); }
  if (act === "side-close") { e.preventDefault(); S.sideOpen = false; render(); }
  if (act === "chat-new" || act === "chat-pick") { S.section = "chat"; S.sideOpen = false; setTimeout(() => { render(); const i = $("#qa-chat-input"); if (i && window.innerWidth > 900) i.focus(); }, 0); }
  if (act === "chat-starter" || act === "chat-rerun") { S.section = "chat"; }
});
document.addEventListener("input", e => { if (e.target.id === "qa-chat-input") autoGrow(); });
document.addEventListener("change", async e => {
  if (e.target.id !== "qa-side-file" || !e.target.files[0]) return;
  const f = e.target.files[0]; e.target.value = ""; S.sideOpen = false;
  await readFile(f);
  if (!S.errors.data) { pushUploadMessage(f.name); S.section = "chat"; }
  render();
});
document.addEventListener("keydown", e => { if (e.key === "Escape" && S.sideOpen) { S.sideOpen = false; render(); } });
document.addEventListener("click", e => { if (e.target.id === "qa-scrim") { S.sideOpen = false; render(); } });
document.addEventListener("toggle", e => { const d = e.target; if (d.classList && d.classList.contains("side-grp")) S.grpOpen[d.dataset.grp] = d.open; }, true);

/* ---------- start-up ---------- */
function start() {
  DATA_TOOLS.filter(t => /^\d+$/.test(t.k)).forEach((t, i) => t.k = String(i + 1).padStart(2, "0"));
  document.addEventListener("click", onClick);
  document.addEventListener("change", onChange);
  document.addEventListener("input", onInput);
  const drop = () => $("#drop");
  document.addEventListener("dragover", e => { const d = drop(); if (d && d.contains(e.target)) { e.preventDefault(); d.classList.add("over"); } });
  document.addEventListener("dragleave", e => { const d = drop(); if (d && d.contains(e.target)) d.classList.remove("over"); });
  document.addEventListener("drop", e => { const d = drop(); if (d && d.contains(e.target)) { e.preventDefault(); d.classList.remove("over"); if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]); } });
  document.addEventListener("keydown", e => { if (e.target.id === "drop" && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); $("#file").click(); } });
  loadExample();
  // a worked example in the results so every page opens in a real working state
  ["compare", "regression", "describe", "survival"].forEach(k => { try { const params = JSON.parse(JSON.stringify(S.form[k])); const result = runAnalysis(k, params); const en = { id: ++S.logSeq, k: 0, kind: k, params, result }; S.log.push(en); S.cur[k] = en; } catch (e) { console.error(e); } });
  S.log.sort((a, b) => ["describe", "compare", "regression", "survival"].indexOf(a.kind) - ["describe", "compare", "regression", "survival"].indexOf(b.kind)); renumber();
  render();
}
start();

})();

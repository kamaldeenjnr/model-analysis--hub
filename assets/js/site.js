/* Model Analysis Hub — small site script (menu, header, reveal, contact form) */
(function () {
  'use strict';
  document.documentElement.classList.remove('no-js');

  var header = document.querySelector('.site-header');
  function onScroll() { if (header) header.classList.toggle('scrolled', window.scrollY > 8); }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  var menuBtn = document.querySelector('.menu-btn');
  if (menuBtn) {
    menuBtn.addEventListener('click', function () {
      var open = document.body.classList.toggle('menu-open');
      menuBtn.setAttribute('aria-expanded', String(open));
      menuBtn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && document.body.classList.contains('menu-open')) {
        document.body.classList.remove('menu-open');
        menuBtn.setAttribute('aria-expanded', 'false');
        menuBtn.focus();
      }
    });
    document.querySelectorAll('.nav a').forEach(function (a) {
      a.addEventListener('click', function () { document.body.classList.remove('menu-open'); menuBtn.setAttribute('aria-expanded', 'false'); });
    });
  }

  /* Gentle reveal, staggered among siblings */
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var items = document.querySelectorAll('.rv');
  var seen = new Map();
  items.forEach(function (el) {
    var p = el.parentElement, i = seen.get(p) || 0;
    el.style.setProperty('--d', Math.min(i * 0.07, 0.35) + 's');
    seen.set(p, i + 1);
  });
  if (!reduce && 'IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -6% 0px', threshold: 0.06 });
    items.forEach(function (el) { io.observe(el); });
    /* Safety net: anything already scrolled past (anchor jumps, fast scrolls) is shown too */
    var sweep = function () {
      var edge = window.innerHeight;
      items.forEach(function (el) { if (!el.classList.contains('in') && el.getBoundingClientRect().top < edge) el.classList.add('in'); });
    };
    window.addEventListener('scroll', function () { window.requestAnimationFrame(sweep); }, { passive: true });
    window.addEventListener('hashchange', sweep);
  } else {
    items.forEach(function (el) { el.classList.add('in'); });
  }

  /* Contact form: validate, then open the visitor's email app with everything filled in */
  var form = document.querySelector('form[data-mail]');
  if (form) {
    var to = form.getAttribute('data-mail');
    var status = form.querySelector('.form-status');
    function check(field) {
      var input = field.querySelector('input, textarea, select');
      var ok = input.checkValidity();
      field.classList.toggle('bad', !ok);
      input.setAttribute('aria-invalid', String(!ok));
      return ok;
    }
    form.querySelectorAll('.field').forEach(function (f) {
      var input = f.querySelector('input, textarea, select');
      input.addEventListener('blur', function () { if (input.value) check(f); });
    });
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var bad = null;
      form.querySelectorAll('.field').forEach(function (f) { if (!check(f) && !bad) bad = f; });
      if (bad) { bad.querySelector('input, textarea, select').focus(); return; }
      var d = new FormData(form);
      var subject = (d.get('topic') || 'Enquiry') + ' – from ' + (d.get('name') || 'website');
      var body = (d.get('message') || '') + '\n\n— ' + (d.get('name') || '') + '\n' + (d.get('email') || '');
      window.location.href = 'mailto:' + to + '?subject=' + encodeURIComponent(subject) + '&body=' + encodeURIComponent(body);
      if (status) {
        status.textContent = 'Your email app should open with the message ready to send. If nothing happens, email ' + to + ' directly.';
        status.classList.add('show');
      }
    });
  }

  document.querySelectorAll('[data-year]').forEach(function (el) { el.textContent = new Date().getFullYear(); });
})();

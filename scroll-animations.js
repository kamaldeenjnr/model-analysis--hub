// scroll-animations.js
// Lightweight scroll-triggered animation engine - no dependencies needed.

(function () {
  'use strict';

  // ─── Config ───────────────────────────────────────────────────────────────
  const ANIMATION_CLASS = 'sa-animate';
  const DONE_CLASS = 'sa-done';

  const CSS = `
    .sa-animate {
      opacity: 0;
      transform: translateY(32px);
      transition: opacity 0.65s cubic-bezier(0.16,1,0.3,1),
                  transform 0.65s cubic-bezier(0.16,1,0.3,1);
    }
    .sa-animate.sa-done {
      opacity: 1;
      transform: translateY(0);
    }
    /* Staggered delay for grid children */
    .sa-stagger > *:nth-child(1) { transition-delay: 0.05s; }
    .sa-stagger > *:nth-child(2) { transition-delay: 0.15s; }
    .sa-stagger > *:nth-child(3) { transition-delay: 0.25s; }
    .sa-stagger > *:nth-child(4) { transition-delay: 0.35s; }
    .sa-stagger > *:nth-child(5) { transition-delay: 0.45s; }
    .sa-stagger > * { opacity: 0; transform: translateY(24px); transition: opacity 0.6s cubic-bezier(0.16,1,0.3,1), transform 0.6s cubic-bezier(0.16,1,0.3,1); }
    .sa-stagger.sa-done > * { opacity: 1; transform: translateY(0); }
  `;

  // Inject styles
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);

  // Mark elements for animation
  function markElements() {
    const selectors = [
      'section', 'article', '.glass-card', '.proj-card', '.proj-card-hero',
      '.stat-card', '.skill-pill', 'h1', 'h2',
    ];
    selectors.forEach(sel => {
      document.querySelectorAll(sel).forEach(el => {
        // Skip if already visible above fold
        if (el.getBoundingClientRect().top < window.innerHeight * 0.85) return;
        el.classList.add(ANIMATION_CLASS);
      });
    });

    // Stagger grid children
    document.querySelectorAll('.grid').forEach(grid => {
      if (grid.getBoundingClientRect().top < window.innerHeight * 0.85) return;
      grid.classList.add('sa-stagger', ANIMATION_CLASS);
    });
  }

  // Observe
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add(DONE_CLASS);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

  function observeAll() {
    document.querySelectorAll(`.${ANIMATION_CLASS}`).forEach(el => observer.observe(el));
  }

  document.addEventListener('DOMContentLoaded', () => {
    markElements();
    observeAll();
  });

  // ─── Mobile Menu ──────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    const header = document.querySelector('header');
    if (!header) return;

    // Find nav links
    const nav = header.querySelector('nav');
    if (!nav) return;

    // Create hamburger button
    const hamburger = document.createElement('button');
    hamburger.id = 'hamburger-btn';
    hamburger.setAttribute('aria-label', 'Open menu');
    hamburger.innerHTML = `
      <svg id="ham-open" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      <svg id="ham-close" class="hidden" xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    `;

    const isDark = document.body.style.background && document.body.style.background.includes('#0f172a')
      || window.getComputedStyle(document.body).background.includes('rgb(15');

    hamburger.style.cssText = `
      display: none; align-items: center; justify-content: center;
      width: 40px; height: 40px; border-radius: 10px; border: none;
      cursor: pointer; background: rgba(0,0,0,0.07);
      color: inherit; transition: background 0.2s;
    `;
    hamburger.addEventListener('mouseenter', () => hamburger.style.background = 'rgba(0,0,0,0.12)');
    hamburger.addEventListener('mouseleave', () => hamburger.style.background = 'rgba(0,0,0,0.07)');

    // Insert before the last child in header's inner div
    const headerInner = header.querySelector('.max-w-7xl');
    if (headerInner) headerInner.appendChild(hamburger);

    // Mobile dropdown
    const mobileMenu = document.createElement('div');
    mobileMenu.id = 'mobile-menu';
    mobileMenu.style.cssText = `
      display: none; position: absolute; top: 100%; left: 0; right: 0;
      background: rgba(255,255,255,0.97); backdrop-filter: blur(20px);
      border-bottom: 1px solid rgba(0,0,0,0.06);
      padding: 16px 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
      z-index: 999; flex-direction: column; gap: 4px;
    `;
    nav.querySelectorAll('a').forEach(link => {
      const clone = link.cloneNode(true);
      clone.style.cssText = 'display:block; padding: 12px 16px; border-radius: 10px; font-weight:600; font-size:14px; color:#1e293b; text-decoration:none; transition: background 0.2s;';
      clone.addEventListener('mouseenter', () => clone.style.background = '#f1f5f9');
      clone.addEventListener('mouseleave', () => clone.style.background = 'transparent');
      mobileMenu.appendChild(clone);
    });

    header.style.position = 'sticky';
    header.style.top = '0';
    header.appendChild(mobileMenu);

    // Responsive visibility
    const mediaQuery = window.matchMedia('(max-width: 767px)');
    function handleMedia(mq) {
      if (mq.matches) {
        hamburger.style.display = 'flex';
        nav.style.display = 'none';
      } else {
        hamburger.style.display = 'none';
        nav.style.display = '';
        mobileMenu.style.display = 'none';
      }
    }
    mediaQuery.addEventListener('change', handleMedia);
    handleMedia(mediaQuery);

    let menuOpen = false;
    hamburger.addEventListener('click', () => {
      menuOpen = !menuOpen;
      mobileMenu.style.display = menuOpen ? 'flex' : 'none';
      document.getElementById('ham-open').classList.toggle('hidden', menuOpen);
      document.getElementById('ham-close').classList.toggle('hidden', !menuOpen);
    });

    // Close on link click
    mobileMenu.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        menuOpen = false;
        mobileMenu.style.display = 'none';
        document.getElementById('ham-open').classList.remove('hidden');
        document.getElementById('ham-close').classList.add('hidden');
      });
    });
  });

})();

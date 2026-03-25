/**
 * AMRScope — shared frontend utilities
 * Loaded on every page via base.html
 */

// ── Navbar active state polish ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link[href]').forEach(a => {
    const href = a.getAttribute('href');
    if (href === '/' ? path === '/' : path.startsWith(href)) {
      a.classList.add('active');
    }
  });
});

// ── Chart.js global defaults ────────────────────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#8b949e';
  Chart.defaults.borderColor = '#30363d';
  Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif';
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = '#1c2128';
  Chart.defaults.plugins.tooltip.borderColor = '#30363d';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = '#e6edf3';
  Chart.defaults.plugins.tooltip.bodyColor = '#8b949e';
  Chart.defaults.plugins.tooltip.padding = 10;
}

// ── Utility: format MRR value with colour ──────────────────────────────────
function mrrColor(v) {
  if (v >= 0.06) return '#3fb950';
  if (v >= 0.03) return '#58a6ff';
  return '#8b949e';
}

// ── Utility: score bar HTML ────────────────────────────────────────────────
function scoreBar(score, maxScore) {
  const pct = maxScore > 0 ? Math.round((score / maxScore) * 100) : 0;
  return `<div class="score-bar-wrap">
    <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%"></div></div>
    <span class="score-val">${score.toFixed(3)}</span>
  </div>`;
}

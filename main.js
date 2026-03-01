// main.js - Slop Engine bootstrap
// - Initializes WASM with an explicit .wasm path for reliable hosting
// - Adds graceful fallback when WebGPU is unavailable
// - Resizes canvas for DPR with compatibility fallback when ResizeObserver isn't present
// - Calls exported `run` with or without canvas argument to support multiple bindings

import init, { run } from './pkg/slop_engine.js';

function createStatusOverlay(message) {
  const overlay = document.createElement('div');
  overlay.setAttribute('role', 'status');
  overlay.style.cssText = [
    'position:fixed',
    'inset:0',
    'display:flex',
    'align-items:center',
    'justify-content:center',
    'background:#000',
    'color:#fff',
    'font:16px system-ui,sans-serif',
    'padding:1.25rem',
    'text-align:center',
    'z-index:9999'
  ].join(';');
  overlay.textContent = message;
  document.body.appendChild(overlay);
  return overlay;
}

function resizeCanvasToDisplaySize(canvas) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    return true;
  }

  return false;
}

function setupCanvasResize(canvas) {
  resizeCanvasToDisplaySize(canvas);

  const resize = () => resizeCanvasToDisplaySize(canvas);
  window.addEventListener('resize', resize, { passive: true });

  let observer = null;
  if (typeof ResizeObserver !== 'undefined') {
    observer = new ResizeObserver(resize);
    observer.observe(canvas);
  }

  let dprMedia = null;
  const onDprChange = () => {
    resize();
    if (dprMedia && typeof dprMedia.removeEventListener === 'function') {
      dprMedia.removeEventListener('change', onDprChange);
    }

    dprMedia = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    if (dprMedia && typeof dprMedia.addEventListener === 'function') {
      dprMedia.addEventListener('change', onDprChange, { passive: true });
    }
  };

  onDprChange();

  return () => {
    window.removeEventListener('resize', resize);
    if (observer) observer.disconnect();
    if (dprMedia && typeof dprMedia.removeEventListener === 'function') {
      dprMedia.removeEventListener('change', onDprChange);
    }
  };
}

async function bootstrap() {
  let overlay = null;

  try {
    const canvas = document.getElementById('slop-canvas');

    if (!canvas) {
      throw new Error('No canvas with id "slop-canvas" found in DOM.');
    }

    if (!('gpu' in navigator)) {
      overlay = createStatusOverlay('WebGPU is not available in this browser. Update your browser or enable the WebGPU feature flag.');
      return;
    }

    await init('./pkg/slop_engine_bg.wasm');
    const cleanupResize = setupCanvasResize(canvas);

    try {
      if (typeof run !== 'function') {
        throw new Error('`run` export not found in wasm package.');
      }

      if (run.length >= 1) {
        await run(canvas);
      } else {
        await run();
      }
    } finally {
      // Engine owns rendering loop; no longer need expensive observers/listeners.
      cleanupResize();
    }

    if (overlay) overlay.remove();
  } catch (err) {
    console.error('Slop Engine failed to initialize:', err);

    if (overlay) overlay.remove();
    createStatusOverlay('Slop Engine initialization failed. Open developer tools for details.');
  }
}

if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', bootstrap, { once: true });
} else {
  bootstrap();
}

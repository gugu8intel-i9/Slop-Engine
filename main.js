// main.js - Slop Engine bootstrap (improved, robust, and compatible)
// - Initializes the wasm package with an explicit .wasm path
// - Verifies WebGPU support early
// - Finds the canvas (#slop-canvas), resizes it for DPR, and keeps it responsive
// - Calls exported `run` with the canvas if the function accepts an argument, otherwise calls it with no args
// - Adds basic error handling and helpful console messages

import init, { run } from './pkg/slop_engine.js';

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

async function bootstrap() {
  try {
    // 1) Check WebGPU support early
    if (!('gpu' in navigator)) {
      console.error('Slop Engine Error: WebGPU not supported in this browser.');
      document.body.innerHTML = '<h1 style="color:#fff;background:#000;padding:2rem;text-align:center">Please use a browser that supports WebGPU.</h1>';
      return;
    }

    // 2) Locate canvas (must match index.html)
    const canvas = document.getElementById('slop-canvas');
    if (!canvas) {
      console.warn('No canvas with id "slop-canvas" found — falling back to document body.');
    }

    // 3) Initialize wasm module with explicit path to the .wasm file
    //    This helps wasm-pack locate the binary reliably in many hosting setups.
    await init('./pkg/slop_engine_bg.wasm');
    console.log('Slop Engine: wasm module initialized.');

    // 4) Ensure canvas is sized correctly before handing control to wasm
    if (canvas) {
      // initial resize
      resizeCanvasToDisplaySize(canvas);

      // keep canvas sized on window resize and DPR changes
      let resizeObserver = new ResizeObserver(() => resizeCanvasToDisplaySize(canvas));
      resizeObserver.observe(canvas);

      // also handle orientation/zoom/devicePixelRatio changes
      window.addEventListener('resize', () => resizeCanvasToDisplaySize(canvas));
      window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener?.('change', () => resizeCanvasToDisplaySize(canvas));
    }

    // 5) Call the engine's run function.
    //    If `run` accepts an argument (e.g., a canvas), pass it; otherwise call with no args.
    //    This keeps compatibility with different Rust bindings.
    try {
      if (typeof run === 'function') {
        if (run.length >= 1 && canvas) {
          // run expects at least one argument — pass the canvas element
          await run(canvas);
        } else {
          // run expects no args or no canvas available
          await run();
        }
      } else {
        throw new Error('`run` export not found in wasm package.');
      }
    } catch (engineErr) {
      console.error('Slop Engine runtime error:', engineErr);
      // Optionally show a user-friendly message in the canvas area
      if (canvas) {
        const ctx = canvas.getContext?.('2d');
        if (ctx) {
          ctx.fillStyle = '#000';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = '#f55';
          ctx.font = '16px sans-serif';
          ctx.fillText('Slop Engine failed to start. See console for details.', 10, 30);
        }
      }
    }

    console.log('Slop Engine: bootstrap complete.');
  } catch (err) {
    console.error('Slop Engine failed to initialize:', err);
    document.body.innerHTML = '<h1 style="color:#fff;background:#000;padding:2rem;text-align:center">Initialization error — check console.</h1>';
  }
}

// Start the engine when the page is ready
if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}

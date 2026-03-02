// sw.js
const CACHE_NAME = 'slop-cache-v1';
const ASSETS = [
  './',
  './index.html',
  './pkg/slop_engine.js',
  './pkg/slop_engine_bg.wasm'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) return caches.delete(key);
          return Promise.resolve();
        })
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // Only handle GET requests
  if (req.method !== 'GET') return;

  // Navigation requests (SPA): try network first, fallback to cached index.html
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((networkResponse) => {
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put('./index.html', responseClone).catch(() => {});
          });
          return networkResponse;
        })
        .catch(() => caches.match('./index.html'))
    );
    return;
  }

  // For other requests (JS, WASM, CSS, Images): Cache first, then network
  event.respondWith(
    caches.match(req).then((cachedResponse) => {
      // Return cached asset if we have it
      if (cachedResponse) {
        return cachedResponse;
      }

      // Otherwise, fetch from network
      return fetch(req).then((networkResponse) => {
        // Only cache successful, non-opaque responses
        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
          return networkResponse;
        }

        // Clone and store dynamically
        const responseClone = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(req, responseClone).catch(() => {});
        });

        return networkResponse;
      }).catch((error) => {
        // IMPORTANT FIX: Do NOT return index.html here. 
        // If a JS/WASM file fails to load, returning HTML will break the app.
        // Let it fail gracefully so the browser handles the network error properly.
        console.error('Fetch failed for asset:', req.url, error);
        throw error;
      });
    })
  );
});

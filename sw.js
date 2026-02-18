// sw.js
// Improved service worker for Slop Engine:
// - versioned cache name
// - install: pre-cache core assets and skip waiting
// - activate: remove old caches and claim clients
// - fetch: network-first for navigations (SPA friendly), cache-first for other GET assets,
//   and dynamically cache successful GET responses (JS/WASM/PNG/etc.)

const CACHE_NAME = 'slop-cache-v1';
const ASSETS = [
  './',
  './index.html',
  './pkg/slop_engine.js',
  './pkg/slop_engine_bg.wasm'
];

self.addEventListener('install', (event) => {
  // Pre-cache core assets and activate immediately
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  // Remove old caches and take control of clients immediately
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
          // Optionally update the cached index.html for offline navigations
          caches.open(CACHE_NAME).then((cache) => {
            // store a copy keyed by '/index.html' so fallback works
            cache.put('./index.html', networkResponse.clone()).catch(() => {});
          });
          return networkResponse;
        })
        .catch(() => caches.match('./index.html'))
    );
    return;
  }

  // For other requests: try cache first, then network; if network succeeds, cache it.
  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;

      return fetch(req)
        .then((networkResponse) => {
          // Only cache successful, same-origin or CORS responses that are not opaque,
          // but still allow opaque responses (e.g., CDN) to pass through without caching.
          if (!networkResponse || networkResponse.status !== 200) {
            return networkResponse;
          }

          // Clone and store a copy in the cache for future requests.
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            // Use request URL as the cache key
            cache.put(req, responseClone).catch(() => {});
          });

          return networkResponse;
        })
        .catch(() => {
          // If fetch fails and nothing in cache, optionally return a fallback (could be an offline page)
          return caches.match('./index.html');
        });
    })
  );
});

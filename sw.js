// sw.js
// Service worker for Slop Engine with resilient caching:
// - Versioned static cache + runtime cache
// - Navigation network-first with offline fallback
// - Stale-while-revalidate for local static assets
// - Runtime caching for successful GET requests with bounded growth

const STATIC_CACHE = 'slop-static-v2';
const RUNTIME_CACHE = 'slop-runtime-v2';
const OFFLINE_FALLBACK = './index.html';
const MAX_RUNTIME_ENTRIES = 80;

const CORE_ASSETS = [
  './',
  './index.html',
  './main.js',
  './sw.js',
  './pkg/slop_engine.js',
  './pkg/slop_engine_bg.wasm'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(STATIC_CACHE)
      .then((cache) => cache.addAll(CORE_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== STATIC_CACHE && key !== RUNTIME_CACHE)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

async function limitRuntimeCache() {
  const cache = await caches.open(RUNTIME_CACHE);
  const keys = await cache.keys();
  if (keys.length <= MAX_RUNTIME_ENTRIES) return;

  const overflow = keys.length - MAX_RUNTIME_ENTRIES;
  await Promise.all(keys.slice(0, overflow).map((request) => cache.delete(request)));
}

function isCacheable(request, response) {
  if (!response) return false;
  if (request.method !== 'GET') return false;
  if (response.status !== 200) return false;
  if (request.url.startsWith('chrome-extension://')) return false;
  return response.type === 'basic' || response.type === 'cors';
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);

  if (request.method !== 'GET') return;

  // Never intercept extension/devtools or websocket URLs.
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;

  // Navigation: network-first, offline fallback to index.
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const responseClone = response.clone();
          caches.open(STATIC_CACHE).then((cache) => cache.put(OFFLINE_FALLBACK, responseClone)).catch(() => {});
          return response;
        })
        .catch(async () => {
          const cached = await caches.match(OFFLINE_FALLBACK);
          return cached || Response.error();
        })
    );
    return;
  }

  // For first-party static files, use stale-while-revalidate.
  const isLocalStatic = url.origin === self.location.origin;
  if (isLocalStatic) {
    event.respondWith(
      caches.match(request).then(async (cached) => {
        const networkPromise = fetch(request)
          .then(async (response) => {
            if (isCacheable(request, response)) {
              const cache = await caches.open(STATIC_CACHE);
              await cache.put(request, response.clone());
            }
            return response;
          })
          .catch(() => cached);

        return cached || networkPromise;
      })
    );
    return;
  }

  // For cross-origin GETs: runtime cache fallback.
  event.respondWith(
    fetch(request)
      .then(async (response) => {
        if (isCacheable(request, response)) {
          const cache = await caches.open(RUNTIME_CACHE);
          await cache.put(request, response.clone());
          await limitRuntimeCache();
        }
        return response;
      })
      .catch(() => caches.match(request))
  );
});

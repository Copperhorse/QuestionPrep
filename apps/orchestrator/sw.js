const CACHE_NAME = "stresscheck-v1";
const MODEL_CACHE = "stresscheck-models-v1";

const STATIC_ASSETS = [
  "/companion",
  "/static/css/styles.css",
  "/static/js/ort.min.js",
  "/static/js/tcn-stress-detector.js",
  "/static/js/tcn-worker.js",
  "/static/js/audio-worklet.js",
  "/static/manifest.json",
];
const MODEL_ASSETS = ["/static/models/tcn_audio_model.onnx"];

self.addEventListener("install", (e) => {
  self.skipWaiting();
  e.waitUntil(
    Promise.all([
      caches.open(CACHE_NAME).then((c) => c.addAll(STATIC_ASSETS)),
      caches.open(MODEL_CACHE).then((c) => c.addAll(MODEL_ASSETS)),
    ]).catch((err) => console.error("[SW] Install failed:", err)),
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(clients.claim());
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);

  // Bypass Service Worker for WASM and MJS files — let browser handle them directly
  if (url.pathname.endsWith(".wasm") || url.pathname.endsWith(".mjs")) {
    return; // Don't call e.respondWith — let browser fetch normally
  }

  if (url.pathname === "/companion") {
    e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
    return;
  }

  e.respondWith(
    caches.match(e.request).then((cached) => {
      if (cached) return cached;
      return fetch(e.request)
        .then((response) => {
          if (
            response.ok &&
            (url.pathname.startsWith("/static/") ||
              url.pathname.startsWith("/companion"))
          ) {
            const cacheName = url.pathname.includes(".onnx")
              ? MODEL_CACHE
              : CACHE_NAME;
            caches
              .open(cacheName)
              .then((c) => c.put(e.request, response.clone()));
          }
          return response;
        })
        .catch(() => {
          if (e.request.mode === "navigate") return caches.match("/companion");
        });
    }),
  );
});

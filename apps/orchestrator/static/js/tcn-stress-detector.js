/**
 * tcn-stress-detector.js  —  Main-thread API for TCN stress detection
 */

console.log("[tcn-stress-detector.js] File loaded, executing...");

const DEBUG = true;

const Log = {
  main: (msg, ...args) => {
    if (DEBUG) console.log(`[Main] ${msg}`, ...args);
  },
  warn: (msg, ...args) => {
    if (DEBUG) console.warn(`[Main] ⚠ ${msg}`, ...args);
  },
  error: (msg, ...args) => {
    if (DEBUG) console.error(`[Main] ✗ ${msg}`, ...args);
  },
};

// ── Error-boundary wrapper ──────────────────────────────────────────────────
try {
  console.log("[Main] Defining TcnStressDetector class...");

  class TcnStressDetector {
    constructor({
      modelPath = "/static/models/tcn_audio_model.onnx",
      wasmDir = "/static/js/",
      workerPath = "/static/js/tcn-worker.js",
      clipSeconds = 5,
      overlapSeconds = 3,
      targetSR = 16000,
    } = {}) {
      this.modelPath = modelPath;
      this.wasmDir = wasmDir;
      this.workerPath = workerPath;
      this.clipSeconds = clipSeconds;
      this.overlapSeconds = overlapSeconds;
      this.targetSR = targetSR;

      this.onResult = null;
      this.onReady = null;
      this.onError = null;

      this._worker = null;
      this._audioCtx = null;
      this._stream = null;
      this._sourceNode = null;
      this._processorNode = null;
      this._workletNode = null;
      this._ringBuffer = null;
      this._ringWritePos = 0;
      this._clipSamples = 0;
      this._hopSamples = 0;
      this._samplesSinceLastClip = 0;
      this._ready = false;
      this._busy = false;
      this._disfluencyFlag = false;

      this._readyPromise = null;
      this._readyResolve = null;
      this._readyReject = null;
    }

    async init() {
      if (this._worker) return;
      await this._preloadWasm();
      this._worker = new Worker(this.workerPath);

      this._readyPromise = new Promise((resolve, reject) => {
        this._readyResolve = resolve;
        this._readyReject = reject;
      });

      this._worker.addEventListener("message", (e) =>
        this._handleWorkerMessage(e),
      );

      this._worker.addEventListener("error", (e) => {
        const msg = `Worker failed to load: ${e.message || e.filename || "Unknown error"}`;
        Log.error(msg);
        if (this._readyReject) this._readyReject(new Error(msg));
        if (this.onError) this.onError(msg);
      });

      this._worker.postMessage({
        type: "INIT",
        modelPath: this.modelPath,
        wasmDir: this.wasmDir,
      });

      await this._readyPromise;
    }
    async _preloadWasm() {
      if (!("caches" in self)) {
        Log.warn("Cache API not available — WASM will not be pre-cached");
        return;
      }

      const WASM_CACHE = "stresscheck-wasm-v1";
      const wasmUrl = "/static/js/ort-wasm-simd-threaded.wasm";

      try {
        const cache = await caches.open(WASM_CACHE);
        const cached = await cache.match(wasmUrl);

        if (cached) {
          Log.main("WASM already in cache — skipping pre-fetch");
          return;
        }

        Log.main("Pre-fetching WASM for offline support…");
        const response = await fetch(wasmUrl);

        if (!response.ok) {
          Log.warn(`WASM pre-fetch failed: HTTP ${response.status}`);
          return;
        }

        const ct = response.headers.get("content-type") || "";
        if (
          !ct.includes("application/wasm") &&
          !ct.includes("application/octet-stream")
        ) {
          Log.warn(
            `WASM Content-Type is '${ct}' — expected application/wasm. Caching anyway but verify your server config.`,
          );
        }

        await cache.put(wasmUrl, response);
        Log.main("WASM pre-cached successfully for offline use");
      } catch (err) {
        // Non-fatal — WASM will be fetched from network on next use
        Log.warn(`WASM pre-cache failed (non-fatal): ${err.message}`);
      }
    }
    waitUntilReady() {
      if (this._ready) return Promise.resolve();
      if (this._readyPromise) return this._readyPromise;
      return Promise.reject(
        new Error("[TcnStressDetector] init() was not called"),
      );
    }

    async analyseClip(pcm, sampleRate = this.targetSR, disfluencyFlag = null) {
      if (!this._ready)
        throw new Error("[TcnStressDetector] Call init() first");
      if (disfluencyFlag !== null)
        this._disfluencyFlag = Boolean(disfluencyFlag);

      return new Promise((resolve, reject) => {
        const onMessage = (e) => {
          if (e.data.type === "RESULT") {
            this._worker.removeEventListener("message", onMessage);
            resolve(e.data.result);
          } else if (e.data.type === "ERROR") {
            this._worker.removeEventListener("message", onMessage);
            reject(new Error(e.data.message));
          }
        };
        this._worker.addEventListener("message", onMessage);
        this._worker.postMessage(
          {
            type: "PREDICT",
            pcm: pcm,
            sampleRate: sampleRate,
            disfluencyFlag: this._disfluencyFlag,
          },
          [pcm.buffer],
        );
      });
    }

    async startListening(externalStream = null) {
      if (!this._ready)
        throw new Error("[TcnStressDetector] Call init() first");
      if (this._stream) return;

      this._stream =
        externalStream ||
        (await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: this.targetSR,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        }));

      this._audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.targetSR,
      });
      if (this._audioCtx.state === "suspended") {
        await this._audioCtx.resume();
      }

      const actualSR = this._audioCtx.sampleRate;
      Log.main(
        `AudioContext sample rate: ${actualSR}Hz (target: ${this.targetSR}Hz)`,
      );

      if (Math.abs(actualSR - this.targetSR) > 1) {
        Log.warn(
          `Sample rate mismatch! Context=${actualSR}Hz, expected=${this.targetSR}Hz. ` +
            `Audio will be resampled in worker.`,
        );
      }

      this._clipSamples = Math.floor(this.clipSeconds * actualSR);
      this._hopSamples = Math.floor(
        (this.clipSeconds - this.overlapSeconds) * actualSR,
      );
      this._ringBuffer = new Float32Array(this._clipSamples);
      this._ringWritePos = 0;
      this._samplesSinceLastClip = 0;

      if (this._audioCtx.audioWorklet) {
        try {
          const workletUrl = new URL(
            "/static/js/audio-worklet.js",
            location.href,
          ).href;
          Log.main(`Loading AudioWorklet from: ${workletUrl}`);
          await this._audioCtx.audioWorklet.addModule(workletUrl);

          this._workletNode = new AudioWorkletNode(
            this._audioCtx,
            "audio-capture-processor",
            {
              processorOptions: { chunkSize: 4096 },
              numberOfInputs: 1,
              numberOfOutputs: 0,
              channelCount: 1,
            },
          );

          this._workletNode.port.onmessage = (e) => this._onWorkletMessage(e);
          this._sourceNode = this._audioCtx.createMediaStreamSource(
            this._stream,
          );
          this._sourceNode.connect(this._workletNode);

          Log.main("AudioWorklet capture started — low latency mode");
        } catch (err) {
          Log.error(
            `AudioWorklet failed: ${err.message}. Falling back to ScriptProcessor.`,
          );
          this._workletNode = null;
          await this._startScriptProcessor(actualSR);
        }
      } else {
        Log.warn(
          "AudioWorklet not available (requires HTTPS). Using ScriptProcessor fallback.",
        );
        await this._startScriptProcessor(actualSR);
      }

      Log.main(
        `Listening at ${actualSR}Hz, clip=${this._clipSamples} samples, hop=${this._hopSamples} samples`,
      );
    }

    async _startScriptProcessor(actualSR) {
      const bufferSize = 4096;
      this._processorNode = this._audioCtx.createScriptProcessor(
        bufferSize,
        1,
        1,
      );
      this._processorNode.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        this._feedSamples(inputData, actualSR);
      };
      this._sourceNode = this._audioCtx.createMediaStreamSource(this._stream);
      this._sourceNode.connect(this._processorNode);
      this._processorNode.connect(this._audioCtx.destination);
      Log.main("ScriptProcessor fallback active");
    }

    _onWorkletMessage(e) {
      const { type, pcm, chunkId, timestamp, sampleRate } = e.data;
      if (type === "CHUNK") {
        Log.main(
          `Received worklet chunk #${chunkId}, samples=${pcm.length}, ts=${timestamp.toFixed(3)}`,
        );
        this._feedSamples(pcm, sampleRate);
      }
    }

    stopListening() {
      if (this._workletNode) {
        this._workletNode.port.postMessage({ type: "STOP" });
        this._workletNode.disconnect();
        this._workletNode = null;
      }
      if (this._processorNode) {
        this._processorNode.disconnect();
        this._processorNode.onaudioprocess = null;
        this._processorNode = null;
      }
      if (this._sourceNode) {
        this._sourceNode.disconnect();
        this._sourceNode = null;
      }
      if (this._stream) {
        this._stream.getTracks().forEach((t) => t.stop());
        this._stream = null;
      }
      if (this._audioCtx) {
        this._audioCtx.close();
        this._audioCtx = null;
      }
      Log.main("Audio capture stopped");
    }

    destroy() {
      this.stopListening();
      if (this._worker) {
        this._worker.terminate();
        this._worker = null;
      }
      this._ready = false;
    }

    setDisfluencyFlag(flag) {
      this._disfluencyFlag = Boolean(flag);
    }

    _feedSamples(chunk, sampleRate) {
      let hasNonZero = false;
      for (let i = 0; i < chunk.length; i++) {
        if (chunk[i] !== 0) {
          hasNonZero = true;
          break;
        }
      }
      if (!hasNonZero) {
        Log.warn(
          `Chunk contains only zeros — microphone may be muted or disconnected`,
        );
      }

      for (let i = 0; i < chunk.length; i++) {
        this._ringBuffer[this._ringWritePos % this._clipSamples] = chunk[i];
        this._ringWritePos++;
      }
      this._samplesSinceLastClip += chunk.length;

      Log.main(
        `Ring buffer: +${chunk.length} samples, total since last clip: ${this._samplesSinceLastClip}/${this._hopSamples}`,
      );

      if (this._samplesSinceLastClip >= this._hopSamples && !this._busy) {
        this._samplesSinceLastClip = 0;
        this._busy = true;

        const clip = new Float32Array(this._clipSamples);
        const writePos = this._ringWritePos;
        for (let i = 0; i < this._clipSamples; i++) {
          clip[i] =
            this._ringBuffer[
              (writePos - this._clipSamples + i + this._clipSamples) %
                this._clipSamples
            ];
        }

        Log.main(
          `Sending clip to worker: ${clip.length} samples, sr=${sampleRate}`,
        );

        this._worker.postMessage(
          {
            type: "PREDICT",
            pcm: clip,
            sampleRate: sampleRate,
            disfluencyFlag: this._disfluencyFlag,
          },
          [clip.buffer],
        );
      }
    }

    _handleWorkerMessage(e) {
      const { type, result, debug } = e.data;

      if (type === "READY") {
        this._ready = true;
        Log.main("Worker reports READY");
        if (this._readyResolve) this._readyResolve();
        if (this.onReady) this.onReady();
      } else if (type === "RESULT") {
        this._busy = false;
        if (DEBUG && debug) {
          Log.main("Worker debug:", debug);
        }
        Log.main(
          `Result: level=${result.stressLevel}, prob=${result.stressProb.toFixed(3)}, ` +
            `emotion=${result.emotion}, processing=${result.processingMs}ms`,
        );
        if (this.onResult) this.onResult(result);
      } else if (type === "ERROR") {
        this._busy = false;
        Log.error(`Worker error: ${e.data.message}`);
        if (this.onError) this.onError(e.data.message);
      }
    }

    get isReady() {
      return this._ready;
    }
    get isListening() {
      return this._stream !== null;
    }
  }

  console.log("[Main] TcnStressDetector class defined successfully");

  // Explicitly attach to window
  window.TcnStressDetector = TcnStressDetector;
  console.log("[Main] TcnStressDetector attached to window");
} catch (err) {
  console.error("[Main] FATAL: Error defining TcnStressDetector class:", err);
  console.error("Stack:", err.stack);
}

// ── Self-Test ───────────────────────────────────────────────────────────────
(function selfTest() {
  if (typeof window === "undefined") {
    console.error(
      "[Main] FATAL: Not running in browser context (window is undefined)",
    );
    return;
  }
  if (typeof Worker === "undefined") {
    console.error("[Main] FATAL: Web Workers not supported");
    return;
  }
  console.log(
    "[Main] Self-test passed: Browser context and Worker support confirmed",
  );

  if (typeof window.TcnStressDetector === "function") {
    console.log(
      "[Main] TcnStressDetector class successfully registered on window",
    );
  } else {
    console.error(
      "[Main] FATAL: TcnStressDetector class NOT found on window after load",
    );
  }
})();

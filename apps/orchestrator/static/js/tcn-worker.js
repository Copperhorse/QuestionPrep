// ── Worker Logger ───────────────────────────────────────────────────────────
const WORKER_DEBUG = true;
console.log("[tcn-worker.js] Worker script loaded, executing...");
const WLog = {
  info: (msg, ...args) => {
    if (WORKER_DEBUG) console.log(`[Worker] ${msg}`, ...args);
  },
  warn: (msg, ...args) => {
    if (WORKER_DEBUG) console.warn(`[Worker] ⚠ ${msg}`, ...args);
  },
  error: (msg, ...args) => {
    if (WORKER_DEBUG) console.error(`[Worker] ✗ ${msg}`, ...args);
  },
};

importScripts("/static/js/ort.min.js");

const SR = 16000;
const N_FFT = 1024;
const HOP_LENGTH = 512;
const N_MELS = 60;
const N_FRAMES = 150;
const FMIN = 0;
const FMAX = 8000;
const N_FREQS = N_FFT / 2 + 1;

const CLASS_MAP = {
  0: "anger",
  1: "calm",
  2: "disgust",
  3: "fear",
  4: "happy",
  5: "neutral",
  6: "sadness",
  7: "surprise",
};
const STRESSED_IDS = [0, 3, 7];

const THRESHOLD_MILD = 0.25;
const THRESHOLD_STRESSED = 0.47;
const THRESHOLD_HIGH = 0.65;

let session = null;
let melFB = null;
let hannWindow = null;

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelFilterbank() {
  const melMin = hzToMel(FMIN);
  const melMax = hzToMel(FMAX);
  const melPts = Array.from(
    { length: N_MELS + 2 },
    (_, i) => melMin + ((melMax - melMin) * i) / (N_MELS + 1),
  );
  const hzPts = melPts.map(melToHz);
  const binPts = hzPts.map((hz) => Math.floor(((N_FFT + 1) * hz) / SR));

  const fb = new Float32Array(N_MELS * N_FREQS);
  for (let m = 1; m <= N_MELS; m++) {
    const lo = binPts[m - 1],
      mid = binPts[m],
      hi = binPts[m + 1];
    for (let k = lo; k < mid; k++) {
      if (k < N_FREQS) fb[(m - 1) * N_FREQS + k] = (k - lo) / (mid - lo);
    }
    for (let k = mid; k < hi; k++) {
      if (k < N_FREQS) fb[(m - 1) * N_FREQS + k] = (hi - k) / (hi - mid);
    }
  }
  return fb;
}

function buildHannWindow() {
  return new Float32Array(N_FFT).map(
    (_, n) => 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N_FFT - 1))),
  );
}

function fft(re, im) {
  const n = re.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1,
        curIm = 0;
      for (let k = 0; k < len >> 1; k++) {
        const uRe = re[i + k],
          uIm = im[i + k];
        const vRe = re[i + k + (len >> 1)],
          vIm = im[i + k + (len >> 1)];
        const tvRe = curRe * vRe - curIm * vIm;
        const tvIm = curRe * vIm + curIm * vRe;
        re[i + k] = uRe + tvRe;
        im[i + k] = uIm + tvIm;
        re[i + k + (len >> 1)] = uRe - tvRe;
        im[i + k + (len >> 1)] = uIm - tvIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

function extractLogMel(pcm) {
  WLog.info(
    `extractLogMel input: length=${pcm.length}, expected≥${N_FRAMES * HOP_LENGTH + N_FFT}`,
  );

  const logMel = new Float32Array(N_FRAMES * N_MELS);
  const reArr = new Float32Array(N_FFT);
  const imArr = new Float32Array(N_FFT);

  for (let t = 0; t < N_FRAMES; t++) {
    const start = t * HOP_LENGTH;
    reArr.fill(0);
    imArr.fill(0);
    for (let i = 0; i < N_FFT; i++) {
      const s = start + i;
      reArr[i] = (s < pcm.length ? pcm[s] : 0) * hannWindow[i];
    }
    fft(reArr, imArr);

    const power = new Float32Array(N_FREQS);
    for (let k = 0; k < N_FREQS; k++) {
      power[k] = reArr[k] * reArr[k] + imArr[k] * imArr[k];
    }

    for (let m = 0; m < N_MELS; m++) {
      let val = 0;
      for (let k = 0; k < N_FREQS; k++) {
        val += melFB[m * N_FREQS + k] * power[k];
      }
      logMel[t * N_MELS + m] = 10 * Math.log10(Math.max(val, 1e-10));
    }
  }

  let mean = 0;
  for (let i = 0; i < logMel.length; i++) mean += logMel[i];
  mean /= logMel.length;

  let variance = 0;
  for (let i = 0; i < logMel.length; i++) variance += (logMel[i] - mean) ** 2;
  const std = Math.sqrt(variance / logMel.length) + 1e-8;

  for (let i = 0; i < logMel.length; i++) logMel[i] = (logMel[i] - mean) / std;

  let hasNaN = false,
    minVal = Infinity,
    maxVal = -Infinity;
  for (let i = 0; i < logMel.length; i++) {
    if (Number.isNaN(logMel[i])) hasNaN = true;
    if (logMel[i] < minVal) minVal = logMel[i];
    if (logMel[i] > maxVal) maxVal = logMel[i];
  }
  if (hasNaN) WLog.warn("NaN detected in log-mel features!");
  if (maxVal === minVal)
    WLog.warn("Feature array is constant — possible silence or input issue");

  WLog.info(
    `Feature extraction complete: shape=[${N_FRAMES},${N_MELS}], min=${minVal.toFixed(3)}, max=${maxVal.toFixed(3)}, mean=${mean.toFixed(3)}, std=${std.toFixed(3)}`,
  );

  return { features: logMel, min: minVal, max: maxVal, mean, std };
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

async function runInference(pcm, disfluencyFlag = false) {
  const t0 = performance.now();

  WLog.info(
    `runInference called: pcm.length=${pcm.length}, disfluency=${disfluencyFlag}`,
  );

  const { features: flat, min, max } = extractLogMel(pcm);

  const expectedSize = 1 * N_FRAMES * N_MELS;
  if (flat.length !== expectedSize) {
    WLog.error(
      `Tensor size mismatch! Expected ${expectedSize}, got ${flat.length}`,
    );
  }

  const tensor = new ort.Tensor("float32", flat, [1, N_FRAMES, N_MELS]);
  WLog.info(
    `Model input tensor: shape=[1,${N_FRAMES},${N_MELS}], dtype=float32`,
  );

  const feeds = { input: tensor };
  const output = await session.run(feeds);
  let probs = output["output"]
    ? output["output"].data
    : output[Object.keys(output)[0]].data;

  WLog.info(
    `Raw model output: length=${probs.length}, values=[${Array.from(probs)
      .map((v) => v.toFixed(4))
      .join(", ")}]`,
  );

  const probSum = probs.reduce((a, b) => a + b, 0);
  const hasNegative = probs.some((p) => p < 0);
  const needsSoftmax = Math.abs(probSum - 1.0) > 0.5 || hasNegative;

  if (needsSoftmax) {
    WLog.warn(
      `Model outputs appear to be logits (sum=${probSum.toFixed(3)}, hasNegative=${hasNegative}). Applying softmax.`,
    );
    probs = softmax(probs);
    WLog.info(
      `After softmax: sum=${probs.reduce((a, b) => a + b, 0).toFixed(6)}, values=[${Array.from(
        probs,
      )
        .map((v) => v.toFixed(4))
        .join(", ")}]`,
    );
  } else {
    WLog.info(
      `Model outputs are valid probabilities (sum≈${probSum.toFixed(3)})`,
    );
  }

  if (probs.some((p) => p < 0 || p > 1)) {
    WLog.warn("Softmax produced values outside [0,1] — model may be corrupted");
  }

  const emotionProbs = {};
  for (let i = 0; i < 8; i++) emotionProbs[CLASS_MAP[i]] = probs[i];

  const stressProb = STRESSED_IDS.reduce((s, i) => s + probs[i], 0);
  WLog.info(
    `Stress probability: ${stressProb.toFixed(4)} (anger=${probs[0].toFixed(4)}, fear=${probs[3].toFixed(4)}, surprise=${probs[7].toFixed(4)})`,
  );

  let topIdx = 0;
  for (let i = 1; i < 8; i++) if (probs[i] > probs[topIdx]) topIdx = i;
  const emotion = CLASS_MAP[topIdx];
  WLog.info(`Top emotion: ${emotion} (${(probs[topIdx] * 100).toFixed(1)}%)`);

  let audioLevel;
  if (stressProb >= THRESHOLD_HIGH) audioLevel = "stressed";
  else if (stressProb >= THRESHOLD_STRESSED) audioLevel = "stressed";
  else if (stressProb >= THRESHOLD_MILD) audioLevel = "mild";
  else audioLevel = "not_stressed";
  WLog.info(`Audio level (before disfluency): ${audioLevel}`);

  let stressLevel;
  if (audioLevel === "stressed" && disfluencyFlag)
    stressLevel = "highly_stressed";
  else if (audioLevel === "stressed") stressLevel = "stressed";
  else if (audioLevel === "mild" && disfluencyFlag) stressLevel = "stressed";
  else if (audioLevel === "mild") stressLevel = "mild";
  else if (audioLevel === "not_stressed" && disfluencyFlag)
    stressLevel = "mild";
  else stressLevel = "not_stressed";
  WLog.info(`Final stress level: ${stressLevel}`);

  const processingMs = Math.round(performance.now() - t0);
  WLog.info(`Inference complete in ${processingMs}ms`);

  const result = {
    stressLevel,
    stressProb,
    emotion,
    emotionProbs,
    disfluencyFlag,
    processingMs,
  };

  const debug = {
    featureMin: min,
    featureMax: max,
    probSum: probs.reduce((a, b) => a + b, 0),
    inputShape: [1, N_FRAMES, N_MELS],
    outputShape: [8],
  };

  return { result, debug };
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;

  if (type === "INIT") {
    try {
      const { modelPath, wasmDir } = e.data;
      WLog.info(`INIT requested: modelPath=${modelPath}, wasmDir=${wasmDir}`);

      ort.env.wasm.wasmPaths = {
        "ort-wasm-simd-threaded.wasm": "/static/js/ort-wasm-simd-threaded.wasm",
      };
      ort.env.wasm.numThreads = 1;

      melFB = buildMelFilterbank();
      hannWindow = buildHannWindow();
      WLog.info(`Mel filterbank built: ${N_MELS} mels × ${N_FREQS} freqs`);

      session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      WLog.info(
        `ONNX session created: ${session.inputNames} → ${session.outputNames}`,
      );

      self.postMessage({ type: "READY" });
    } catch (err) {
      WLog.error(`INIT failed: ${err.message}`);
      self.postMessage({
        type: "ERROR",
        message: `INIT failed: ${err.message}`,
      });
    }
  } else if (type === "PREDICT") {
    try {
      const { pcm, sampleRate, disfluencyFlag = false } = e.data;
      WLog.info(
        `PREDICT received: pcm.length=${pcm.length}, sampleRate=${sampleRate}, disfluency=${disfluencyFlag}`,
      );

      let audio = pcm;
      if (sampleRate !== SR) {
        WLog.info(`Resampling: ${sampleRate}Hz → ${SR}Hz`);
        audio = resampleLinear(pcm, sampleRate, SR);
        WLog.info(`Resampled length: ${audio.length} samples`);
      }

      const { result, debug } = await runInference(audio, disfluencyFlag);

      self.postMessage({ type: "RESULT", result, debug });
      WLog.info("RESULT posted to main thread with debug payload");
    } catch (err) {
      WLog.error(`PREDICT failed: ${err.message}`);
      self.postMessage({
        type: "ERROR",
        message: `PREDICT failed: ${err.message}`,
      });
    }
  }
  // In the message handler, add this case:
  if (type === "PING") {
    self.postMessage({ type: "PONG", timestamp: performance.now() });
    return;
  }
});

function resampleLinear(input, fromSR, toSR) {
  if (fromSR === toSR) return input;
  const ratio = fromSR / toSR;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const pos = i * ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, input.length - 1);
    output[i] = input[lo] + (pos - lo) * (input[hi] - input[lo]);
  }
  return output;
}
// ── Self-Test ───────────────────────────────────────────────────────────────
(function selfTest() {
  if (typeof self === "undefined") {
    console.error(
      "[Worker] FATAL: Not running in Worker context (self is undefined)",
    );
    return;
  }
  if (typeof importScripts !== "function") {
    console.error("[Worker] FATAL: importScripts not available");
    return;
  }
  if (typeof ort === "undefined") {
    console.error("[Worker] FATAL: ONNX Runtime (ort) not loaded");
    return;
  }
  console.log(
    "[Worker] Self-test passed: Worker context, importScripts, and ort are available",
  );
})();

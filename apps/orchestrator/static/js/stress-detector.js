/**
 * stress_detector.js
 * -------------------
 * Client-side stress detection using three ONNX models.
 * Extracts 85 audio features from raw PCM audio, runs soft voting
 * across XGBoost, Random Forest, and Logistic Regression models,
 * and returns a stress prediction with probability.
 *
 * Usage:
 *   const detector = new StressDetector();
 *   await detector.load();
 *   const result = await detector.predict(audioBuffer);
 *
 * Dependencies:
 *   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
 *
 * Model files required (same directory or update paths below):
 *   xgboost.onnx
 *   random_forest.onnx
 *   logistic_regression.onnx
 */

// ─── Config ───────────────────────────────────────────────────────────────────

const SR = 16000; // expected sample rate
const N_MFCC = 13;
const HOP_LENGTH = 512;
const N_FFT = 1024;
const FRAME_LENGTH = 1024;
const F0_MIN = 75;
const F0_MAX = 300;

// Stress threshold for soft voting probability
const STRESS_THRESHOLD = 0.4;

// The 85 selected features — order must match Python training pipeline exactly
const SELECTED_FEATURES = [
  "rms_std",
  "rms_max",
  "mfcc_1_std",
  "rms_mean",
  "mfcc_1_max",
  "f0_mean",
  "mfcc_d1_1_std",
  "mfcc_d1_1_min",
  "mfcc_4_mean",
  "mfcc_1_mean",
  "mfcc_5_min",
  "mfcc_d1_1_max",
  "mfcc_4_min",
  "mfcc_5_mean",
  "mfcc_d2_1_std",
  "mfcc_2_mean",
  "mfcc_13_std",
  "mfcc_d2_13_std",
  "mfcc_10_max",
  "mfcc_d2_1_max",
  "mfcc_3_mean",
  "mfcc_5_std",
  "mfcc_8_mean",
  "mfcc_11_mean",
  "mfcc_7_mean",
  "mfcc_1_min",
  "mfcc_d1_5_std",
  "mfcc_d2_1_min",
  "mfcc_6_min",
  "mfcc_10_mean",
  "mfcc_d1_13_std",
  "mfcc_12_std",
  "mfcc_d2_6_std",
  "mfcc_d2_2_max",
  "rolloff_std",
  "mfcc_d2_12_std",
  "mfcc_6_std",
  "rms_min",
  "mfcc_9_mean",
  "mfcc_5_max",
  "mfcc_8_max",
  "mfcc_2_max",
  "mfcc_13_max",
  "mfcc_6_mean",
  "mfcc_4_max",
  "mfcc_2_min",
  "mfcc_d2_3_std",
  "flatness_std",
  "mfcc_4_std",
  "mfcc_d1_4_std",
  "mfcc_d2_5_mean",
  "mfcc_11_min",
  "mfcc_d2_13_max",
  "mfcc_d1_2_mean",
  "mfcc_3_min",
  "mfcc_d1_5_max",
  "mfcc_12_min",
  "mfcc_d2_3_min",
  "mfcc_13_min",
  "mfcc_11_max",
  "mfcc_3_max",
  "mfcc_d2_11_min",
  "mfcc_3_std",
  "mfcc_d2_11_mean",
  "mfcc_12_mean",
  "mfcc_9_min",
  "mfcc_12_max",
  "mfcc_8_std",
  "mfcc_9_max",
  "mfcc_d1_12_std",
  "mfcc_d1_12_min",
  "mfcc_13_mean",
  "mfcc_d2_8_mean",
  "mfcc_d1_7_max",
  "mfcc_d2_5_std",
  "mfcc_d2_13_min",
  "zcr_mean",
  "mfcc_d2_3_mean",
  "mfcc_d1_5_min",
  "rolloff_mean",
  "mfcc_d2_2_min",
  "mfcc_d2_9_min",
  "mfcc_d2_12_min",
  "mfcc_d2_6_mean",
  "mfcc_d2_7_std",
];

// ══════════════════════════════════════════════════════════════════════════════
//  DSP UTILITIES
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Frame a signal into overlapping windows.
 * Returns a 2D array [n_frames][frame_length].
 */
function frame(signal, frameLength, hopLength) {
  const nFrames = Math.floor((signal.length - frameLength) / hopLength) + 1;
  const frames = [];
  for (let i = 0; i < nFrames; i++) {
    frames.push(signal.slice(i * hopLength, i * hopLength + frameLength));
  }
  return frames;
}

/**
 * Apply a Hann window to a frame.
 */
function hannWindow(frameLength) {
  const window = new Float32Array(frameLength);
  for (let i = 0; i < frameLength; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frameLength - 1)));
  }
  return window;
}

/**
 * Compute the power spectrum of a windowed frame via FFT.
 * Returns magnitude spectrum of length N_FFT/2 + 1.
 */
function powerSpectrum(frame, fftSize) {
  const windowed = new Float32Array(fftSize);
  const hann = hannWindow(frame.length);
  for (let i = 0; i < frame.length; i++) {
    windowed[i] = frame[i] * hann[i];
  }

  // Real FFT via split-radix (simple DFT for correctness, fast enough for 1024)
  const real = Array.from(windowed);
  const imag = new Array(fftSize).fill(0);
  fftInPlace(real, imag);

  const halfLen = Math.floor(fftSize / 2) + 1;
  const power = new Float32Array(halfLen);
  for (let i = 0; i < halfLen; i++) {
    power[i] = real[i] * real[i] + imag[i] * imag[i];
  }
  return power;
}

/**
 * In-place Cooley-Tukey FFT (radix-2, power-of-2 size).
 */
function fftInPlace(real, imag) {
  const n = real.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  // Butterfly operations
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = -Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1,
        curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = real[i + k];
        const uIm = imag[i + k];
        const vRe =
          real[i + k + len / 2] * curRe - imag[i + k + len / 2] * curIm;
        const vIm =
          real[i + k + len / 2] * curIm + imag[i + k + len / 2] * curRe;
        real[i + k] = uRe + vRe;
        imag[i + k] = uIm + vIm;
        real[i + k + len / 2] = uRe - vRe;
        imag[i + k + len / 2] = uIm - vIm;
        const newRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newRe;
      }
    }
  }
}

/**
 * Build a mel filterbank matrix.
 * Returns Float32Array[nMels][nFFT/2+1].
 */
function melFilterbank(sr, nFFT, nMels = 128, fMin = 0, fMax = null) {
  fMax = fMax || sr / 2;

  const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);

  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = [];
  for (let i = 0; i <= nMels + 1; i++) {
    melPoints.push(melToHz(melMin + (i / (nMels + 1)) * (melMax - melMin)));
  }

  const fftFreqs = [];
  const halfLen = Math.floor(nFFT / 2) + 1;
  for (let i = 0; i < halfLen; i++) {
    fftFreqs.push((i * sr) / nFFT);
  }

  const filters = [];
  for (let m = 1; m <= nMels; m++) {
    const filter = new Float32Array(halfLen);
    for (let f = 0; f < halfLen; f++) {
      const freq = fftFreqs[f];
      if (freq >= melPoints[m - 1] && freq <= melPoints[m]) {
        filter[f] =
          (freq - melPoints[m - 1]) / (melPoints[m] - melPoints[m - 1]);
      } else if (freq >= melPoints[m] && freq <= melPoints[m + 1]) {
        filter[f] =
          (melPoints[m + 1] - freq) / (melPoints[m + 1] - melPoints[m]);
      }
    }
    filters.push(filter);
  }
  return filters;
}

// Pre-compute mel filterbank once
const MEL_FILTERS = melFilterbank(SR, N_FFT, 128);

/**
 * Compute log-mel spectrogram frames.
 * Returns 2D array [n_frames][128].
 */
function logMelSpectrogram(signal) {
  const frames = frame(signal, N_FFT, HOP_LENGTH);
  const melFrames = [];
  for (const f of frames) {
    const padded = new Float32Array(N_FFT);
    padded.set(f.length < N_FFT ? f : f.slice(0, N_FFT));
    const power = powerSpectrum(padded, N_FFT);
    const melRow = new Float32Array(128);
    for (let m = 0; m < 128; m++) {
      let val = 0;
      for (let k = 0; k < power.length; k++)
        val += MEL_FILTERS[m][k] * power[k];
      melRow[m] = Math.log(Math.max(val, 1e-10));
    }
    melFrames.push(melRow);
  }
  return melFrames; // [n_frames][128]
}

/**
 * Compute MFCCs from log-mel spectrogram via DCT-II.
 * Returns 2D array [N_MFCC][n_frames].
 */
function computeMFCC(signal) {
  const melFrames = logMelSpectrogram(signal);
  const nFrames = melFrames.length;
  const nMels = 128;

  // DCT-II: mfcc[k] = sum_n logMel[n] * cos(pi*k*(2n+1) / (2*nMels))
  const mfccs = [];
  for (let k = 0; k < N_MFCC; k++) {
    const row = new Float32Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      let val = 0;
      for (let n = 0; n < nMels; n++) {
        val +=
          melFrames[t][n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * nMels));
      }
      row[t] = val;
    }
    mfccs.push(row);
  }
  return mfccs; // [N_MFCC][n_frames]
}

/**
 * Compute delta (first derivative) of a 2D feature matrix.
 * Returns same shape [nCoeff][nFrames].
 */
function delta(features, width = 9) {
  const nCoeff = features.length;
  const nFrames = features[0].length;
  const result = [];
  const halfW = Math.floor(width / 2);
  const norm =
    2 *
    Array.from({ length: halfW }, (_, i) => (i + 1) ** 2).reduce(
      (a, b) => a + b,
      0,
    );

  for (let k = 0; k < nCoeff; k++) {
    const row = new Float32Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      let val = 0;
      for (let n = 1; n <= halfW; n++) {
        const tPlus = Math.min(t + n, nFrames - 1);
        const tMinus = Math.max(t - n, 0);
        val += n * (features[k][tPlus] - features[k][tMinus]);
      }
      row[t] = val / norm;
    }
    result.push(row);
  }
  return result;
}

// ══════════════════════════════════════════════════════════════════════════════
//  AGGREGATION
// ══════════════════════════════════════════════════════════════════════════════

function mean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr, m = null) {
  if (!arr.length) return 0;
  const mu = m !== null ? m : mean(arr);
  const variance = arr.reduce((a, b) => a + (b - mu) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

function min(arr) {
  return arr.length ? Math.min(...arr) : 0;
}
function max(arr) {
  return arr.length ? Math.max(...arr) : 0;
}

/**
 * Return {mean, std, min, max} for a Float32Array or Array.
 */
function aggAll(arr) {
  const a = Array.from(arr);
  return { mean: mean(a), std: std(a), min: min(a), max: max(a) };
}

// ══════════════════════════════════════════════════════════════════════════════
//  FEATURE EXTRACTION
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Extract all features into a named dict, then select and order
 * by SELECTED_FEATURES to produce the final 85-dim Float32Array.
 *
 * @param {Float32Array} signal  — mono PCM audio at SR (16kHz)
 * @returns {Float32Array}       — 85-dim feature vector
 */
function extractFeatures(signal) {
  const feat = {};

  // ── MFCCs ──────────────────────────────────────────────────────────────────
  const mfcc = computeMFCC(signal); // [13][n_frames]
  const mfccD1 = delta(mfcc); // [13][n_frames]
  const mfccD2 = delta(mfccD1); // [13][n_frames]

  for (let i = 0; i < N_MFCC; i++) {
    const c = i + 1; // 1-indexed
    const a = aggAll(mfcc[i]);
    const a1 = aggAll(mfccD1[i]);
    const a2 = aggAll(mfccD2[i]);

    feat[`mfcc_${c}_mean`] = a.mean;
    feat[`mfcc_${c}_std`] = a.std;
    feat[`mfcc_${c}_min`] = a.min;
    feat[`mfcc_${c}_max`] = a.max;

    feat[`mfcc_d1_${c}_mean`] = a1.mean;
    feat[`mfcc_d1_${c}_std`] = a1.std;
    feat[`mfcc_d1_${c}_min`] = a1.min;
    feat[`mfcc_d1_${c}_max`] = a1.max;

    feat[`mfcc_d2_${c}_mean`] = a2.mean;
    feat[`mfcc_d2_${c}_std`] = a2.std;
    feat[`mfcc_d2_${c}_min`] = a2.min;
    feat[`mfcc_d2_${c}_max`] = a2.max;
  }

  // ── RMS Energy ─────────────────────────────────────────────────────────────
  const audioFrames = frame(signal, FRAME_LENGTH, HOP_LENGTH);
  const rmsFrames = audioFrames.map((f) => {
    const sumSq = f.reduce((a, b) => a + b * b, 0);
    return Math.sqrt(sumSq / f.length);
  });
  const rmsAgg = aggAll(rmsFrames);
  feat["rms_mean"] = rmsAgg.mean;
  feat["rms_std"] = rmsAgg.std;
  feat["rms_min"] = rmsAgg.min;
  feat["rms_max"] = rmsAgg.max;

  // ── Zero Crossing Rate (mean only) ─────────────────────────────────────────
  const zcrFrames = audioFrames.map((f) => {
    let crossings = 0;
    for (let i = 1; i < f.length; i++) {
      if (f[i] >= 0 !== f[i - 1] >= 0) crossings++;
    }
    return crossings / f.length;
  });
  feat["zcr_mean"] = mean(zcrFrames);

  // ── Spectral Features ──────────────────────────────────────────────────────
  const specFrames = frame(signal, N_FFT, HOP_LENGTH);
  const hann = hannWindow(N_FFT);
  const freqBins = Array.from(
    { length: Math.floor(N_FFT / 2) + 1 },
    (_, i) => (i * SR) / N_FFT,
  );

  const rolloffVals = [];
  const flatnessVals = [];

  for (const f of specFrames) {
    const padded = new Float32Array(N_FFT);
    padded.set(f.length < N_FFT ? f : f.slice(0, N_FFT));
    for (let i = 0; i < N_FFT; i++) padded[i] *= hann[i];

    const power = powerSpectrum(padded, N_FFT);
    const totalE = power.reduce((a, b) => a + b, 0);

    // Spectral rolloff (85th percentile of energy)
    let cumE = 0;
    const threshold = 0.85 * totalE;
    let rolloff = freqBins[freqBins.length - 1];
    for (let k = 0; k < power.length; k++) {
      cumE += power[k];
      if (cumE >= threshold) {
        rolloff = freqBins[k];
        break;
      }
    }
    rolloffVals.push(rolloff);

    // Spectral flatness: geometric mean / arithmetic mean
    let logSum = 0;
    for (let k = 0; k < power.length; k++) {
      logSum += Math.log(Math.max(power[k], 1e-10));
    }
    const geomMean = Math.exp(logSum / power.length);
    const arithMean = totalE / power.length;
    flatnessVals.push(arithMean > 0 ? geomMean / arithMean : 0);
  }

  const rolloffAgg = aggAll(rolloffVals);
  feat["rolloff_mean"] = rolloffAgg.mean;
  feat["rolloff_std"] = rolloffAgg.std;
  feat["flatness_std"] = std(flatnessVals);

  // ── Pitch / F0 (mean only) ─────────────────────────────────────────────────
  // Autocorrelation-based F0 estimation
  feat["f0_mean"] = estimateF0Mean(signal);

  // ── Assemble final vector in SELECTED_FEATURES order ──────────────────────
  const vector = new Float32Array(SELECTED_FEATURES.length);
  for (let i = 0; i < SELECTED_FEATURES.length; i++) {
    const val = feat[SELECTED_FEATURES[i]];
    vector[i] = val === undefined || !isFinite(val) ? 0 : val;
  }
  return vector;
}

/**
 * Estimate mean F0 using autocorrelation on overlapping frames.
 * Only uses voiced frames (those with a clear periodic peak).
 */
function estimateF0Mean(signal) {
  const minLag = Math.round(SR / F0_MAX); // ~53 samples
  const maxLag = Math.round(SR / F0_MIN); // ~213 samples
  const frames = frame(signal, FRAME_LENGTH, HOP_LENGTH);
  const f0s = [];

  for (const f of frames) {
    // Normalised autocorrelation
    const acf = new Float32Array(maxLag + 1);
    const energy = f.reduce((a, b) => a + b * b, 0);
    if (energy < 1e-6) continue; // silent frame

    for (let lag = minLag; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < f.length - lag; i++) sum += f[i] * f[i + lag];
      acf[lag] = sum / energy;
    }

    // Find peak in valid range
    let peakVal = 0,
      peakLag = -1;
    for (let lag = minLag; lag <= maxLag; lag++) {
      if (acf[lag] > peakVal) {
        peakVal = acf[lag];
        peakLag = lag;
      }
    }

    // Only accept voiced frames (strong periodic peak)
    if (peakVal > 0.3 && peakLag > 0) {
      f0s.push(SR / peakLag);
    }
  }

  return f0s.length > 0 ? mean(f0s) : 0;
}

// ══════════════════════════════════════════════════════════════════════════════
//  STRESS DETECTOR CLASS
// ══════════════════════════════════════════════════════════════════════════════

class StressDetector {
  constructor(modelPaths = {}) {
    this.modelPaths = {
      xgboost: modelPaths.xgboost || "xgboost.onnx",
      randomForest: modelPaths.randomForest || "./models/random_forest.onnx",
      logisticRegression:
        modelPaths.logisticRegression || "./models/logistic_regression.onnx",
    };
    this.sessions = {};
    this.loaded = false;
  }

  /**
   * Load all three ONNX models. Call once before predict().
   */
  async load() {
    console.log("[StressDetector] Loading ONNX models...");
    const entries = Object.entries(this.modelPaths);

    await Promise.all(
      entries.map(async ([name, path]) => {
        try {
          this.sessions[name] = await ort.InferenceSession.create(path);
          console.log(`  [OK] ${name} loaded from ${path}`);
        } catch (e) {
          console.error(`  [FAIL] ${name}: ${e.message}`);
          throw e;
        }
      }),
    );

    this.loaded = true;
    console.log("[StressDetector] All models loaded.");
  }

  /**
   * Run inference on a single model and return stressed probability.
   * Handles both array output (RF/LR) and dict output (some XGBoost configs).
   */
  async _runModel(sessionName, tensor) {
    const session = this.sessions[sessionName];
    const inputName = session.inputNames[0];
    const feeds = { [inputName]: tensor };
    const results = await session.run(feeds);

    // Try to find probability output
    // Models may output: [label, probabilities] or just [probabilities]
    for (const outputName of session.outputNames) {
      const output = results[outputName];
      // Float32 array of shape [1, 2] — [prob_class0, prob_class1]
      if (
        output.type === "float32" &&
        output.dims[output.dims.length - 1] === 2
      ) {
        return output.data[1]; // probability of class 1 (stressed)
      }
    }

    // Fallback: binary output — return as probability
    const firstOutput = results[session.outputNames[0]];
    return Number(firstOutput.data[0]);
  }

  /**
   * Predict stress from a Float32Array of mono PCM audio at 16kHz.
   *
   * @param {Float32Array} audioData  — mono float32 PCM at 16kHz
   * @returns {Object} {
   *   stressed:     boolean,
   *   probability:  float [0-1],  averaged across models
   *   level:        string,       "not_stressed" | "stressed" | "highly_stressed"
   *   features:     Float32Array, the 85-dim feature vector
   *   modelScores:  object,       per-model probabilities
   * }
   */
  async predict(audioData, stutterFlag = false) {
    if (!this.loaded) throw new Error("Models not loaded. Call load() first.");

    // Extract features
    const features = extractFeatures(audioData);

    // Build ONNX tensor — shape [1, 85]
    const tensor = new ort.Tensor("float32", features, [
      1,
      SELECTED_FEATURES.length,
    ]);

    // Run all three models in parallel
    const [pXgb, pRf, pLr] = await Promise.all([
      this._runModel("xgboost", tensor),
      this._runModel("randomForest", tensor),
      this._runModel("logisticRegression", tensor),
    ]);

    // Soft voting: average probabilities
    const avgProb = (pXgb + pRf + pLr) / 3;
    const isStressed = avgProb >= STRESS_THRESHOLD;

    // Combine with stutter flag for final stress level
    let level;
    if (isStressed && stutterFlag) {
      level = "highly_stressed";
    } else if (isStressed) {
      level = "stressed";
    } else if (!isStressed && stutterFlag) {
      level = "mild";
    } else {
      level = "not_stressed";
    }

    return {
      stressed: isStressed,
      probability: avgProb,
      level,
      features,
      modelScores: {
        xgboost: pXgb,
        randomForest: pRf,
        logisticRegression: pLr,
      },
    };
  }

  /**
   * Convenience: decode a Web Audio API AudioBuffer and predict.
   * Resamples to 16kHz mono if needed.
   *
   * @param {AudioBuffer} audioBuffer
   */
  async predictFromAudioBuffer(audioBuffer, stutterFlag = false) {
    let signal = audioBuffer.getChannelData(0); // mono / left channel

    // Resample to 16kHz if necessary
    if (audioBuffer.sampleRate !== SR) {
      signal = resample(signal, audioBuffer.sampleRate, SR);
    }

    return this.predict(signal, stutterFlag);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  RESAMPLING  (simple linear interpolation)
// ══════════════════════════════════════════════════════════════════════════════

function resample(signal, fromSR, toSR) {
  if (fromSR === toSR) return signal;
  const ratio = fromSR / toSR;
  const outLength = Math.round(signal.length / ratio);
  const out = new Float32Array(outLength);

  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = signal[idx] || 0;
    const b = signal[idx + 1] || 0;
    out[i] = a + frac * (b - a);
  }
  return out;
}

// ══════════════════════════════════════════════════════════════════════════════
//  EXPORTS
// ══════════════════════════════════════════════════════════════════════════════

// Browser global
if (typeof window !== "undefined") {
  window.StressDetector = StressDetector;
}

// Node / ES module
if (typeof module !== "undefined") {
  module.exports = { StressDetector, extractFeatures, SELECTED_FEATURES };
}

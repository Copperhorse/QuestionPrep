/**
 * stress-detector.js — Client-side Stress & Disfluency Analyser
 *
 * Fixes applied:
 *   B29 - XGBoost default model path was 'xgboost.onnx' (bare filename).
 *         Fixed: default is now './models/xgboost.onnx' — consistent with the others.
 *
 *   OPT1 - _melFilterbank() was recomputed on every call to _extractFeatures()
 *          even though the parameters (N_MELS, FFT_SIZE, sampleRate) never change
 *          after construction.  The result is now cached in this._cachedMelFB on
 *          the first prediction and reused on all subsequent ones.
 *
 *   OPT2 - predict() now returns a `modelScores` object exposing the individual
 *          probability from each of the three classifiers.  interview.html uses
 *          these values to render the per-model pill badges in the stress panel.
 *          Previously the field was missing, causing a TypeError in showStressBanner.
 */

class StressDetector {
  /**
   * @param {object} opts
   * @param {string} [opts.xgboostPath='./models/xgboost.onnx']       B29: was 'xgboost.onnx'
   * @param {string} [opts.rfPath='./models/random_forest.onnx']
   * @param {string} [opts.lrPath='./models/logistic_regression.onnx']
   * @param {number} [opts.sampleRate=16000]
   */
  constructor({
    xgboostPath = "./models/xgboost.onnx", // B29: was 'xgboost.onnx'
    rfPath = "./models/random_forest.onnx",
    lrPath = "./models/logistic_regression.onnx",
    sampleRate = 16000,
  } = {}) {
    this.paths = { xgboost: xgboostPath, rf: rfPath, lr: lrPath };
    this.sampleRate = sampleRate;
    this.sessions = {};
    this.ready = false;

    // OPT1: Mel filterbank cache — computed once on first prediction.
    this._cachedMelFB = null;

    this._initPromise = this._loadModels();
  }

  // ── Model loading ──────────────────────────────────────────────────────────

  async _loadModels() {
    try {
      const [xgb, rf, lr] = await Promise.all([
        ort.InferenceSession.create(this.paths.xgboost),
        ort.InferenceSession.create(this.paths.rf),
        ort.InferenceSession.create(this.paths.lr),
      ]);
      this.sessions = { xgboost: xgb, rf, lr };
      this.ready = true;
      console.log("[StressDetector] All 3 ONNX models loaded:", this.paths);
    } catch (err) {
      console.error("[StressDetector] Model load failed:", err);
      this.ready = false;
    }
  }

  async waitUntilReady(timeoutMs = 15000) {
    await this._initPromise;
    if (this.ready) return;
    // Poll briefly in case the promise resolved without setting ready
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 100));
      if (this.ready) return;
    }
    throw new Error("[StressDetector] Timed out waiting for models");
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Predict stress level from raw audio and an optional stutter flag.
   * @param {Float32Array} audioBuffer  — mono PCM at this.sampleRate
   * @param {boolean}      stutterFlag  — from the server's disfluency analyser
   * @returns {{
   *   level: string,
   *   probability: number,
   *   stutter_flag: boolean,
   *   modelScores: { xgboost: number, randomForest: number, logisticRegression: number }
   * }}
   */
  async predict(audioBuffer, stutterFlag = false) {
    if (!this.ready) {
      console.warn("[StressDetector] Models not ready — returning default");
      return {
        level: "unknown",
        probability: 0,
        stutter_flag: stutterFlag,
        // OPT2: always include modelScores so callers don't need to guard
        modelScores: { xgboost: 0, randomForest: 0, logisticRegression: 0 },
      };
    }

    try {
      const features = this._extractFeatures(audioBuffer);
      const inputTensor = new ort.Tensor("float32", features, [
        1,
        features.length,
      ]);
      const inputName = "input"; // adjust to match your ONNX export

      const [pXgb, pRf, pLr] = await Promise.all([
        this._runSession(this.sessions.xgboost, inputName, inputTensor),
        this._runSession(this.sessions.rf, inputName, inputTensor),
        this._runSession(this.sessions.lr, inputName, inputTensor),
      ]);

      // Soft voting
      const avgProb = (pXgb + pRf + pLr) / 3;
      const isStressed = avgProb >= 0.5;

      return {
        level: this._toLevel(isStressed, stutterFlag),
        probability: Math.round(avgProb * 100) / 100,
        stutter_flag: stutterFlag,
        // OPT2: expose individual model scores so the UI can render per-model pills
        modelScores: {
          xgboost: Math.round(pXgb * 100) / 100,
          randomForest: Math.round(pRf * 100) / 100,
          logisticRegression: Math.round(pLr * 100) / 100,
        },
      };
    } catch (err) {
      console.error("[StressDetector] predict() failed:", err);
      return {
        level: "unknown",
        probability: 0,
        stutter_flag: stutterFlag,
        modelScores: { xgboost: 0, randomForest: 0, logisticRegression: 0 },
      };
    }
  }

  // ── Internal helpers ───────────────────────────────────────────────────────

  async _runSession(session, inputName, tensor) {
    const feeds = { [inputName]: tensor };
    const results = await session.run(feeds);
    // Expect a probability output named 'probabilities' or 'output_probability'
    const out =
      results["probabilities"] ||
      results["output_probability"] ||
      results[Object.keys(results)[0]];
    const data = out.data;
    // Binary classifier: data[1] = P(stressed)
    return data.length >= 2 ? data[1] : data[0];
  }

  _toLevel(isStressed, stutterFlag) {
    if (isStressed && stutterFlag) return "highly_stressed";
    if (isStressed && !stutterFlag) return "stressed";
    if (!isStressed && stutterFlag) return "mild";
    return "not_stressed";
  }

  // ── Feature extraction (85-dim) ────────────────────────────────────────────

  _extractFeatures(audio) {
    const N_MELS = 128;
    const N_MFCC = 13;
    const FFT_SIZE = 512;
    const HOP = 256;

    const frames = this._frame(audio, FFT_SIZE, HOP);
    const spectra = frames.map((f) => this._powerSpectrum(f, FFT_SIZE));

    // OPT1: Build the mel filterbank once and cache it.
    // N_MELS, FFT_SIZE, and sampleRate are fixed after construction so the
    // result never changes between calls — no need to recompute every time.
    if (!this._cachedMelFB) {
      this._cachedMelFB = this._melFilterbank(N_MELS, FFT_SIZE, this.sampleRate);
    }
    const melFB = this._cachedMelFB;

    // MFCCs
    const mfccs = spectra.map((s) => {
      const mel = melFB.map((fb) => {
        const energy = fb.reduce((sum, w, i) => sum + w * s[i], 0);
        return Math.log(Math.max(energy, 1e-10));
      });
      return this._dct(mel, N_MFCC);
    });

    const mfccMean = this._colMean(mfccs);
    const mfccDelta = this._colMean(this._delta(mfccs));

    // Spectral descriptors
    const rms = spectra.map((s) =>
      Math.sqrt(s.reduce((a, v) => a + v * v, 0) / s.length),
    );
    const zcr = frames.map((f) => this._zeroCrossingRate(f));
    const rolloff = spectra.map((s) =>
      this._spectralRolloff(s, this.sampleRate, FFT_SIZE),
    );
    const flatness = spectra.map((s) => this._spectralFlatness(s));
    const f0 = frames.map((f) => this._estimateF0(f, this.sampleRate));

    const features = [
      ...mfccMean, // 13
      ...mfccDelta, // 13
      this._mean(rms), //  1
      this._std(rms), //  1
      this._mean(zcr), //  1
      this._std(zcr), //  1
      this._mean(rolloff), //  1
      this._std(rolloff), //  1
      this._mean(flatness), //  1
      this._std(flatness), //  1
      this._mean(f0), //  1
      this._std(f0), //  1
      ...this._melBandEnergies(spectra, melFB, 24), // 24
      this._dynamicRange(rms), //  1
      this._spectralCentroid(spectra, this.sampleRate, FFT_SIZE), // 1
      this._jitter(f0), //  1
      this._shimmer(rms), //  1
      this._hnr(audio, this.sampleRate), //  1
      // pad or trim to exactly 85
    ];

    return Float32Array.from(this._padOrTrim(features, 85));
  }

  // ── DSP helpers ────────────────────────────────────────────────────────────

  _frame(audio, size, hop) {
    const frames = [];
    for (let i = 0; i + size <= audio.length; i += hop) {
      frames.push(audio.slice(i, i + size));
    }
    return frames.length ? frames : [new Float32Array(size)];
  }

  _powerSpectrum(frame, fftSize) {
    // Apply Hann window
    const windowed = new Float32Array(fftSize);
    for (let i = 0; i < frame.length && i < fftSize; i++) {
      windowed[i] =
        frame[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (fftSize - 1)));
    }
    const { re, im } = this._fft(windowed);
    const half = Math.floor(fftSize / 2) + 1;
    return Float32Array.from(
      { length: half },
      (_, k) => re[k] ** 2 + im[k] ** 2,
    );
  }

  _fft(x) {
    const N = x.length;
    const re = Float32Array.from(x);
    const im = new Float32Array(N);

    // Cooley-Tukey iterative FFT
    let j = 0;
    for (let i = 1; i < N; i++) {
      let bit = N >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }

    for (let len = 2; len <= N; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang),
        wIm = Math.sin(ang);
      for (let i = 0; i < N; i += len) {
        let curRe = 1,
          curIm = 0;
        for (let k = 0; k < len / 2; k++) {
          const uRe = re[i + k],
            uIm = im[i + k];
          const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
          const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
          re[i + k] = uRe + vRe;
          im[i + k] = uIm + vIm;
          re[i + k + len / 2] = uRe - vRe;
          im[i + k + len / 2] = uIm - vIm;
          const newRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = newRe;
        }
      }
    }
    return { re, im };
  }

  _melFilterbank(nMels, fftSize, sr) {
    const half = Math.floor(fftSize / 2) + 1;
    const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
    const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

    const melMin = hzToMel(0),
      melMax = hzToMel(sr / 2);
    const melPts = Array.from({ length: nMels + 2 }, (_, i) =>
      melToHz(melMin + (i * (melMax - melMin)) / (nMels + 1)),
    );

    const bins = melPts.map((hz) => Math.floor((hz * fftSize) / sr));

    return Array.from({ length: nMels }, (_, m) => {
      const fb = new Float32Array(half);
      for (let k = bins[m]; k < bins[m + 1]; k++)
        fb[k] = (k - bins[m]) / Math.max(1, bins[m + 1] - bins[m]);
      for (let k = bins[m + 1]; k < bins[m + 2]; k++)
        fb[k] = (bins[m + 2] - k) / Math.max(1, bins[m + 2] - bins[m + 1]);
      return fb;
    });
  }

  _dct(x, nCoeffs) {
    const N = x.length;
    return Array.from({ length: nCoeffs }, (_, k) =>
      x.reduce(
        (s, v, n) => s + v * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N)),
        0,
      ),
    );
  }

  _delta(matrix) {
    const n = matrix.length;
    return matrix.map((row, i) => {
      const prev = matrix[Math.max(0, i - 1)];
      const next = matrix[Math.min(n - 1, i + 1)];
      return row.map((_, j) => (next[j] - prev[j]) / 2);
    });
  }

  _colMean(matrix) {
    const cols = matrix[0].length;
    return Array.from({ length: cols }, (_, j) =>
      this._mean(matrix.map((r) => r[j])),
    );
  }

  _melBandEnergies(spectra, melFB, nBands) {
    const bandSize = Math.floor(melFB.length / nBands);
    return Array.from({ length: nBands }, (_, b) => {
      const start = b * bandSize,
        end = start + bandSize;
      const bandFB = melFB.slice(start, end);
      const energies = spectra.map((s) =>
        bandFB.reduce(
          (sum, fb) => sum + fb.reduce((a, w, i) => a + w * s[i], 0),
          0,
        ),
      );
      return this._mean(energies);
    });
  }

  _zeroCrossingRate(frame) {
    let zc = 0;
    for (let i = 1; i < frame.length; i++)
      if (frame[i - 1] * frame[i] < 0) zc++;
    return zc / frame.length;
  }

  _spectralRolloff(spectrum, sr, fftSize) {
    const total = spectrum.reduce((a, v) => a + v, 0);
    let cum = 0;
    for (let i = 0; i < spectrum.length; i++) {
      cum += spectrum[i];
      if (cum >= 0.85 * total) return (i * sr) / fftSize;
    }
    return sr / 2;
  }

  _spectralFlatness(spectrum) {
    const n = spectrum.length;
    const gm = Math.exp(
      spectrum.reduce((a, v) => a + Math.log(Math.max(v, 1e-10)), 0) / n,
    );
    const am = spectrum.reduce((a, v) => a + v, 0) / n;
    return gm / Math.max(am, 1e-10);
  }

  _spectralCentroid(spectra, sr, fftSize) {
    const cents = spectra.map((s) => {
      const total = s.reduce((a, v) => a + v, 0);
      return (
        s.reduce((a, v, i) => a + (v * (i * sr)) / fftSize, 0) /
        Math.max(total, 1e-10)
      );
    });
    return this._mean(cents);
  }

  _estimateF0(frame, sr) {
    // Simple autocorrelation pitch estimate
    const minPeriod = Math.floor(sr / 500); // ~500 Hz max
    const maxPeriod = Math.floor(sr / 50); // ~50 Hz min
    let bestCorr = -Infinity,
      bestPeriod = 0;
    for (let t = minPeriod; t <= maxPeriod && t < frame.length; t++) {
      let corr = 0;
      for (let i = 0; i + t < frame.length; i++)
        corr += frame[i] * frame[i + t];
      if (corr > bestCorr) {
        bestCorr = corr;
        bestPeriod = t;
      }
    }
    return bestPeriod > 0 ? sr / bestPeriod : 0;
  }

  _jitter(f0Array) {
    if (f0Array.length < 2) return 0;
    const voiced = f0Array.filter((v) => v > 0);
    if (voiced.length < 2) return 0;
    let sum = 0;
    for (let i = 1; i < voiced.length; i++)
      sum += Math.abs(voiced[i] - voiced[i - 1]);
    return sum / (voiced.length - 1) / Math.max(this._mean(voiced), 1);
  }

  _shimmer(rmsArray) {
    if (rmsArray.length < 2) return 0;
    let sum = 0;
    for (let i = 1; i < rmsArray.length; i++)
      sum += Math.abs(rmsArray[i] - rmsArray[i - 1]);
    return sum / (rmsArray.length - 1) / Math.max(this._mean(rmsArray), 1e-10);
  }

  _hnr(audio, sr) {
    // Simplified HNR via autocorrelation ratio
    const period = Math.floor(sr / 120); // assume 120 Hz fundamental
    if (period <= 0 || period >= audio.length) return 0;
    let signal = 0,
      noise = 0;
    for (let i = period; i < audio.length; i++) {
      signal += audio[i] * audio[i - period];
      noise += (audio[i] - audio[i - period]) ** 2;
    }
    return 10 * Math.log10(Math.max(signal, 1e-10) / Math.max(noise, 1e-10));
  }

  _dynamicRange(rmsArray) {
    const max = Math.max(...rmsArray),
      min = Math.min(...rmsArray);
    return max - min;
  }

  _mean(arr) {
    return arr.length ? arr.reduce((a, v) => a + v, 0) / arr.length : 0;
  }
  _std(arr) {
    const m = this._mean(arr);
    return Math.sqrt(
      arr.reduce((a, v) => a + (v - m) ** 2, 0) / Math.max(arr.length, 1),
    );
  }
  _padOrTrim(arr, n) {
    if (arr.length >= n) return arr.slice(0, n);
    return [...arr, ...new Array(n - arr.length).fill(0)];
  }
}

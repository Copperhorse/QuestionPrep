/**
 * audio-worklet-processor.js — AudioWorklet for low-latency PCM streaming
 *
 * Runs on the audio rendering thread (not main thread).
 * Captures raw Float32 PCM and streams chunks to the main thread via MessagePort.
 *
 * Message protocol (processor → main):
 *   { type: 'CHUNK', pcm: Float32Array, timestamp: number }
 *
 * Message protocol (main → processor):
 *   { type: 'STOP' } — cleanly stop processing
 */

const DEBUG = true; // Set true for worklet-level logging

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._active = true;
    this._chunkSize = 4096; // ~256ms @ 16kHz — matches old ScriptProcessor buffer
    this._buffer = new Float32Array(this._chunkSize);
    this._writePos = 0;
    this._chunkId = 0;

    this.port.onmessage = (e) => {
      if (e.data?.type === "STOP") {
        this._active = false;
        if (DEBUG) console.log("[AudioWorklet] Received STOP signal");
      }
    };
  }

  process(inputs, outputs, parameters) {
    if (!this._active) return false;

    const input = inputs[0];
    if (!input || !input[0]) return true; // no input connected

    const channelData = input[0];
    let readPos = 0;

    while (readPos < channelData.length) {
      const remaining = this._chunkSize - this._writePos;
      const toCopy = Math.min(remaining, channelData.length - readPos);

      this._buffer.set(
        channelData.subarray(readPos, readPos + toCopy),
        this._writePos,
      );
      this._writePos += toCopy;
      readPos += toCopy;

      if (this._writePos >= this._chunkSize) {
        // Buffer full — send to main thread (transfer ownership for zero-copy)
        const chunk = new Float32Array(this._buffer); // copy out
        this.port.postMessage(
          {
            type: "CHUNK",
            pcm: chunk,
            chunkId: this._chunkId++,
            timestamp: currentTime,
            sampleRate: sampleRate,
          },
          [chunk.buffer],
        );

        if (DEBUG) {
          console.log(
            `[AudioWorklet] Sent chunk #${this._chunkId - 1}, ` +
              `samples=${chunk.length}, sr=${sampleRate}`,
          );
        }
        this._writePos = 0;
      }
    }

    return true; // keep alive
  }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);

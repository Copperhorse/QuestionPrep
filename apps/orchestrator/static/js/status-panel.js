/**
 * status-panel.js — QuestionPrep live backend status panel
 *
 * Connects to GET /api/events (Server-Sent Events) and renders a
 * floating terminal that shows exactly what the backend is doing.
 * Also polls GET /api/status every 5 s to keep the header badge current.
 *
 * Improvements (senior engineer pass):
 *
 *   OPT1 - highlightMsg() previously applied 6 sequential regex replacements,
 *          each of which scanned the entire string from the beginning.
 *          Replaced with a single compiled alternation regex that makes one
 *          pass and dispatches to the right CSS class via a lookup Map.
 *          The regex literals are module-level constants — compiled once by
 *          the JS engine at parse time, not re-created on every log line.
 *
 *   OPT2 - DOM pruning previously looped with removeChild(firstChild) until
 *          the count was within MAX_LINES.  Each removeChild triggers a
 *          style/layout recalculation.  Replaced with a document.createRange()
 *          + deleteContents() call that removes all excess nodes in a single
 *          DOM mutation, reducing layout work from O(excess) to O(1).
 */

"use strict";

(function QuestionPrepStatusPanel() {
  // ── Config ──────────────────────────────────────────────────────────────────
  const MAX_LINES = 200;
  const STATUS_POLL_MS = 5_000;
  const RECONNECT_MS = 3_000;

  // ── OPT1: Highlight pattern — compiled ONCE at module parse time ────────────
  //
  // The original code ran six .replace() calls in sequence, each with its own
  // regex literal.  Even though modern JS engines cache regex literals, six
  // separate string traversals still happen per log line.
  //
  // This single alternation regex does one traversal.  The capturing group
  // lets the replacer determine which branch matched by inspecting the first
  // character — same information the original six patterns encoded, just
  // unified into one compiled automaton.
  //
  // Branch order matters: longer/more-specific patterns come first so they
  // shadow overlapping shorter ones (e.g. a hex hash that also starts with a
  // digit should match the hash branch, not the timing branch).
  const _HL_RE = /(✓[^<]*)|(✗[^<]*)|([▶■][^<]*)|([a-f0-9]{8,})|(\d+\.\d+s)/g;

  // OPT1: Map from capturing-group index (1-based) to CSS class.
  const _HL_CLASS = [
    "qp-hl-ok",
    "qp-hl-err",
    "qp-hl-warn",
    "qp-hl-path",
    "qp-hl-ok",
  ];

  function highlightMsg(msg) {
    // Reset lastIndex so the stateful /g regex starts fresh on every call.
    _HL_RE.lastIndex = 0;
    return msg.replace(_HL_RE, (...args) => {
      // args: [fullMatch, g1, g2, g3, g4, g5, offset, string]
      // Find which capture group matched (first non-undefined after fullMatch).
      for (let i = 1; i <= 5; i++) {
        if (args[i] !== undefined) {
          return `<span class="${_HL_CLASS[i - 1]}">${args[i]}</span>`;
        }
      }
      return args[0]; // should never reach here
    });
  }

  // ── Colours for log level tokens ───────────────────────────────────────────
  const LEVEL_COLOURS = {
    INFO: "#7ec8a0",
    WARNING: "#f0c060",
    ERROR: "#e07070",
    DEBUG: "#8ab4d4",
    CRITICAL: "#e07070",
  };

  // ── CSS injected once ───────────────────────────────────────────────────────
  const CSS = `
    #qp-status-root {
      position: fixed;
      bottom: 1.25rem;
      right: 1.25rem;
      z-index: 10000;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 0.5rem;
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
      font-size: 12px;
    }

    #qp-status-toggle {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: #1e1e2e;
      color: #cdd6f4;
      border: 1px solid #313244;
      border-radius: 8px;
      padding: 0.45rem 0.9rem;
      cursor: pointer;
      user-select: none;
      box-shadow: 0 4px 14px rgba(0,0,0,0.4);
      transition: background 0.2s;
      white-space: nowrap;
    }
    #qp-status-toggle:hover { background: #2a2a3e; }

    .qp-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: #45475a;
      flex-shrink: 0;
      transition: background 0.4s;
    }
    .qp-dot.active   { background: #a6e3a1; box-shadow: 0 0 6px #a6e3a1aa; }
    .qp-dot.busy     { background: #f9e2af; box-shadow: 0 0 6px #f9e2afaa;
                        animation: qp-pulse 1s ease-in-out infinite; }
    .qp-dot.error    { background: #f38ba8; box-shadow: 0 0 6px #f38ba8aa; }
    @keyframes qp-pulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.45; }
    }

    #qp-status-panel {
      width: 520px;
      max-width: calc(100vw - 2.5rem);
      background: #1e1e2e;
      border: 1px solid #313244;
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 8px 30px rgba(0,0,0,0.5);
      display: flex;
      flex-direction: column;
    }
    #qp-status-panel.hidden { display: none; }

    .qp-panel-header {
      background: #181825;
      padding: 0.55rem 0.9rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #313244;
      flex-shrink: 0;
    }
    .qp-panel-title {
      color: #cdd6f4;
      font-weight: 700;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .qp-panel-controls { display: flex; gap: 0.5rem; align-items: center; }
    .qp-ctrl-btn {
      background: none;
      border: none;
      color: #6c7086;
      cursor: pointer;
      font-size: 13px;
      padding: 0 3px;
      line-height: 1;
      transition: color 0.15s;
    }
    .qp-ctrl-btn:hover { color: #cdd6f4; }

    .qp-chips {
      padding: 0.45rem 0.9rem;
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      border-bottom: 1px solid #313244;
      flex-shrink: 0;
    }
    .qp-chip {
      font-size: 10px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
    }
    .qp-chip.off  { color: #6c7086; border-color: #313244; }
    .qp-chip.on   { color: #a6e3a1; border-color: rgba(166,227,161,0.35);
                    background: rgba(166,227,161,0.08); }
    .qp-chip.busy { color: #f9e2af; border-color: rgba(249,226,175,0.35);
                    background: rgba(249,226,175,0.08); }

    #qp-log-output {
      height: 260px;
      overflow-y: auto;
      padding: 0.6rem 0.9rem;
      display: flex;
      flex-direction: column;
      gap: 1px;
      scroll-behavior: smooth;
    }
    #qp-log-output::-webkit-scrollbar { width: 5px; }
    #qp-log-output::-webkit-scrollbar-track { background: transparent; }
    #qp-log-output::-webkit-scrollbar-thumb { background: #45475a; border-radius: 3px; }

    .qp-log-line { display: flex; gap: 0.5rem; line-height: 1.55; flex-shrink: 0; }
    .qp-log-ts   { color: #45475a; flex-shrink: 0; font-size: 10px; padding-top: 1px; }
    .qp-log-level { flex-shrink: 0; font-weight: 700; font-size: 10px; padding-top: 1px; min-width: 44px; }
    .qp-log-msg  { color: #cdd6f4; word-break: break-all; }
    .qp-log-msg .qp-hl-path { color: #89b4fa; }
    .qp-log-msg .qp-hl-ok   { color: #a6e3a1; }
    .qp-log-msg .qp-hl-warn { color: #f9e2af; }
    .qp-log-msg .qp-hl-err  { color: #f38ba8; }

    .qp-panel-footer {
      background: #181825;
      padding: 0.3rem 0.9rem;
      border-top: 1px solid #313244;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    .qp-conn-indicator { font-size: 10px; display: flex; align-items: center; gap: 4px; }
    .qp-conn-dot { width: 6px; height: 6px; border-radius: 50%; background: #45475a; }
    .qp-conn-dot.connected    { background: #a6e3a1; }
    .qp-conn-dot.reconnecting { background: #f9e2af; animation: qp-pulse 1s infinite; }
    .qp-conn-dot.disconnected { background: #f38ba8; }
    .qp-conn-label { color: #6c7086; }
    .qp-line-count { color: #45475a; font-size: 10px; }
  `;

  const styleEl = document.createElement("style");
  styleEl.textContent = CSS;
  document.head.appendChild(styleEl);

  // ── Build DOM ───────────────────────────────────────────────────────────────
  const root = document.createElement("div");
  root.id = "qp-status-root";
  root.innerHTML = `
    <div id="qp-status-panel" class="hidden">
      <div class="qp-panel-header">
        <span class="qp-panel-title">⚙ Backend Status</span>
        <div class="qp-panel-controls">
          <button class="qp-ctrl-btn" id="qp-clear-btn"  title="Clear log">⊘</button>
          <button class="qp-ctrl-btn" id="qp-scroll-btn" title="Scroll to bottom">↓</button>
          <button class="qp-ctrl-btn" id="qp-close-btn"  title="Close">✕</button>
        </div>
      </div>
      <div class="qp-chips">
        <span class="qp-chip off" id="qp-chip-llm"><span>◉</span> LLM server</span>
        <span class="qp-chip off" id="qp-chip-enrich"><span>◎</span> Enrichment</span>
        <span class="qp-chip off" id="qp-chip-asr"><span>◎</span> ASR</span>
        <span class="qp-chip off" id="qp-chip-sessions"><span>◎</span> Sessions: 0</span>
      </div>
      <div id="qp-log-output"></div>
      <div class="qp-panel-footer">
        <div class="qp-conn-indicator">
          <div class="qp-conn-dot disconnected" id="qp-conn-dot"></div>
          <span class="qp-conn-label" id="qp-conn-label">Disconnected</span>
        </div>
        <span class="qp-line-count" id="qp-line-count">0 lines</span>
      </div>
    </div>
    <button id="qp-status-toggle" title="Toggle backend status panel">
      <span class="qp-dot" id="qp-main-dot"></span>
      <span id="qp-toggle-label">Status</span>
    </button>`;
  document.body.appendChild(root);

  // ── Element refs ────────────────────────────────────────────────────────────
  const panel = document.getElementById("qp-status-panel");
  const toggle = document.getElementById("qp-status-toggle");
  const logOutput = document.getElementById("qp-log-output");
  const connDot = document.getElementById("qp-conn-dot");
  const connLabel = document.getElementById("qp-conn-label");
  const lineCount = document.getElementById("qp-line-count");
  const mainDot = document.getElementById("qp-main-dot");
  const toggleLabel = document.getElementById("qp-toggle-label");
  const chipLLM = document.getElementById("qp-chip-llm");
  const chipEnrich = document.getElementById("qp-chip-enrich");
  const chipASR = document.getElementById("qp-chip-asr");
  const chipSessions = document.getElementById("qp-chip-sessions");

  let panelOpen = false;
  let totalLines = 0;
  let autoScroll = true;
  let eventSource = null;

  // ── Panel open/close ────────────────────────────────────────────────────────
  toggle.addEventListener("click", () => {
    panelOpen = !panelOpen;
    panel.classList.toggle("hidden", !panelOpen);
    if (panelOpen && autoScroll) scrollToBottom();
  });

  document.getElementById("qp-close-btn").addEventListener("click", () => {
    panelOpen = false;
    panel.classList.add("hidden");
  });

  document.getElementById("qp-clear-btn").addEventListener("click", () => {
    logOutput.innerHTML = "";
    totalLines = 0;
    lineCount.textContent = "0 lines";
  });

  document
    .getElementById("qp-scroll-btn")
    .addEventListener("click", scrollToBottom);

  logOutput.addEventListener("scroll", () => {
    autoScroll =
      logOutput.scrollHeight - logOutput.scrollTop - logOutput.clientHeight <
      40;
  });

  // ── Log rendering ───────────────────────────────────────────────────────────
  function appendLogLine(rawMsg) {
    const now = new Date();
    const ts =
      `${now.getHours().toString().padStart(2, "0")}:` +
      `${now.getMinutes().toString().padStart(2, "0")}:` +
      `${now.getSeconds().toString().padStart(2, "0")}`;

    const levelMatch = rawMsg.match(/^\[(\w+)\]\s*/);
    let level = "INFO";
    let msg = rawMsg;
    if (levelMatch) {
      level = levelMatch[1].toUpperCase();
      msg = rawMsg.slice(levelMatch[0].length);
    }

    const colour = LEVEL_COLOURS[level] || "#cdd6f4";

    const line = document.createElement("div");
    line.className = "qp-log-line";
    line.innerHTML =
      `<span class="qp-log-ts">${ts}</span>` +
      `<span class="qp-log-level" style="color:${colour}">${level.padEnd(5)}</span>` +
      `<span class="qp-log-msg">${highlightMsg(escHtml(msg))}</span>`;
    logOutput.appendChild(line);

    totalLines++;

    // OPT2: Batch-prune excess lines with a Range deletion.
    // The old while-loop called removeChild() up to `excess` times, each
    // triggering a style/layout recalculation.  createRange().deleteContents()
    // removes all excess nodes in a single DOM mutation — O(1) layout cost.
    const excess = logOutput.children.length - MAX_LINES;
    if (excess > 0) {
      const range = document.createRange();
      range.setStartBefore(logOutput.firstChild);
      range.setEndBefore(logOutput.children[excess]);
      range.deleteContents();
    }

    lineCount.textContent = `${totalLines} lines`;
    if (autoScroll) scrollToBottom();
    updateMainDotFromMsg(rawMsg, level);
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function scrollToBottom() {
    logOutput.scrollTop = logOutput.scrollHeight;
  }

  // ── Main dot state ──────────────────────────────────────────────────────────
  let _dotState = "idle";

  function setDotState(state) {
    if (_dotState === state) return;
    _dotState = state;
    mainDot.className = "qp-dot";
    switch (state) {
      case "idle":
        toggleLabel.textContent = "Status";
        break;
      case "loading":
        mainDot.classList.add("busy");
        toggleLabel.textContent = "Loading\u2026";
        break;
      case "busy":
        mainDot.classList.add("active");
        toggleLabel.textContent = "Working\u2026";
        break;
      case "error":
        mainDot.classList.add("error");
        toggleLabel.textContent = "Error";
        break;
    }
  }

  function updateMainDotFromMsg(msg, level) {
    const lc = msg.toLowerCase();
    if (level === "ERROR" || level === "CRITICAL") {
      setDotState("error");
      return;
    }
    if (lc.includes("loading") || lc.includes("starting")) {
      setDotState("loading");
      return;
    }
    if (lc.includes("enrichment") || lc.includes("indexing")) {
      setDotState("busy");
      return;
    }
    if (
      lc.includes("\u2713") ||
      lc.includes("complete") ||
      lc.includes("ready")
    ) {
      if (_dotState !== "error") setDotState("idle");
    }
  }

  // ── SSE connection ──────────────────────────────────────────────────────────
  function setConnState(state) {
    connDot.className = `qp-conn-dot ${state}`;
    connLabel.textContent =
      {
        connected: "Connected",
        reconnecting: "Reconnecting\u2026",
        disconnected: "Disconnected",
      }[state] || state;
  }

  function connectSSE() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    eventSource = new EventSource("/api/events");
    eventSource.onopen = () => {
      setConnState("connected");
      appendLogLine(
        "[INFO] orchestrator: \u2500\u2500 SSE stream connected \u2500\u2500",
      );
    };
    eventSource.onmessage = (e) => {
      if (!e.data || e.data.trim() === "") return;
      appendLogLine(e.data);
    };
    eventSource.onerror = () => {
      setConnState("reconnecting");
      eventSource.close();
      eventSource = null;
      setTimeout(connectSSE, RECONNECT_MS);
    };
  }

  connectSSE();

  // ── Status badge polling ────────────────────────────────────────────────────
  async function pollStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const s = await res.json();

      if (s.llm_server_running) {
        chipLLM.className =
          s.active_enrichments > 0 ? "qp-chip busy" : "qp-chip on";
        chipLLM.innerHTML = `<span>${s.active_enrichments > 0 ? "\u25c9" : "\u25cf"}</span> LLM server`;
      } else {
        chipLLM.className = "qp-chip off";
        chipLLM.innerHTML = "<span>\u25c9</span> LLM server";
      }

      if (s.active_enrichments > 0) {
        chipEnrich.className = "qp-chip busy";
        chipEnrich.innerHTML = `<span>\u25ce</span> Enriching (${s.active_enrichments})`;
        setDotState("busy");
      } else {
        chipEnrich.className = "qp-chip off";
        chipEnrich.innerHTML = "<span>\u25ce</span> Enrichment idle";
      }

      chipASR.className =
        s.sessions_with_asr > 0 ? "qp-chip on" : "qp-chip off";
      chipASR.innerHTML =
        s.sessions_with_asr > 0
          ? `<span>\u25cf</span> ASR loaded (${s.sessions_with_asr})`
          : "<span>\u25ce</span> ASR unloaded";

      chipSessions.className =
        s.active_sessions > 0 ? "qp-chip on" : "qp-chip off";
      chipSessions.innerHTML = `<span>${s.active_sessions > 0 ? "\u25cf" : "\u25ce"}</span> Sessions: ${s.active_sessions}`;
    } catch (_) {
      /* silent — SSE stream surfaces backend errors */
    }
  }

  pollStatus();
  setInterval(pollStatus, STATUS_POLL_MS);
})();

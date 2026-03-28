/* =========================================
   interview.js — Interview Session Logic

   Original fixes:
     B03 - session_id stored in localStorage; resumed on reload.
     B04 - Replaced undefined CSS variables.
     B20 - Only the submit button is disabled during evaluation, not the textarea.
     B02 - TTS now integrated correctly as const arrow (no hoisting race).

   UX improvements (senior engineer pass):
     UX1 - Ctrl+Enter (or Cmd+Enter on Mac) submits the answer without
           reaching for the mouse. A hint is shown in the textarea placeholder.

     UX2 - Auto-resizing textarea. The answer box grows as the user types
           (up to 200px) instead of forcing a fixed 2-row box.

     UX3 - Draft persistence. The current answer text is saved to
           sessionStorage on every keystroke and restored when a new question
           is displayed. Accidental refreshes no longer lose in-progress work.
           The draft is cleared on successful submit or intentional session end.

     UX4 - Character counter. A small live counter sits below the textarea,
           turning amber at 800 chars and red at 1200 chars.

     UX5 - Question progress badge. A "Question N" chip appears above the
           chat window as the session advances.
========================================= */

const API_BASE = "";
const SESSION_KEY = "qp_session_id";
const DRAFT_KEY = "qp_draft_answer"; // UX3

// ── State ──────────────────────────────────────────────────────────────────────
let sessionId = null;
const userId = localStorage.getItem("qp_user_id");
let _questionNumber = 0; // UX5

// ── DOM refs ───────────────────────────────────────────────────────────────────
const chatContainer = document.querySelector(".chat-container");
const chatForm = document.getElementById("chat-form");
const answerTextarea = document.getElementById("answer-text");
const endSessionBtn = document.getElementById("end-session-btn");
let charCounterEl = null; // UX4: injected on DOMContentLoaded

// ── Auth guard ─────────────────────────────────────────────────────────────────
if (!userId) window.location.href = "/login";

// ── HTML helpers ───────────────────────────────────────────────────────────────
function escapeHtml(text) {
  const div = document.createElement("div");
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

function appendMessage(role, html, extraStyle = "") {
  const div = document.createElement("div");
  div.className = `message ${role === "ai" ? "msg-ai" : "msg-user"}`;
  if (extraStyle) div.style.cssText = extraStyle;
  div.innerHTML = html;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// B02 FIX: const arrow — not hoistable, eliminates the aliasing/stack-overflow
// risk that the old function-declaration wrapper introduced.
const appendQuestion = (text) => {
  _questionNumber++;
  _updateProgressBadge(); // UX5
  appendMessage("ai", `<strong>Interviewer:</strong><br />${escapeHtml(text)}`);
  _restoreDraft(); // UX3: put back any saved draft text
  speakQuestion(text);
};

function appendUserAnswer(text) {
  appendMessage("user", `<strong>You:</strong><br />${escapeHtml(text)}`);
}

function appendFeedback(evaluation) {
  const pct = (evaluation.similarity * 100).toFixed(0);
  const barColor =
    evaluation.similarity >= 0.65
      ? "var(--success)"
      : evaluation.similarity >= 0.45
        ? "var(--warning)"
        : "var(--danger)";

  appendMessage(
    "ai",
    `<strong>Feedback:</strong><br />
     ${escapeHtml(evaluation.feedback)}<br />
     <div style="margin-top:10px;font-size:0.85rem;display:flex;align-items:center;gap:10px;">
       <div style="flex:1;height:6px;background:var(--paper-deep);border-radius:99px;overflow:hidden;">
         <div style="width:${pct}%;height:100%;background:${barColor};border-radius:99px;transition:width 0.4s;"></div>
       </div>
       <span style="opacity:0.7">Relevance: ${pct}%</span>
     </div>`,
    "background:var(--paper-light);border:1px solid var(--border);border-left:3px solid var(--warm);",
  );
}

function appendSystemMessage(text) {
  const div = document.createElement("div");
  div.style.cssText =
    "text-align:center;color:var(--ink-light);opacity:0.7;padding:12px;font-style:italic;font-size:0.95rem;";
  div.textContent = text;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ── UX5: Progress badge ────────────────────────────────────────────────────────
function _updateProgressBadge() {
  let badge = document.getElementById("qp-question-badge");
  if (!badge) {
    badge = document.createElement("div");
    badge.id = "qp-question-badge";
    badge.className = "qp-question-badge";
    chatContainer?.parentNode?.insertBefore(badge, chatContainer);
  }
  badge.textContent = `Question ${_questionNumber}`;
  badge.style.display = "inline-flex";
}

// ── UX2: Auto-resize ──────────────────────────────────────────────────────────
function _autoResize() {
  if (!answerTextarea) return;
  answerTextarea.style.height = "auto";
  answerTextarea.style.height =
    Math.min(answerTextarea.scrollHeight, 200) + "px";
}

// ── UX4: Character counter ─────────────────────────────────────────────────────
function _updateCharCounter() {
  if (!charCounterEl || !answerTextarea) return;
  const len = answerTextarea.value.length;
  charCounterEl.textContent = `${len} char${len !== 1 ? "s" : ""}`;
  charCounterEl.classList.toggle("char-counter-warn", len > 800 && len <= 1200);
  charCounterEl.classList.toggle("char-counter-limit", len > 1200);
}

// ── UX3: Draft persistence ─────────────────────────────────────────────────────
function _saveDraft() {
  sessionStorage.setItem(DRAFT_KEY, answerTextarea?.value ?? "");
}
function _restoreDraft() {
  const draft = sessionStorage.getItem(DRAFT_KEY);
  if (draft && answerTextarea) {
    answerTextarea.value = draft;
    _autoResize();
    _updateCharCounter();
  }
}
function _clearDraft() {
  sessionStorage.removeItem(DRAFT_KEY);
}

// ── Input state helpers ────────────────────────────────────────────────────────
function setSubmitEnabled(enabled) {
  const submitBtn = chatForm?.querySelector("button[type='submit']");
  if (submitBtn) submitBtn.disabled = !enabled;
  const micBtn = document.getElementById("mic-btn");
  if (micBtn) micBtn.disabled = !enabled;
}

function setSubmitLabel(text) {
  const btn = chatForm?.querySelector("button[type='submit']");
  if (btn) btn.textContent = text;
}

// ── B03: Session Resume Logic ──────────────────────────────────────────────────
async function tryResumeSession() {
  const storedId = localStorage.getItem(SESSION_KEY);
  if (!storedId) return false;
  try {
    const res = await fetch(`${API_BASE}/api/interview/${storedId}/status`);
    if (!res.ok) {
      localStorage.removeItem(SESSION_KEY);
      return false;
    }
    const data = await res.json();
    if (data.state === "TERMINAL") {
      localStorage.removeItem(SESSION_KEY);
      return false;
    }
    sessionId = storedId;
    _questionNumber = data.questions_answered ?? 0; // UX5: restore correct count
    appendSystemMessage(
      `Resuming your session (${data.questions_answered} question(s) answered so far).`,
    );
    if (data.current_question) {
      appendQuestion(data.current_question);
      setSubmitEnabled(true);
    }
    return true;
  } catch {
    localStorage.removeItem(SESSION_KEY);
    return false;
  }
}

// ── Session start ──────────────────────────────────────────────────────────────
async function startInterview() {
  chatContainer.innerHTML = "";
  appendSystemMessage("Starting your interview session…");
  setSubmitEnabled(false);

  const resumed = await tryResumeSession();
  if (resumed) return;

  try {
    const res = await fetch(`${API_BASE}/api/interview/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId }),
    });
    const data = await res.json();
    chatContainer.innerHTML = "";
    if (res.ok) {
      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);
      appendQuestion(data.first_question);
      setSubmitEnabled(true);
    } else {
      const isNoQuestions = data.detail?.includes("No questions available");
      appendSystemMessage(
        isNoQuestions
          ? 'No questions are ready yet. Upload a PDF on your profile page, then click "Generate Questions".'
          : `Could not start session: ${data.detail}`,
      );
    }
  } catch (err) {
    console.error("Failed to start interview:", err);
    chatContainer.innerHTML = "";
    appendSystemMessage(
      "Could not connect to the server. Is the FastAPI backend running?",
    );
  }
}

// ── Answer submission ──────────────────────────────────────────────────────────
async function submitAnswer() {
  const answer = answerTextarea?.value?.trim();
  if (!answer || !sessionId) return;

  setSubmitEnabled(false);
  setSubmitLabel("Evaluating…");
  appendUserAnswer(answer);
  answerTextarea.value = "";
  _clearDraft(); // UX3
  _autoResize();
  _updateCharCounter();

  try {
    const res = await fetch(`${API_BASE}/api/interview/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, user_answer: answer }),
    });
    const data = await res.json();
    if (res.ok) {
      appendFeedback(data.evaluation);
      if (data.is_terminal) {
        appendSystemMessage(
          "You've answered all available questions — great work!",
        );
        localStorage.removeItem(SESSION_KEY);
        showSummaryButton();
      } else if (data.next_question) {
        appendQuestion(data.next_question);
        setSubmitEnabled(true);
        setSubmitLabel("Send");
      }
    } else {
      appendSystemMessage(`Evaluation error: ${data.detail}`);
      setSubmitEnabled(true);
      setSubmitLabel("Send");
    }
  } catch (err) {
    console.error("Evaluation failed:", err);
    appendSystemMessage("Evaluation failed. Please check your connection.");
    setSubmitEnabled(true);
    setSubmitLabel("Send");
  }
}

// ── Session end ────────────────────────────────────────────────────────────────
async function endSession() {
  if (sessionId) {
    try {
      await fetch(`${API_BASE}/api/interview/${sessionId}`, {
        method: "DELETE",
      });
    } catch (_) {}
    localStorage.removeItem(SESSION_KEY);
  }
  _clearDraft(); // UX3: clean up on intentional exit
  window.location.href = "/profile";
}

// ── Summary button ─────────────────────────────────────────────────────────────
function showSummaryButton() {
  const div = document.createElement("div");
  div.style.cssText = "text-align:center;margin-top:20px;padding:10px;";
  div.innerHTML = `
    <a href="/api/interview/${sessionId}/summary" target="_blank"
       class="btn btn-primary" style="text-decoration:none;margin-right:10px;">
      View Summary
    </a>
    <a href="/profile" class="btn btn-dark" style="text-decoration:none;">
      Back to Profile
    </a>`;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ── B02: TTS helper ─────────────────────────────────────────────────────────────
async function speakQuestion(text) {
  try {
    const res = await fetch(`${API_BASE}/api/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.play();
    audio.onended = () => URL.revokeObjectURL(url);
  } catch (err) {
    console.warn("TTS unavailable:", err);
  }
}

// ── Event listeners ────────────────────────────────────────────────────────────
chatForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  submitAnswer();
});
endSessionBtn?.addEventListener("click", endSession);

// UX1: Ctrl+Enter / Cmd+Enter submits from the textarea
answerTextarea?.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    submitAnswer();
  }
});

// UX2 + UX3 + UX4: textarea input handling
answerTextarea?.addEventListener("input", () => {
  _autoResize();
  _updateCharCounter();
  _saveDraft();
});

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // UX4: Inject character counter element below the textarea
  if (answerTextarea) {
    charCounterEl = document.createElement("div");
    charCounterEl.id = "qp-char-counter";
    charCounterEl.className = "char-counter";
    charCounterEl.textContent = "0 chars";
    answerTextarea.parentNode.insertBefore(
      charCounterEl,
      answerTextarea.nextSibling,
    );
    _updateCharCounter();

    // UX1: Update placeholder to hint at keyboard shortcut
    answerTextarea.placeholder =
      "Type your answer or speak\u2026 (Ctrl+Enter to submit)";
  }

  startInterview();
});

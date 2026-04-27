/* =========================================
   interview.js — Interview Session Logic
   (Silent evaluation version)
========================================= */

const API_BASE = "";
const SESSION_KEY = "qp_session_id";
const DRAFT_KEY = "qp_draft_answer";

let sessionId = null;
const userId = localStorage.getItem("qp_user_id");
let _questionNumber = 0;

const chatContainer = document.querySelector(".chat-container");
const chatForm = document.getElementById("chat-form");
const answerTextarea = document.getElementById("answer-text");
const endSessionBtn = document.getElementById("end-session-btn");
let charCounterEl = null;

if (!userId) window.location.href = "/login";

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

const appendQuestion = (text) => {
  _questionNumber++;
  _updateProgressBadge();
  appendMessage("ai", `<strong>Interviewer:</strong><br />${escapeHtml(text)}`);
  _restoreDraft();
  speakQuestion(text);
};

function appendUserAnswer(text) {
  appendMessage("user", `<strong>You:</strong><br />${escapeHtml(text)}`);
}

function appendSystemMessage(text) {
  const div = document.createElement("div");
  div.style.cssText =
    "text-align:center;color:var(--ink-light);opacity:0.7;padding:12px;font-style:italic;font-size:0.95rem;";
  div.textContent = text;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

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

function _autoResize() {
  if (!answerTextarea) return;
  answerTextarea.style.height = "auto";
  answerTextarea.style.height =
    Math.min(answerTextarea.scrollHeight, 200) + "px";
}

function _updateCharCounter() {
  if (!charCounterEl || !answerTextarea) return;
  const len = answerTextarea.value.length;
  charCounterEl.textContent = `${len} char${len !== 1 ? "s" : ""}`;
  charCounterEl.classList.toggle("char-counter-warn", len > 800 && len <= 1200);
  charCounterEl.classList.toggle("char-counter-limit", len > 1200);
}

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

function setSubmitEnabled(enabled) {
  const submitBtn = chatForm?.querySelector("button[type='submit']");
  if (submitBtn) submitBtn.disabled = !enabled;
  const micBtn = document.getElementById("mic-btn");
  if (micBtn) micBtn.disabled = !enabled;
}

function setSubmitLabel(text) {
  const btn = chatForm?.querySelector("button[type='submit']");
  if (!btn) return;
  if (text === "Send") {
    btn.innerHTML = '<i data-lucide="send"></i> Send';
    if (window.lucide) lucide.createIcons();
  } else {
    btn.textContent = text;
  }
}

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
    _questionNumber = data.questions_answered ?? 0;
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

async function startInterview() {
  chatContainer.innerHTML = "";
  appendSystemMessage("Starting your interview session…");
  setSubmitEnabled(false);

  const resumed = await tryResumeSession();
  if (resumed) return;

  // Add this to grab the file_id from the URL
  const urlParams = new URLSearchParams(window.location.search);
  const fileIdParam = urlParams.get("file_id");

  const payload = { user_id: userId };
  if (fileIdParam) {
    payload.file_id = fileIdParam;
  }

  try {
    const res = await fetch(`${API_BASE}/api/interview/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), // Send the updated payload here
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

async function submitAnswer() {
  const answer = answerTextarea?.value?.trim();
  if (!answer || !sessionId) return;

  setSubmitEnabled(false);
  setSubmitLabel("Evaluating…");
  appendUserAnswer(answer);
  answerTextarea.value = "";
  _clearDraft();
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
      // FEEDBACK REMOVED — evaluation happens silently in the background.
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

async function endSession() {
  const sid = sessionId;
  if (sid) {
    try {
      await fetch(`${API_BASE}/api/interview/${sid}`, { method: "DELETE" });
    } catch (_) {}
    localStorage.removeItem(SESSION_KEY);
  }
  _clearDraft();
  window.location.href = sid ? `/session?sid=${sid}` : "/profile";
}

function showSummaryButton() {
  const div = document.createElement("div");
  div.style.cssText = "text-align:center;margin-top:20px;padding:10px;";
  div.innerHTML = `
    <a href="/session?sid=${sessionId}" class="btn btn-primary" style="text-decoration:none;margin-right:10px;">
      View Session Results
    </a>
    <a href="/profile" class="btn btn-dark" style="text-decoration:none;">
      Back to Profile
    </a>`;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

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
    try {
      await audio.play();
    } catch (playErr) {
      console.warn("TTS autoplay blocked:", playErr);
    }
    audio.onended = () => URL.revokeObjectURL(url);
  } catch (err) {
    console.warn("TTS unavailable:", err);
  }
}

chatForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  submitAnswer();
});
endSessionBtn?.addEventListener("click", endSession);

answerTextarea?.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    submitAnswer();
  }
});

answerTextarea?.addEventListener("input", () => {
  _autoResize();
  _updateCharCounter();
  _saveDraft();
});

document.addEventListener("DOMContentLoaded", () => {
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
    answerTextarea.placeholder =
      "Type your answer or speak\u2026 (Ctrl+Enter to submit)";
  }
  startInterview();
});

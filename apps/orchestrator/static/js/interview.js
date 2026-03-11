/* =========================================
   interview.js — Interview Session Logic
   Connects the interview page to the FastAPI backend.
   Handles: start session, evaluate answers, end session, terminal state.
========================================= */

const API_BASE = "";

// ── State ──────────────────────────────────────────────────────────────────────
let sessionId = null;
const userId = localStorage.getItem("qp_user_id");

// ── DOM refs ───────────────────────────────────────────────────────────────────
const chatContainer = document.querySelector(".chat-container");
const chatForm = document.getElementById("chat-form");
const answerTextarea = document.getElementById("answer-text");
const endSessionBtn = document.getElementById("end-session-btn");

// ── Auth guard ─────────────────────────────────────────────────────────────────
if (!userId) {
  window.location.href = "/login";
}

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

function appendQuestion(text) {
  appendMessage("ai", `<strong>Interviewer:</strong><br />${escapeHtml(text)}`);
}

function appendUserAnswer(text) {
  appendMessage("user", `<strong>You:</strong><br />${escapeHtml(text)}`);
}

function appendFeedback(evaluation) {
  const pct = (evaluation.similarity * 100).toFixed(0);
  const barColor =
    evaluation.similarity >= 0.65
      ? "var(--soft-lime)"
      : evaluation.similarity >= 0.45
        ? "var(--peach-pastel)"
        : "#fecaca";

  appendMessage(
    "ai",
    `<strong>Feedback:</strong><br />
     ${escapeHtml(evaluation.feedback)}<br />
     <div style="margin-top:10px; font-size:0.85rem; display:flex; align-items:center; gap:10px;">
       <div style="flex:1; height:6px; background:#e5e7eb; border-radius:99px; overflow:hidden;">
         <div style="width:${pct}%; height:100%; background:${barColor}; border-radius:99px; transition:width 0.4s;"></div>
       </div>
       <span style="opacity:0.7">Relevance: ${pct}%</span>
     </div>`,
    "background-color: var(--baby-aqua);",
  );
}

function appendSystemMessage(text) {
  const div = document.createElement("div");
  div.style.cssText =
    "text-align:center; color:var(--soft-charcoal); opacity:0.7; padding:12px; font-style:italic; font-size:0.95rem;";
  div.textContent = text;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ── Input state helpers ────────────────────────────────────────────────────────
function setInputEnabled(enabled) {
  if (answerTextarea) answerTextarea.disabled = !enabled;
  const submitBtn = chatForm?.querySelector("button[type='submit']");
  if (submitBtn) submitBtn.disabled = !enabled;
  const micBtn = document.getElementById("mic-btn");
  if (micBtn) micBtn.disabled = !enabled;
}

function setSubmitLabel(text) {
  const btn = chatForm?.querySelector("button[type='submit']");
  if (btn) btn.textContent = text;
}

// ── Session start ──────────────────────────────────────────────────────────────
async function startInterview() {
  chatContainer.innerHTML = "";
  appendSystemMessage("Starting your interview session…");
  setInputEnabled(false);

  try {
    const res = await fetch(`${API_BASE}/api/interview/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId }),
    });

    const data = await res.json();
    chatContainer.innerHTML = ""; // clear loading message

    if (res.ok) {
      sessionId = data.session_id;
      appendQuestion(data.first_question);
      setInputEnabled(true);
    } else {
      const isNoQuestions = data.detail?.includes("No questions available");
      appendSystemMessage(
        isNoQuestions
          ? 'No questions are ready yet. Upload a PDF on your profile page, then click "Generate Questions".'
          : `Could not start session: ${data.detail}`,
      );
      // Keep input disabled — there's nothing to answer
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

  setInputEnabled(false);
  setSubmitLabel("Evaluating…");
  appendUserAnswer(answer);
  answerTextarea.value = "";

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
        showSummaryButton();
        // Input stays disabled — session is over
      } else if (data.next_question) {
        appendQuestion(data.next_question);
        setInputEnabled(true);
        setSubmitLabel("Send");
      }
    } else {
      appendSystemMessage(`Evaluation error: ${data.detail}`);
      setInputEnabled(true);
      setSubmitLabel("Send");
    }
  } catch (err) {
    console.error("Evaluation failed:", err);
    appendSystemMessage("Evaluation failed. Please check your connection.");
    setInputEnabled(true);
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
    } catch (_) {
      // best-effort cleanup
    }
  }
  window.location.href = "/profile";
}

// ── Summary button ─────────────────────────────────────────────────────────────
function showSummaryButton() {
  const div = document.createElement("div");
  div.style.cssText = "text-align:center; margin-top:20px; padding:10px;";
  div.innerHTML = `
    <a href="/api/interview/${sessionId}/summary"
       target="_blank"
       class="cta-button"
       style="background-color:var(--periwinkle); text-decoration:none; margin-right:10px;">
      View Summary
    </a>
    <a href="/profile"
       class="cta-button"
       style="background-color:var(--soft-charcoal); text-decoration:none;">
      Back to Profile
    </a>
  `;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ── Event listeners ────────────────────────────────────────────────────────────
chatForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  submitAnswer();
});

endSessionBtn?.addEventListener("click", endSession);

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", startInterview);

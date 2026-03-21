/* =========================================
   app.js - QuestionPrep Frontend Logic
========================================= */

const API_BASE = "";

// ── Inline SVG icon helper (for dynamically injected cards) ──────────────────
const ICON_PATHS = {
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  sparkles:
    '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
};

function icon(name, size = 16) {
  const p = ICON_PATHS[name];
  if (!p) return "";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;">${p}</svg>`;
}

// ── Modal helper — falls back to native alert() on non-profile pages ──────────
function notify(title, body, type = "info") {
  if (typeof showModal === "function") {
    showModal(title, body, type);
  } else {
    alert(`${title}\n\n${body}`);
  }
}

// ==========================================
// 1. AUTHENTICATION LOGIC (Login & Signup)
// ==========================================

// --- LOGIN ---
const loginForm = document.getElementById("login-form");
if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const username = document.getElementById("login-username").value;
    const password = document.getElementById("login-password").value;
    const submitBtn = loginForm.querySelector("button[type='submit']");

    const originalText = submitBtn.innerText;
    submitBtn.innerText = "Logging in...";
    submitBtn.disabled = true;

    try {
      const response = await fetch(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (response.ok) {
        const userId = data.user.user_id || data.user.id;
        localStorage.setItem("qp_token", data.token);
        localStorage.setItem("qp_user_id", userId);
        window.location.href = "/profile";
      } else {
        notify("Login failed", data.detail, "error");
      }
    } catch (error) {
      console.error("Error logging in:", error);
      notify(
        "Network error",
        "Could not reach the server. Is the FastAPI server running?",
        "error",
      );
    } finally {
      submitBtn.innerText = originalText;
      submitBtn.disabled = false;
    }
  });
}

// --- SIGNUP ---
const signupForm = document.getElementById("signup-form");
if (signupForm) {
  signupForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const username = document.getElementById("signup-username").value;
    const email = document.getElementById("signup-email").value;
    const password = document.getElementById("signup-password").value;
    const submitBtn = signupForm.querySelector("button[type='submit']");

    const originalText = submitBtn.innerText;
    submitBtn.innerText = "Creating account...";
    submitBtn.disabled = true;

    try {
      const response = await fetch(`${API_BASE}/api/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        notify(
          "Account created",
          "Your account is ready. Please log in to continue.",
          "success",
        );
        signupForm.reset();
        document.getElementById("signup-view").style.display = "none";
        document.getElementById("login-view").style.display = "block";
      } else {
        notify("Sign up failed", data.detail, "error");
      }
    } catch (error) {
      console.error("Error signing up:", error);
      notify("Network error", "A network error occurred.", "error");
    } finally {
      submitBtn.innerText = originalText;
      submitBtn.disabled = false;
    }
  });
}

// ==========================================
// 2. QUESTION GENERATION (Profile Page)
// ==========================================

async function generateQuestions(fileId, btn) {
  const originalHTML = btn.innerHTML;
  btn.innerHTML = `${icon("sparkles")} Starting…`;
  btn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/questions/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: fileId }),
    });

    const data = await response.json();

    if (response.ok) {
      btn.innerHTML = `${icon("check")} Generating in background`;
      btn.style.background = "var(--bg-success)";
      btn.style.color = "var(--success)";
      btn.style.borderColor = "var(--success)";
    } else {
      notify("Generation failed", data.detail, "error");
      btn.innerHTML = originalHTML;
      btn.disabled = false;
    }
  } catch (error) {
    console.error("Generate questions failed:", error);
    notify(
      "Network error",
      "A network error occurred during question generation.",
      "error",
    );
    btn.innerHTML = originalHTML;
    btn.disabled = false;
  }
}

// ==========================================
// 3. DYNAMIC FILE FETCHING (Profile Page)
// ==========================================

async function fetchUserFiles() {
  const userId = localStorage.getItem("qp_user_id");
  const filesGrid = document.getElementById("files-grid");

  if (!userId || !filesGrid) return;

  filesGrid.innerHTML = `
    <p style="grid-column:1/-1; text-align:center; color:var(--ink-very-light); font-style:italic; font-family:'Lora',serif;">
      Loading your files…
    </p>`;

  try {
    const response = await fetch(`${API_BASE}/api/files?user_id=${userId}`);
    const data = await response.json();

    filesGrid.innerHTML = "";

    if (!response.ok) {
      filesGrid.innerHTML = `<p style="grid-column:1/-1; color:var(--danger);">Failed to load files.</p>`;
      return;
    }

    if (!data.files || data.files.length === 0) {
      filesGrid.innerHTML = `
        <p style="grid-column:1/-1; text-align:center; color:var(--ink-very-light); font-style:italic; font-family:'Lora',serif;">
          No PDFs uploaded yet. Start by uploading a document above.
        </p>`;
      return;
    }

    data.files.forEach((file) => {
      const fileName =
        file.filename || file.file_name || file.name || "Document";
      const fileId = file.file_id || file.id || "unknown";
      const uploaded = file.assigned_at
        ? new Date(file.assigned_at).toLocaleDateString()
        : "—";

      const cardHtml = `
        <div class="file-card">
          <div class="file-card-title">${fileName}</div>
          <div class="file-card-meta">Uploaded: ${uploaded}</div>
          <div class="file-card-actions">
            <button
              class="btn btn-outline btn-sm"
              onclick="generateQuestions('${fileId}', this)">
              ${icon("sparkles")} Generate Questions
            </button>
            <a
              href="/interview"
              class="btn btn-primary btn-sm"
              style="justify-content:center; text-align:center;">
              ${icon("play")} Start Interview
            </a>
          </div>
        </div>
      `;

      filesGrid.innerHTML += cardHtml;
    });
  } catch (error) {
    console.error("Error fetching files:", error);
    filesGrid.innerHTML = `<p style="grid-column:1/-1; color:var(--danger);">Failed to load your files.</p>`;
  }
}

document.addEventListener("DOMContentLoaded", fetchUserFiles);

// ==========================================
// 4. FILE UPLOAD LOGIC (Profile Page)
// ==========================================

const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("pdf-upload");

if (uploadBtn && fileInput) {
  uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    const userId = localStorage.getItem("qp_user_id");

    if (!file) {
      notify(
        "No file selected",
        "Please select a PDF file before uploading.",
        "info",
      );
      return;
    }

    if (!userId) {
      notify(
        "Not logged in",
        "You must be logged in to upload files.",
        "error",
      );
      window.location.href = "/login";
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const originalHTML = uploadBtn.innerHTML;
    uploadBtn.innerHTML = `${icon("sparkles")} Uploading…`;
    uploadBtn.disabled = true;

    try {
      const response = await fetch(
        `${API_BASE}/api/files/ingest?user_id=${userId}`,
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await response.json();

      if (response.ok) {
        fileInput.value = "";
        notify(
          "Upload successful",
          'Your file is being ingested in the background.\n\nOnce it appears in your file list, click "Generate Questions" to create interview questions — this may take a few minutes.',
          "success",
        );
        pollForNewFile();
      } else {
        notify("Upload failed", data.detail, "error");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      notify(
        "Network error",
        "A network error occurred during upload.",
        "error",
      );
    } finally {
      uploadBtn.innerHTML = originalHTML;
      uploadBtn.disabled = false;
    }
  });
}

// ==========================================
// 5. POLL FOR NEW FILE AFTER UPLOAD
// ==========================================

async function pollForNewFile() {
  const userId = localStorage.getItem("qp_user_id");
  if (!userId) return;

  let previousCount = document.querySelectorAll(
    "#files-grid .file-card",
  ).length;
  let attempts = 0;
  const maxAttempts = 18;

  const interval = setInterval(async () => {
    attempts++;
    try {
      const res = await fetch(`${API_BASE}/api/files?user_id=${userId}`);
      const data = await res.json();
      const newCount = data.files?.length ?? 0;

      if (newCount > previousCount || attempts >= maxAttempts) {
        clearInterval(interval);
        await fetchUserFiles();
      }
    } catch (_) {
      if (attempts >= maxAttempts) clearInterval(interval);
    }
  }, 5000);
}

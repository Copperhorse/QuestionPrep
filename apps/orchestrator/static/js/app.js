/* =========================================
   app.js — QuestionPrep Frontend Logic

   Original fixes:
     B07 - showModal() implemented using .dialog / .alert CSS classes.
     B24 - pollForNewFile() no longer reads the file count from the DOM.

   Improvements (senior engineer pass):

     OPT1 - fetchUserFiles() builds cards with DocumentFragment + createElement
            instead of innerHTML +=.  The old approach re-parsed and re-rendered
            the entire grid innerHTML on every iteration — O(n) re-parses for n
            files.  A DocumentFragment defers all DOM mutations to a single
            appendChild call (one layout pass, O(1)).

     OPT2 - _knownFileIds is a Set<string> keyed on file_id.  pollForNewFile()
            snapshots this Set at upload time and detects arrivals with
            Set.has() — O(1) per check vs O(n) count comparison.
            fetchUserFiles() rebuilds the Set from every API response, keeping
            it perfectly in sync with the server.

     OPT3 - _inFlightGenerations is a Set<string> that guards generateQuestions
            against duplicate calls for the same file_id (e.g. double-click,
            rapid re-click).  Prevents the unnecessary second network round-trip;
            the backend has its own guard but eliminating the call client-side
            is cheaper.

     OPT4 - Toast notification system replaces notify() for success / info /
            warning cases.  Toasts are non-blocking: no backdrop, no required
            click, auto-dismiss after a configurable delay, and stack vertically.
            showModal is retained only for ERROR-level alerts.

     OPT5 - Drag-and-drop file upload.  The upload zone accepts dragged PDF
            files.  A shared uploadFile(file) function handles both the
            button-click and drop paths, eliminating duplicated fetch logic.

     OPT6 - Upload is guarded by a boolean _uploading flag, making it
            idempotent against multiple clicks during an in-flight request.
========================================= */

const API_BASE = "";

// ── OPT2: Canonical file-ID registry ─────────────────────────────────────────
let _knownFileIds = new Set(); // Set<string> of file_id values
let _lastKnownFileCount = 0; // kept for any legacy references

// ── OPT3: In-flight generation guard ─────────────────────────────────────────
const _inFlightGenerations = new Set(); // Set<string> of file_ids with pending requests

// ── OPT6: Upload guard ────────────────────────────────────────────────────────
let _uploading = false;

// ── Inline SVG icon helper ────────────────────────────────────────────────────
const ICON_PATHS = {
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  sparkles:
    '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  upload:
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
};

function icon(name, size = 16) {
  const p = ICON_PATHS[name];
  if (!p) return "";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;">${p}</svg>`;
}

// ── Minimal HTML escaper for user-supplied strings rendered into cards ────────
function _escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ==========================================
// OPT4: Toast notification system
// ==========================================

let _toastContainer = null;

function _getToastContainer() {
  if (_toastContainer) return _toastContainer;
  _toastContainer = document.createElement("div");
  _toastContainer.id = "qp-toast-container";
  document.body.appendChild(_toastContainer);
  return _toastContainer;
}

/**
 * Display a small non-blocking toast.
 * @param {string} message
 * @param {'success'|'info'|'warning'|'error'} type
 * @param {number} durationMs  0 = manual dismiss only
 * @returns {Function} dismiss function (call early if needed)
 */
function toast(message, type = "info", durationMs = 4500) {
  const container = _getToastContainer();
  const el = document.createElement("div");
  el.className = `qp-toast qp-toast-${type}`;
  el.innerHTML = `<span class="qp-toast-msg">${message}</span>
    <button class="qp-toast-close" aria-label="Dismiss">\u00d7</button>`;

  const remove = () => {
    el.classList.add("qp-toast-out");
    el.addEventListener("animationend", () => el.remove(), { once: true });
  };

  el.querySelector(".qp-toast-close").addEventListener("click", remove);
  container.appendChild(el);
  if (durationMs > 0) setTimeout(remove, durationMs);
  return remove;
}

// ==========================================
// B07: showModal — blocking, for errors only
// ==========================================
function showModal(title, body, type = "info") {
  const alertClass =
    {
      success: "alert-success",
      error: "alert-danger",
      warning: "alert-warning",
      info: "alert-info",
    }[type] || "alert-info";

  const backdrop = document.createElement("div");
  backdrop.className = "dialog-backdrop";
  backdrop.innerHTML = `
    <div class="dialog" role="dialog" aria-modal="true" aria-label="${title}">
      <div class="dialog-header">${title}</div>
      <div class="alert ${alertClass}" style="margin:0.5rem 0 1.2rem;border-radius:6px;">${body}</div>
      <div style="text-align:right;">
        <button class="btn btn-outline btn-sm" id="modal-close-btn">${icon("x")} Dismiss</button>
      </div>
    </div>`;

  document.body.appendChild(backdrop);

  const close = () => {
    backdrop.remove();
    clearTimeout(autoClose);
  };
  document.getElementById("modal-close-btn").addEventListener("click", close);
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) close();
  });
  const autoClose = type !== "error" ? setTimeout(close, 5000) : null;
}

/**
 * Route success/info/warning to toast (non-blocking),
 * error to showModal (blocking, requires acknowledgement).
 */
function notify(title, body, type = "info") {
  if (type === "error") {
    showModal(title, body, type);
  } else {
    toast(`<strong>${title}:</strong> ${body}`, type);
  }
}

// ==========================================
// 1. AUTHENTICATION LOGIC
// ==========================================

const loginForm = document.getElementById("login-form");
if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = document.getElementById("login-username").value;
    const password = document.getElementById("login-password").value;
    const submitBtn = loginForm.querySelector("button[type='submit']");
    const orig = submitBtn.innerText;
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
        localStorage.setItem("qp_token", data.token);
        localStorage.setItem("qp_user_id", data.user.user_id || data.user.id);
        window.location.href = "/profile";
      } else {
        notify("Login failed", data.detail, "error");
      }
    } catch {
      notify(
        "Network error",
        "Could not reach the server. Is the FastAPI server running?",
        "error",
      );
    } finally {
      submitBtn.innerText = orig;
      submitBtn.disabled = false;
    }
  });
}

const signupForm = document.getElementById("signup-form");
if (signupForm) {
  signupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = document.getElementById("signup-username").value;
    const email = document.getElementById("signup-email").value;
    const password = document.getElementById("signup-password").value;
    const submitBtn = signupForm.querySelector("button[type='submit']");
    const orig = submitBtn.innerText;
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
          "Your account is ready. Please log in.",
          "success",
        );
        signupForm.reset();
        document.getElementById("signup-view").style.display = "none";
        document.getElementById("login-view").style.display = "block";
      } else {
        notify("Sign up failed", data.detail, "error");
      }
    } catch {
      notify("Network error", "A network error occurred.", "error");
    } finally {
      submitBtn.innerText = orig;
      submitBtn.disabled = false;
    }
  });
}

// ==========================================
// 2. QUESTION GENERATION
// ==========================================

async function generateQuestions(fileId, btn) {
  // OPT3: Prevent duplicate in-flight requests for the same file.
  if (_inFlightGenerations.has(fileId)) return;
  _inFlightGenerations.add(fileId);

  const originalHTML = btn.innerHTML;
  btn.innerHTML = `${icon("sparkles")} Starting\u2026`;
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
  } catch {
    notify(
      "Network error",
      "A network error occurred during question generation.",
      "error",
    );
    btn.innerHTML = originalHTML;
    btn.disabled = false;
  } finally {
    _inFlightGenerations.delete(fileId); // always release the guard
  }
}

// ==========================================
// 3. DYNAMIC FILE FETCHING
// ==========================================

async function fetchUserFiles() {
  const userId = localStorage.getItem("qp_user_id");
  const filesGrid = document.getElementById("files-grid");
  if (!userId || !filesGrid) return;

  filesGrid.innerHTML = `
    <p style="grid-column:1/-1;text-align:center;color:var(--ink-very-light);font-style:italic;font-family:'Lora',serif;">
      Loading your files\u2026
    </p>`;

  try {
    const response = await fetch(`${API_BASE}/api/files?user_id=${userId}`);
    const data = await response.json();

    if (!response.ok) {
      filesGrid.innerHTML = `<p style="grid-column:1/-1;color:var(--danger);">Failed to load files.</p>`;
      return;
    }

    const files = data.files ?? [];

    // OPT2: Rebuild the canonical Set from the authoritative server response.
    _knownFileIds = new Set(files.map((f) => f.file_id || f.id));
    _lastKnownFileCount = _knownFileIds.size;

    filesGrid.innerHTML = "";

    if (files.length === 0) {
      filesGrid.innerHTML = `
        <p style="grid-column:1/-1;text-align:center;color:var(--ink-very-light);font-style:italic;font-family:'Lora',serif;">
          No PDFs uploaded yet. Start by uploading a document above.
        </p>`;
      return;
    }

    // OPT1: Build all card elements in a DocumentFragment.
    // innerHTML += triggers a full re-parse of everything already in filesGrid
    // on every iteration — O(chars_in_grid × n_files) work.
    // A DocumentFragment accumulates all nodes off-screen, then a single
    // appendChild flushes them in one layout/paint pass.
    const frag = document.createDocumentFragment();

    files.forEach((file) => {
      const fileName =
        file.filename || file.file_name || file.name || "Document";
      const fileId = file.file_id || file.id || "unknown";
      const uploaded = file.assigned_at
        ? new Date(file.assigned_at).toLocaleDateString()
        : "\u2014";

      const card = document.createElement("div");
      card.className = "file-card";
      card.dataset.fileId = fileId; // useful for incremental diffing in future

      card.innerHTML = `
        <div class="file-card-title">${_escHtml(fileName)}</div>
        <div class="file-card-meta">Uploaded: ${uploaded}</div>
        <div class="file-card-actions">
          <button class="btn btn-outline btn-sm"
                  onclick="generateQuestions('${fileId}', this)">
            ${icon("sparkles")} Generate Questions
          </button>
          <a href="/interview?file_id=${fileId}" class="btn btn-primary btn-sm"
             style="justify-content:center;text-align:center;">
            ${icon("play")} Start Interview
          </a>
        </div>`;

      frag.appendChild(card);
    });

    filesGrid.appendChild(frag); // single DOM mutation
  } catch (error) {
    console.error("Error fetching files:", error);
    filesGrid.innerHTML = `<p style="grid-column:1/-1;color:var(--danger);">Failed to load your files.</p>`;
  }
}

document.addEventListener("DOMContentLoaded", fetchUserFiles);

// ==========================================
// 4. FILE UPLOAD  (shared logic for button + drop)
// ==========================================

/**
 * OPT5 + OPT6: Central upload function.  Called by the button-click handler
 * and the drag-and-drop handler so there is a single code path.
 * Guarded by _uploading to prevent concurrent submissions.
 */
async function uploadFile(file) {
  if (_uploading) return;

  const userId = localStorage.getItem("qp_user_id");
  if (!userId) {
    notify("Not logged in", "You must be logged in to upload files.", "error");
    window.location.href = "/login";
    return;
  }
  if (!file) {
    notify(
      "No file selected",
      "Please select a PDF file before uploading.",
      "info",
    );
    return;
  }

  const uploadBtn = document.getElementById("upload-btn");
  const fileInput = document.getElementById("pdf-upload");
  const originalHTML = uploadBtn?.innerHTML ?? "";

  _uploading = true;
  if (uploadBtn) {
    uploadBtn.innerHTML = `${icon("sparkles")} Uploading\u2026`;
    uploadBtn.disabled = true;
  }
  document.getElementById("upload-drop-zone")?.classList.remove("drag-over");

  const formData = new FormData();
  formData.append("file", file);

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
      if (fileInput) fileInput.value = "";
      notify(
        "Upload successful",
        "Your file is being ingested. Once it appears below, click \u201cGenerate Questions\u201d.",
        "success",
      );
      pollForNewFile();
    } else {
      notify("Upload failed", data.detail, "error");
    }
  } catch {
    notify("Network error", "A network error occurred during upload.", "error");
  } finally {
    _uploading = false;
    if (uploadBtn) {
      uploadBtn.innerHTML = originalHTML;
      uploadBtn.disabled = false;
    }
  }
}

// ── Button path ────────────────────────────────────────────────────────────────
const _uploadBtn = document.getElementById("upload-btn");
const _fileInput = document.getElementById("pdf-upload");
if (_uploadBtn && _fileInput) {
  _uploadBtn.addEventListener("click", () => uploadFile(_fileInput.files[0]));
}

// ── OPT5: Drag-and-drop path ──────────────────────────────────────────────────
const dropZone = document.getElementById("upload-drop-zone");
if (dropZone) {
  dropZone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", (e) => {
    // Only remove the class when the pointer genuinely leaves the zone
    // (not when it enters a child element inside it).
    if (!dropZone.contains(e.relatedTarget))
      dropZone.classList.remove("drag-over");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file?.type === "application/pdf") {
      uploadFile(file);
    } else if (file) {
      toast("Only PDF files are accepted.", "warning");
    }
  });

  // Clicking the zone (but not the button or input directly) opens the picker
  dropZone.addEventListener("click", (e) => {
    if (e.target !== _fileInput && e.target !== _uploadBtn) _fileInput?.click();
  });
}

// ==========================================
// 5. POLL FOR NEW FILE AFTER UPLOAD
// ==========================================

async function pollForNewFile() {
  const userId = localStorage.getItem("qp_user_id");
  if (!userId) return;

  // OPT2: Snapshot the current known IDs.  The poller looks for any file_id
  // that was absent from this snapshot — O(1) per check with Set.has().
  const baselineIds = new Set(_knownFileIds);
  let attempts = 0;
  const maxAttempts = 18; // 90 s total at 5 s intervals

  const interval = setInterval(async () => {
    attempts++;
    try {
      const res = await fetch(`${API_BASE}/api/files?user_id=${userId}`);
      const data = await res.json();
      const freshIds = new Set(
        (data.files ?? []).map((f) => f.file_id || f.id),
      );

      const hasNew = [...freshIds].some((id) => !baselineIds.has(id));
      if (hasNew || attempts >= maxAttempts) {
        clearInterval(interval);
        await fetchUserFiles();
      }
    } catch (_) {
      if (attempts >= maxAttempts) clearInterval(interval);
    }
  }, 5000);
}

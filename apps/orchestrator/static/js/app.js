/* =========================================
   app.js — QuestionPrep Frontend Logic
========================================= */

const API_BASE = "";

let _knownFileIds = new Set();
let _lastKnownFileCount = 0;
const _inFlightGenerations = new Set();
let _uploading = false;

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

function _escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

let _toastContainer = null;
function _getToastContainer() {
  if (_toastContainer) return _toastContainer;
  _toastContainer = document.createElement("div");
  _toastContainer.id = "qp-toast-container";
  document.body.appendChild(_toastContainer);
  return _toastContainer;
}

function toast(message, type = "info", durationMs = 4500) {
  const container = _getToastContainer();
  const el = document.createElement("div");
  el.className = `qp-toast qp-toast-${type}`;
  el.innerHTML = `<span class="qp-toast-msg">${message}</span><button class="qp-toast-close" aria-label="Dismiss">\u00d7</button>`;
  const remove = () => {
    el.classList.add("qp-toast-out");
    el.addEventListener("animationend", () => el.remove(), { once: true });
  };
  el.querySelector(".qp-toast-close").addEventListener("click", remove);
  container.appendChild(el);
  if (durationMs > 0) setTimeout(remove, durationMs);
  return remove;
}

function showModal(title, body, type = "info") {
  toast(`<strong>${title}:</strong> ${body}`, type, 0);
}

function notify(title, body, type = "info") {
  toast(`<strong>${title}:</strong> ${body}`, type);
}

// ==========================================
// AUTHENTICATION LOGIC
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
      notify("Network error", "Could not reach the server.", "error");
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
// QUESTION GENERATION
// ==========================================
async function generateQuestions(fileId, btn) {
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
      if (data.skipped) {
        notify("Skipped", data.message, "info");
        btn.innerHTML = originalHTML;
        btn.disabled = false;
        return;
      }
      btn.innerHTML = `${icon("check")} Generating`;
      btn.style.background = "var(--bg-success)";
      btn.style.color = "var(--success)";
      btn.style.borderColor = "var(--success)";
    } else {
      notify("Generation failed", data.detail || data.message, "warning");
      btn.innerHTML = originalHTML;
      btn.disabled = false;
    }
  } catch {
    notify("Network error", "A network error occurred.", "error");
    btn.innerHTML = originalHTML;
    btn.disabled = false;
  } finally {
    _inFlightGenerations.delete(fileId);
  }
}

// ==========================================
// FILE FETCHING & MODAL LOGIC
// ==========================================
// ==========================================
// DELETE SINGLE FILE
// ==========================================
async function deleteFile(fileId, fileName, btn) {
  if (
    !confirm(
      `Are you sure you want to delete "${fileName}"? This will also remove all generated questions and embeddings.`,
    )
  ) {
    return;
  }

  const userId = localStorage.getItem("qp_user_id");
  if (!userId) {
    window.location.href = "/login";
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `${icon("sparkles")} Deleting…`;

  try {
    const response = await fetch(
      `${API_BASE}/api/files/${fileId}?user_id=${userId}`,
      {
        method: "DELETE",
      },
    );
    const data = await response.json();

    if (response.ok) {
      notify("Deleted", `"${fileName}" has been removed.`, "success");
      // Remove the card from the DOM
      const card = btn.closest(".file-card");
      if (card) {
        card.style.transition = "opacity 0.3s ease, transform 0.3s ease";
        card.style.opacity = "0";
        card.style.transform = "scale(0.95)";
        setTimeout(() => card.remove(), 300);
      }
      // Refresh the file list to update state
      setTimeout(fetchUserFiles, 350);
    } else {
      notify("Delete failed", data.detail || "Could not delete file.", "error");
      btn.disabled = false;
      btn.innerHTML = `${icon("x")} Delete`;
    }
  } catch {
    notify(
      "Network error",
      "A network error occurred while deleting.",
      "error",
    );
    btn.disabled = false;
    btn.innerHTML = `${icon("x")} Delete`;
  }
}
async function fetchUserFiles() {
  const userId = localStorage.getItem("qp_user_id");
  const filesGrid = document.getElementById("files-grid");
  if (!userId || !filesGrid) return;

  filesGrid.innerHTML = `<p style="grid-column:1/-1;text-align:center;color:var(--ink-very-light);font-style:italic;">Loading your files\u2026</p>`;

  try {
    const response = await fetch(`${API_BASE}/api/files?user_id=${userId}`);
    const data = await response.json();

    if (!response.ok) {
      filesGrid.innerHTML = `<p style="grid-column:1/-1;color:var(--danger);">Failed to load files.</p>`;
      return;
    }

    const files = data.files ?? [];
    _knownFileIds = new Set(files.map((f) => f.file_id || f.id));

    filesGrid.innerHTML = "";
    if (files.length === 0) {
      filesGrid.innerHTML = `<p style="grid-column:1/-1;text-align:center;color:var(--ink-very-light);font-style:italic;">No PDFs uploaded yet. Start by uploading a document above.</p>`;
      return;
    }

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
      card.dataset.fileId = fileId;

      // Attach Audit Listener
      card.addEventListener("click", (e) => {
        // Ignore clicks on inner buttons/links
        if (e.target.closest("button") || e.target.closest("a")) return;
        openAuditModal(fileId, fileName);
      });

      card.innerHTML = `
        <div class="file-card-title">${_escHtml(fileName)}</div>
        <div class="file-card-meta">Uploaded: ${uploaded}</div>
        <div class="file-card-actions">
          <button class="btn btn-outline btn-sm" onclick="generateQuestions('${fileId}', this)">
            ${icon("sparkles")} Generate Questions
          </button>
          <a href="/interview" class="btn btn-primary btn-sm" style="justify-content:center;text-align:center;">
            ${icon("play")} Start Interview
          </a>
          <button class="btn btn-danger btn-sm" onclick="deleteFile('${fileId}', '${_escHtml(fileName).replace(/'/g, "\\'")}', this)" style="margin-left:auto;">
            ${icon("x")} Delete
          </button>
        </div>`;

      frag.appendChild(card);
    });

    filesGrid.appendChild(frag);
  } catch (error) {
    filesGrid.innerHTML = `<p style="grid-column:1/-1;color:var(--danger);">Failed to load your files.</p>`;
  }
}

document.addEventListener("DOMContentLoaded", fetchUserFiles);

// ==========================================
// X-RAY AUDIT MODAL
// ==========================================
async function openAuditModal(fileId, fileName) {
  const modal = document.getElementById("file-summary-modal");
  if (!modal) return;

  try {
    const res = await fetch(`/api/files/${fileId}/audit`);
    const data = await res.json();

    document.getElementById("modal-file-title").innerText =
      fileName || "File Details";
    document.getElementById("modal-q-count").innerText =
      data.total_questions || 0;

    const qList = document.getElementById("modal-questions-list");
    qList.innerHTML = "";
    if (data.questions && data.questions.length > 0) {
      data.questions.forEach((q) => {
        const li = document.createElement("li");
        li.innerHTML = `<strong>[${q.difficulty || "N/A"}]</strong> ${_escHtml(q.question_text)}`;
        qList.appendChild(li);
      });
    } else {
      qList.innerHTML = `<li style="border-left:none; font-style:italic;">No questions generated yet.</li>`;
    }

    const rejList = document.getElementById("modal-rejected-list");
    rejList.innerHTML = "";
    if (data.rejected && data.rejected.length > 0) {
      data.rejected.forEach((r) => {
        const li = document.createElement("li");
        li.style.borderLeftColor = "var(--danger)";
        li.innerHTML = `<strong>[${_escHtml(r.reason)}]</strong> ${_escHtml(r.question_text || "Unknown Fragment")}`;
        rejList.appendChild(li);
      });
    } else {
      rejList.innerHTML = `<li style="border-left:none; font-style:italic;">No rejected fragments.</li>`;
    }

    modal.showModal();
  } catch (error) {
    notify("Error", "Failed to load audit data.", "error");
  }
}

// ==========================================
// SETTINGS DRAWER & THEME TOGGLE
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
  const userId = localStorage.getItem("qp_user_id");

  // Theme logic
  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const root = document.documentElement;
      const isLight = root.getAttribute("data-theme") === "light";
      if (isLight) {
        root.removeAttribute("data-theme");
        localStorage.setItem("qp-theme", "dark");
      } else {
        root.setAttribute("data-theme", "light");
        localStorage.setItem("qp-theme", "light");
      }
    });
  }

  // Drawer toggles
  const drawer = document.getElementById("settings-drawer");
  const menuBtn = document.getElementById("menu-toggle");
  const closeBtn = document.getElementById("close-drawer");

  if (menuBtn && drawer) {
    menuBtn.addEventListener("click", () =>
      drawer.classList.add("drawer-open"),
    );
    closeBtn.addEventListener("click", () =>
      drawer.classList.remove("drawer-open"),
    );
  }

  // Modal close
  const modal = document.getElementById("file-summary-modal");
  if (modal) {
    document
      .getElementById("close-modal")
      .addEventListener("click", () => modal.close());
    modal.addEventListener("click", (e) => {
      const dims = modal.getBoundingClientRect();
      if (
        e.clientX < dims.left ||
        e.clientX > dims.right ||
        e.clientY < dims.top ||
        e.clientY > dims.bottom
      ) {
        modal.close();
      }
    });
  }

  // Action Logic
  document.getElementById("btn-logout")?.addEventListener("click", () => {
    localStorage.clear();
    window.location.href = "/login";
  });

  document
    .getElementById("btn-delete-account")
    ?.addEventListener("click", async () => {
      if (!userId) return;
      if (
        confirm(
          "WARNING: This will permanently delete your account, all files, and all interview history. Proceed?",
        )
      ) {
        try {
          const res = await fetch(`/api/users/${userId}`, { method: "DELETE" });
          if (res.ok) {
            localStorage.clear();
            window.location.href = "/login";
          }
        } catch (e) {
          notify("Error", "Failed to delete account", "error");
        }
      }
    });

  document
    .getElementById("btn-delete-files")
    ?.addEventListener("click", async () => {
      if (!userId) return;
      if (
        confirm(
          "Are you sure you want to delete all uploaded files and generated questions?",
        )
      ) {
        try {
          const res = await fetch(`/api/files/all/${userId}`, {
            method: "DELETE",
          });
          if (res.ok) {
            notify("Success", "All files deleted.", "success");
            fetchUserFiles();
            if (drawer) drawer.classList.remove("drawer-open");
          }
        } catch (e) {
          notify("Error", "Failed to delete files", "error");
        }
      }
    });

  document
    .getElementById("btn-change-email")
    ?.addEventListener("click", async () => {
      if (!userId) return;
      const newEmail = prompt("Enter your new email address:");
      if (newEmail) {
        try {
          const res = await fetch(`/api/users/${userId}/email`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ new_email: newEmail }),
          });
          if (res.ok)
            notify("Success", "Email updated successfully.", "success");
          else notify("Error", "Failed to update email.", "error");
        } catch (e) {
          notify("Error", "Failed to reach server.", "error");
        }
      }
    });
});

// ==========================================
// FILE UPLOAD (shared logic for button + drop)
// ==========================================
async function uploadFile(file) {
  if (_uploading) return;
  const userId = localStorage.getItem("qp_user_id");
  if (!userId) {
    window.location.href = "/login";
    return;
  }
  if (!file) return;

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
      notify("Upload successful", "Your file is being ingested.", "success");
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

const _uploadBtn = document.getElementById("upload-btn");
const _fileInput = document.getElementById("pdf-upload");
if (_uploadBtn && _fileInput) {
  _uploadBtn.addEventListener("click", () => uploadFile(_fileInput.files[0]));
}

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
    if (!dropZone.contains(e.relatedTarget))
      dropZone.classList.remove("drag-over");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file?.type === "application/pdf") uploadFile(file);
    else if (file) toast("Only PDF files are accepted.", "warning");
  });
  dropZone.addEventListener("click", (e) => {
    if (e.target !== _fileInput && e.target !== _uploadBtn) _fileInput?.click();
  });
}

// ==========================================
// POLL FOR NEW FILE AFTER UPLOAD
// ==========================================
async function pollForNewFile() {
  const userId = localStorage.getItem("qp_user_id");
  if (!userId) return;

  const baselineIds = new Set(_knownFileIds);
  let attempts = 0;
  const maxAttempts = 18;

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

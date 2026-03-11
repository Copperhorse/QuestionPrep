/* =========================================
   app.js - QuestionPrep Frontend Logic
========================================= */

const API_BASE = "";

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
        // DBManager returns 'user_id' as the primary key field name
        const userId = data.user.user_id || data.user.id;

        localStorage.setItem("qp_token", data.token);
        localStorage.setItem("qp_user_id", userId);

        window.location.href = "/profile";
      } else {
        alert(`Login failed: ${data.detail}`);
      }
    } catch (error) {
      console.error("Error logging in:", error);
      alert("A network error occurred. Is the FastAPI server running?");
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
        alert("Account created successfully! Please log in.");
        signupForm.reset();
        document.getElementById("signup-view").style.display = "none";
        document.getElementById("login-view").style.display = "block";
      } else {
        alert(`Sign up failed: ${data.detail}`);
      }
    } catch (error) {
      console.error("Error signing up:", error);
      alert("A network error occurred.");
    } finally {
      submitBtn.innerText = originalText;
      submitBtn.disabled = false;
    }
  });
}

// ==========================================
// 2. QUESTION GENERATION (Profile Page)
// ==========================================

/**
 * Trigger LLM enrichment + vector indexing for a specific file.
 * Called from the "Generate Questions" button on each file card.
 * The backend task is async — this just kicks it off.
 */
async function generateQuestions(fileId, btn) {
  const originalText = btn.innerText;
  btn.innerText = "Starting...";
  btn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/questions/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: fileId }),
    });

    const data = await response.json();

    if (response.ok) {
      btn.innerText = "✓ Generating in background";
      btn.style.backgroundColor = "var(--soft-lime)";
      btn.style.color = "var(--very-soft-navy)";
      // Leave disabled — re-triggering enrichment is wasteful
    } else {
      alert(`Failed to start generation: ${data.detail}`);
      btn.innerText = originalText;
      btn.disabled = false;
    }
  } catch (error) {
    console.error("Generate questions failed:", error);
    alert("A network error occurred during question generation.");
    btn.innerText = originalText;
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

  // Show a loading state while fetching
  filesGrid.innerHTML =
    "<p style='grid-column: 1 / -1; text-align: center; color: var(--soft-charcoal); opacity: 0.6;'>Loading your files…</p>";

  try {
    const response = await fetch(`${API_BASE}/api/files?user_id=${userId}`);
    const data = await response.json();

    filesGrid.innerHTML = "";

    if (!response.ok) {
      filesGrid.innerHTML =
        "<p style='grid-column: 1 / -1; color: red;'>Failed to load files.</p>";
      return;
    }

    if (!data.files || data.files.length === 0) {
      filesGrid.innerHTML =
        "<p style='grid-column: 1 / -1; text-align: center; color: var(--soft-charcoal);'>No PDFs uploaded yet. Start by uploading a document above!</p>";
      return;
    }

    data.files.forEach((file) => {
      const fileName =
        file.filename || file.file_name || file.name || "Document";
      const fileId = file.file_id || file.id || "unknown";

      const cardHtml = `
        <div class="card" style="background-color: var(--soft-lime);">
          <h3 style="word-break: break-word;">${fileName}</h3>
          <p style="opacity: 0.7; font-size: 0.85rem;">Uploaded: ${
            file.assigned_at
              ? new Date(file.assigned_at).toLocaleDateString()
              : "—"
          }</p>
          <div class="actions" style="margin-top: 20px; display: flex; flex-direction: column; gap: 10px;">
            <button
              class="cta-button"
              onclick="generateQuestions('${fileId}', this)"
              style="background-color: var(--periwinkle); border: none; cursor: pointer; padding: 10px 20px; font-size: 0.9rem;">
              Generate Questions
            </button>
            <a
              href="/interview"
              class="cta-button"
              style="background-color: var(--sky-blue); padding: 10px 20px; font-size: 0.9rem; text-align: center;">
              Start Interview
            </a>
          </div>
        </div>
      `;
      filesGrid.innerHTML += cardHtml;
    });
  } catch (error) {
    console.error("Error fetching files:", error);
    filesGrid.innerHTML =
      "<p style='grid-column: 1 / -1; color: red;'>Failed to load your files.</p>";
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
      alert("Please select a PDF file first.");
      return;
    }

    if (!userId) {
      alert("You must be logged in to upload files.");
      window.location.href = "/login";
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const originalText = uploadBtn.innerText;
    uploadBtn.innerText = "Uploading…";
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
        alert(
          "File uploaded! Ingestion is running in the background.\n\n" +
            'Once it appears in your file list below, click "Generate Questions" to create interview questions (this may take a few minutes).',
        );
        // Poll the file list until the new file appears (up to 90 seconds)
        pollForNewFile();
      } else {
        alert(`Upload failed: ${data.detail}`);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("A network error occurred during upload.");
    } finally {
      uploadBtn.innerText = originalText;
      uploadBtn.disabled = false;
    }
  });
}

/**
 * Poll /api/files every 5 seconds after an upload until the file
 * count increases, then refresh the grid. Stops after 18 attempts (~90s).
 */
async function pollForNewFile() {
  const userId = localStorage.getItem("qp_user_id");
  if (!userId) return;

  // Capture current file count
  let previousCount = document.querySelectorAll("#files-grid .card").length;
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

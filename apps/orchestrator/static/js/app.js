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
        // IMPORTANT: Ensure your DB returns 'id'. If it returns 'user_id', change this to data.user.user_id
        const userId = data.user.id || data.user.user_id;

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
// 2. DYNAMIC FILE FETCHING (Profile Page)
// ==========================================
async function fetchUserFiles() {
  const userId = localStorage.getItem("qp_user_id");
  const filesGrid = document.getElementById("files-grid");

  if (!userId || !filesGrid) return;

  try {
    const response = await fetch(`${API_BASE}/api/files?user_id=${userId}`);
    const data = await response.json();

    if (response.ok) {
      filesGrid.innerHTML = "";

      if (!data.files || data.files.length === 0) {
        filesGrid.innerHTML =
          "<p style='grid-column: 1 / -1; text-align: center; color: var(--soft-charcoal);'>No PDFs uploaded yet. Start by uploading a document!</p>";
        return;
      }

      data.files.forEach((file) => {
        const fileName = file.filename || file.name || "Document";
        const fileId = file.id || file.file_id || "unknown";

        const cardHtml = `
                    <div class="card" style="background-color: var(--soft-lime);">
                        <h3>${fileName}</h3>
                        <p>Status: Ready</p>
                        <div class="actions" style="margin-top: 20px;">
                            <a href="/interview?file_id=${fileId}" class="cta-button" style="background-color: var(--periwinkle); padding: 10px 20px; font-size: 0.95rem;">Start Interview</a>
                        </div>
                    </div>
                `;
        filesGrid.innerHTML += cardHtml;
      });
    }
  } catch (error) {
    console.error("Error fetching files:", error);
    filesGrid.innerHTML =
      "<p style='color: red;'>Failed to load your files.</p>";
  }
}

document.addEventListener("DOMContentLoaded", fetchUserFiles);

// ==========================================
// 3. FILE UPLOAD LOGIC (Profile Page)
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
    uploadBtn.innerText = "Uploading & Processing...";
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
        alert("File uploaded successfully! It is now processing.");
        fileInput.value = "";
        await fetchUserFiles();
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

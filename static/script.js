"use strict";

/* ==========================================================================
   Constants
   ========================================================================== */

const EMOTION_META = {
  sadness: { emoji: "😢", color: "#6699ff", ambient: "#2a3a66" },
  joy: { emoji: "😄", color: "#f6c453", ambient: "#4a3a1a" },
  love: { emoji: "❤️", color: "#f4708f", ambient: "#4a2436" },
  anger: { emoji: "😠", color: "#ef7454", ambient: "#4a2a1c" },
  fear: { emoji: "😨", color: "#a68df0", ambient: "#332a52" },
  surprise: { emoji: "😲", color: "#52d6d1", ambient: "#1c4442" },
};

const EMOTION_ORDER = ["sadness", "joy", "love", "anger", "fear", "surprise"];
const RING_CIRCUMFERENCE = 2 * Math.PI * 70; // r = 70, matches SVG
const MAX_CHARS = 2000;

/* ==========================================================================
   Elements
   ========================================================================== */

const el = {
  textInput: document.getElementById("textInput"),
  charCounter: document.getElementById("charCounter"),
  chipRow: document.getElementById("chipRow"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusDot: document.getElementById("statusDot"),
  statusLabel: document.getElementById("statusLabel"),
  footerDot: document.getElementById("footerDot"),
  footerStatus: document.getElementById("footerStatus"),
  resultsEmpty: document.getElementById("resultsEmpty"),
  resultsBody: document.getElementById("resultsBody"),
  resultEmoji: document.getElementById("resultEmoji"),
  resultEmotion: document.getElementById("resultEmotion"),
  resultConfidence: document.getElementById("resultConfidence"),
  ringProgress: document.getElementById("ringProgress"),
  breakdownList: document.getElementById("breakdownList"),
  insightText: document.getElementById("insightText"),
  toastStack: document.getElementById("toastStack"),
};

let isAnalyzing = false;

/* ==========================================================================
   Health check
   ========================================================================== */

async function checkHealth() {
  setStatus("loading");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("bad status");
    const data = await res.json();
    setStatus(data.model_loaded ? "online" : "offline");
  } catch (err) {
    setStatus("offline");
  }
}

function setStatus(state) {
  const labels = {
    online: "AI Online",
    offline: "AI Offline",
    loading: "Connecting…",
  };
  const footer = {
    online: "Model online",
    offline: "Model unavailable",
    loading: "Checking model status…",
  };

  el.statusDot.className = "status-dot " + state;
  el.statusLabel.textContent = labels[state];
  el.footerDot.className = "status-dot small " + state;
  el.footerStatus.textContent = footer[state];
}

/* ==========================================================================
   Input handling
   ========================================================================== */

function updateCharCounter() {
  const len = el.textInput.value.length;
  el.charCounter.textContent = `${len} / ${MAX_CHARS}`;
  el.charCounter.classList.toggle("warn", len >= MAX_CHARS * 0.9 && len < MAX_CHARS);
  el.charCounter.classList.toggle("max", len >= MAX_CHARS);
}

function validateInput(text) {
  return typeof text === "string" && text.trim().length > 0;
}

el.textInput.addEventListener("input", updateCharCounter);

el.textInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    analyzeEmotion();
  }
});

el.chipRow.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  el.textInput.value = chip.textContent;
  updateCharCounter();
  el.textInput.focus();
});

el.analyzeBtn.addEventListener("click", analyzeEmotion);

/* ==========================================================================
   Loading state
   ========================================================================== */

function setLoadingState(loading) {
  isAnalyzing = loading;
  el.analyzeBtn.classList.toggle("loading", loading);
  el.analyzeBtn.disabled = loading;
}

/* ==========================================================================
   Core analyze flow
   ========================================================================== */

async function analyzeEmotion() {
  if (isAnalyzing) return;

  const text = el.textInput.value;

  if (!validateInput(text)) {
    showToast("Please enter some text before analyzing.", "error");
    el.textInput.focus();
    return;
  }

  setLoadingState(true);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (res.status === 503) {
      showToast("The AI model is still loading. Please try again shortly.", "info");
      return;
    }

    if (!res.ok) {
      showToast("Something went wrong while analyzing your text. Please try again.", "error");
      return;
    }

    const data = await res.json();
    renderPrediction(data);
  } catch (err) {
    showToast("The Emotion AI engine is unavailable. Please try again.", "error");
  } finally {
    setLoadingState(false);
  }
}

/* ==========================================================================
   Rendering
   ========================================================================== */

function extractEmotionKey(predictionEmotion) {
  const token = String(predictionEmotion || "").trim().split(/\s+/)[0].toLowerCase();
  return EMOTION_ORDER.includes(token) ? token : EMOTION_ORDER[0];
}

function renderPrediction(data) {
  const key = extractEmotionKey(data.predicted_emotion || data.prediction_emotion);
  const meta = EMOTION_META[key];

  updateEmotionTheme(key);
  renderConfidence(key, meta, data.confidence);
  renderEmotionBreakdown(data.all_probabilities || data.all_probabilites, key);
  generateInsight(data.confidence);

  el.resultsEmpty.hidden = true;
  el.resultsBody.hidden = false;
  el.resultsBody.classList.remove("reveal");
  // Force reflow so the animation replays on repeated analyses
  void el.resultsBody.offsetWidth;
  el.resultsBody.classList.add("reveal");

  el.resultsBody.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderConfidence(key, meta, confidence) {
  const pct = Math.max(0, Math.min(1, confidence)) * 100;
  el.resultEmoji.textContent = meta.emoji;
  el.resultEmotion.textContent = key;
  el.resultConfidence.textContent = `${pct.toFixed(1)}%`;

  el.ringProgress.style.stroke = meta.color;
  const offset = RING_CIRCUMFERENCE * (1 - pct / 100);
  el.ringProgress.style.strokeDasharray = `${RING_CIRCUMFERENCE}`;
  // Reset then animate for a clean sweep on repeated analyses
  el.ringProgress.style.transition = "none";
  el.ringProgress.style.strokeDashoffset = `${RING_CIRCUMFERENCE}`;
  void el.ringProgress.getBoundingClientRect();
  el.ringProgress.style.transition = "";
  requestAnimationFrame(() => {
    el.ringProgress.style.strokeDashoffset = `${offset}`;
  });
}

function renderEmotionBreakdown(allProbabilities, leadKey) {
  const entries = Object.entries(allProbabilities || {}).sort((a, b) => b[1] - a[1]);

  el.breakdownList.innerHTML = "";

  entries.forEach(([label, prob]) => {
    const meta = EMOTION_META[label] || { emoji: "•", color: "#8888a0" };
    const pct = Math.max(0, Math.min(1, prob)) * 100;

    const li = document.createElement("li");
    li.className = "breakdown-row" + (label === leadKey ? " lead" : "");
    li.innerHTML = `
      <span class="breakdown-emoji">${meta.emoji}</span>
      <span class="breakdown-name">${label}</span>
      <span class="breakdown-track"><span class="breakdown-fill" style="--bar-color:${meta.color}"></span></span>
      <span class="breakdown-pct">${pct.toFixed(1)}%</span>
    `;
    el.breakdownList.appendChild(li);

    const fill = li.querySelector(".breakdown-fill");
    requestAnimationFrame(() => {
      fill.style.width = `${pct}%`;
    });
  });
}

function generateInsight(confidence) {
  let message;
  if (confidence >= 0.85) {
    message = "The model shows a strong and highly confident signal for this emotion.";
  } else if (confidence >= 0.6) {
    message = "The model identifies a clear emotional signal, although some secondary emotions are present.";
  } else {
    message = "The text contains mixed emotional signals, so the prediction is less certain.";
  }
  el.insightText.textContent = message;
}

function updateEmotionTheme(key) {
  const meta = EMOTION_META[key];
  document.documentElement.style.setProperty("--ambient", meta.ambient);
}

function resetResults() {
  el.resultsBody.hidden = true;
  el.resultsBody.classList.remove("reveal");
  el.resultsEmpty.hidden = false;
  document.documentElement.style.setProperty("--ambient", "transparent");
}

/* ==========================================================================
   Toasts
   ========================================================================== */

const TOAST_ICONS = {
  success: '<svg viewBox="0 0 20 20" fill="none"><path d="M4 10.5L8 14.5L16 6.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  error: '<svg viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="1.6"/><path d="M10 6.5V11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="10" cy="13.6" r="0.9" fill="currentColor"/></svg>',
  info: '<svg viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="1.6"/><path d="M10 9V14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="10" cy="6.4" r="0.9" fill="currentColor"/></svg>',
};

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.setAttribute("role", "status");
  toast.innerHTML = `
    <span class="toast-icon">${TOAST_ICONS[type] || TOAST_ICONS.info}</span>
    <span class="toast-message">${escapeHtml(message)}</span>
  `;
  el.toastStack.appendChild(toast);

  const dismiss = () => {
    toast.classList.add("leaving");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  };

  setTimeout(dismiss, 4200);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/* ==========================================================================
   Init
   ========================================================================== */

updateCharCounter();
checkHealth();

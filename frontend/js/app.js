/* ═══════════════════════════════════════════════════════════
   DermaScan AI — Frontend Application (No Auth)
   ═══════════════════════════════════════════════════════════ */

const API = "/api";
let currentView = null;

// ─── Toast ──────────────────────────────────────────────────
function toast(msg, type = "info") {
    const c = document.getElementById("toast-container");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; setTimeout(() => el.remove(), 300); }, 3500);
}

// ─── API Helper ─────────────────────────────────────────────
async function api(endpoint, opts = {}) {
    let res;
    try {
        res = await fetch(`${API}${endpoint}`, opts);
    } catch (e) {
        throw new Error("Backend not responding. Run: python start.py");
    }
    if (!res.ok) {
        let msg = `Error ${res.status}`;
        try { const j = await res.json(); msg = j.detail || msg; } catch (_) { }
        throw new Error(msg);
    }
    return res.json();
}

// ─── Navigation ─────────────────────────────────────────────
function navigate(view, data = null) {
    currentView = view;
    // Update nav highlights
    document.querySelectorAll(".nav-link").forEach(btn => {
        const id = btn.id.replace("nav-", "");
        if (id === view) {
            btn.className = "nav-link flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-headline font-bold tracking-tight bg-accent/10 text-accent border-l-2 border-accent transition-all duration-200";
        } else {
            btn.className = "nav-link flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-headline font-medium tracking-tight text-txt-secondary hover:text-txt-primary hover:bg-bg-hover transition-all duration-200";
        }
    });

    const main = document.getElementById("main-content");
    if (view === "dashboard") renderDashboard(main);
    else if (view === "history") renderHistory(main);
    else if (view === "results") renderResults(main, data);
}

// ═══════════════════════════════════════════════════════════
// DASHBOARD VIEW (Upload)
// ═══════════════════════════════════════════════════════════
function renderDashboard(container) {
    container.innerHTML = `
    <div class="view-transition p-8 lg:p-12 max-w-4xl mx-auto flex flex-col items-center justify-center min-h-screen">
        <!-- Header -->
        <div class="w-full mb-10 text-center">
            <h2 class="font-headline text-4xl font-bold tracking-tight text-txt-primary mb-3">Diagnostic Console</h2>
            <p class="font-mono text-xs text-txt-muted uppercase tracking-widest">Upload a dermoscopic image for AI-powered lesion analysis</p>
        </div>

        <!-- Upload Zone -->
        <div id="drop-zone" class="relative w-full max-w-[560px] aspect-[4/3] cursor-pointer group">
            <div class="absolute inset-0 border-2 border-dashed border-border-accent rounded-2xl bg-bg-card/40 backdrop-blur-sm
                        flex flex-col items-center justify-center transition-all duration-300
                        group-hover:border-accent/50 group-hover:bg-accent/[0.03]">
                <div id="upload-content" class="flex flex-col items-center text-center px-6">
                    <div class="w-20 h-20 mb-5 bg-accent/10 rounded-2xl flex items-center justify-center group-hover:bg-accent/15 transition-colors">
                        <span class="material-symbols-outlined text-4xl text-accent">upload_file</span>
                    </div>
                    <h3 class="font-headline text-xl font-semibold text-txt-primary mb-2">Drop Scan for Analysis</h3>
                    <p class="text-txt-secondary text-sm mb-6 max-w-[300px] leading-relaxed">
                        Drag JPEG or PNG dermoscopic images here, or click to browse your files.
                    </p>
                    <button id="browse-btn" class="bg-bg-hover hover:bg-accent/10 border border-border-accent text-txt-primary
                                                    font-headline text-xs uppercase tracking-widest px-8 py-3 rounded-xl transition-all hover:text-accent">
                        Browse Local Files
                    </button>
                    <span class="font-mono text-[9px] text-txt-muted mt-3 uppercase">Supported: JPG, PNG • Max 50MB</span>
                </div>

                <!-- Preview (hidden initially) -->
                <div id="preview-wrap" class="hidden absolute inset-0 p-3">
                    <img id="preview-img" class="w-full h-full object-contain rounded-xl" />
                    <button id="clear-preview" class="absolute top-5 right-5 w-8 h-8 bg-bg-primary/80 border border-border-subtle rounded-lg
                                                       flex items-center justify-center text-txt-secondary hover:text-danger transition-colors">
                        <span class="material-symbols-outlined text-lg">close</span>
                    </button>
                </div>
            </div>
        </div>

        <!-- Analyze Button -->
        <button id="analyze-btn" disabled
            class="w-full max-w-[560px] mt-6 py-4 rounded-xl font-headline font-bold text-base tracking-wide uppercase
                   flex items-center justify-center gap-3 transition-all duration-300
                   bg-bg-card text-txt-muted cursor-not-allowed border border-border-subtle">
            <span class="material-symbols-outlined">lock</span>
            Upload Image to Analyze
        </button>

        <!-- Info Cards -->
        <div class="w-full max-w-[560px] mt-8 grid grid-cols-3 gap-4">
            <div class="bg-bg-card/60 border border-border-subtle rounded-xl p-4 text-center">
                <p class="font-mono text-[9px] text-txt-muted uppercase mb-1">Model</p>
                <p class="font-headline text-sm font-bold text-txt-primary">EfficientNet-B3</p>
            </div>
            <div class="bg-bg-card/60 border border-border-subtle rounded-xl p-4 text-center">
                <p class="font-mono text-[9px] text-txt-muted uppercase mb-1">Classes</p>
                <p class="font-headline text-sm font-bold text-txt-primary">7 Types</p>
            </div>
            <div class="bg-bg-card/60 border border-border-subtle rounded-xl p-4 text-center">
                <p class="font-mono text-[9px] text-txt-muted uppercase mb-1">Accuracy</p>
                <p class="font-headline text-sm font-bold text-safe">86.4%</p>
            </div>
        </div>
    </div>`;

    setupUploadHandlers();
}

// ─── Upload Logic ───────────────────────────────────────────
let selectedFile = null;

function setupUploadHandlers() {
    const zone = document.getElementById("drop-zone");
    const fileInput = document.createElement("input");
    fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    const browse = document.getElementById("browse-btn");
    if (browse) browse.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
    zone.addEventListener("click", () => fileInput.click());

    zone.addEventListener("dragover", e => { e.preventDefault(); zone.querySelector(".absolute").classList.add("drag-active"); });
    zone.addEventListener("dragleave", e => { e.preventDefault(); zone.querySelector(".absolute").classList.remove("drag-active"); });
    zone.addEventListener("drop", e => {
        e.preventDefault(); zone.querySelector(".absolute").classList.remove("drag-active");
        if (e.dataTransfer.files.length) showPreview(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", e => { if (e.target.files.length) showPreview(e.target.files[0]); });

    document.getElementById("analyze-btn").addEventListener("click", runAnalysis);
}

function showPreview(file) {
    selectedFile = file;
    const img = document.getElementById("preview-img");
    const preview = document.getElementById("preview-wrap");
    const content = document.getElementById("upload-content");

    img.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
    content.classList.add("hidden");

    // Enable analyze button
    const btn = document.getElementById("analyze-btn");
    btn.disabled = false;
    btn.className = "w-full max-w-[560px] mt-6 py-4 rounded-xl font-headline font-bold text-base tracking-wide uppercase flex items-center justify-center gap-3 transition-all duration-300 bg-accent hover:brightness-110 text-bg-primary cursor-pointer glow-accent";
    btn.innerHTML = `<span class="material-symbols-outlined">neurology</span> Run Neural Analysis`;

    document.getElementById("clear-preview").addEventListener("click", (e) => {
        e.stopPropagation();
        selectedFile = null;
        preview.classList.add("hidden");
        content.classList.remove("hidden");
        btn.disabled = true;
        btn.className = "w-full max-w-[560px] mt-6 py-4 rounded-xl font-headline font-bold text-base tracking-wide uppercase flex items-center justify-center gap-3 transition-all duration-300 bg-bg-card text-txt-muted cursor-not-allowed border border-border-subtle";
        btn.innerHTML = `<span class="material-symbols-outlined">lock</span> Upload Image to Analyze`;
    });
}

async function runAnalysis() {
    if (!selectedFile) return;
    const btn = document.getElementById("analyze-btn");
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-outlined animate-spin">progress_activity</span> Computing Diagnostics...`;
    btn.className = "w-full max-w-[560px] mt-6 py-4 rounded-xl font-headline font-bold text-base tracking-wide uppercase flex items-center justify-center gap-3 bg-bg-card text-accent border border-accent/30 cursor-wait";

    try {
        const fd = new FormData();
        fd.append("file", selectedFile);
        const data = await api("/analyze", { method: "POST", body: fd });
        data._imageUrl = URL.createObjectURL(selectedFile);
        selectedFile = null;
        navigate("results", data);
    } catch (err) {
        toast(err.message, "error");
        navigate("dashboard");
    }
}

// ═══════════════════════════════════════════════════════════
// RESULTS VIEW
// ═══════════════════════════════════════════════════════════

const CLASS_LABELS = {
    akiec: "Actinic Keratosis", bcc: "Basal Cell Carcinoma", bkl: "Benign Keratosis",
    df: "Dermatofibroma", mel: "Melanoma", nv: "Melanocytic Nevus", vasc: "Vascular Lesion"
};

function renderResults(container, data) {
    if (!data) return navigate("dashboard");

    const isHigh = data.is_high_risk;
    const confPct = (data.confidence * 100).toFixed(1);
    const dashOffset = 264 - (264 * data.confidence);
    const label = CLASS_LABELS[data.prediction] || data.prediction;

    // Sort scores descending
    const sorted = Object.entries(data.scores || {}).sort((a, b) => b[1] - a[1]);

    container.innerHTML = `
    <div class="view-transition p-8 lg:p-12 max-w-5xl mx-auto">
        <!-- Banner -->
        <div class="mb-8 px-6 py-4 rounded-xl flex items-center justify-between ${isHigh ? 'bg-danger/10 border border-danger/30' : 'bg-safe/10 border border-safe/30'}">
            <div class="flex items-center gap-3">
                <span class="material-symbols-outlined text-2xl ${isHigh ? 'text-danger animate-pulse' : 'text-safe'}"
                      style="font-variation-settings:'FILL' 1;">${isHigh ? 'warning' : 'check_circle'}</span>
                <span class="font-mono text-sm font-bold tracking-tight uppercase ${isHigh ? 'text-danger' : 'text-safe'}">
                    ${isHigh ? '⚠ HIGH RISK — Immediate Review Recommended' : '✓ LOW RISK — Benign Morphology Detected'}
                </span>
            </div>
        </div>

        <div class="grid grid-cols-12 gap-8">
            <!-- LEFT: Main prediction + image -->
            <div class="col-span-12 lg:col-span-7 space-y-6">
                <!-- Prediction Card -->
                <div class="bg-bg-card border ${isHigh ? 'border-danger/30 glow-danger' : 'border-border-subtle'} rounded-2xl p-8 relative overflow-hidden">
                    <div class="absolute top-0 right-0 w-48 h-48 ${isHigh ? 'bg-danger/5' : 'bg-safe/5'} blur-[60px] -mr-24 -mt-24"></div>
                    <div class="relative z-10 flex justify-between items-start">
                        <div>
                            <p class="font-mono text-xs ${isHigh ? 'text-danger' : 'text-safe'} font-semibold tracking-widest uppercase mb-1">Pathology Prediction</p>
                            <h3 class="font-headline text-4xl font-black text-txt-primary tracking-tight uppercase mb-3">${label}</h3>
                            <div class="inline-flex items-center px-3 py-1 ${isHigh ? 'bg-danger/15 text-danger border-danger/30' : 'bg-safe/15 text-safe border-safe/30'} text-[11px] font-mono rounded-lg border">
                                Threshold: ${data.threshold?.toFixed(2) || 'N/A'} — ${isHigh ? 'EXCEEDED' : 'WITHIN RANGE'}
                            </div>
                        </div>
                        <!-- Confidence Ring -->
                        <div class="relative w-28 h-28 shrink-0">
                            <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="42" fill="transparent" stroke="#1e2740" stroke-width="6"></circle>
                                <circle class="conf-ring ${isHigh ? 'text-danger' : 'text-safe'}" cx="50" cy="50" r="42" fill="transparent"
                                    stroke="currentColor" stroke-width="6" stroke-linecap="round"
                                    stroke-dasharray="264" stroke-dashoffset="${dashOffset}"></circle>
                            </svg>
                            <div class="absolute inset-0 flex flex-col items-center justify-center">
                                <span class="font-mono text-2xl font-black text-txt-primary">${confPct}%</span>
                                <span class="font-mono text-[8px] text-txt-muted uppercase">Confidence</span>
                            </div>
                        </div>
                    </div>

                    <!-- Image preview -->
                    ${data._imageUrl ? `
                    <div class="mt-6 rounded-xl overflow-hidden border border-border-subtle bg-bg-primary">
                        <img src="${data._imageUrl}" class="w-full max-h-[300px] object-contain" alt="Uploaded lesion scan"/>
                    </div>` : ''}
                </div>

                <!-- Actions -->
                <div class="grid grid-cols-2 gap-4">
                    <button onclick="navigate('dashboard')"
                        class="bg-accent hover:brightness-110 text-bg-primary px-6 py-4 rounded-xl font-headline font-bold text-sm tracking-widest uppercase flex items-center justify-center gap-3 glow-accent transition-all">
                        <span class="material-symbols-outlined text-lg">add_a_photo</span> New Scan
                    </button>
                    <button onclick="navigate('history')"
                        class="bg-bg-card border border-border-subtle text-txt-primary px-6 py-4 rounded-xl font-headline font-bold text-sm tracking-widest uppercase flex items-center justify-center gap-3 hover:bg-bg-hover transition-all">
                        <span class="material-symbols-outlined text-lg">history</span> View History
                    </button>
                </div>
            </div>

            <!-- RIGHT: All class scores -->
            <div class="col-span-12 lg:col-span-5 space-y-6">
                <div class="bg-bg-card border border-border-subtle rounded-2xl p-6">
                    <h4 class="font-mono text-xs text-accent font-bold uppercase tracking-widest mb-6">Full Probability Breakdown</h4>
                    <div class="space-y-4">
                        ${sorted.map(([cls, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const isTop = cls === data.prediction;
        const barColor = isTop ? (isHigh ? 'bg-danger' : 'bg-safe') : 'bg-txt-muted/40';
        const textColor = isTop ? (isHigh ? 'text-danger font-bold' : 'text-safe font-bold') : 'text-txt-secondary';
        return `
                            <div>
                                <div class="flex justify-between text-[11px] font-mono mb-1.5">
                                    <span class="${textColor}">${CLASS_LABELS[cls] || cls}</span>
                                    <span class="${textColor}">${pct}%</span>
                                </div>
                                <div class="h-1.5 bg-bg-hover rounded-full overflow-hidden">
                                    <div class="${barColor} h-full rounded-full transition-all duration-700" style="width:${Math.max(pct, 1)}%"></div>
                                </div>
                            </div>`;
    }).join('')}
                    </div>
                </div>
            </div>
        </div>
    </div>`;
}

// ═══════════════════════════════════════════════════════════
// HISTORY VIEW
// ═══════════════════════════════════════════════════════════
async function renderHistory(container) {
    container.innerHTML = `
    <div class="view-transition p-8 lg:p-12 max-w-5xl mx-auto">
        <div class="flex justify-between items-end mb-8">
            <div>
                <h2 class="font-headline text-4xl font-bold tracking-tight text-txt-primary mb-2">Diagnostic Archive</h2>
                <p class="font-mono text-xs text-txt-muted uppercase tracking-widest flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-safe"></span> Past scan results
                </p>
            </div>
            <button id="delete-all-btn"
                class="flex items-center gap-2 px-4 py-2 bg-danger/10 border border-danger/30 rounded-xl text-danger text-xs font-headline font-bold uppercase tracking-widest hover:bg-danger/20 transition-all">
                <span class="material-symbols-outlined text-sm">delete_sweep</span> Clear All
            </button>
        </div>
        <div id="history-list" class="space-y-3">
            <div class="flex items-center justify-center py-20">
                <span class="material-symbols-outlined text-3xl text-accent animate-spin">progress_activity</span>
            </div>
        </div>
    </div>`;

    document.getElementById("delete-all-btn").addEventListener("click", async () => {
        if (!confirm("Delete ALL history? This cannot be undone.")) return;
        try {
            await api("/history/all", { method: "DELETE" });
            toast("All history cleared", "success");
            renderHistory(container);
        } catch (e) { toast(e.message, "error"); }
    });

    try {
        const history = await api("/history");
        const list = document.getElementById("history-list");

        if (!history.length) {
            list.innerHTML = `
            <div class="flex flex-col items-center justify-center py-20 text-center">
                <span class="material-symbols-outlined text-5xl text-txt-muted mb-4">inbox</span>
                <p class="font-headline text-lg font-semibold text-txt-secondary mb-1">No scans yet</p>
                <p class="text-sm text-txt-muted">Upload an image from the Dashboard to get started.</p>
                <button onclick="navigate('dashboard')" class="mt-6 px-6 py-3 bg-accent/10 border border-accent/30 text-accent rounded-xl font-headline text-xs uppercase tracking-widest hover:bg-accent/20 transition-all">
                    Go to Dashboard
                </button>
            </div>`;
            return;
        }

        list.innerHTML = history.map(log => {
            const dt = new Date(log.created_at);
            const dateStr = dt.toLocaleDateString("en-CA"); // YYYY-MM-DD
            const timeStr = dt.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
            const label = CLASS_LABELS[log.prediction] || log.prediction;
            const confPct = (log.confidence * 100).toFixed(1);
            const isHigh = log.is_high_risk;
            return `
            <div class="bg-bg-card border border-border-subtle rounded-xl p-5 flex items-center gap-6 hover:bg-bg-hover transition-colors group">
                <!-- Date -->
                <div class="w-28 shrink-0">
                    <p class="font-mono text-xs text-txt-secondary">${dateStr}</p>
                    <p class="font-mono text-[10px] text-txt-muted">${timeStr}</p>
                </div>
                <!-- Prediction -->
                <div class="flex-1">
                    <p class="font-headline font-bold text-base ${isHigh ? 'text-danger' : 'text-txt-primary'}">${label}</p>
                    <p class="font-mono text-[10px] text-txt-muted uppercase mt-0.5">Session #${log.id}</p>
                </div>
                <!-- Confidence -->
                <div class="text-right w-20">
                    <p class="font-mono text-sm font-bold ${isHigh ? 'text-danger' : 'text-txt-primary'}">${confPct}%</p>
                    <p class="font-mono text-[9px] text-txt-muted uppercase">conf</p>
                </div>
                <!-- Risk Badge -->
                <div class="w-24 flex justify-center">
                    ${isHigh
                    ? `<span class="flex items-center gap-1.5 px-3 py-1 bg-danger/15 border border-danger/30 rounded-lg">
                             <span class="w-2 h-2 rounded-full bg-danger pulse-dot"></span>
                             <span class="text-[10px] font-black uppercase text-danger tracking-tight">Critical</span>
                           </span>`
                    : `<span class="flex items-center gap-1.5 px-3 py-1 bg-bg-hover border border-border-subtle rounded-lg">
                             <span class="w-2 h-2 rounded-full bg-safe"></span>
                             <span class="text-[10px] font-black uppercase text-txt-muted tracking-tight">Normal</span>
                           </span>`
                }
                </div>
                <!-- Delete -->
                <button onclick="deleteEntry(${log.id}, event)"
                    class="w-8 h-8 rounded-lg flex items-center justify-center text-txt-muted hover:text-danger hover:bg-danger/10 transition-all opacity-0 group-hover:opacity-100">
                    <span class="material-symbols-outlined text-lg">delete</span>
                </button>
            </div>`;
        }).join("");

    } catch (e) { toast(e.message, "error"); }
}

window.deleteEntry = async function (id, event) {
    event.stopPropagation();
    try {
        await api(`/history/${id}`, { method: "DELETE" });
        toast("Entry deleted", "success");
        navigate("history");
    } catch (e) { toast(e.message, "error"); }
};

// ─── Init ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => navigate("dashboard"));

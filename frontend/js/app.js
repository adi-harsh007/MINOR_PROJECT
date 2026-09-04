/* ═══════════════════════════════════════════════════════════════════════════
   DermaScan AI 4.0 — Cyber-Clinical SPA Engine
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = "/api";

// ─── Application State ──────────────────────────────────────────────────────
const state = {
    selectedFile: null,
    selectedAnatomicSite: "Anterior Torso",
    zoom: 1.0,
    panX: 0,
    panY: 0,
    heatmapVisible: false,
    heatmapBase64: null,
    calibration: {
        brightness: 100,
        contrast: 100,
        saturation: 100
    },
    latestResult: null,
    history: []
};

// ─── Pathology Class Descriptions & Codes ───────────────────────────────────
const PATHOLOGY_META = {
    akiec: {
        name: "Actinic Keratosis",
        code: "ICD-11: EK90",
        type: "Pre-Malignant",
        desc: "Rough, scaly patch on sun-damaged skin. Precursor to squamous cell carcinoma requiring topical treatment or cryotherapy."
    },
    bcc: {
        name: "Basal Cell Carcinoma",
        code: "ICD-11: 2C31",
        type: "Malignant",
        desc: "Slow-growing malignant epidermal tumor. Common in sun-exposed areas; rarely metastasizes but causes local damage."
    },
    bkl: {
        name: "Benign Keratosis",
        code: "ICD-11: ED50",
        type: "Benign",
        desc: "Non-cancerous skin growth including seborrheic keratosis and solar lentigines. No malignant potential."
    },
    df: {
        name: "Dermatofibroma",
        code: "ICD-11: EA81",
        type: "Benign",
        desc: "Common benign cutaneous nodule consisting of fibrous tissue. Typical 'dimple sign' upon lateral compression."
    },
    mel: {
        name: "Melanoma",
        code: "ICD-11: 2C30",
        type: "Malignant / High Risk",
        desc: "Aggressive malignant tumor derived from melanocytes. Requires urgent surgical excision and dermatological evaluation."
    },
    nv: {
        name: "Melanocytic Nevus",
        code: "ICD-11: ED20",
        type: "Benign",
        desc: "Common benign nevomelanocytic proliferation (mole). Benign morphology with uniform pigmentation."
    },
    vasc: {
        name: "Vascular Lesion",
        code: "ICD-11: ED90",
        type: "Benign",
        desc: "Vascular proliferation including cherry angiomas and pyogenic granulomas. Non-malignant vascular structure."
    }
};

// ─── Toast Notifications ────────────────────────────────────────────────────
function toast(message, type = "info") {
    // The container is created on demand: index.html does not define one, which
    // previously made every notification in the app a silent no-op.
    let container = document.getElementById("toast-container");
    if (!container) {
        container = document.createElement("div");
        container.id = "toast-container";
        container.className = "fixed top-4 right-4 z-50 flex flex-col gap-2";
        document.body.appendChild(container);
    }
    
    const colors = {
        info: "bg-surface-elevated border-cyan-500/50 text-cyan-400",
        success: "bg-surface-elevated border-emerald-500/50 text-emerald-400",
        warning: "bg-surface-elevated border-amber-500/50 text-amber-400",
        error: "bg-surface-elevated border-rose-500/50 text-rose-400"
    };

    const el = document.createElement("div");
    el.className = `toast px-4 py-3 rounded-xl border font-mono text-xs shadow-2xl flex items-center gap-3 transition-all duration-300 transform translate-y-2 opacity-0 ${colors[type] || colors.info}`;
    el.innerHTML = `
        <span class="material-symbols-outlined text-base font-bold">${type === 'error' ? 'error' : type === 'success' ? 'check_circle' : 'info'}</span>
        <span>${message}</span>
    `;
    
    container.appendChild(el);
    requestAnimationFrame(() => {
        el.classList.remove("translate-y-2", "opacity-0");
    });

    setTimeout(() => {
        el.classList.add("opacity-0", "-translate-y-2");
        setTimeout(() => el.remove(), 300);
    }, 4000);
}

// ─── API Client ─────────────────────────────────────────────────────────────
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        if (!response.ok) {
            let errorMsg = `Server Error (${response.status})`;
            try {
                const errJson = await response.json();
                // OOD rejections send {message, reason}; everything else a string.
                const detail = errJson.detail;
                if (typeof detail === "string") {
                    errorMsg = detail;
                } else if (detail && detail.message) {
                    errorMsg = detail.message;
                }
            } catch (_) {}
            const error = new Error(errorMsg);
            error.status = response.status;
            throw error;
        }
        return await response.json();
    } catch (err) {
        const error = new Error(err.message || "Failed to communicate with DermaScan AI server.");
        error.status = err.status;
        throw error;
    }
}

// ─── SPA Navigation Router ──────────────────────────────────────────────────
function navigate(viewId, payload = null) {

    // 1. Hide all view containers
    const views = document.querySelectorAll(".view-container");
    views.forEach(view => {
        view.classList.add("hidden");
    });

    // 2. Show target view container
    const targetView = document.getElementById(viewId);
    if (targetView) {
        targetView.classList.remove("hidden");
        targetView.scrollIntoView({ behavior: "smooth" });
    }

    // 3. Highlight sidebar links
    document.querySelectorAll(".nav-item").forEach(link => {
        const linkView = link.getAttribute("data-view");
        if (linkView === viewId) {
            link.className = "nav-item active flex items-center px-gutter py-4 transition-all group bg-primary/10 text-primary border-l-4 border-primary font-medium w-full";
        } else {
            link.className = "nav-item flex items-center px-gutter py-4 transition-all group text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface border-l-4 border-transparent font-medium w-full";
        }
    });

    // 4. View-specific initialization
    if (viewId === "view-analytics") {
        loadAnalyticsData();
    } else if (viewId === "view-results" && payload) {
        renderResultsView(payload);
    }
}

// ─── Viewport Zoom & Pan Control Engine ─────────────────────────────────────
function updateViewportTransform() {
    const img = document.getElementById("console-viewport-img");
    const canvas = document.getElementById("console-heatmap-canvas");
    
    if (img) {
        img.style.transform = `scale(${state.zoom}) translate(${state.panX}px, ${state.panY}px)`;
        img.style.transition = "transform 0.15s ease-out";
    }
    if (canvas) {
        canvas.style.transform = `scale(${state.zoom}) translate(${state.panX}px, ${state.panY}px)`;
        canvas.style.transition = "transform 0.15s ease-out";
    }
    
    const resTag = document.getElementById("image-resolution-tag");
    if (resTag) {
        const dim = Math.round(224 * state.zoom);
        resTag.textContent = `${dim} x ${dim} px (${Math.round(state.zoom * 100)}%)`;
    }
}

function zoomIn() {
    state.zoom = Math.min(state.zoom + 0.25, 3.0);
    updateViewportTransform();
    toast(`Viewport Zoom: ${Math.round(state.zoom * 100)}%`, "info");
}

function zoomOut() {
    state.zoom = Math.max(state.zoom - 0.25, 0.5);
    updateViewportTransform();
    toast(`Viewport Zoom: ${Math.round(state.zoom * 100)}%`, "info");
}

function resetViewportTransform() {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    updateViewportTransform();
    toast("Viewport Zoom & Pan Reset", "info");
}

// ─── Grad-CAM AI Activation Heatmap Engine ──────────────────────────────────
function renderGradCamHeatmap(base64Data) {
    const canvas = document.getElementById("console-heatmap-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Only ever draw attribution the model actually produced. With no data the
    // canvas stays empty and the overlay is reported as unavailable.
    if (!base64Data) return;

    const img = new Image();
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = base64Data;
}

function toggleGradCamHeatmap() {
    if (!state.heatmapBase64) {
        toast("No Grad-CAM attribution is available for this scan.", "warning");
        return;
    }
    state.heatmapVisible = !state.heatmapVisible;
    const canvas = document.getElementById("console-heatmap-canvas");
    const toggleBtn = document.getElementById("btn-toggle-heatmap");

    if (canvas) {
        if (state.heatmapVisible) {
            renderGradCamHeatmap(state.heatmapBase64);
            canvas.classList.remove("hidden");
            if (toggleBtn) {
                toggleBtn.className = "flex items-center gap-1.5 px-3 py-1 bg-primary/20 text-primary rounded border border-primary/50 transition-all font-data-sm text-data-sm shadow-[0_0_12px_rgba(0,212,255,0.4)] font-bold";
            }
            toast("Grad-CAM AI Activation Heatmap Enabled", "info");
        } else {
            canvas.classList.add("hidden");
            if (toggleBtn) {
                toggleBtn.className = "flex items-center gap-1.5 px-3 py-1 bg-surface-container hover:bg-surface-container-highest text-on-surface-variant hover:text-primary rounded border border-outline-variant/30 transition-all font-data-sm text-data-sm";
            }
            toast("Grad-CAM AI Activation Heatmap Disabled", "info");
        }
    }
}

// ─── Image Calibration Logic ────────────────────────────────────────────────
function updateCalibrationFilters() {
    const img = document.getElementById("console-viewport-img");
    if (!img) return;

    const { brightness, contrast, saturation } = state.calibration;
    img.style.filter = `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturation}%)`;

    // Update labels (support both val- and slider--val IDs for resilience)
    const bVal = document.getElementById("val-brightness");
    const cVal = document.getElementById("val-contrast");
    const sVal = document.getElementById("val-saturation");

    if (bVal) bVal.textContent = `${brightness}%`;
    if (cVal) cVal.textContent = `${contrast}%`;
    if (sVal) sVal.textContent = `${saturation}%`;
}

function resetCalibration() {
    state.calibration = { brightness: 100, contrast: 100, saturation: 100 };
    const bSlider = document.getElementById("slider-brightness");
    const cSlider = document.getElementById("slider-contrast");
    const sSlider = document.getElementById("slider-saturation");

    if (bSlider) bSlider.value = 100;
    if (cSlider) cSlider.value = 100;
    if (sSlider) sSlider.value = 100;

    updateCalibrationFilters();
    resetViewportTransform();
    toast("Image calibration parameters & viewport reset to baseline defaults", "success");
}

// ─── Image Loading & Preview Handlers ───────────────────────────────────────
function handleFileSelected(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
        toast("Please upload a valid image file (JPEG, PNG, WEBP).", "error");
        return;
    }

    state.selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const imageUrl = e.target.result;
        setConsoleViewportImage(imageUrl);
    };
    reader.readAsDataURL(file);
}

function setConsoleViewportImage(src) {
    const viewportPlaceholder = document.getElementById("console-viewport-placeholder");
    const viewportImg = document.getElementById("console-viewport-img");
    const analyzeBtn = document.getElementById("btn-run-analysis");

    if (viewportImg && viewportPlaceholder) {
        viewportImg.src = src;
        viewportImg.classList.remove("hidden");
        viewportPlaceholder.classList.add("hidden");
    }

    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.className = "flex items-center gap-2 px-6 py-2 bg-primary text-on-primary font-headline-md text-headline-md rounded-lg shadow-[0_0_20px_rgba(0,212,255,0.4)] hover:shadow-[0_0_30px_rgba(0,212,255,0.6)] transition-all transform hover:-translate-y-0.5 cursor-pointer";
        analyzeBtn.innerHTML = `<span class="material-symbols-outlined">analytics</span> Begin AI Analysis`;
    }

    resetCalibration();
}

// ─── Clinical Sample Library Engine ─────────────────────────────────────────
// Reference cases served from the repository's own `samples/` directory
// (HAM10000 dermoscopic images, 600x450), one per diagnostic class. Ground truth
// and body site come from the HAM10000 metadata; every case is drawn from the
// held-out test split of the partition the deployed model was trained under
// (models/splits/split_test.csv), so none of them was a training image.
//
// Regenerate the files with:
//   python scripts/build_test_samples.py --manifest models/splits/split_test.csv \
//          --image-root path/to/ham10000
// and update this list to match. Do not add an entry whose diagnosis you cannot
// source from the manifest.
const CLINICAL_SAMPLES = [
    {
        id: "sample-mel",
        name: "Melanoma (MEL)",
        site: "Back",
        category: "Malignant",
        desc: "Histopathology-confirmed melanoma, HAM10000 held-out test case",
        url: "/samples/mel_1_ISIC_0026120.jpg"
    },
    {
        id: "sample-bcc",
        name: "Basal Cell Carcinoma (BCC)",
        site: "Trunk",
        category: "Malignant",
        desc: "Histopathology-confirmed basal cell carcinoma",
        url: "/samples/bcc_1_ISIC_0029230.jpg"
    },
    {
        id: "sample-akiec",
        name: "Actinic Keratosis (AKIEC)",
        site: "Face",
        category: "Pre-Malignant",
        desc: "Histopathology-confirmed actinic keratosis / intraepithelial carcinoma",
        url: "/samples/akiec_1_ISIC_0029659.jpg"
    },
    {
        id: "sample-nv",
        name: "Melanocytic Nevus (NV)",
        site: "Abdomen",
        category: "Benign",
        desc: "Benign nevomelanocytic mole, confirmed by clinical follow-up",
        url: "/samples/nv_1_ISIC_0032285.jpg"
    },
    {
        id: "sample-bkl",
        name: "Benign Keratosis (BKL)",
        site: "Upper Extremity",
        category: "Benign",
        desc: "Histopathology-confirmed benign keratosis-like lesion",
        url: "/samples/bkl_1_ISIC_0025915.jpg"
    },
    {
        id: "sample-df",
        name: "Dermatofibroma (DF)",
        site: "Lower Extremity",
        category: "Benign",
        desc: "Histopathology-confirmed dermatofibroma",
        url: "/samples/df_1_ISIC_0029760.jpg"
    },
    {
        id: "sample-vasc",
        name: "Vascular Lesion (VASC)",
        site: "Back",
        category: "Benign",
        desc: "Histopathology-confirmed vascular lesion",
        url: "/samples/vasc_1_ISIC_0029486.jpg"
    },
    {
        id: "sample-ood",
        name: "Non-Skin Control",
        site: "Not applicable",
        category: "Control",
        desc: "Non-clinical image. Rejected only once the feature-space OOD stage is calibrated",
        url: "/samples/cat.jpg"
    }
];

function renderSampleGallery() {
    const container = document.getElementById("sample-gallery-list");
    if (!container) return;

    container.innerHTML = CLINICAL_SAMPLES.map(s => {
        const badgeColor = s.category === 'Malignant' ? 'text-rose-400 bg-rose-500/10 border-rose-500/20' : s.category === 'Pre-Malignant' ? 'text-amber-400 bg-amber-500/10 border-amber-500/20' : 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
        return `
            <div data-sample-id="${s.id}" class="sample-card p-3 rounded-lg bg-surface-container hover:bg-surface-container-high border border-outline-variant/20 hover:border-primary/40 cursor-pointer transition-all flex items-center justify-between group">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-md overflow-hidden bg-surface-dim border border-outline-variant/30 flex-shrink-0">
                        <img src="${s.url}" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" alt="${s.name}" />
                    </div>
                    <div>
                        <h4 class="text-xs font-bold text-on-surface group-hover:text-primary transition-colors">${s.name}</h4>
                        <p class="text-[10px] text-on-surface-variant flex items-center gap-1 mt-0.5">
                            <span>${s.site}</span> • 
                            <span class="px-1.5 py-0.2 border rounded text-[9px] ${badgeColor}">${s.category}</span>
                        </p>
                    </div>
                </div>
                <span class="material-symbols-outlined text-on-surface-variant group-hover:text-primary text-[18px]">chevron_right</span>
            </div>
        `;
    }).join('');

    container.querySelectorAll(".sample-card").forEach(card => {
        card.addEventListener("click", () => {
            const sampleId = card.getAttribute("data-sample-id");
            const sample = CLINICAL_SAMPLES.find(s => s.id === sampleId);
            if (sample) loadSampleImage(sample);
        });
    });
}

// The sample cards show each case's real HAM10000 `localization` value ("Back",
// "Trunk", "Face"...). The API only accepts the six sites the tagger offers and
// silently discards anything else, so a sample previously recorded no site at all
// and cleared every tag button. This maps the dataset vocabulary onto the API's.
const HAM_SITE_TO_API_SITE = {
    "Back": "Posterior Torso",
    "Trunk": "Anterior Torso",
    "Abdomen": "Anterior Torso",
    "Chest": "Anterior Torso",
    "Face": "Head & Neck",
    "Scalp": "Head & Neck",
    "Neck": "Head & Neck",
    "Ear": "Head & Neck",
    "Upper Extremity": "Upper Extremities",
    "Hand": "Upper Extremities",
    "Lower Extremity": "Lower Extremities",
    "Foot": "Lower Extremities",
    "Acral": "Palms & Soles"
};

function toApiSite(localization) {
    if (!localization) return null;
    if (ANATOMIC_SITES.includes(localization)) return localization;
    return HAM_SITE_TO_API_SITE[localization] || null;
}

// Mirrors ANATOMIC_SITES in backend/config.py. Anything outside this list is
// discarded server-side, so the client must not invent values.
const ANATOMIC_SITES = [
    "Head & Neck", "Anterior Torso", "Posterior Torso",
    "Upper Extremities", "Lower Extremities", "Palms & Soles"
];

function loadSampleImage(sampleInput) {
    let sample = sampleInput;
    if (typeof sampleInput === "string") {
        sample = CLINICAL_SAMPLES.find(s => s.id === sampleInput || s.url === sampleInput) || {
            name: "Clinical Case",
            site: "Anterior Torso",
            url: sampleInput
        };
    }

    state.selectedFile = null;
    setConsoleViewportImage(sample.url);

    // Synchronize anatomic site selection UI & state. The dataset's own wording
    // is translated to the six values the API accepts; an unmappable site (the
    // non-skin control) leaves the current selection alone rather than clearing it.
    const apiSite = toApiSite(sample.site);
    if (apiSite) {
        state.selectedAnatomicSite = apiSite;
        document.querySelectorAll(".anatomic-tag").forEach(b => {
            if (b.getAttribute("data-site") === apiSite) {
                b.className = "anatomic-tag active px-3 py-2 bg-primary/10 border border-primary/30 text-primary rounded-lg font-data-sm text-data-sm hover:bg-primary/20 transition-all text-left font-bold";
            } else {
                b.className = "anatomic-tag px-3 py-2 bg-surface-container border border-outline-variant/20 text-on-surface-variant rounded-lg font-data-sm text-data-sm hover:bg-surface-container-high transition-all text-left";
            }
        });
    }

    const statusTag = document.getElementById("image-status-tag");
    if (statusTag) statusTag.textContent = `SOURCE: SAMPLE (${(sample.name || 'CASE').toUpperCase()})`;

    toast(`Loaded clinical sample: ${sample.name || 'Case'}`, "info");
}

// ─── Diagnostic API Analysis Trigger ────────────────────────────────────────
async function runAnalysis() {
    const analyzeBtn = document.getElementById("btn-run-analysis");
    if (!analyzeBtn || analyzeBtn.disabled) return;

    const viewportImg = document.getElementById("console-viewport-img");
    if (!state.selectedFile && (!viewportImg || !viewportImg.src)) {
        toast("Please upload or select an image scan first.", "warning");
        return;
    }

    // Lock button & show loading indicator
    analyzeBtn.disabled = true;
    analyzeBtn.className = "flex items-center gap-2 px-6 py-2 bg-surface-container border border-primary/40 text-primary font-headline-md text-headline-md rounded-lg cursor-wait animate-pulse";
    analyzeBtn.innerHTML = `<span class="material-symbols-outlined animate-spin text-lg">progress_activity</span> Computing Analysis...`;

    try {
        const formData = new FormData();
        
        if (state.selectedFile) {
            formData.append("file", state.selectedFile);
        } else {
            // Fetch the selected reference image. If it cannot be retrieved we
            // abort — analysing a stand-in image would report a diagnosis for
            // something the user never submitted.
            const res = await fetch(viewportImg.src);
            if (!res.ok) {
                throw new Error("Could not load the selected image. Please re-select or upload it.");
            }
            const blob = await res.blob();
            formData.append("file", blob, "lesion_scan.jpg");
        }
        
        formData.append("site", state.selectedAnatomicSite);

        const result = await apiCall("/analyze", {
            method: "POST",
            body: formData
        });

        // Store image preview URL in result
        result._previewUrl = viewportImg ? viewportImg.src : "";

        if (result.heatmap_base64) {
            state.heatmapBase64 = result.heatmap_base64;
            renderGradCamHeatmap(state.heatmapBase64);
            toast("Diagnostic inference & Grad-CAM map generated!", "success");
        } else {
            state.heatmapBase64 = null;
            toast("Inference complete. Grad-CAM attribution unavailable for this scan.", "warning");
        }

        state.latestResult = result;

        // Transition to Diagnostic Results view
        navigate("view-results", result);

    } catch (err) {
        toast(err.message, "error");
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.className = "flex items-center gap-2 px-6 py-2 bg-primary text-on-primary font-headline-md text-headline-md rounded-lg shadow-[0_0_20px_rgba(0,212,255,0.4)] hover:shadow-[0_0_30px_rgba(0,212,255,0.6)] transition-all transform hover:-translate-y-0.5 cursor-pointer";
        analyzeBtn.innerHTML = `<span class="material-symbols-outlined">analytics</span> Begin AI Analysis`;
    }
}

// ─── Render Diagnostic Results View ─────────────────────────────────────────
function renderResultsView(data) {
    if (!data) return;

    const isHighRisk = data.is_high_risk;
    const confidencePct = (data.confidence * 100).toFixed(1);
    const meta = PATHOLOGY_META[data.prediction] || {
        name: data.prediction.toUpperCase(),
        code: "ICD-11: N/A",
        type: isHighRisk ? "High Risk" : "Low Risk",
        desc: "Dermoscopic diagnostic result computed by EfficientNet-B3 neural network."
    };

    // 1. Update Alert Banner
    const banner = document.getElementById("results-alert-banner");
    const bannerIcon = document.getElementById("results-alert-icon");
    const bannerText = document.getElementById("results-alert-text");

    if (banner && bannerIcon && bannerText) {
        // Precedence: malignant prediction, then an unresolved melanoma alert,
        // then benign. The alert must outrank the benign banner - reassuring the
        // user while p(melanoma) is high is the dangerous combination.
        if (!isHighRisk && data.melanoma_alert) {
            banner.className = "mb-6 px-6 py-4 rounded-xl border bg-amber-500/10 border-amber-500/40 flex items-center justify-between";
            bannerIcon.className = "material-symbols-outlined text-amber-400 text-2xl animate-pulse";
            bannerIcon.textContent = "release_alert";
            bannerText.className = "font-mono text-xs font-bold uppercase tracking-wider text-amber-400";
            bannerText.textContent =
                `MELANOMA NOT EXCLUDED — p(mel) ${(data.melanoma_probability * 100).toFixed(1)}% — SPECIALIST REVIEW RECOMMENDED`;
        } else if (isHighRisk) {
            banner.className = "mb-6 px-6 py-4 rounded-xl border bg-rose-500/10 border-rose-500/30 flex items-center justify-between";
            bannerIcon.className = "material-symbols-outlined text-rose-400 text-2xl animate-pulse";
            bannerIcon.textContent = "warning";
            bannerText.className = "font-mono text-xs font-bold uppercase tracking-wider text-rose-400";
            bannerText.textContent = "CRITICAL PATHOLOGY ALERT — HIGH RISK MALIGNANT MORPHOLOGY DETECTED";
        } else {
            banner.className = "mb-6 px-6 py-4 rounded-xl border bg-emerald-500/10 border-emerald-500/30 flex items-center justify-between";
            bannerIcon.className = "material-symbols-outlined text-emerald-400 text-2xl";
            bannerIcon.textContent = "check_circle";
            bannerText.className = "font-mono text-xs font-bold uppercase tracking-wider text-emerald-400";
            bannerText.textContent = "NO MALIGNANT PATTERN IDENTIFIED — THIS IS NOT A CLINICAL CLEARANCE";
        }
    }

    // 2. Radial Gauge Animation (SVG stroke-dashoffset: 264 - (264 * confidence))
    const gaugeValue = document.getElementById("gauge-confidence-val");
    const gaugeCircle = document.getElementById("gauge-confidence-circle");
    const gaugeStatus = document.getElementById("gauge-risk-status");

    const dashOffset = 264 - (264 * data.confidence);

    if (gaugeValue) gaugeValue.textContent = `${confidencePct}%`;
    if (gaugeCircle) {
        gaugeCircle.style.strokeDashoffset = dashOffset;
        gaugeCircle.style.stroke = isHighRisk ? "#f43f5e" : "#00d4ff";
    }
    if (gaugeStatus) {
        gaugeStatus.textContent = isHighRisk ? "HIGH RISK" : "BENIGN";
        gaugeStatus.className = `font-mono text-xs font-bold uppercase ${isHighRisk ? 'text-rose-400' : 'text-cyan-400'}`;
    }

    // 3. Pathology Header Card
    const pathologyTitle = document.getElementById("results-pathology-title");
    const pathologyCode = document.getElementById("results-pathology-code");
    const pathologyType = document.getElementById("results-pathology-type");
    const pathologyDesc = document.getElementById("results-pathology-desc");

    if (pathologyTitle) pathologyTitle.textContent = meta.name;
    if (pathologyCode) pathologyCode.textContent = meta.code;
    if (pathologyType) {
        pathologyType.textContent = meta.type;
        pathologyType.className = `px-3 py-1 rounded-lg text-xs font-mono font-bold uppercase border ${isHighRisk ? 'bg-rose-500/15 border-rose-500/40 text-rose-400' : 'bg-cyan-500/15 border-cyan-500/40 text-cyan-400'}`;
    }
    if (pathologyDesc) pathologyDesc.textContent = meta.desc;

    // 4. 7-Class Probability Matrix
    const matrixContainer = document.getElementById("results-probability-matrix");
    if (matrixContainer && data.scores) {
        const sortedScores = Object.entries(data.scores).sort((a, b) => b[1] - a[1]);
        
        matrixContainer.innerHTML = sortedScores.map(([clsKey, prob]) => {
            const clsMeta = PATHOLOGY_META[clsKey] || { name: clsKey.toUpperCase() };
            const pct = (prob * 100).toFixed(1);
            const isTop = clsKey === data.prediction;
            const barWidth = Math.max(pct, 1.5);

            const barColor = isTop ? (isHighRisk ? 'bg-rose-500' : 'bg-cyan-400') : 'bg-slate-700';
            const labelColor = isTop ? (isHighRisk ? 'text-rose-400 font-bold' : 'text-cyan-400 font-bold') : 'text-slate-400';

            return `
                <div>
                    <div class="flex justify-between text-xs font-mono mb-1.5">
                        <span class="${labelColor}">${clsMeta.name} (${clsKey.toUpperCase()})</span>
                        <span class="${labelColor}">${pct}%</span>
                    </div>
                    <div class="h-2 bg-surface-base rounded-full overflow-hidden border border-slate-800">
                        <div class="${barColor} h-full rounded-full transition-all duration-700" style="width: ${barWidth}%"></div>
                    </div>
                </div>
            `;
        }).join('');
    }

    // 5. Clinical summary — derived from this result, never a fixed string.
    //    A hardcoded "immediate biopsy recommended" previously appeared on every
    //    result, including benign ones, directly contradicting the banner above it.
    const summaryEl = document.getElementById("res-clinical-summary");
    if (summaryEl) summaryEl.textContent = buildClinicalSummary(data, meta);

    // 6. OOD gate — the statistics the gate actually measured on this image, and
    //    the honest state of its calibration. Every value here comes from the API;
    //    if the API did not send it, nothing is displayed.
    renderOodPanel(data);
}

// Wording is derived from the predicted class and the melanoma alert. It states
// what the system found and what that does not rule out; it does not issue
// treatment instructions.
function buildClinicalSummary(data, meta) {
    const pct = (data.confidence * 100).toFixed(1);
    const name = meta.name;
    const parts = [];

    if (data.is_high_risk) {
        parts.push(
            `The model's leading classification is ${name} at ${pct}% confidence, ` +
            `which falls in the group flagged as high risk (melanoma, basal cell carcinoma, ` +
            `actinic keratosis). Specialist dermatological assessment is indicated.`);
    } else {
        parts.push(
            `The model's leading classification is ${name} at ${pct}% confidence, ` +
            `a class not flagged as high risk.`);
    }

    if (data.melanoma_alert) {
        const mp = data.melanoma_probability != null
            ? ` (p(mel) ${(data.melanoma_probability * 100).toFixed(1)}%)` : "";
        parts.push(
            `The independent melanoma alert channel fired for this scan${mp}: melanoma is ` +
            `not excluded regardless of the leading class, and the scan is flagged for review.`);
    }

    parts.push(
        `Measured melanoma recall for this model is 0.800 on a lesion-disjoint hold-out — ` +
        `about one melanoma in five is missed by the prediction itself. A result that is not ` +
        `flagged is not a clinical clearance.`);

    return parts.join(" ");
}

function renderOodPanel(data) {
    const badge = document.getElementById("ood-status-badge");
    const statusText = document.getElementById("ood-status-text");
    const note = document.getElementById("ood-calibration-note");
    const cells = {
        "ood-rel-contrast": "rel_contrast",
        "ood-hf-ratio": "hf_ratio",
        "ood-blue-green": "blue_green"
    };

    const m = data.ood_metrics;
    for (const [elId, key] of Object.entries(cells)) {
        const el = document.getElementById(elId);
        if (!el) continue;
        el.textContent = (m && typeof m[key] === "number") ? m[key].toFixed(3) : "—";
    }

    // The scan reached this view, so stage 1 accepted it. That is all that can be
    // claimed — and only stage 1 ran unless the feature-space stage is fitted.
    if (badge) {
        badge.textContent = m ? "STAGE 1 PASSED" : "NOT REPORTED";
        badge.className = "px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase " +
            (m ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
               : "bg-surface-container text-on-surface-variant border border-outline-variant/40");
    }
    if (statusText) {
        statusText.textContent = m
            ? "Image statistics fell inside the accepted range, so the colour gate did not reject this scan. " +
              "The colour gate rejects flat fields, pixel noise and non-skin hues; it cannot reject a " +
              "photograph of another real object."
            : "The server did not report out-of-distribution metrics for this scan.";
    }
    if (note) {
        const bits = [];
        bits.push(data.ood_calibrated
            ? "Gate thresholds: fitted to data."
            : "Gate thresholds: provisional defaults, not yet fitted (scripts/calibrate_ood.py).");
        bits.push(data.ood_feature_stage_active
            ? "Feature-space stage: active."
            : "Feature-space stage: not fitted, did not run.");
        note.textContent = bits.join(" ");
        note.className = "text-[10px] font-data-sm " +
            ((data.ood_calibrated && data.ood_feature_stage_active)
                ? "text-on-surface-variant" : "text-amber-400");
    }
}

// ─── Clinical PDF Report Generator ─────────────────────────────────────────
function downloadClinicalReport() {
    const reportElement = document.getElementById("view-results");
    if (!reportElement) {
        toast("No diagnostic results available to export.", "warning");
        return;
    }

    // The export button lives in the always-visible top bar, so it can be pressed
    // with no scan run and the results panel hidden. html2pdf would then render a
    // blank or placeholder page and present it as a clinical report.
    if (!state.latestResult) {
        toast("Run a scan before exporting a report.", "warning");
        return;
    }
    if (reportElement.classList.contains("hidden")) {
        navigate("view-results", state.latestResult);
    }

    toast("Generating Clinical Consultation PDF Report...", "info");

    const opt = {
        margin:       0.5,
        filename:     `DermaScan_Report_${state.latestResult?.prediction || 'Scan'}_${new Date().toISOString().slice(0,10)}.pdf`,
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#090d16' },
        jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    if (window.html2pdf) {
        window.html2pdf().set(opt).from(reportElement).save()
            .then(() => toast("Report downloaded successfully!", "success"))
            .catch(err => toast("PDF generation error: " + err.message, "error"));
    } else {
        window.print();
    }
}

// ─── Analytics & History View Handler ───────────────────────────────────────
async function loadAnalyticsData() {
    const historyTableBody = document.getElementById("analytics-history-tbody");
    if (!historyTableBody) return;

    historyTableBody.innerHTML = `
        <tr>
            <td colspan="6" class="py-12 text-center text-slate-500 font-mono text-xs">
                <span class="material-symbols-outlined animate-spin text-cyan-400 text-2xl mb-2">progress_activity</span>
                <p>Loading historical diagnostic records...</p>
            </td>
        </tr>
    `;

    try {
        const logs = await apiCall("/history");
        state.history = logs;

        // Calculate KPI summaries
        const totalScans = logs.length;
        const highRiskCount = logs.filter(l => l.is_high_risk).length;
        const lowRiskCount = totalScans - highRiskCount;
        const avgConfidence = totalScans > 0 ? (logs.reduce((acc, l) => acc + l.confidence, 0) / totalScans * 100).toFixed(1) : 0;

        const kpiTotal = document.getElementById("kpi-total-scans");
        const kpiHigh = document.getElementById("kpi-high-risk");
        const kpiLow = document.getElementById("kpi-low-risk");
        const kpiAvgConf = document.getElementById("kpi-avg-confidence");

        if (kpiTotal) kpiTotal.textContent = totalScans;
        if (kpiHigh) kpiHigh.textContent = highRiskCount;
        if (kpiLow) kpiLow.textContent = lowRiskCount;
        if (kpiAvgConf) kpiAvgConf.textContent = `${avgConfidence}%`;

        renderHistoryTable(logs);

    } catch (err) {
        toast("Failed to load historical analytics: " + err.message, "error");
    }
}

function renderHistoryTable(records) {
    const tbody = document.getElementById("analytics-history-tbody");
    if (!tbody) return;

    if (!records.length) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="py-12 text-center text-slate-500 font-mono text-xs">
                    <span class="material-symbols-outlined text-3xl mb-2 text-slate-600">inbox</span>
                    <p>No historical scans found in SQLite database.</p>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = records.map(log => {
        // The API sends UTC with an explicit offset; render both halves in the
        // viewer's local zone. Previously the date came from toISOString() (UTC)
        // and the time from toTimeString() (local), so around midnight a row
        // could show one day's date beside the next day's time.
        const dt = new Date(log.created_at);
        const pad = (n) => String(n).padStart(2, "0");
        const dateStr = `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}`;
        const timeStr = `${pad(dt.getHours())}:${pad(dt.getMinutes())}`;
        const meta = PATHOLOGY_META[log.prediction] || { name: log.prediction.toUpperCase() };
        const confPct = (log.confidence * 100).toFixed(1);
        const isHigh = log.is_high_risk;

        return `
            <tr class="border-b border-surface-elevated hover:bg-surface-elevated/40 transition-colors">
                <td class="py-4 px-4 font-mono text-xs text-slate-400">#${log.id}</td>
                <td class="py-4 px-4 font-mono text-xs text-slate-300">${dateStr} <span class="text-slate-500">${timeStr}</span></td>
                <td class="py-4 px-4 font-mono text-xs font-bold text-slate-200">${meta.name} <span class="text-cyan-400">(${log.prediction.toUpperCase()})</span></td>
                <td class="py-4 px-4 font-mono text-xs font-bold ${isHigh ? 'text-rose-400' : 'text-cyan-400'}">${confPct}%</td>
                <td class="py-4 px-4">
                    ${isHigh
                        ? `<span class="px-2.5 py-1 bg-rose-500/15 border border-rose-500/30 text-rose-400 font-mono text-[10px] font-bold uppercase rounded-md">HIGH RISK</span>`
                        : `<span class="px-2.5 py-1 bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-mono text-[10px] font-bold uppercase rounded-md">BENIGN</span>`
                    }
                </td>
                <td class="py-4 px-4 text-right">
                    <button onclick="deleteHistoryRecord(${log.id})" class="p-1.5 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors" title="Delete Log">
                        <span class="material-symbols-outlined text-base">delete</span>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}


// ─── Admin Token (required by the destructive endpoints) ─────────────────────
const ADMIN_TOKEN_KEY = "dermascan.adminToken";

function getStoredAdminToken() {
    try {
        return window.localStorage.getItem(ADMIN_TOKEN_KEY) || "";
    } catch (_) {
        return "";   // private mode / storage blocked
    }
}

function setStoredAdminToken(value) {
    try {
        if (value) window.localStorage.setItem(ADMIN_TOKEN_KEY, value);
        else window.localStorage.removeItem(ADMIN_TOKEN_KEY);
    } catch (_) { /* non-fatal: the token simply won't persist */ }
}

function requestAdminToken() {
    const existing = getStoredAdminToken();
    if (existing) return existing;
    const entered = window.prompt(
        "Deleting a record requires the admin token.\n\n" +
        "This is the ADMIN_TOKEN value set in the server's .env file. " +
        "It is stored in this browser only."
    );
    const token = (entered || "").trim();
    if (token) setStoredAdminToken(token);
    return token;
}

async function deleteHistoryRecord(id) {
    if (!confirm(`Are you sure you want to delete scan log #${id}?`)) return;

    const token = requestAdminToken();
    if (!token) {
        toast("Delete cancelled: no admin token provided.", "warning");
        return;
    }

    try {
        await apiCall(`/history/${id}`, {
            method: "DELETE",
            headers: { "X-Admin-Token": token }
        });
        toast(`Scan log #${id} deleted.`, "success");
        loadAnalyticsData();
    } catch (err) {
        if (err.status === 401) {
            // Wrong token: forget it so the next attempt asks again.
            setStoredAdminToken("");
            toast("Admin token rejected. Check ADMIN_TOKEN and try again.", "error");
        } else if (err.status === 403) {
            toast("Deletion is disabled: the server has no ADMIN_TOKEN configured.", "warning");
        } else {
            toast(err.message, "error");
        }
    }
}

// ─── CSV Export Handler ──────────────────────────────────────────────────────
function exportHistoryCSV() {
    if (!state.history || !state.history.length) {
        toast("No diagnostic scan records available to export.", "warning");
        return;
    }

    const headers = ["ID", "Timestamp", "Prediction", "Pathology_Name", "Confidence", "Is_High_Risk", "Anatomic_Site"];
    const rows = state.history.map(item => {
        const meta = PATHOLOGY_META[item.prediction] || { name: item.prediction };
        return [
            item.id,
            `"${item.created_at}"`,
            `"${item.prediction}"`,
            `"${meta.name}"`,
            (item.confidence * 100).toFixed(2) + "%",
            item.is_high_risk ? "Yes" : "No",
            `"${item.anatomic_site || ""}"`
        ].join(",");
    });

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `DermaScan_Diagnostic_History_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast("CSV history report exported successfully!", "success");
}

// ─── Setup Event Listeners ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    // 1. Sidebar & Inter-View SPA Router
    document.querySelectorAll("[data-view]").forEach(elem => {
        elem.addEventListener("click", (e) => {
            e.preventDefault();
            const viewId = elem.getAttribute("data-view");
            if (viewId) navigate(viewId);
        });
    });

    // 2. Viewport Zoom & Pan Controls
    const btnZoomIn = document.getElementById("btn-zoom-in");
    const btnZoomOut = document.getElementById("btn-zoom-out");
    const btnPanReset = document.getElementById("btn-pan-reset");

    if (btnZoomIn) btnZoomIn.addEventListener("click", zoomIn);
    if (btnZoomOut) btnZoomOut.addEventListener("click", zoomOut);
    if (btnPanReset) btnPanReset.addEventListener("click", resetViewportTransform);

    // 3. Populate Clinical Sample Gallery
    renderSampleGallery();

    // 4. Anatomic Site Tagger Buttons
    document.querySelectorAll(".anatomic-tag").forEach(tagBtn => {
        tagBtn.addEventListener("click", () => {
            document.querySelectorAll(".anatomic-tag").forEach(b => {
                b.className = "anatomic-tag px-3 py-2 bg-surface-container border border-outline-variant/20 text-on-surface-variant rounded-lg font-data-sm text-data-sm hover:bg-surface-container-high transition-all text-left";
            });
            tagBtn.className = "anatomic-tag active px-3 py-2 bg-primary/10 border border-primary/30 text-primary rounded-lg font-data-sm text-data-sm hover:bg-primary/20 transition-all text-left font-bold";
            state.selectedAnatomicSite = tagBtn.getAttribute("data-site");
            toast(`Anatomic site set to: ${state.selectedAnatomicSite}`, "info");
        });
    });

    // 5. Image Calibration Sliders & Viewport Toolbar
    const bSlider = document.getElementById("slider-brightness");
    const cSlider = document.getElementById("slider-contrast");
    const sSlider = document.getElementById("slider-saturation");
    const btnResetCal = document.getElementById("btn-reset-calibration");
    const btnZoomInEl = document.getElementById("btn-zoom-in");
    const btnZoomOutEl = document.getElementById("btn-zoom-out");
    const btnPanResetEl = document.getElementById("btn-pan-reset");
    const btnToggleHeatmapEl = document.getElementById("btn-toggle-heatmap");

    if (bSlider) bSlider.addEventListener("input", (e) => { state.calibration.brightness = e.target.value; updateCalibrationFilters(); });
    if (cSlider) cSlider.addEventListener("input", (e) => { state.calibration.contrast = e.target.value; updateCalibrationFilters(); });
    if (sSlider) sSlider.addEventListener("input", (e) => { state.calibration.saturation = e.target.value; updateCalibrationFilters(); });
    if (btnResetCal) btnResetCal.addEventListener("click", resetCalibration);
    if (btnZoomInEl) btnZoomInEl.addEventListener("click", zoomIn);
    if (btnZoomOutEl) btnZoomOutEl.addEventListener("click", zoomOut);
    if (btnPanResetEl) btnPanResetEl.addEventListener("click", resetViewportTransform);
    if (btnToggleHeatmapEl) btnToggleHeatmapEl.addEventListener("click", toggleGradCamHeatmap);

    // 6. File Drag & Drop Handlers
    const dropZone = document.getElementById("console-drop-zone");
    const fileInput = document.getElementById("console-file-input");

    if (dropZone && fileInput) {
        dropZone.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) handleFileSelected(e.target.files[0]);
        });

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("border-primary", "bg-primary/5");
        });
        dropZone.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dropZone.classList.remove("border-primary", "bg-primary/5");
        });
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("border-primary", "bg-primary/5");
            if (e.dataTransfer.files.length > 0) handleFileSelected(e.dataTransfer.files[0]);
        });
    }

    // 7. Run Analysis Button
    const runBtn = document.getElementById("btn-run-analysis");
    if (runBtn) runBtn.addEventListener("click", runAnalysis);

    // 8. Clinical Report Export (Bind to all export buttons)
    document.querySelectorAll(".btn-export-pdf-trigger").forEach(btn => {
        btn.addEventListener("click", downloadClinicalReport);
    });

    // 9. Knowledge Hub Filtering
    document.querySelectorAll(".kh-filter-btn").forEach(filterBtn => {
        filterBtn.addEventListener("click", () => {
            const cat = filterBtn.getAttribute("data-cat");
            
            document.querySelectorAll(".kh-filter-btn").forEach(b => {
                b.className = "kh-filter-btn px-4 py-2 bg-surface-container text-on-surface-variant hover:text-on-surface font-data-sm text-data-sm rounded-lg transition-all cursor-pointer";
            });
            filterBtn.className = "kh-filter-btn px-4 py-2 bg-primary text-on-primary font-data-sm text-data-sm rounded-lg shadow-[0_0_15px_rgba(0,212,255,0.4)] transition-all cursor-pointer";

            const cards = document.querySelectorAll("#knowledge-cards-container .kh-card");
            cards.forEach(card => {
                const cardCat = card.getAttribute("data-category");
                if (cat === "all" || cardCat === cat) {
                    card.classList.remove("hidden");
                } else {
                    card.classList.add("hidden");
                }
            });
        });
    });

    // 10. History Table Search Filter
    const historySearchInput = document.getElementById("history-search-input");
    if (historySearchInput) {
        historySearchInput.addEventListener("input", (e) => {
            const q = e.target.value.toLowerCase().trim();
            if (!q) {
                renderHistoryTable(state.history);
                return;
            }
            const filtered = state.history.filter(item => {
                const meta = PATHOLOGY_META[item.prediction] || { name: item.prediction };
                return item.id.toString().includes(q) ||
                       item.prediction.toLowerCase().includes(q) ||
                       meta.name.toLowerCase().includes(q);
            });
            renderHistoryTable(filtered);
        });
    }

    // 11. CSV Export Button
    const btnExportCSV = document.getElementById("btn-export-csv");
    if (btnExportCSV) btnExportCSV.addEventListener("click", exportHistoryCSV);

    // 12. Compare Mode Dual Viewport Sync
    const syncToggle = document.getElementById("syncToggle");
    const syncThumb = document.getElementById("syncThumb");
    let syncActive = true;
    if (syncToggle && syncThumb) {
        syncToggle.addEventListener("click", () => {
            syncActive = !syncActive;
            if (syncActive) {
                syncThumb.style.transform = "translateX(0)";
                syncToggle.className = "w-8 h-4 bg-primary/40 rounded-full relative ml-2 transition-colors duration-300";
                toast("Viewport Sync Active", "info");
            } else {
                syncThumb.style.transform = "translateX(-16px)";
                syncToggle.className = "w-8 h-4 bg-surface-container-high rounded-full relative ml-2 transition-colors duration-300";
                toast("Viewport Sync Paused", "info");
            }
        });
    }

    // Compare Viewport Zoom Sync Engine
    let compareScale = 1.0;
    const imgLeft = document.getElementById("img-compare-left");
    const imgRight = document.getElementById("img-compare-right");
    const vLeft = document.getElementById("viewportLeft");
    const vRight = document.getElementById("viewportRight");

    function updateCompareZoom(delta) {
        compareScale = Math.max(0.5, Math.min(3.0, compareScale + delta));
        if (imgLeft) imgLeft.style.transform = `scale(${compareScale})`;
        if (syncActive && imgRight) imgRight.style.transform = `scale(${compareScale})`;
    }

    if (vLeft) {
        vLeft.addEventListener("wheel", (e) => {
            e.preventDefault();
            updateCompareZoom(e.deltaY < 0 ? 0.1 : -0.1);
        }, { passive: false });
    }

    if (vRight) {
        vRight.addEventListener("wheel", (e) => {
            e.preventDefault();
            if (syncActive) {
                updateCompareZoom(e.deltaY < 0 ? 0.1 : -0.1);
            } else if (imgRight) {
                let currentScale = parseFloat(imgRight.style.transform.replace(/scale\((.*?)\)/, '$1')) || 1.0;
                currentScale = Math.max(0.5, Math.min(3.0, currentScale + (e.deltaY < 0 ? 0.1 : -0.1)));
                imgRight.style.transform = `scale(${currentScale})`;
            }
        }, { passive: false });
    }

    // 13. Dynamic Compare Mode Uploads & Scan Sync
    function setCompareSlotImage(slot, imageSrc, labelText, dateText) {
        const isA = slot === 'A';
        const imgEl = document.getElementById(isA ? "img-compare-left" : "img-compare-right");
        const tagEl = document.getElementById(isA ? "tag-compare-a" : "tag-compare-b");
        const dateEl = document.getElementById(isA ? "compare-a-date" : "compare-b-date");
        const phEl = document.getElementById(isA ? "compare-a-placeholder" : "compare-b-placeholder");

        if (imgEl) {
            imgEl.src = imageSrc;
            imgEl.classList.remove("hidden");
        }
        if (phEl) phEl.classList.add("hidden");
        if (tagEl && labelText) {
            tagEl.textContent = labelText;
            tagEl.classList.remove("hidden");
        }
        if (dateEl) dateEl.textContent = dateText || new Date().toISOString().split('T')[0];

        // The label is the source of the image, not a diagnosis. Nothing here is
        // measured or compared: this view only places two scans side by side.
        toast(`Loaded image into Scan ${slot}`, "success");
    }

    function handleCompareFileUpload(slot, file) {
        if (!file || !file.type.startsWith("image/")) {
            toast("Please select a valid image file", "error");
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            setCompareSlotImage(slot, e.target.result, file.name.substring(0, 18), new Date().toISOString().split('T')[0]);
        };
        reader.readAsDataURL(file);
    }

    // Scan A Uploads & Drag-Drop
    const btnUploadA = document.getElementById("btn-upload-compare-a");
    const inputCompareA = document.getElementById("input-compare-a");
    const dropOverlayA = document.getElementById("drop-overlay-a");

    if (btnUploadA && inputCompareA) {
        btnUploadA.addEventListener("click", () => inputCompareA.click());
        inputCompareA.addEventListener("change", (e) => {
            if (e.target.files.length > 0) handleCompareFileUpload('A', e.target.files[0]);
        });
    }

    if (vLeft && dropOverlayA) {
        vLeft.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropOverlayA.classList.remove("hidden");
        });
        vLeft.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dropOverlayA.classList.add("hidden");
        });
        vLeft.addEventListener("drop", (e) => {
            e.preventDefault();
            dropOverlayA.classList.add("hidden");
            if (e.dataTransfer.files.length > 0) handleCompareFileUpload('A', e.dataTransfer.files[0]);
        });
    }

    // Scan B Uploads & Drag-Drop
    const btnUploadB = document.getElementById("btn-upload-compare-b");
    const inputCompareB = document.getElementById("input-compare-b");
    const dropOverlayB = document.getElementById("drop-overlay-b");
    const btnLoadCurrentB = document.getElementById("btn-load-current-to-b");

    if (btnUploadB && inputCompareB) {
        btnUploadB.addEventListener("click", () => inputCompareB.click());
        inputCompareB.addEventListener("change", (e) => {
            if (e.target.files.length > 0) handleCompareFileUpload('B', e.target.files[0]);
        });
    }

    if (vRight && dropOverlayB) {
        vRight.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropOverlayB.classList.remove("hidden");
        });
        vRight.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dropOverlayB.classList.add("hidden");
        });
        vRight.addEventListener("drop", (e) => {
            e.preventDefault();
            dropOverlayB.classList.add("hidden");
            if (e.dataTransfer.files.length > 0) handleCompareFileUpload('B', e.dataTransfer.files[0]);
        });
    }

    function syncActiveScanToB() {
        let imgSrc = null;
        let label = "ACTIVE SCAN";

        if (state.latestResult && state.latestResult.image_url) {
            imgSrc = state.latestResult.image_url;
            const topPred = state.latestResult.prediction;
            const meta = PATHOLOGY_META[topPred];
            label = meta ? `${topPred} (${meta.name})` : topPred;
        } else if (state.selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => setCompareSlotImage('B', e.target.result, state.selectedFile.name, "TODAY");
            reader.readAsDataURL(state.selectedFile);
            return;
        } else {
            const viewportImg = document.getElementById("console-viewport-img");
            if (viewportImg && viewportImg.src && !viewportImg.src.includes("placeholder")) {
                imgSrc = viewportImg.src;
            }
        }

        if (imgSrc) {
            setCompareSlotImage('B', imgSrc, label, "TODAY");
        } else {
            toast("No active scan found in Console to load", "warning");
        }
    }

    if (btnLoadCurrentB) {
        btnLoadCurrentB.addEventListener("click", syncActiveScanToB);
    }

    // Auto-sync active scan when navigating to Compare View if available
    const compareNavBtn = document.querySelector('[data-view="view-compare"]');
    if (compareNavBtn) {
        compareNavBtn.addEventListener("click", () => {
            if (state.latestResult || state.selectedFile) {
                setTimeout(syncActiveScanToB, 100);
            }
        });
    }

    // Initial view load
    navigate("view-console");
});

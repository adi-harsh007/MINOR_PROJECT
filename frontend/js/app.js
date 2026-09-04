/* ═══════════════════════════════════════════════════════════════════════════
   DermaScan — single-page client for the lesion classifier API.

   Presentation note: state that used to be applied by rewriting whole Tailwind
   class strings is now a single class toggle against the component styles in
   index.html (.nav-item.is-active, .seg.is-active, .btn.is-on, .chip--high…).
   That keeps hover and selected visually distinct and stops the markup and the
   script drifting apart.
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
    resultsHeatmapVisible: false,
    heatmapBase64: null,
    calibration: {
        brightness: 100,
        contrast: 100,
        saturation: 100
    },
    latestResult: null,
    history: [],
    // The two records the comparison view is showing, plus a slot id parked by
    // the results view on its way here.
    compare: { a: null, b: null, pendingA: null }
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
        // Below the 56px header, so notifications never sit on top of the
        // export button they are often reporting on.
        container.className = "fixed top-[68px] right-4 z-[60] flex flex-col gap-2 items-end";
        document.body.appendChild(container);
    }
    
    const colors = {
        info: "border-outline text-on-surface",
        success: "border-risk-low/50 text-risk-low",
        warning: "border-risk-mid/50 text-risk-mid",
        error: "border-risk-high/50 text-risk-high"
    };

    const el = document.createElement("div");
    el.className = "toast max-w-[22rem] px-4 py-3 rounded-xl border bg-surface-container-high " +
        "text-[13px] leading-5 shadow-lg flex items-start gap-2.5 transition-all duration-300 " +
        `transform translate-y-2 opacity-0 ${colors[type] || colors.info}`;
    el.innerHTML = `
        <span class="material-symbols-outlined text-[18px] shrink-0 mt-px">${type === 'error' ? 'error' : type === 'success' ? 'check_circle' : type === 'warning' ? 'warning' : 'info'}</span>
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

// ─── Mobile navigation drawer ───────────────────────────────────────────────
// Below the `lg` breakpoint the sidebar is translated off-canvas by CSS and
// only `.is-open` brings it back, so the 256px rail no longer eats two thirds
// of a phone screen.
function openNavDrawer() {
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("nav-backdrop");
    const opener = document.getElementById("btn-nav-open");
    if (sidebar) sidebar.classList.add("is-open");
    if (backdrop) backdrop.classList.remove("hidden");
    if (opener) opener.setAttribute("aria-expanded", "true");
}

function closeNavDrawer() {
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("nav-backdrop");
    const opener = document.getElementById("btn-nav-open");
    if (sidebar) sidebar.classList.remove("is-open");
    if (backdrop) backdrop.classList.add("hidden");
    if (opener) opener.setAttribute("aria-expanded", "false");
}

// ─── Anatomic site selection (shared by the tag buttons and the gallery) ─────
function setAnatomicSiteUI(site) {
    document.querySelectorAll(".anatomic-tag").forEach(b => {
        const isActive = b.getAttribute("data-site") === site;
        b.classList.toggle("is-active", isActive);
        b.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
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
        // Views are full-height panes, not anchors: scrollIntoView() on a pane
        // taller than the viewport left the page part-scrolled, so the header
        // and the first row of controls were cut off on arrival.
        window.scrollTo({ top: 0, behavior: "auto" });
    }

    // 3. Highlight sidebar links. `is-active` carries the whole selected style
    //    (rail, tint, filled icon) so it cannot be confused with plain hover.
    document.querySelectorAll(".nav-item").forEach(link => {
        const isActive = link.getAttribute("data-view") === viewId;
        link.classList.toggle("is-active", isActive);
        if (isActive) {
            link.setAttribute("aria-current", "page");
        } else {
            link.removeAttribute("aria-current");
        }
    });

    // 4. On small screens the sidebar is a drawer; a selection closes it.
    closeNavDrawer();

    // 5. View-specific initialization
    if (viewId === "view-analytics") {
        loadAnalyticsData();
    } else if (viewId === "view-compare") {
        loadCompareRecords();
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
// The attribution map is computed over the whole 224x224 model input, so it
// corresponds to the whole image and must cover exactly the pixels the image
// occupies. The canvas carries intrinsic 224x224 dimensions and only
// `max-h-full max-w-full`, which left it a 224px square floating in the middle
// of a 600x450 photo — the hot region pointed at the wrong part of the lesion.
// Measure the image's rendered content box (it is laid out with object-contain)
// and place the canvas on top of it.
function syncHeatmapCanvasToImage(
        containerId = "preview-container",
        imgId = "console-viewport-img",
        canvasId = "console-heatmap-canvas") {
    const container = document.getElementById(containerId);
    const img = document.getElementById(imgId);
    const canvas = document.getElementById(canvasId);
    if (!container || !img || !canvas) return;
    if (!img.naturalWidth || !img.naturalHeight) return;

    const box = img.getBoundingClientRect();
    const containerBox = container.getBoundingClientRect();
    if (!box.width || !box.height) return;

    // The <img> box is already capped to the container; object-contain then
    // letterboxes the bitmap inside that box.
    const scale = Math.min(box.width / img.naturalWidth, box.height / img.naturalHeight);
    const drawnW = img.naturalWidth * scale;
    const drawnH = img.naturalHeight * scale;

    canvas.style.left = `${(box.left - containerBox.left) + (box.width - drawnW) / 2}px`;
    canvas.style.top = `${(box.top - containerBox.top) + (box.height - drawnH) / 2}px`;
    canvas.style.width = `${drawnW}px`;
    canvas.style.height = `${drawnH}px`;
    canvas.style.maxWidth = "none";
    canvas.style.maxHeight = "none";
}

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
        syncHeatmapCanvasToImage();
    };
    img.src = base64Data;
}

// ─── The analysed image on the result panel ─────────────────────────────────
// The same attribution map as the console, over the stored image rather than
// whatever the console happens to be showing.
function syncResultsHeatmapCanvas() {
    syncHeatmapCanvasToImage("results-image-wrap", "results-image", "results-heatmap-canvas");
}

function drawResultsHeatmap() {
    const canvas = document.getElementById("results-heatmap-canvas");
    if (!canvas || !state.heatmapBase64) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const img = new Image();
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        syncResultsHeatmapCanvas();
    };
    img.src = state.heatmapBase64;
}

function setResultsHeatmapVisible(visible) {
    const canvas = document.getElementById("results-heatmap-canvas");
    const btn = document.getElementById("btn-results-heatmap");
    state.resultsHeatmapVisible = visible;
    if (canvas) canvas.classList.toggle("hidden", !visible);
    if (btn) {
        btn.classList.toggle("is-on", visible);
        btn.setAttribute("aria-pressed", visible ? "true" : "false");
    }
    if (visible) drawResultsHeatmap();
}

// The stored upload is fetched back from the record rather than reused from the
// console: it is the image the model was actually given, and it is unaffected by
// the console's display sliders or by loading a different sample afterwards.
function renderResultsImage(data) {
    const img = document.getElementById("results-image");
    const missing = document.getElementById("results-image-missing");
    const btn = document.getElementById("btn-results-heatmap");
    const note = document.getElementById("results-image-note");
    if (!img || !missing) return;

    setResultsHeatmapVisible(false);

    // Grad-CAM is offered only when the API actually returned attribution.
    if (btn) {
        const available = Boolean(state.heatmapBase64);
        btn.disabled = !available;
        btn.title = available
            ? "Overlay the model's Grad-CAM attribution map"
            : "No Grad-CAM attribution was returned for this scan";
    }
    if (note) {
        note.textContent = state.heatmapBase64
            ? "As submitted. Display adjustments in the console are not applied here."
            : "As submitted. No Grad-CAM attribution was returned for this scan.";
    }

    const showMissing = () => {
        img.classList.add("hidden");
        missing.classList.remove("hidden");
        missing.classList.add("flex");
    };

    if (!data.session_id) {
        showMissing();
        return;
    }

    img.onerror = showMissing;
    img.onload = () => {
        missing.classList.add("hidden");
        missing.classList.remove("flex");
        img.classList.remove("hidden");
        syncResultsHeatmapCanvas();
    };
    img.src = `${API_BASE}/history/${encodeURIComponent(data.session_id)}/image`;
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
            syncHeatmapCanvasToImage();
            if (toggleBtn) {
                toggleBtn.classList.add("is-on");
                toggleBtn.setAttribute("aria-pressed", "true");
            }
            toast("Grad-CAM overlay on", "info");
        } else {
            canvas.classList.add("hidden");
            if (toggleBtn) {
                toggleBtn.classList.remove("is-on");
                toggleBtn.setAttribute("aria-pressed", "false");
            }
            toast("Grad-CAM overlay off", "info");
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

    // The source chip kept naming the last reference sample after an upload
    // replaced it, so the toolbar described an image that was no longer shown.
    const statusTag = document.getElementById("image-status-tag");
    if (statusTag) statusTag.textContent = `Uploaded · ${file.name}`;

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
        setAnalyzeButtonIdle(analyzeBtn);
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
        const chipClass = s.category === 'Malignant' ? 'chip chip--high'
            : s.category === 'Pre-Malignant' ? 'chip chip--mid'
            : s.category === 'Benign' ? 'chip chip--low'
            : 'chip';
        return `
            <button type="button" data-sample-id="${s.id}"
                    class="sample-card w-full text-left p-2.5 rounded-lg bg-surface-container border border-outline-variant hover:border-outline hover:bg-surface-container-high cursor-pointer transition-colors flex items-center gap-3">
                <img src="${s.url}" class="w-11 h-11 rounded-md object-cover bg-surface-dim shrink-0" alt="" />
                <span class="min-w-0 flex-1">
                    <span class="block text-[13px] font-medium text-on-surface truncate">${s.name}</span>
                    <span class="block text-[12px] text-on-surface-muted truncate mt-0.5">${s.site}</span>
                </span>
                <span class="${chipClass} shrink-0">${s.category}</span>
            </button>
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
        setAnatomicSiteUI(apiSite);
    }

    const statusTag = document.getElementById("image-status-tag");
    if (statusTag) statusTag.textContent = `Sample · ${sample.name || 'Case'}`;

    toast(`Loaded sample: ${sample.name || 'Case'}`, "info");
}

// ─── Analysis button and overlay states ─────────────────────────────────────
function setAnalyzeButtonIdle(btn) {
    if (!btn) return;
    btn.classList.remove("is-busy");
    btn.innerHTML = `<span class="material-symbols-outlined">play_arrow</span> Run analysis`;
}

function setAnalysisOverlay(visible) {
    const overlay = document.getElementById("analysis-overlay");
    if (!overlay) return;
    // The overlay is a flex column; `hidden` alone loses to `flex`, so toggle both.
    overlay.classList.toggle("hidden", !visible);
    overlay.classList.toggle("flex", visible);
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
    analyzeBtn.classList.add("is-busy");
    analyzeBtn.innerHTML = `<span class="material-symbols-outlined animate-spin">progress_activity</span> Analysing…`;
    // The overlay existed in the markup but nothing ever unhid it, so a scan gave
    // no feedback over the image itself. Show it for the duration of the request.
    setAnalysisOverlay(true);

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
        setAnalyzeButtonIdle(analyzeBtn);
        setAnalysisOverlay(false);
    }
}

// Millisecond timings from the API. Sub-millisecond values are common for the
// queue wait on an idle server; rounding those to "0 ms" would read as a missing
// measurement, so they keep one decimal.
function formatMs(value) {
    if (typeof value !== "number" || !isFinite(value)) return "—";
    if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
    if (value < 10) return `${value.toFixed(1)} ms`;
    return `${Math.round(value)} ms`;
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
        // Sentence case, not shouted mono: the banner has to be read carefully,
        // and all-caps at 12px is the hardest thing on the page to read.
        const bannerBase = "px-4 py-3 rounded-xl border flex items-center gap-3";
        const iconBase = "material-symbols-outlined text-[22px] shrink-0";
        const textBase = "text-[13px] font-semibold leading-5";

        if (!isHighRisk && data.melanoma_alert) {
            banner.className = `${bannerBase} border-risk-mid/45 bg-risk-mid/10`;
            bannerIcon.className = `${iconBase} text-risk-mid`;
            bannerIcon.textContent = "release_alert";
            bannerText.className = `${textBase} text-risk-mid`;
            bannerText.textContent =
                `Melanoma not excluded — p(mel) ${(data.melanoma_probability * 100).toFixed(1)}%. Specialist review recommended.`;
        } else if (isHighRisk) {
            banner.className = `${bannerBase} border-risk-high/45 bg-risk-high/10`;
            bannerIcon.className = `${iconBase} text-risk-high`;
            bannerIcon.textContent = "warning";
            bannerText.className = `${textBase} text-risk-high`;
            bannerText.textContent = "Predicted class is in the high-risk group. Specialist assessment is indicated.";
        } else {
            banner.className = `${bannerBase} border-risk-low/45 bg-risk-low/10`;
            bannerIcon.className = `${iconBase} text-risk-low`;
            bannerIcon.textContent = "check_circle";
            bannerText.className = `${textBase} text-risk-low`;
            bannerText.textContent = "No malignant pattern identified. This is not a clinical clearance.";
        }
    }

    // 2. Confidence gauge. Circumference is read off the SVG rather than
    //    hardcoded, so changing the radius in the markup cannot silently
    //    desynchronise the arc from the number printed inside it.
    const gaugeValue = document.getElementById("gauge-confidence-val");
    const gaugeCircle = document.getElementById("gauge-confidence-circle");
    const gaugeStatus = document.getElementById("gauge-risk-status");

    if (gaugeValue) gaugeValue.textContent = `${confidencePct}%`;
    if (gaugeCircle) {
        const r = parseFloat(gaugeCircle.getAttribute("r")) || 46;
        const circumference = 2 * Math.PI * r;
        gaugeCircle.setAttribute("stroke-dasharray", circumference.toFixed(2));
        gaugeCircle.style.strokeDashoffset = circumference * (1 - data.confidence);
        gaugeCircle.style.stroke = isHighRisk ? "#e8746c" : "#6fc0a4";
    }
    if (gaugeStatus) {
        gaugeStatus.textContent = isHighRisk ? "High risk" : "Not flagged";
        gaugeStatus.className = `label mt-1.5 ${isHighRisk ? 'text-risk-high' : 'text-risk-low'}`;
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
        pathologyType.className = `chip ${isHighRisk ? 'chip--high' : meta.type === 'Pre-Malignant' ? 'chip--mid' : 'chip--low'}`;
    }
    if (pathologyDesc) pathologyDesc.textContent = meta.desc;

    // 3b. Fields that the previous markup shipped as fixed placeholder values —
    //     a made-up session id, "Anterior Torso" and "142ms" — and that nothing
    //     ever updated. They now show what the API actually returned, and the
    //     latency tile (which the API does not report at all) was replaced by
    //     the decision threshold, which it does.
    const sessionEl = document.getElementById("res-session-id");
    if (sessionEl) {
        sessionEl.textContent = data.session_id
            ? `Session #${data.session_id} · EfficientNet-B3`
            : "EfficientNet-B3";
    }
    const siteEl = document.getElementById("res-anatomic-site");
    if (siteEl) siteEl.textContent = data.anatomic_site || "Not recorded";

    //     `threshold` is the decision threshold of the predicted class, not a
    //     global one, so name the class alongside it — a bare "0.000" reads as
    //     a missing value rather than a permissive threshold.
    const thresholdEl = document.getElementById("res-threshold");
    if (thresholdEl) {
        thresholdEl.textContent = (typeof data.threshold === "number")
            ? `${data.threshold.toFixed(3)} · ${data.prediction.toUpperCase()}`
            : "—";
    }

    //     Server-side timings, shown separately because they are measured
    //     separately: `inference_ms` is the forward pass, `queue_ms` the wait
    //     for a concurrency slot. An older build printed a hardcoded "142ms"
    //     here that no code path ever wrote to, so anything not sent by the
    //     API renders as an em dash rather than a plausible-looking number.
    const latencyEl = document.getElementById("res-latency");
    if (latencyEl) latencyEl.textContent = formatMs(data.inference_ms);

    const queueEl = document.getElementById("res-queue");
    if (queueEl) queueEl.textContent = formatMs(data.queue_ms);

    // 4. 7-Class Probability Matrix
    const matrixContainer = document.getElementById("results-probability-matrix");
    if (matrixContainer && data.scores) {
        const sortedScores = Object.entries(data.scores).sort((a, b) => b[1] - a[1]);
        
        matrixContainer.innerHTML = sortedScores.map(([clsKey, prob]) => {
            const clsMeta = PATHOLOGY_META[clsKey] || { name: clsKey.toUpperCase() };
            const pct = (prob * 100).toFixed(1);
            const isTop = clsKey === data.prediction;
            const barWidth = Math.max(pct, 1.5);

            // Only the predicted class is coloured, and it is coloured by risk.
            // Everything else stays neutral so the eye lands on the one row that
            // decides the outcome instead of on seven competing colours.
            const barColor = isTop ? (isHighRisk ? 'bg-risk-high' : 'bg-risk-low') : 'bg-outline';
            const nameColor = isTop ? 'text-on-surface font-semibold' : 'text-on-surface-variant';
            const pctColor = isTop
                ? (isHighRisk ? 'text-risk-high font-semibold' : 'text-risk-low font-semibold')
                : 'text-on-surface-variant';

            return `
                <div>
                    <div class="flex items-baseline justify-between gap-3 mb-1.5">
                        <span class="text-[13px] ${nameColor} truncate">
                            ${clsMeta.name}
                            <span class="font-data-sm text-[12px] text-on-surface-muted">${clsKey.toUpperCase()}</span>
                        </span>
                        <span class="font-data-sm text-[13px] tabular ${pctColor}">${pct}%</span>
                    </div>
                    <div class="h-1.5 bg-surface-container rounded-full overflow-hidden" role="img" aria-label="${clsMeta.name}: ${pct} percent">
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

    // 5b. The image the scan was run on.
    renderResultsImage(data);

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

    // The recall figure comes from the API, which reads it out of the threshold
    // file shipped with the weights. It used to be the literal 0.800 written
    // here, which would have kept asserting itself through any retrain with
    // nothing able to catch it. If the server does not report one, the sentence
    // is dropped rather than filled with a remembered number.
    //
    // Deliberately not described as measured "on a lesion-disjoint hold-out",
    // which the old string claimed: the served artifact records `fitted_on` for
    // its thresholds, not the split these metrics were measured on, so that
    // provenance is not ours to assert here.
    const recall = data.operating_point && data.operating_point.melanoma_recall;
    if (typeof recall === "number" && recall >= 0 && recall <= 1) {
        // The "one in N" gloss is only worth stating where it is both meaningful
        // and not misleadingly precise: a recall of 0.999 is not "one in 1000",
        // and a recall of 0 is not "one in 1". Zero itself is still reported —
        // dropping the sentence would hide the worst case rather than state it.
        const missedInEvery = Math.round(1 / (1 - recall));
        const showGloss = recall < 0.995 && isFinite(missedInEvery) && missedInEvery >= 2;
        parts.push(
            `The melanoma recall recorded for this configuration is ${recall.toFixed(3)}` +
            (showGloss
                ? ` — roughly one melanoma in ${missedInEvery} is missed by the prediction itself.`
                : `.`));
    }

    parts.push("A result that is not flagged is not a clinical clearance.");

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
        badge.textContent = m ? "Stage 1 passed" : "Not reported";
        badge.className = "chip " + (m ? "chip--low" : "");
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
        note.className = "text-[12px] leading-5 " +
            ((data.ood_calibrated && data.ood_feature_stage_active)
                ? "text-on-surface-muted" : "text-risk-mid");
    }
}

// ─── Printed / exported report ──────────────────────────────────────────────
// A document built for paper, not a screenshot of the screen. The previous
// export handed html2pdf the live results view, so the "clinical report" was a
// picture of a dark dashboard — neon on near-black, cropped mid-card by the page
// break, and with no image of the lesion anywhere in it, because the panel had
// none. Everything below is drawn from `state.latestResult`; nothing is invented
// to fill a field, and a value the API did not send prints as an em dash.

function escapeHtml(value) {
    return String(value == null ? "" : value)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function reportFlagClass(data) {
    if (data.is_high_risk) return "pr-flag--high";
    if (data.melanoma_alert) return "pr-flag--mid";
    return "pr-flag--low";
}

function reportFlagText(data) {
    if (data.is_high_risk) return "High-risk group";
    if (data.melanoma_alert) return "Melanoma not excluded";
    return "Not flagged";
}

function buildPrintReport(data) {
    const host = document.getElementById("print-report");
    if (!host || !data) return null;

    const meta = PATHOLOGY_META[data.prediction] || {
        name: data.prediction.toUpperCase(), code: "—", type: "—", desc: ""
    };
    const generated = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const stamp = `${generated.getFullYear()}-${pad(generated.getMonth() + 1)}-` +
                  `${pad(generated.getDate())} ${pad(generated.getHours())}:${pad(generated.getMinutes())}`;

    const scores = data.scores || {};
    const rows = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .map(([cls, p]) => {
            const m = PATHOLOGY_META[cls] || { name: cls.toUpperCase() };
            const pct = (p * 100);
            const isTop = cls === data.prediction;
            return `
                <tr class="${isTop ? "pr-top" : ""}">
                    <td>${escapeHtml(m.name)}</td>
                    <td style="color:#5b6672">${escapeHtml(cls.toUpperCase())}</td>
                    <td class="pr-num">${pct.toFixed(1)}%</td>
                    <td style="width:38%"><div class="pr-bar"><span style="width:${Math.max(pct, 0.6)}%"></span></div></td>
                </tr>`;
        }).join("");

    const m = data.ood_metrics || {};
    const oodValue = (key) => (typeof m[key] === "number" ? m[key].toFixed(3) : "—");
    const gateNote = [
        data.ood_calibrated
            ? "Gate thresholds: fitted to data."
            : "Gate thresholds: provisional defaults, not yet fitted.",
        data.ood_feature_stage_active
            ? "Feature-space stage: active."
            : "Feature-space stage: not fitted, did not run."
    ].join(" ");

    // Both figures are assembled here rather than patched in afterwards, so the
    // node is complete the moment it is built — the print fallback renders the
    // same document as the PDF export, and an absent image is a labelled
    // placeholder instead of a broken-image glyph.
    const submittedFigure = data.session_id ? `
        <figure class="pr-col">
            <img id="pr-image" src="${API_BASE}/history/${encodeURIComponent(data.session_id)}/image" alt="" />
            <figcaption>The image the model was given, as stored.</figcaption>
        </figure>` : `
        <figure class="pr-col">
            <div class="pr-nofig">Not available</div>
            <figcaption>The stored image for this scan could not be located.</figcaption>
        </figure>`;

    const heatmapFigure = state.heatmapBase64 ? `
        <figure class="pr-col">
            <img src="${state.heatmapBase64}" alt="" />
            <figcaption>Grad-CAM attribution over the 224&times;224 model input. Indicates where
            the network attended, not lesion boundaries.</figcaption>
        </figure>` : `
        <figure class="pr-col">
            <div class="pr-nofig">Not available</div>
            <figcaption>No Grad-CAM attribution was returned for this scan.</figcaption>
        </figure>`;

    host.innerHTML = `
        <div class="pr-head">
            <div>
                <h1>Lesion classification report</h1>
                <div class="pr-sub">DermaScan &middot; EfficientNet-B3, 7-class, trained on HAM10000</div>
            </div>
            <div class="pr-meta">
                Session ${data.session_id ? "#" + escapeHtml(data.session_id) : "—"}<br />
                Generated ${escapeHtml(stamp)}
            </div>
        </div>

        <div class="pr-block">
            <h2>Submitted image</h2>
            <div class="pr-cols">
                ${submittedFigure}
                ${heatmapFigure}
            </div>
        </div>

        <div class="pr-block">
            <h2>Classification</h2>
            <div style="display:flex;justify-content:space-between;align-items:baseline;gap:16px">
                <p class="pr-headline">${escapeHtml(meta.name)}</p>
                <span class="pr-flag ${reportFlagClass(data)}">${reportFlagText(data)}</span>
            </div>
            <div class="pr-sub" style="margin-bottom:10px">
                ${escapeHtml(data.prediction.toUpperCase())} &middot; ${escapeHtml(meta.code)} &middot;
                confidence ${(data.confidence * 100).toFixed(1)}%
            </div>
            <dl class="pr-kv">
                <div><dt>Anatomic site</dt><dd>${escapeHtml(data.anatomic_site || "Not recorded")}</dd></div>
                <div><dt>Decision threshold</dt><dd>${typeof data.threshold === "number"
                    ? data.threshold.toFixed(3) + " · " + escapeHtml(data.prediction.toUpperCase()) : "—"}</dd></div>
                <div><dt>p(melanoma)</dt><dd>${typeof data.melanoma_probability === "number"
                    ? (data.melanoma_probability * 100).toFixed(1) + "%" : "—"}</dd></div>
                <div><dt>Model inference</dt><dd>${formatMs(data.inference_ms)}</dd></div>
            </dl>
        </div>

        <div class="pr-block">
            <h2>Probability across all seven classes</h2>
            <table>
                <thead>
                    <tr><th>Class</th><th>Code</th><th class="pr-num">Probability</th><th></th></tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>

        <div class="pr-block">
            <h2>Input screening</h2>
            <dl class="pr-kv" style="margin-bottom:8px">
                <div><dt>Rel. contrast</dt><dd>${oodValue("rel_contrast")}</dd></div>
                <div><dt>High-freq ratio</dt><dd>${oodValue("hf_ratio")}</dd></div>
                <div><dt>Blue-green hue</dt><dd>${oodValue("blue_green")}</dd></div>
                <div><dt>Stage 1</dt><dd>${data.ood_metrics ? "Passed" : "Not reported"}</dd></div>
            </dl>
            <div class="pr-sub">${escapeHtml(gateNote)}</div>
        </div>

        <div class="pr-block">
            <h2>Summary</h2>
            <p style="margin:0 0 10px">${escapeHtml(buildClinicalSummary(data, meta))}</p>
            <div class="pr-note">
                <strong>This is not a diagnosis.</strong> The output above is a single
                classifier's probability distribution over seven lesion classes, produced from
                one image with no clinical context, history or examination. It requires
                correlation by a qualified clinician and is not a substitute for one. The
                system measures nothing about the lesion itself &mdash; no size, growth,
                pigmentation or border metric is computed &mdash; and a negative result is not
                a clearance.
            </div>
        </div>

        <div class="pr-foot">
            <span>DermaScan &middot; research decision-support tool</span>
            <span>Session ${data.session_id ? "#" + escapeHtml(data.session_id) : "—"} &middot; ${escapeHtml(stamp)}</span>
        </div>
    `;

    return host;
}

// html2canvas cannot render a `display:none` subtree, so the report has to be
// laid out for the capture. It must NOT be pushed off-screen to hide it: given
// an absolutely or fixed positioned element, html2pdf returns a blank page with
// no error (measured here as a 3 KB PDF against 728 KB for the same content in
// normal flow). Visibility is handled by #print-report-clip, a zero-height
// overflow:hidden parent, which leaves the report statically positioned.
function showReportForCapture(host) {
    host.style.display = "block";
}

function hideReport(host) {
    host.style.display = "";
}

// Resolve once every image in the report has settled. Capturing before they load
// produced a report with empty frames where the lesion should be.
function whenReportImagesSettle(host) {
    const images = [...host.querySelectorAll("img")].filter(img => img.getAttribute("src"));
    return Promise.all(images.map(img => (img.complete && img.naturalWidth)
        ? Promise.resolve()
        : new Promise(resolve => {
            img.addEventListener("load", resolve, { once: true });
            img.addEventListener("error", resolve, { once: true });
        })));
}

async function downloadClinicalReport() {
    // The export button lives in the always-visible top bar, so it can be pressed
    // with no scan run. html2pdf would then render a blank or placeholder page and
    // present it as a clinical report.
    if (!state.latestResult) {
        toast("Run a scan before exporting a report.", "warning");
        return;
    }

    const data = state.latestResult;
    const host = buildPrintReport(data);
    if (!host) {
        toast("Could not assemble the report.", "error");
        return;
    }

    if (!window.html2pdf) {
        // The print stylesheet renders the same node, so the fallback produces
        // the same document rather than a printout of the dashboard.
        await whenReportImagesSettle(host);
        window.print();
        return;
    }

    toast("Building report…", "info");
    showReportForCapture(host);
    try {
        await whenReportImagesSettle(host);
        await window.html2pdf().set({
            margin:      [14, 14, 14, 14],
            filename:    `DermaScan_${data.prediction}_${data.session_id || "scan"}_` +
                         `${new Date().toISOString().slice(0, 10)}.pdf`,
            image:       { type: "jpeg", quality: 0.98 },
            html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
            jsPDF:       { unit: "mm", format: "letter", orientation: "portrait" },
            // Honour the page-break rules on .pr-block so a section is never
            // sliced in half by the page boundary.
            pagebreak:   { mode: ["css", "legacy"] }
        }).from(host).save();
        toast("Report downloaded.", "success");
    } catch (err) {
        toast("Could not generate the PDF: " + err.message, "error");
    } finally {
        hideReport(host);
    }
}

// ─── Analytics & History View Handler ───────────────────────────────────────
async function loadAnalyticsData() {
    const historyTableBody = document.getElementById("analytics-history-tbody");
    if (!historyTableBody) return;

    historyTableBody.innerHTML = `
        <tr>
            <td colspan="6" class="py-14 text-center text-on-surface-muted text-[13px]">
                <span class="material-symbols-outlined animate-spin text-[22px] block mb-2">progress_activity</span>
                Loading records&hellip;
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
                <td colspan="6" class="py-14 text-center text-on-surface-muted">
                    <span class="material-symbols-outlined text-[28px] block mb-2">inbox</span>
                    <p class="text-[14px] text-on-surface-variant">No scans recorded yet</p>
                    <p class="text-[13px] mt-1">Run one from the scan console and it will appear here.</p>
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
            <tr class="border-b border-outline-variant hover:bg-surface-container transition-colors">
                <td class="py-3 px-5 font-data-sm text-[12px] text-on-surface-muted">#${log.id}</td>
                <td class="py-3 px-5 font-data-sm text-[12px] text-on-surface-variant tabular">${dateStr} <span class="text-on-surface-muted">${timeStr}</span></td>
                <td class="py-3 px-5 text-[13px] text-on-surface">${meta.name} <span class="font-data-sm text-[12px] text-on-surface-muted">${log.prediction.toUpperCase()}</span></td>
                <td class="py-3 px-5 font-data-sm text-[13px] tabular text-right ${isHigh ? 'text-risk-high' : 'text-on-surface'}">${confPct}%</td>
                <td class="py-3 px-5">
                    ${isHigh
                        ? `<span class="chip chip--high">High risk</span>`
                        : `<span class="chip chip--low">Not flagged</span>`
                    }
                </td>
                <td class="py-3 px-5 text-right">
                    <button onclick="deleteHistoryRecord(${log.id})" class="btn-icon hover:text-risk-high" aria-label="Delete record ${log.id}" title="Delete record">
                        <span class="material-symbols-outlined">delete</span>
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

/* ═══════════════════════════════════════════════════════════════════════════
   Compare view

   Two recorded scans, chosen from history. The previous version accepted loose
   image uploads into both slots, so it could only ever show pixels: an upload
   has not been through the model, so there was nothing to put beside it. Every
   figure rendered here comes from a stored inference on the image next to it.

   What this deliberately does not do: assert that the two images show the same
   lesion. Nothing in the schema links a scan to a lesion or a patient, so that
   claim is not available to the application. Differences shown are differences
   between two model outputs, never measurements of skin. This is the same line
   the removed "differential engine" crossed when it derived lesion growth from
   the length of a base64 string.
   ═══════════════════════════════════════════════════════════════════════════ */

// Fixed order, high-risk group first. A table whose rows move between record
// pairs cannot be read across pairs, so this deliberately does not sort by
// probability the way the single-result panel does.
const COMPARE_CLASS_ORDER = ["mel", "bcc", "akiec", "nv", "bkl", "df", "vasc"];
const HIGH_RISK_CLASSES = ["mel", "bcc", "akiec"];

const compareZoom = { a: 1.0, b: 1.0, linked: true };

function applyCompareZoom() {
    const imgLeft = document.getElementById("img-compare-left");
    const imgRight = document.getElementById("img-compare-right");
    if (imgLeft) imgLeft.style.transform = `scale(${compareZoom.a})`;
    if (imgRight) imgRight.style.transform = `scale(${compareZoom.b})`;
}

function formatPct(value) {
    if (typeof value !== "number" || !isFinite(value)) return "—";
    return `${(value * 100).toFixed(1)}%`;
}

// Differences between two percentages are percentage points, not percent.
function formatDeltaPP(value) {
    if (typeof value !== "number" || !isFinite(value)) return "—";
    const pp = value * 100;
    if (Math.abs(pp) < 0.05) return "0.0 pp";
    return `${pp > 0 ? "+" : "−"}${Math.abs(pp).toFixed(1)} pp`;
}

function formatInterval(msA, msB) {
    if (!isFinite(msA) || !isFinite(msB)) return "—";
    const ms = Math.abs(msB - msA);
    const minutes = ms / 60000;
    if (minutes < 1) return "under a minute";
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const hours = minutes / 60;
    if (hours < 48) return `${hours.toFixed(hours < 10 ? 1 : 0)} h`;
    const days = hours / 24;
    if (days < 60) return `${Math.round(days)} days`;
    return `${(days / 30.44).toFixed(1)} months`;
}

function formatRecordedAt(iso) {
    const dt = new Date(iso);
    if (isNaN(dt)) return "—";
    const pad = (n) => String(n).padStart(2, "0");
    return `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())} ` +
           `${pad(dt.getHours())}:${pad(dt.getMinutes())}`;
}

function compareOptionLabel(rec) {
    const meta = PATHOLOGY_META[rec.prediction] || { name: rec.prediction };
    return `#${rec.id} · ${meta.name} · ${formatPct(rec.confidence)} · ${formatRecordedAt(rec.created_at)}`;
}

// Populate both pickers from whatever history is loaded, keeping any selection
// the user has already made.
function populateCompareSelectors() {
    const records = state.history || [];
    ["A", "B"].forEach(slot => {
        const select = document.getElementById(`compare-select-${slot.toLowerCase()}`);
        if (!select) return;
        const chosen = select.value;
        select.innerHTML = `<option value="">Select a recorded scan…</option>` +
            records.map(r => `<option value="${r.id}">${compareOptionLabel(r)}</option>`).join("");
        if (chosen && records.some(r => String(r.id) === String(chosen))) {
            select.value = chosen;
        }
    });
}

async function loadCompareRecords(force = false) {
    if (force || !state.history || !state.history.length) {
        try {
            state.history = await apiCall("/history");
        } catch (err) {
            toast("Could not load scan records: " + err.message, "error");
            return;
        }
    }
    populateCompareSelectors();

    // A scan that has just finished is the one you want in slot A.
    if (state.compare.pendingA) {
        const id = state.compare.pendingA;
        state.compare.pendingA = null;
        const select = document.getElementById("compare-select-a");
        if (select && state.history.some(r => String(r.id) === String(id))) {
            select.value = String(id);
            await setCompareSlot("A", id);
        }
    }
    renderCompare();
}

// Each slot's in-flight request is tagged, so changing the picker again before
// the previous fetch returns cannot let the older response land last and
// silently show a record the picker is no longer pointing at.
const compareRequest = { a: 0, b: 0 };

async function setCompareSlot(slot, id) {
    const key = slot === "A" ? "a" : "b";
    const ticket = ++compareRequest[key];

    if (!id) {
        state.compare[key] = null;
        renderCompare();
        return;
    }
    try {
        // Read the record back rather than trusting the cached list row: the
        // list is capped and can be stale, and a record deleted in another tab
        // should fail here rather than render as a comparison.
        const record = await apiCall(`/history/${encodeURIComponent(id)}`);
        if (ticket !== compareRequest[key]) return;
        state.compare[key] = record;
    } catch (err) {
        if (ticket !== compareRequest[key]) return;
        state.compare[key] = null;
        toast(`Could not load scan #${id}: ${err.message}`, "error");
    }
    renderCompare();
}

function swapCompareSlots() {
    const selectA = document.getElementById("compare-select-a");
    const selectB = document.getElementById("compare-select-b");
    const { a, b } = state.compare;
    state.compare.a = b;
    state.compare.b = a;
    if (selectA && selectB) {
        const va = selectA.value;
        selectA.value = selectB.value;
        selectB.value = va;
    }
    const za = compareZoom.a;
    compareZoom.a = compareZoom.b;
    compareZoom.b = za;
    renderCompare();
}

function renderCompareImage(slot, rec) {
    const isA = slot === "A";
    const img = document.getElementById(isA ? "img-compare-left" : "img-compare-right");
    const missing = document.getElementById(isA ? "compare-a-missing" : "compare-b-missing");
    if (!img || !missing) return;

    const showMissing = () => {
        img.classList.add("hidden");
        missing.classList.remove("hidden");
        missing.classList.add("flex");
    };

    if (!rec.has_image) {
        showMissing();
        return;
    }

    // Retention and the orphan sweep can remove a file while its row survives,
    // so a 404 here is an expected outcome, not a failure to report.
    img.onerror = showMissing;
    img.onload = () => {
        missing.classList.add("hidden");
        missing.classList.remove("flex");
        img.classList.remove("hidden");
    };
    img.src = `${API_BASE}/history/${encodeURIComponent(rec.id)}/image`;
}

function renderCompareHeader(slot, rec) {
    const isA = slot === "A";
    const caption = document.getElementById(isA ? "compare-a-caption" : "compare-b-caption");
    const chip = document.getElementById(isA ? "compare-a-chip" : "compare-b-chip");
    const meta = PATHOLOGY_META[rec.prediction] || { name: rec.prediction };

    if (caption) caption.textContent = `#${rec.id} · ${formatRecordedAt(rec.created_at)}`;
    if (chip) {
        chip.textContent = `${meta.name} ${formatPct(rec.confidence)}`;
        chip.className = "chip shrink-0 " + (rec.is_high_risk ? "chip--high" : "chip--low");
    }
}

function renderCompareSummary(a, b) {
    const agrees = a.prediction === b.prediction;
    const metaA = PATHOLOGY_META[a.prediction] || { name: a.prediction };
    const metaB = PATHOLOGY_META[b.prediction] || { name: b.prediction };

    const set = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };

    set("compare-agreement", agrees ? "Same" : "Different");
    set("compare-agreement-detail", agrees
        ? metaA.name
        : `A: ${metaA.name} · B: ${metaB.name}`);

    const confDelta = b.confidence - a.confidence;
    set("compare-conf-delta", formatDeltaPP(confDelta));
    set("compare-conf-detail", agrees
        ? `${formatPct(a.confidence)} → ${formatPct(b.confidence)}`
        : "Different classes — not a like-for-like change");

    set("compare-interval", formatInterval(
        new Date(a.created_at).getTime(), new Date(b.created_at).getTime()));

    const siteA = a.anatomic_site || "Not recorded";
    const siteB = b.anatomic_site || "Not recorded";
    set("compare-site", siteA === siteB ? siteA : `A: ${siteA} · B: ${siteB}`);
}

function renderCompareClassTable(a, b) {
    const tbody = document.getElementById("compare-class-table");
    if (!tbody) return;

    const scoresA = a.scores || {};
    const scoresB = b.scores || {};

    tbody.innerHTML = COMPARE_CLASS_ORDER.map(cls => {
        const meta = PATHOLOGY_META[cls] || { name: cls.toUpperCase() };
        const pA = typeof scoresA[cls] === "number" ? scoresA[cls] : null;
        const pB = typeof scoresB[cls] === "number" ? scoresB[cls] : null;
        const delta = (pA === null || pB === null) ? null : pB - pA;

        // Diverging bar around a centre line. The magnitude can reach a full
        // 100 points either way, so half the track is one point per percent.
        const half = delta === null ? 0 : Math.min(Math.abs(delta) * 100 / 2, 50);
        const positive = (delta || 0) > 0;
        const bar = delta === null ? "" : `
            <div class="relative h-1.5 w-full bg-surface-container rounded-full">
                <span class="absolute inset-y-0 left-1/2 w-px bg-outline"></span>
                <span class="absolute inset-y-0 ${positive ? 'left-1/2' : 'right-1/2'} bg-primary rounded-full"
                      style="width: ${half}%"></span>
            </div>`;

        const isHigh = HIGH_RISK_CLASSES.includes(cls);
        const topA = cls === a.prediction;
        const topB = cls === b.prediction;

        return `
            <tr class="border-b border-outline-variant/60">
                <td class="py-2.5 pr-4 ${isHigh ? 'border-l-2 border-l-risk-high/60 pl-3' : 'pl-3'}">
                    <span class="text-[13px] ${topA || topB ? 'text-on-surface font-medium' : 'text-on-surface-variant'}">${meta.name}</span>
                    <span class="font-data-sm text-[12px] text-on-surface-muted ml-1.5">${cls.toUpperCase()}</span>
                </td>
                <td class="py-2.5 px-3 text-right font-data-sm text-[13px] tabular ${topA ? 'text-on-surface font-semibold' : 'text-on-surface-variant'}">${formatPct(pA)}</td>
                <td class="py-2.5 px-3 text-right font-data-sm text-[13px] tabular ${topB ? 'text-on-surface font-semibold' : 'text-on-surface-variant'}">${formatPct(pB)}</td>
                <td class="py-2.5 px-3 text-right font-data-sm text-[13px] tabular text-on-surface-variant">${formatDeltaPP(delta)}</td>
                <td class="py-2.5 pl-3">${bar}</td>
            </tr>
        `;
    }).join("");
}

function renderCompareDetailTable(a, b) {
    const tbody = document.getElementById("compare-detail-table");
    if (!tbody) return;

    const yesNo = (v) => (v ? "Yes" : "No");
    const metaName = (rec) => (PATHOLOGY_META[rec.prediction] || { name: rec.prediction }).name;
    const num = (v, digits) => (typeof v === "number" ? v.toFixed(digits) : "—");

    const rows = [
        ["Session id", `#${a.id}`, `#${b.id}`],
        ["Recorded", formatRecordedAt(a.created_at), formatRecordedAt(b.created_at)],
        ["Predicted class", `${metaName(a)} (${a.prediction.toUpperCase()})`,
                            `${metaName(b)} (${b.prediction.toUpperCase()})`],
        ["Confidence", formatPct(a.confidence), formatPct(b.confidence)],
        ["High-risk group", yesNo(a.is_high_risk), yesNo(b.is_high_risk)],
        ["Melanoma alert", yesNo(a.melanoma_alert), yesNo(b.melanoma_alert)],
        ["p(melanoma)", formatPct(a.melanoma_probability), formatPct(b.melanoma_probability)],
        ["Decision threshold", num(a.threshold_used, 3), num(b.threshold_used, 3)],
        ["Anatomic site", a.anatomic_site || "Not recorded", b.anatomic_site || "Not recorded"],
        ["Image retained", yesNo(a.has_image), yesNo(b.has_image)],
    ];

    tbody.innerHTML = rows.map(([field, va, vb]) => `
        <tr class="border-b border-outline-variant/60">
            <td class="py-2.5 pr-4 text-[13px] text-on-surface-variant">${field}</td>
            <td class="py-2.5 px-3 text-[13px] text-on-surface">${va}</td>
            <td class="py-2.5 px-3 text-[13px] text-on-surface">${vb}</td>
        </tr>
    `).join("");
}

function renderCompare() {
    const empty = document.getElementById("compare-empty");
    const content = document.getElementById("compare-content");
    const { a, b } = state.compare;

    if (!a || !b) {
        if (empty) empty.classList.remove("hidden");
        if (content) {
            content.classList.add("hidden");
            content.classList.remove("flex");
        }
        return;
    }

    if (empty) empty.classList.add("hidden");
    if (content) {
        content.classList.remove("hidden");
        content.classList.add("flex");
    }

    renderCompareHeader("A", a);
    renderCompareHeader("B", b);
    renderCompareImage("A", a);
    renderCompareImage("B", b);
    renderCompareSummary(a, b);
    renderCompareClassTable(a, b);
    renderCompareDetailTable(a, b);
    applyCompareZoom();
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

    // 1a2. Grad-CAM toggle on the result panel.
    const btnResultsHeatmap = document.getElementById("btn-results-heatmap");
    if (btnResultsHeatmap) {
        btnResultsHeatmap.addEventListener("click", () => {
            if (!state.heatmapBase64) {
                toast("No Grad-CAM attribution is available for this scan.", "warning");
                return;
            }
            setResultsHeatmapVisible(!state.resultsHeatmapVisible);
        });
    }

    // 1a. Keep the Grad-CAM overlay locked to the image it explains.
    const resyncHeatmaps = () => {
        syncHeatmapCanvasToImage();
        syncResultsHeatmapCanvas();
    };
    window.addEventListener("resize", resyncHeatmaps);
    const viewportImgEl = document.getElementById("console-viewport-img");
    if (viewportImgEl) viewportImgEl.addEventListener("load", () => syncHeatmapCanvasToImage());

    // 1b. Mobile navigation drawer
    const btnNavOpen = document.getElementById("btn-nav-open");
    const btnNavClose = document.getElementById("btn-nav-close");
    const navBackdrop = document.getElementById("nav-backdrop");

    if (btnNavOpen) btnNavOpen.addEventListener("click", openNavDrawer);
    if (btnNavClose) btnNavClose.addEventListener("click", closeNavDrawer);
    if (navBackdrop) navBackdrop.addEventListener("click", closeNavDrawer);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeNavDrawer();
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
            state.selectedAnatomicSite = tagBtn.getAttribute("data-site");
            setAnatomicSiteUI(state.selectedAnatomicSite);
            toast(`Anatomic site: ${state.selectedAnatomicSite}`, "info");
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
        // The drop zone is exposed as a button, so it has to answer to the keys
        // a button answers to.
        dropZone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                fileInput.click();
            }
        });
        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) handleFileSelected(e.target.files[0]);
        });

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("ring-2", "ring-inset", "ring-primary", "bg-primary/5");
        });
        dropZone.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dropZone.classList.remove("ring-2", "ring-inset", "ring-primary", "bg-primary/5");
        });
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("ring-2", "ring-inset", "ring-primary", "bg-primary/5");
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
                const isActive = b === filterBtn;
                b.classList.toggle("is-active", isActive);
                b.setAttribute("aria-pressed", isActive ? "true" : "false");
            });

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

    // 12. Compare view
    const btnCompareRefresh = document.getElementById("btn-compare-refresh");
    const btnCompareSwap = document.getElementById("btn-compare-swap");
    const selectA = document.getElementById("compare-select-a");
    const selectB = document.getElementById("compare-select-b");

    if (selectA) selectA.addEventListener("change", (e) => setCompareSlot("A", e.target.value));
    if (selectB) selectB.addEventListener("change", (e) => setCompareSlot("B", e.target.value));
    if (btnCompareSwap) btnCompareSwap.addEventListener("click", swapCompareSlots);
    if (btnCompareRefresh) {
        btnCompareRefresh.addEventListener("click", () => loadCompareRecords(true));
    }

    // Linked zoom. Both viewports are fixed-aspect boxes now, so a single scale
    // factor per side is all this needs.
    const syncToggle = document.getElementById("syncToggle");
    const syncThumb = document.getElementById("syncThumb");
    if (syncToggle && syncThumb) {
        syncToggle.addEventListener("click", () => {
            compareZoom.linked = !compareZoom.linked;
            if (compareZoom.linked) {
                syncThumb.style.transform = "translateX(0)";
                syncToggle.className = "w-9 h-5 rounded-full relative transition-colors duration-200 bg-primary/50 border border-primary/60";
                syncToggle.setAttribute("aria-checked", "true");
                // Re-link from A so the two images do not stay at whatever
                // independent scales they drifted to while unlinked.
                compareZoom.b = compareZoom.a;
                applyCompareZoom();
                toast("Zoom linked", "info");
            } else {
                syncThumb.style.transform = "translateX(-16px)";
                syncToggle.className = "w-9 h-5 rounded-full relative transition-colors duration-200 bg-surface-container-high border border-outline";
                syncToggle.setAttribute("aria-checked", "false");
                toast("Zoom unlinked", "info");
            }
        });
    }

    const vLeft = document.getElementById("viewportLeft");
    const vRight = document.getElementById("viewportRight");

    function wheelZoom(slot) {
        return (e) => {
            e.preventDefault();
            const step = e.deltaY < 0 ? 0.1 : -0.1;
            const clamp = (v) => Math.max(1.0, Math.min(4.0, v + step));
            if (compareZoom.linked) {
                compareZoom.a = clamp(compareZoom.a);
                compareZoom.b = compareZoom.a;
            } else if (slot === "A") {
                compareZoom.a = clamp(compareZoom.a);
            } else {
                compareZoom.b = clamp(compareZoom.b);
            }
            applyCompareZoom();
        };
    }

    if (vLeft) vLeft.addEventListener("wheel", wheelZoom("A"), { passive: false });
    if (vRight) vRight.addEventListener("wheel", wheelZoom("B"), { passive: false });

    // 13. Arriving at the comparison from a finished scan preselects it, which is
    //     the only reason someone presses that button. The old handler tried to
    //     push the console's *file* into slot B, which carried no model output
    //     with it and so could not be compared against anything.
    const btnResultsCompare = document.getElementById("btn-results-compare");
    if (btnResultsCompare) {
        // Capture phase on purpose. The button also carries `data-view`, so the
        // router's own bubble-phase listener — registered first — would call
        // navigate() before this ran, and loadCompareRecords() reads pendingA
        // synchronously whenever history is already cached. The slot has to be
        // parked before navigation, not after it.
        btnResultsCompare.addEventListener("click", () => {
            const id = state.latestResult && state.latestResult.session_id;
            if (id) state.compare.pendingA = String(id);
        }, true);
    }

    // Initial view load
    navigate("view-console");
});

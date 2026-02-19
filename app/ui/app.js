// ============================================================================
// Model Hub UI — Full overhaul with drag-from-catalog, log viewer, stats,
// stop/shorten, ASAP booking
// ============================================================================

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ─── Theme ──────────────────────────────────────────────────────────────────
const themeToggle = $('#themeToggle');
const iconSun = $('#iconSun');
const iconMoon = $('#iconMoon');

function isDark() { return document.documentElement.classList.contains('dark'); }

function updateThemeIcons() {
  if (isDark()) { iconSun.classList.remove('hidden'); iconMoon.classList.add('hidden'); }
  else { iconSun.classList.add('hidden'); iconMoon.classList.remove('hidden'); }
}

// ─── Logout ─────────────────────────────────────────────────────────────────
$('#logoutBtn').addEventListener('click', async () => {
  try {
    await fetch('/api/logout', { method: 'POST' });
  } catch (e) {
    // ignore
  }
  window.location.href = '/login';
});

themeToggle.addEventListener('click', () => {
  document.documentElement.classList.toggle('dark');
  updateThemeIcons();
  renderTimeline();
});
updateThemeIcons();

// ─── Toast Notifications ────────────────────────────────────────────────────
function toast(msg, type = 'success') {
  const container = $('#toastContainer');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3100);
}

// ─── API Helper ─────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (res.status === 401) {
    // Session expired or not authenticated — redirect to login
    window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
    throw new Error('Session expired. Redirecting to login…');
  }
  const txt = await res.text();
  const data = txt ? JSON.parse(txt) : {};
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// ─── Global State ───────────────────────────────────────────────────────────
let DASH = null;
let MODEL_MAP = new Map();
let EDITING_LEASE_ID = null;
let REFRESH_INTERVAL = null;

// Timeline geometry
let TL = {
  leftPad: 70,
  headerH: 34,
  laneH: 44,
  pxPerHour: 90,
  hours: 24,
  start: null,
  end: null,
  gpuTotal: 8,
  width: 0,
  height: 0,
};

// ─── Utilities ──────────────────────────────────────────────────────────────
function parseDT(s) { return s ? new Date(s) : null; }
function fmtTime(d) { return d.toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }); }
function fmtHour(d) { return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function escapeHtml(s) {
  return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');
}
function snapTo15Min(date) {
  const d = new Date(date);
  d.setMinutes(Math.round(d.getMinutes() / 15) * 15, 0, 0);
  return d;
}
function toLocalDTInput(date) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}
function formatDuration(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0 && m > 0) return `${h}h ${m}m`;
  if (h > 0) return `${h}h`;
  return `${m}m`;
}

function svgPoint(svg, clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX; pt.y = clientY;
  const ctm = svg.getScreenCTM();
  return ctm ? pt.matrixTransform(ctm.inverse()) : pt;
}

function xToDate(x) {
  const hoursFromStart = (x - TL.leftPad) / TL.pxPerHour;
  return new Date(TL.start.getTime() + hoursFromStart * 3600000);
}

function dateToX(d) {
  return TL.leftPad + ((d.getTime() - TL.start.getTime()) / 3600000) * TL.pxPerHour;
}

// ─── SVG Helpers ────────────────────────────────────────────────────────────
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  }
  return el;
}

function drawRect(parent, x, y, w, h, attrs = {}) {
  const el = svgEl('rect', { x, y, width: w, height: h, ...attrs });
  parent.appendChild(el);
  return el;
}

function drawLine(parent, x1, y1, x2, y2, attrs = {}) {
  const el = svgEl('line', { x1, y1, x2, y2, ...attrs });
  parent.appendChild(el);
  return el;
}

function drawText(parent, x, y, text, attrs = {}) {
  const el = svgEl('text', { x, y, ...attrs });
  el.textContent = text;
  parent.appendChild(el);
  return el;
}

function drawGroup(parent, attrs = {}) {
  const el = svgEl('g', attrs);
  parent.appendChild(el);
  return el;
}

// ─── Data Fetch ─────────────────────────────────────────────────────────────
async function refresh() {
  try {
    DASH = await api('/admin/dashboard');
    MODEL_MAP = new Map(DASH.models.map(m => [m.id, m]));
    $('#subtitle').textContent = `Single node · ${DASH.total_gpus} GPUs · ${fmtTime(new Date(DASH.now))}`;
    renderCatalog();
    renderTimeline();
    renderTable();
    renderStatusIndicators();

    // Only repopulate modal models if the modal is NOT currently open,
    // or preserve the current selection if it is.
    const modalOpen = !modalBackdrop.classList.contains('hidden');
    if (modalOpen) {
      const currentSelection = $('#modalModel').value;
      populateModalModels(currentSelection);  // preserve selection
    } else {
      populateModalModels();
    }
  } catch (e) {
    toast(e.message, 'error');
  }
}

function startAutoRefresh() {
  if (REFRESH_INTERVAL) clearInterval(REFRESH_INTERVAL);
  REFRESH_INTERVAL = setInterval(() => {
    if (dragState.active || catalogDragState.active) return;
    refresh();
  }, 8000);
}

// ─── Status Indicators (header) ─────────────────────────────────────────────
function renderStatusIndicators() {
  const container = $('#statusIndicators');
  if (!DASH || !DASH.endpoint_stats) { container.innerHTML = ''; return; }

  const stats = DASH.endpoint_stats || [];
  const readyCount = stats.filter(s => s.state === 'READY').length;
  const startingCount = stats.filter(s => s.state === 'STARTING').length;

  let html = '';
  if (readyCount > 0) {
    html += `<span class="flex items-center gap-1.5"><span class="status-dot-ready"></span>${readyCount} running</span>`;
  }
  if (startingCount > 0) {
    html += `<span class="flex items-center gap-1.5"><span class="status-dot-starting"></span>${startingCount} starting</span>`;
  }
  if (readyCount === 0 && startingCount === 0) {
    html = `<span class="text-slate-500">No models active</span>`;
  }
  container.innerHTML = html;
}

// ─── Catalog Rendering (with drag support) ──────────────────────────────────
function renderCatalog() {
  const q = $('#searchInput').value.trim().toLowerCase();
  const filter = $('#filterSelect').value;

  const models = DASH.models
    .filter(m => {
      if (q && !m.id.toLowerCase().includes(q)) return false;
      if (filter === 'ready' && !m.ready) return false;
      if (filter === 'stopped' && m.ready) return false;
      return true;
    })
    .sort((a, b) => a.id.localeCompare(b.id));

  $('#modelCount').textContent = `${models.length} model${models.length !== 1 ? 's' : ''}`;

  const list = $('#modelsList');
  list.innerHTML = models.map(m => {
    const meta = m.meta || {};
    const g = meta.gpus ?? '?';
    const tp = meta.tensor_parallel_size ?? '?';
    const notes = meta.notes || '';
    const isRunning = m.ready;

    // Find endpoint stats for this model
    const epStats = (DASH.endpoint_stats || []).filter(s => s.model === m.id && s.state === 'READY');
    let statsHtml = '';
    if (epStats.length > 0) {
      const ep = epStats[0];
      const uptime = ep.uptime_seconds ? formatDuration(Math.floor(ep.uptime_seconds)) : '—';
      statsHtml = `<div class="mt-1.5 text-xs text-slate-500 flex items-center gap-3">
        <span>⏱ Uptime: ${uptime}</span>
        <span>📡 ${ep.host}:${ep.port}</span>
      </div>`;
    }

    const badge = isRunning
      ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
           <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>Running
         </span>`
      : `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-slate-500/15 text-slate-400 border border-slate-500/30">Idle</span>`;

    return `
      <div class="catalog-card rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4 hover:shadow-md hover:border-brand-500/30 transition-all duration-150"
           draggable="true" data-model-id="${escapeHtml(m.id)}">
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <div class="font-semibold text-sm break-all">${escapeHtml(m.id)}</div>
            <div class="mt-1 text-xs text-slate-500 font-mono">${g} GPUs · TP ${tp}</div>
            ${notes ? `<div class="mt-1 text-xs text-slate-500 italic">${escapeHtml(notes)}</div>` : ''}
            ${statsHtml}
          </div>
          <div class="shrink-0">${badge}</div>
        </div>
        <div class="mt-3 flex gap-2">
          <button class="flex-1 px-3 py-2 text-xs rounded-lg bg-brand-600 text-white hover:bg-brand-500 transition font-medium"
            onclick="event.stopPropagation(); openNewBookingForModel('${escapeHtml(m.id)}')">
            <svg class="w-3.5 h-3.5 inline mr-1 -mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
            Schedule
          </button>
          ${isRunning ? `
            <button class="px-3 py-2 text-xs rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-800 transition"
              onclick="event.stopPropagation(); viewLogs('${escapeHtml(m.id)}')">Logs</button>
          ` : ''}
        </div>
      </div>
    `;
  }).join('');

  // Attach drag events to catalog cards
  list.querySelectorAll('.catalog-card[draggable="true"]').forEach(card => {
    card.addEventListener('dragstart', onCatalogDragStart);
    card.addEventListener('dragend', onCatalogDragEnd);
  });
}

window.openChat = (modelId) => {
  window.open(`/v1/chat/ui?model=${encodeURIComponent(modelId)}`, '_blank');
};

window.openNewBookingForModel = (modelId) => {
  openModal({ model: modelId });
};

// ─── Catalog Drag-and-Drop onto Timeline ────────────────────────────────────
const catalogDragState = {
  active: false,
  modelId: null,
  ghostEl: null,
};

function onCatalogDragStart(e) {
  const modelId = e.currentTarget.dataset.modelId;
  catalogDragState.active = true;
  catalogDragState.modelId = modelId;

  // Set drag data
  e.dataTransfer.setData('text/plain', modelId);
  e.dataTransfer.effectAllowed = 'copy';

  // Create a small custom drag image
  const ghost = document.createElement('div');
  ghost.className = 'catalog-drag-ghost';
  ghost.textContent = `📦 ${modelId}`;
  document.body.appendChild(ghost);
  catalogDragState.ghostEl = ghost;
  e.dataTransfer.setDragImage(ghost, 0, 0);

  // Show drop zone overlay
  setTimeout(() => {
    $('#dropZoneOverlay').classList.remove('hidden');
  }, 0);
}

function onCatalogDragEnd(e) {
  catalogDragState.active = false;
  if (catalogDragState.ghostEl) {
    catalogDragState.ghostEl.remove();
    catalogDragState.ghostEl = null;
  }
  $('#dropZoneOverlay').classList.add('hidden');
  // Remove any SVG ghost
  const existing = document.querySelector('.catalog-drop-ghost');
  if (existing) existing.remove();
}

// Timeline drop zone handlers
(function setupTimelineDrop() {
  const container = $('#timelineContainer');

  container.addEventListener('dragover', (e) => {
    if (!catalogDragState.active) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';

    // Show a ghost block on the SVG at the hovered time
    const svg = $('#timelineSvg');
    const pt = svgPoint(svg, e.clientX, e.clientY);

    // Remove old ghost
    const old = svg.querySelector('.catalog-drop-ghost');
    if (old) old.remove();

    if (pt.x < TL.leftPad || pt.y < TL.headerH || pt.y > TL.height) return;

    const modelId = catalogDragState.modelId;
    const m = MODEL_MAP.get(modelId);
    if (!m) return;
    const gpus = m.meta?.gpus || 1;

    const hoverDate = snapDate(xToDate(pt.x));
    // Default 2h duration for preview
    const endDate = new Date(hoverDate.getTime() + 2 * 3600000);

    const x = dateToX(hoverDate);
    const w = dateToX(endDate) - x;
    const clickedLane = Math.floor((pt.y - TL.headerH) / TL.laneH);
    const laneStart = clamp(clickedLane, 0, TL.gpuTotal - gpus);
    const y = TL.headerH + laneStart * TL.laneH + 4;
    const h = gpus * TL.laneH - 8;

    const hasConflict = checkVisualConflict(hoverDate, endDate, gpus, null);

    const g = drawGroup(svg, { class: 'catalog-drop-ghost ghost-block' });
    drawRect(g, x, y, Math.max(8, w), h, {
      fill: hasConflict ? 'rgba(239, 68, 68, 0.25)' : 'rgba(14, 165, 233, 0.25)',
      stroke: hasConflict ? 'rgba(239, 68, 68, 0.8)' : 'rgba(14, 165, 233, 0.8)',
      'stroke-width': 2,
      'stroke-dasharray': '6 3',
      rx: 10,
    });
    drawText(g, x + 10, y + 18, modelId, {
      fill: hasConflict ? '#fca5a5' : '#7dd3fc',
      'font-size': 11,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
    drawText(g, x + 10, y + 34, `${fmtHour(hoverDate)} → ${fmtHour(endDate)} · ${gpus} GPUs`, {
      fill: hasConflict ? '#fca5a5' : '#7dd3fc',
      'font-size': 10,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
  });

  container.addEventListener('dragleave', (e) => {
    // Only if actually leaving the container
    if (!container.contains(e.relatedTarget)) {
      const old = document.querySelector('.catalog-drop-ghost');
      if (old) old.remove();
    }
  });

    container.addEventListener('drop', (e) => {
    e.preventDefault();
    const old = document.querySelector('.catalog-drop-ghost');
    if (old) old.remove();
    $('#dropZoneOverlay').classList.add('hidden');

    // Grab model ID from dataTransfer (reliable even if dragend already fired)
    let modelId = e.dataTransfer.getData('text/plain');
    if (!modelId && catalogDragState.active) {
      modelId = catalogDragState.modelId;
    }

    catalogDragState.active = false;
    catalogDragState.modelId = null;

    if (!modelId) return;

    const svg = $('#timelineSvg');
    const pt = svgPoint(svg, e.clientX, e.clientY);

    if (pt.x < TL.leftPad) return;

    const dropDate = snapDate(xToDate(pt.x));

    // Open booking modal pre-filled with the model and time
    openModal({ model: modelId, beginAt: dropDate, durationHours: 2 });
  });
})();

// ─── Log Viewer ─────────────────────────────────────────────────────────────
const logModalBackdrop = $('#logModalBackdrop');
let logState = {
  leaseId: null,
  activeTab: 'stdout',
  data: null,
};

function openLogModal(leaseId) {
  logState.leaseId = leaseId;
  logState.activeTab = 'stdout';

  const lease = DASH?.leases?.find(l => l.id === leaseId);
  const jobId = lease?.slurm_job_id || '?';
  $('#logModalTitle').textContent = `Slurm Logs — ${lease?.model || '?'}`;
  $('#logModalSubtitle').textContent = `Job ID: ${jobId}`;
  $('#logContent').textContent = 'Loading…';
  $('#logTruncatedBanner').classList.add('hidden');

  updateLogTabs();
  logModalBackdrop.classList.remove('hidden');
  logModalBackdrop.setAttribute('aria-hidden', 'false');

  fetchLogs(leaseId);
}

function closeLogModal() {
  logModalBackdrop.classList.add('hidden');
  logModalBackdrop.setAttribute('aria-hidden', 'true');
  logState.leaseId = null;
  logState.data = null;
}

$('#logModalClose').addEventListener('click', closeLogModal);
logModalBackdrop.addEventListener('click', (e) => {
  if (e.target === logModalBackdrop) closeLogModal();
});

$('#logRefreshBtn').addEventListener('click', () => {
  if (logState.leaseId) fetchLogs(logState.leaseId);
});

$('#logTabStdout').addEventListener('click', () => {
  logState.activeTab = 'stdout';
  updateLogTabs();
  renderLogContent();
});

$('#logTabStderr').addEventListener('click', () => {
  logState.activeTab = 'stderr';
  updateLogTabs();
  renderLogContent();
});

function updateLogTabs() {
  const stdoutTab = $('#logTabStdout');
  const stderrTab = $('#logTabStderr');
  if (logState.activeTab === 'stdout') {
    stdoutTab.classList.add('border-brand-500', 'text-brand-400');
    stdoutTab.classList.remove('border-transparent', 'text-slate-400');
    stderrTab.classList.remove('border-brand-500', 'text-brand-400');
    stderrTab.classList.add('border-transparent', 'text-slate-400');
  } else {
    stderrTab.classList.add('border-brand-500', 'text-brand-400');
    stderrTab.classList.remove('border-transparent', 'text-slate-400');
    stdoutTab.classList.remove('border-brand-500', 'text-brand-400');
    stdoutTab.classList.add('border-transparent', 'text-slate-400');
  }
}

async function fetchLogs(leaseId) {
  try {
    const data = await api(`/admin/leases/${leaseId}/logs`);
    logState.data = data;
    renderLogContent();
  } catch (e) {
    $('#logContent').textContent = `Error loading logs: ${e.message}`;
  }
}

function renderLogContent() {
  if (!logState.data) return;
  const content = logState.activeTab === 'stdout'
    ? logState.data.log_stdout
    : logState.data.log_stderr;

  $('#logContent').textContent = content || '(empty)';

  if (logState.data.truncated) {
    $('#logTruncatedBanner').classList.remove('hidden');
  } else {
    $('#logTruncatedBanner').classList.add('hidden');
  }

  // Auto-scroll to bottom
  const scrollContainer = $('#logContent').parentElement;
  scrollContainer.scrollTop = scrollContainer.scrollHeight;
}

// Global helper to open logs for a model (finds the active lease)
window.viewLogs = (modelId) => {
  const lease = DASH?.leases?.find(l =>
    l.model === modelId && ['RUNNING', 'SUBMITTED', 'STARTING'].includes(l.state) && l.slurm_job_id
  );
  if (lease) {
    openLogModal(lease.id);
  } else {
    toast('No active Slurm job found for this model', 'info');
  }
};


window.openLogModalForLease = (leaseId) => {
  openLogModal(leaseId);
};

// ─── Booking Modal ──────────────────────────────────────────────────────────
const modalBackdrop = $('#modalBackdrop');
const modalBeginAt = $('#modalBeginAt');
const modalDurationRange = $('#modalDurationRange');
const modalDurationLabel = $('#modalDurationLabel');

let modalState = {
  mode: 'create',
  leaseId: null,
  selectedStartOffset: 0,
  startDate: null,
  durationHours: 4,
  asap: false,
};

function openModal({ model = null, beginAt = null, durationHours = 4, leaseId = null } = {}) {
  modalState.mode = leaseId ? 'edit' : 'create';
  modalState.leaseId = leaseId;
  modalState.durationHours = durationHours;
  modalState.asap = false;

  $('#modalTitle').textContent = leaseId ? 'Edit Booking' : 'New Booking';
  $('#modalSubtitle').textContent = leaseId ? 'Modify a planned booking' : 'Schedule a model on the GPUs';
  $('#modalSaveText').textContent = leaseId ? 'Update Booking' : 'Create Booking';
  $('#modalError').classList.add('hidden');

  // Populate notes from existing lease if editing
  const existingLease = leaseId ? DASH?.leases?.find(l => l.id === leaseId) : null;
  $('#modalNotes').value = existingLease?.notes || '';

  populateModalModels(model);

  // Duration
  modalDurationRange.value = durationHours;
  modalDurationLabel.textContent = `${durationHours}h`;
  updateDurationButtons(durationHours);

  // Start time
  if (beginAt) {
    modalState.startDate = beginAt;
    modalState.selectedStartOffset = 'custom';
    modalBeginAt.value = toLocalDTInput(beginAt);
    modalBeginAt.classList.remove('hidden');
    updateQuickTimeButtons('custom');
    $('#modalStartPreview').textContent = `Starting: ${fmtTime(beginAt)}`;
  } else {
    modalState.selectedStartOffset = 0;
    modalState.startDate = null;
    modalBeginAt.classList.add('hidden');
    updateQuickTimeButtons('0');
    $('#modalStartPreview').textContent = 'Starting: Now';
  }

  modalBackdrop.classList.remove('hidden');
  modalBackdrop.setAttribute('aria-hidden', 'false');
}

function closeModal() {
  modalBackdrop.classList.add('hidden');
  modalBackdrop.setAttribute('aria-hidden', 'true');
  EDITING_LEASE_ID = null;
}

$('#modalClose').addEventListener('click', closeModal);
$('#modalCancel').addEventListener('click', closeModal);
modalBackdrop.addEventListener('click', (e) => {
  if (e.target === modalBackdrop) closeModal();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeModal();
    closeLogModal();
    hideBlockPopover();
  }
});

function populateModalModels(selected = null) {
  if (!DASH) return;
  const sel = $('#modalModel');
  const models = DASH.models.slice().sort((a, b) => a.id.localeCompare(b.id));
  sel.innerHTML = models.map(m =>
    `<option value="${m.id}">${escapeHtml(m.id)}</option>`
  ).join('');
  if (selected) {
    sel.value = selected;
  }
  // Trigger change to update meta display
  sel.dispatchEvent(new Event('change'));
}

$('#modalModel').addEventListener('change', () => {
  const id = $('#modalModel').value;
  const m = MODEL_MAP.get(id);
  if (!m) return;
  const meta = m.meta || {};
  $('#modalModelMeta').textContent = `${meta.notes || ''}`;
  $('#modalGpuInfo').textContent = `Requires ${meta.gpus} GPUs · TP ${meta.tensor_parallel_size}`;
});

// Quick time buttons (including ASAP)
$$('#quickTimeButtons .qt-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const offset = btn.dataset.offset;
    modalState.selectedStartOffset = offset;
    modalState.asap = (offset === 'asap');
    updateQuickTimeButtons(offset);

    if (offset === 'asap') {
      modalBeginAt.classList.add('hidden');
      modalState.startDate = null;
      $('#modalStartPreview').textContent = 'Starting: As soon as possible (auto-find earliest slot)';
    } else if (offset === 'custom') {
      modalBeginAt.classList.remove('hidden');
      const now = new Date();
      now.setMinutes(Math.ceil(now.getMinutes() / 15) * 15, 0, 0);
      modalBeginAt.value = toLocalDTInput(now);
      modalState.startDate = now;
      $('#modalStartPreview').textContent = `Starting: ${fmtTime(now)}`;
    } else if (offset === 'tomorrow9') {
      modalBeginAt.classList.add('hidden');
      const d = new Date();
      d.setDate(d.getDate() + 1);
      d.setHours(9, 0, 0, 0);
      modalState.startDate = d;
      $('#modalStartPreview').textContent = `Starting: ${fmtTime(d)}`;
    } else {
      modalBeginAt.classList.add('hidden');
      const mins = parseInt(offset, 10);
      if (mins === 0) {
        modalState.startDate = null;
        $('#modalStartPreview').textContent = 'Starting: Now';
      } else {
        const d = new Date(Date.now() + mins * 60000);
        d.setMinutes(Math.round(d.getMinutes() / 15) * 15, 0, 0);
        modalState.startDate = d;
        $('#modalStartPreview').textContent = `Starting: ${fmtTime(d)}`;
      }
    }
  });
});

function updateQuickTimeButtons(activeOffset) {
  $$('#quickTimeButtons .qt-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.offset === String(activeOffset));
  });
}

modalBeginAt.addEventListener('change', () => {
  const v = modalBeginAt.value;
  if (v) {
    modalState.startDate = new Date(v);
    $('#modalStartPreview').textContent = `Starting: ${fmtTime(modalState.startDate)}`;
  }
});

// Duration buttons
$$('#durationButtons .dur-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const h = parseInt(btn.dataset.hours, 10);
    modalState.durationHours = h;
    modalDurationRange.value = h;
    modalDurationLabel.textContent = `${h}h`;
    updateDurationButtons(h);
  });
});

function updateDurationButtons(hours) {
  $$('#durationButtons .dur-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.hours, 10) === hours);
  });
}

modalDurationRange.addEventListener('input', () => {
  const h = parseInt(modalDurationRange.value, 10);
  modalState.durationHours = h;
  modalDurationLabel.textContent = `${h}h`;
  updateDurationButtons(h);
});

// Save
$('#modalSave').addEventListener('click', async () => {
  try {
    $('#modalError').classList.add('hidden');
    const model = $('#modalModel').value;
    if (!model) { showModalError("Please choose a model."); return; }

    const durationHours = modalState.durationHours;
    if (durationHours < 1) { showModalError("Duration must be at least 1 hour."); return; }

    if (modalState.mode === 'create') {
      const payload = {
        model,
        duration_seconds: durationHours * 3600,
        notes: $('#modalNotes').value.trim() || null,
      };

      if (modalState.asap) {
        payload.asap = true;
        // begin_at is null — server finds earliest slot
      } else {
        payload.begin_at = modalState.startDate ? modalState.startDate.toISOString() : null;
      }

      await api("/admin/leases", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      toast(`Booking created for ${model}`, 'success');
    } else {
      // Edit (PATCH)
      const begin = modalState.startDate || new Date();
      const end = new Date(begin.getTime() + durationHours * 3600000);
      await api(`/admin/leases/${modalState.leaseId}`, {
        method: "PATCH",
        body: JSON.stringify({
          begin_at: begin.toISOString(),
          end_at: end.toISOString(),
          notes: $('#modalNotes').value.trim() || null,
        })
      });
      toast('Booking updated', 'success');
    }
    closeModal();
    await refresh();
  } catch (e) {
    showModalError(e.message);
  }
});

function showModalError(msg) {
  const el = $('#modalError');
  el.textContent = msg;
  el.classList.remove('hidden');
}

// ─── Block Popover (with stop, shorten, logs) ───────────────────────────────
const blockPopover = $('#blockPopover');
let popoverLeaseId = null;

function showBlockPopover(lease, anchorX, anchorY) {
  popoverLeaseId = lease.id;

  const b = new Date(lease.begin_at || lease.created_at);
  const e = new Date(lease.end_at);
  const durationSec = (e - b) / 1000;
  const isFailed = lease.state === 'FAILED';

  $('#popoverModel').textContent = lease.model;
  $('#popoverTime').innerHTML = `${fmtTime(b)} → ${fmtTime(e)} (${formatDuration(durationSec)})` +
    (lease.notes ? `<div class="mt-1.5 text-xs text-slate-400 italic">📝 ${escapeHtml(lease.notes)}</div>` : '');

  const epStats = (DASH?.endpoint_stats || []).find(s => s.model === lease.model && s.state === 'READY');
  if (epStats) {
    const uptime = epStats.uptime_seconds ? formatDuration(Math.floor(epStats.uptime_seconds)) : '—';
    $('#popoverStats').innerHTML = `
      <div class="flex items-center gap-3 text-xs text-slate-400">
        <span>⏱ ${uptime}</span>
        <span>📡 ${epStats.host}:${epStats.port}</span>
      </div>
    `;
  } else {
    $('#popoverStats').innerHTML = '';
  }

  const stateColors = {
    RUNNING: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30',
    SUBMITTED: 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30',
    STARTING: 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30',
    PLANNED: 'bg-sky-500/15 text-sky-700 dark:text-sky-300 border-sky-500/30',
    FAILED: 'bg-red-500/15 text-red-700 dark:text-red-300 border-red-500/30',
  };
  const colorClass = stateColors[lease.state] || 'bg-slate-500/15 text-slate-600 dark:text-slate-300 border-slate-500/30';
  $('#popoverStatus').innerHTML = `
    <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs border ${colorClass}">
      ${lease.state}${lease.conflict ? ' · CONFLICT' : ''}
    </span>
  `;

  const actionsEl = $('#popoverActions');
  actionsEl.innerHTML = '';

  if (isFailed) {
    if (lease.slurm_job_id) {
      const logBtn = document.createElement('button');
      logBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-600/20 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-500/30 hover:bg-slate-200 dark:hover:bg-slate-600/30 transition font-medium flex items-center justify-center gap-1.5';
      logBtn.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg> View Logs`;
      logBtn.addEventListener('click', () => {
        hideBlockPopover();
        openLogModal(lease.id);
      });
      actionsEl.appendChild(logBtn);
    }

    const dismissBtn = document.createElement('button');
    dismissBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-red-50 dark:bg-red-600/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-500/30 hover:bg-red-100 dark:hover:bg-red-600/30 transition font-medium';
    dismissBtn.textContent = 'Dismiss Failed Booking';
    dismissBtn.addEventListener('click', async () => {
      try {
        await api(`/admin/leases/${lease.id}`, { method: "DELETE" });
        toast('Failed booking dismissed', 'success');
        hideBlockPopover();
        await refresh();
      } catch (err) {
        toast(err.message, 'error');
      }
    });
    actionsEl.appendChild(dismissBtn);

  } else {
    // Extend buttons — now includes STARTING
    if (['RUNNING', 'SUBMITTED', 'STARTING', 'PLANNED'].includes(lease.state)) {
      const extendRow = document.createElement('div');
      extendRow.className = 'flex gap-2';
      [
        { label: '+1h', secs: 3600 },
        { label: '+2h', secs: 7200 },
        { label: '+4h', secs: 14400 },
      ].forEach(({ label, secs }) => {
        const btn = document.createElement('button');
        btn.className = 'flex-1 px-2 py-1.5 text-xs rounded-lg bg-emerald-50 dark:bg-emerald-600/20 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30 hover:bg-emerald-100 dark:hover:bg-emerald-600/30 transition font-medium';
        btn.textContent = label;
        btn.addEventListener('click', async () => {
          try {
            await api(`/admin/leases/${lease.id}/extend`, {
              method: "POST",
              body: JSON.stringify({ duration_seconds: secs })
            });
            toast(`Extended by ${label}`, 'success');
            hideBlockPopover();
            await refresh();
          } catch (err) {
            toast(err.message, 'error');
          }
        });
        extendRow.appendChild(btn);
      });
      actionsEl.appendChild(extendRow);
    }

    // Shorten — now includes STARTING
    if (['RUNNING', 'SUBMITTED', 'STARTING'].includes(lease.state)) {
      const shortenRow = document.createElement('div');
      shortenRow.className = 'flex gap-2';

      const now = new Date();
      const currentEnd = new Date(lease.end_at);
      const remainingSec = (currentEnd - now) / 1000;

      const shortenOptions = [];
      if (remainingSec > 1800 + 300) {
        shortenOptions.push({ label: 'End in 30m', newEnd: new Date(now.getTime() + 30 * 60000) });
      }
      if (remainingSec > 3600 + 300) {
        shortenOptions.push({ label: 'End in 1h', newEnd: new Date(now.getTime() + 3600000) });
      }
      if (remainingSec > 7200 + 300) {
        shortenOptions.push({ label: '-2h', newEnd: new Date(currentEnd.getTime() - 7200000) });
      }

      shortenOptions.forEach(({ label, newEnd }) => {
        const btn = document.createElement('button');
        btn.className = 'flex-1 px-2 py-1.5 text-xs rounded-lg bg-amber-50 dark:bg-amber-600/20 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-500/30 hover:bg-amber-100 dark:hover:bg-amber-600/30 transition font-medium';
        btn.textContent = label;
        btn.addEventListener('click', async () => {
          try {
            await api(`/admin/leases/${lease.id}/shorten`, {
              method: "POST",
              body: JSON.stringify({ new_end_at: newEnd.toISOString() })
            });
            toast(`Shortened: new end ${fmtTime(newEnd)}`, 'success');
            hideBlockPopover();
            await refresh();
          } catch (err) {
            toast(err.message, 'error');
          }
        });
        shortenRow.appendChild(btn);
      });

      if (shortenOptions.length > 0) {
        const label = document.createElement('div');
        label.className = 'text-xs text-slate-500 mb-1';
        label.textContent = 'Shorten:';
        actionsEl.appendChild(label);
        actionsEl.appendChild(shortenRow);
      }
    }

    // Edit (PLANNED only)
    if (lease.state === 'PLANNED') {
      const editBtn = document.createElement('button');
      editBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-sky-50 dark:bg-brand-600/20 text-sky-700 dark:text-brand-300 border border-sky-300 dark:border-brand-500/30 hover:bg-sky-100 dark:hover:bg-brand-600/30 transition font-medium';
      editBtn.textContent = 'Edit Booking';
      editBtn.addEventListener('click', () => {
        hideBlockPopover();
        const hours = Math.max(1, Math.round((e - b) / 3600000));
        openModal({ model: lease.model, beginAt: b, durationHours: hours, leaseId: lease.id });
      });
      actionsEl.appendChild(editBtn);
    }

    // Edit notes (any active state)
    if (['RUNNING', 'SUBMITTED', 'STARTING'].includes(lease.state)) {
      const notesBtn = document.createElement('button');
      notesBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-gray-100 dark:bg-slate-600/20 text-gray-700 dark:text-slate-300 border border-gray-300 dark:border-slate-500/30 hover:bg-gray-200 dark:hover:bg-slate-600/30 transition font-medium';
      notesBtn.textContent = '📝 Edit Notes';
      notesBtn.addEventListener('click', async () => {
        const newNotes = prompt('Booking notes:', lease.notes || '');
        if (newNotes === null) return; // cancelled
        try {
          await api(`/admin/leases/${lease.id}/notes`, {
            method: "PATCH",
            body: JSON.stringify({ notes: newNotes })
          });
          toast('Notes updated', 'success');
          hideBlockPopover();
          await refresh();
        } catch (err) {
          toast(err.message, 'error');
        }
      });
      actionsEl.appendChild(notesBtn);
    }

    // Logs (if has slurm job)
    if (lease.slurm_job_id) {

      const logBtn = document.createElement('button');
      logBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-gray-100 dark:bg-slate-600/20 text-gray-700 dark:text-slate-300 border border-gray-300 dark:border-slate-500/30 hover:bg-gray-200 dark:hover:bg-slate-600/30 transition font-medium flex items-center justify-center gap-1.5';
      logBtn.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg> View Logs`;
      logBtn.addEventListener('click', () => {
        hideBlockPopover();
        openLogModal(lease.id);
      });
      actionsEl.appendChild(logBtn);
    }

    // Stop Now — now includes STARTING
    if (['RUNNING', 'SUBMITTED', 'STARTING'].includes(lease.state)) {
      const stopNowBtn = document.createElement('button');
      stopNowBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-red-50 dark:bg-red-600/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-500/30 hover:bg-red-100 dark:hover:bg-red-600/30 transition font-medium';
      stopNowBtn.textContent = '⏹ Stop Now';
      stopNowBtn.addEventListener('click', async () => {
        if (!confirm(`Stop ${lease.model} immediately? This will cancel the Slurm job.`)) return;
        try {
          await api(`/admin/leases/${lease.id}/stop`, { method: "POST" });
          toast('Model stopped', 'success');
          hideBlockPopover();
          await refresh();
        } catch (err) {
          toast(err.message, 'error');
        }
      });
      actionsEl.appendChild(stopNowBtn);
    }

    // Cancel/Remove (for PLANNED)
    if (lease.state === 'PLANNED') {
      const removeBtn = document.createElement('button');
      removeBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-red-50 dark:bg-red-600/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-500/30 hover:bg-red-100 dark:hover:bg-red-600/30 transition font-medium';
      removeBtn.textContent = 'Remove Booking';
      removeBtn.addEventListener('click', async () => {
        try {
          await api(`/admin/leases/${lease.id}`, { method: "DELETE" });
          toast('Booking removed', 'success');
          hideBlockPopover();
          await refresh();
        } catch (err) {
          toast(err.message, 'error');
        }
      });
      actionsEl.appendChild(removeBtn);
    }
  }

  // Position
  const popW = 320;
  const popH = actionsEl.childElementCount * 44 + 160;
  let left = anchorX + 8;
  let top = anchorY - 20;

  if (left + popW > window.innerWidth - 16) left = anchorX - popW - 8;
  if (top + popH > window.innerHeight - 16) top = window.innerHeight - popH - 16;
  if (top < 8) top = 8;

  blockPopover.style.left = `${left}px`;
  blockPopover.style.top = `${top}px`;
  blockPopover.classList.remove('hidden');
}

function hideBlockPopover() {
  blockPopover.classList.add('hidden');
  popoverLeaseId = null;
}

document.addEventListener('mousedown', (e) => {
  if (!blockPopover.classList.contains('hidden') && !blockPopover.contains(e.target)) {
    hideBlockPopover();
  }
});

// ─── Drag State Machine ─────────────────────────────────────────────────────
const dragState = {
  active: false,
  type: null,
  leaseId: null,
  lease: null,
  startX: 0,
  startY: 0,
  currentX: 0,
  origBegin: null,
  origEnd: null,
  ghostEl: null,
  snapLines: [],
  committed: false,
};

const DRAG_THRESHOLD = 5;
const SNAP_MINUTES = 15;

function snapDate(d) {
  const snapped = new Date(d);
  snapped.setMinutes(Math.round(snapped.getMinutes() / SNAP_MINUTES) * SNAP_MINUTES, 0, 0);
  return snapped;
}

function initDragHandlers() {
  const svg = $('#timelineSvg');
  svg.addEventListener('pointerdown', onPointerDown);
  document.addEventListener('pointermove', onPointerMove);
  document.addEventListener('pointerup', onPointerUp);
}

function onPointerDown(e) {
  // Only left button
  if (e.button !== 0) return;

  const svg = $('#timelineSvg');
  const pt = svgPoint(svg, e.clientX, e.clientY);

  // Check if we hit a block handle or body
  const target = e.target;
  const blockGroup = target.closest?.('[data-lease-id]') || (target.dataset?.leaseId ? target : null);

  if (blockGroup) {
    const leaseId = parseInt(blockGroup.dataset?.leaseId || target.dataset?.leaseId, 10);
    const lease = DASH.leases.find(l => l.id === leaseId);
    if (!lease) return;

    // Determine drag type based on what was clicked
    const handleType = target.dataset?.handle;

    if (handleType === 'left' && lease.state === 'PLANNED') {
      startDrag(e, 'resize-left', lease, pt);
    } else if (handleType === 'right') {
      startDrag(e, 'resize-right', lease, pt);
    } else if (lease.state === 'PLANNED' && !handleType) {
      startDrag(e, 'move', lease, pt);
    }
    // For non-PLANNED, clicking body opens popover (handled in pointerUp)
    return;
  }

  // Click on empty space → drag to create
  if (pt.x > TL.leftPad && pt.y > TL.headerH && pt.y < TL.height) {
    startDrag(e, 'create', null, pt);
  }
}

function startDrag(e, type, lease, pt) {
  e.preventDefault();
  const svg = $('#timelineSvg');
  svg.setPointerCapture?.(e.pointerId);

  dragState.active = true;
  dragState.type = type;
  dragState.lease = lease;
  dragState.leaseId = lease?.id || null;
  dragState.startX = pt.x;
  dragState.startY = pt.y;
  dragState.currentX = pt.x;
  dragState.committed = false;

  if (lease) {
    dragState.origBegin = new Date(lease.begin_at || lease.created_at);
    dragState.origEnd = new Date(lease.end_at);
  } else {
    dragState.origBegin = snapDate(xToDate(pt.x));
    dragState.origEnd = snapDate(xToDate(pt.x));
  }

  document.body.style.cursor =
    type === 'create' ? 'crosshair' :
    type.startsWith('resize') ? 'ew-resize' : 'grabbing';
}

function onPointerMove(e) {
  if (!dragState.active) return;

  const svg = $('#timelineSvg');
  const pt = svgPoint(svg, e.clientX, e.clientY);
  dragState.currentX = pt.x;

  const dx = Math.abs(pt.x - dragState.startX);
  if (!dragState.committed && dx < DRAG_THRESHOLD) return;
  dragState.committed = true;

  // Hide popover during drag
  hideBlockPopover();

  updateDragGhost(pt);
  updateDragTooltip(e.clientX, e.clientY);
}

function onPointerUp(e) {
  if (!dragState.active) return;

  const svg = $('#timelineSvg');
  svg.releasePointerCapture?.(e.pointerId);

  document.body.style.cursor = '';
  hideTimelineTooltip();

  if (!dragState.committed) {
    // It was a click, not a drag
    const pt = svgPoint(svg, e.clientX, e.clientY);

    if (dragState.type === 'create') {
      // Click on empty space → open modal with time
      const clickedDate = snapDate(xToDate(pt.x));
      if (clickedDate >= TL.start) {
        openModal({ beginAt: clickedDate });
      }
    } else if (dragState.lease) {
      // Click on block → show popover
      showBlockPopover(dragState.lease, e.clientX, e.clientY);
    }

    resetDrag();
    return;
  }

  // Commit the drag operation
  commitDrag();
}

function updateDragGhost(pt) {
  const svg = $('#timelineSvg');

  // Remove old ghost
  if (dragState.ghostEl) {
    dragState.ghostEl.remove();
    dragState.ghostEl = null;
  }

  let ghostBegin, ghostEnd, laneStart, laneCount;

  if (dragState.type === 'create') {
    const d1 = snapDate(xToDate(dragState.startX));
    const d2 = snapDate(xToDate(pt.x));
    ghostBegin = new Date(Math.min(d1, d2));
    ghostEnd = new Date(Math.max(d1, d2));
    if (ghostEnd - ghostBegin < 15 * 60000) {
      ghostEnd = new Date(ghostBegin.getTime() + 15 * 60000);
    }
    const clickedLane = Math.floor((pt.y - TL.headerH) / TL.laneH);
    const defaultGpus = DASH?.models?.[0]?.meta?.gpus || 1;
    laneStart = clamp(clickedLane, 0, TL.gpuTotal - defaultGpus);
    laneCount = defaultGpus;

  } else if (dragState.type === 'resize-right') {
    ghostBegin = dragState.origBegin;
    ghostEnd = snapDate(xToDate(pt.x));
    if (ghostEnd <= ghostBegin) ghostEnd = new Date(ghostBegin.getTime() + 15 * 60000);
    laneStart = dragState.lease.lane_start ?? 0;
    laneCount = dragState.lease.lane_count ?? dragState.lease.requested_gpus ?? 1;

  } else if (dragState.type === 'resize-left') {
    ghostBegin = snapDate(xToDate(pt.x));
    ghostEnd = dragState.origEnd;
    if (ghostBegin >= ghostEnd) ghostBegin = new Date(ghostEnd.getTime() - 15 * 60000);
    laneStart = dragState.lease.lane_start ?? 0;
    laneCount = dragState.lease.lane_count ?? dragState.lease.requested_gpus ?? 1;

  } else if (dragState.type === 'move') {
    const deltaMs = ((pt.x - dragState.startX) / TL.pxPerHour) * 3600000;
    ghostBegin = snapDate(new Date(dragState.origBegin.getTime() + deltaMs));
    ghostEnd = snapDate(new Date(dragState.origEnd.getTime() + deltaMs));
    laneStart = dragState.lease.lane_start ?? 0;
    laneCount = dragState.lease.lane_count ?? dragState.lease.requested_gpus ?? 1;
  }

  // Draw ghost
  const x = dateToX(ghostBegin);
  const w = dateToX(ghostEnd) - x;
  const y = TL.headerH + laneStart * TL.laneH + 4;
  const h = laneCount * TL.laneH - 8;

  const hasConflict = checkVisualConflict(ghostBegin, ghostEnd, laneCount, dragState.leaseId);

  const ghost = drawGroup(svg, { class: 'ghost-block' });
  drawRect(ghost, x, y, Math.max(8, w), h, {
    fill: hasConflict ? 'rgba(239, 68, 68, 0.25)' : 'rgba(14, 165, 233, 0.25)',
    stroke: hasConflict ? 'rgba(239, 68, 68, 0.8)' : 'rgba(14, 165, 233, 0.8)',
    'stroke-width': 2,
    'stroke-dasharray': '6 3',
    rx: 10,
  });

  const model = dragState.lease?.model || (DASH?.models?.[0]?.id || 'New');
  drawText(ghost, x + 10, y + 20, `${model}`, {
    fill: hasConflict ? '#fca5a5' : '#7dd3fc',
    'font-size': 11,
    'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
  });
  drawText(ghost, x + 10, y + 36, `${fmtHour(ghostBegin)} → ${fmtHour(ghostEnd)}`, {
    fill: hasConflict ? '#fca5a5' : '#7dd3fc',
    'font-size': 10,
    'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
  });

  dragState.ghostEl = ghost;
  dragState._ghostBegin = ghostBegin;
  dragState._ghostEnd = ghostEnd;
  dragState._hasConflict = hasConflict;
}

function checkVisualConflict(begin, end, gpusNeeded, excludeLeaseId = null) {
  if (!DASH) return false;

  const activeLeases = DASH.leases.filter(l =>
    ['PLANNED', 'SUBMITTED', 'STARTING', 'RUNNING'].includes(l.state) && l.id !== excludeLeaseId
  );

  const step = 15 * 60000;
  for (let t = begin.getTime(); t < end.getTime(); t += step) {
    const moment = new Date(t);
    let usedGpus = 0;
    for (const l of activeLeases) {
      const lb = new Date(l.begin_at || l.created_at);
      const le = new Date(l.end_at);
      if (moment >= lb && moment < le) {
        usedGpus += l.requested_gpus || 0;
      }
    }
    if (usedGpus + gpusNeeded > TL.gpuTotal) {
      return true;
    }
  }
  return false;
}

function updateDragTooltip(clientX, clientY) {
  const tip = $('#timelineTooltip');
  if (!dragState.committed || !dragState._ghostBegin || !dragState._ghostEnd) {
    tip.classList.add('hidden');
    return;
  }

  const begin = dragState._ghostBegin;
  const end = dragState._ghostEnd;
  const durationMin = Math.round((end - begin) / 60000);
  const hours = Math.floor(durationMin / 60);
  const mins = durationMin % 60;

  let text = `${fmtHour(begin)} → ${fmtHour(end)} (${hours}h${mins > 0 ? ` ${mins}m` : ''})`;
  if (dragState._hasConflict) {
    text += ' ⚠ CONFLICT';
  }

  tip.textContent = text;
  tip.classList.remove('hidden');

  const container = $('#timelineContainer');
  const rect = container.getBoundingClientRect();
  tip.style.left = `${clientX - rect.left + 12}px`;
  tip.style.top = `${clientY - rect.top - 36}px`;
}

function hideTimelineTooltip() {
  $('#timelineTooltip').classList.add('hidden');
}

async function commitDrag() {
  const begin = dragState._ghostBegin;
  const end = dragState._ghostEnd;
  const hasConflict = dragState._hasConflict;

  if (hasConflict) {
    toast('Cannot place booking — GPU conflict detected', 'error');
    resetDrag();
    return;
  }

  try {
    if (dragState.type === 'create') {
      const durationHours = Math.max(1, Math.round((end - begin) / 3600000));
      openModal({ beginAt: begin, durationHours });
      resetDrag();
      return;
    }

    if (dragState.type === 'resize-right') {
      const lease = dragState.lease;
      const origEnd = dragState.origEnd;
      const diffSeconds = Math.round((end - origEnd) / 1000);

      if (Math.abs(diffSeconds) < 60) {
        resetDrag();
        return;
      }

      if (lease.state === 'PLANNED') {
        await api(`/admin/leases/${lease.id}`, {
          method: "PATCH",
          body: JSON.stringify({ end_at: end.toISOString() })
        });
        toast('Booking resized', 'success');
      } else {
        // RUNNING/SUBMITTED — use extend or shorten
        if (diffSeconds > 0) {
          await api(`/admin/leases/${lease.id}/extend`, {
            method: "POST",
            body: JSON.stringify({ duration_seconds: diffSeconds })
          });
          toast(`Extended by ${Math.round(diffSeconds / 60)}m`, 'success');
        } else {
          // Shorten via new endpoint
          await api(`/admin/leases/${lease.id}/shorten`, {
            method: "POST",
            body: JSON.stringify({ new_end_at: end.toISOString() })
          });
          toast(`Shortened to ${fmtTime(end)}`, 'success');
        }
      }
    }

    if (dragState.type === 'resize-left') {
      const lease = dragState.lease;
      await api(`/admin/leases/${lease.id}`, {
        method: "PATCH",
        body: JSON.stringify({ begin_at: begin.toISOString() })
      });
      toast('Booking start moved', 'success');
    }

    if (dragState.type === 'move') {
      const lease = dragState.lease;
      await api(`/admin/leases/${lease.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          begin_at: begin.toISOString(),
          end_at: end.toISOString(),
        })
      });
      toast('Booking moved', 'success');
    }

    await refresh();
  } catch (e) {
    toast(e.message, 'error');
  }

  resetDrag();
}

function resetDrag() {
  if (dragState.ghostEl) {
    dragState.ghostEl.remove();
    dragState.ghostEl = null;
  }
  dragState.active = false;
  dragState.type = null;
  dragState.lease = null;
  dragState.leaseId = null;
  dragState.committed = false;
  dragState._ghostBegin = null;
  dragState._ghostEnd = null;
  dragState._hasConflict = false;
  document.body.style.cursor = '';
  hideTimelineTooltip();
}

// Helper for catalog drag ghost conflict detection
function snapDate(d) {
  // Already defined above but needed for catalog drag context
  const snapped = new Date(d);
  snapped.setMinutes(Math.round(snapped.getMinutes() / SNAP_MINUTES) * SNAP_MINUTES, 0, 0);
  return snapped;
}

// ─── Timeline Rendering ─────────────────────────────────────────────────────
function renderTimeline() {
  const svg = $('#timelineSvg');
  const hours = parseInt($('#windowSelect').value, 10);
  const pxPerHour = parseInt($('#zoomSelect').value, 10);
  const gpuTotal = DASH?.total_gpus || 8;

  TL.hours = hours;
  TL.pxPerHour = pxPerHour;
  TL.gpuTotal = gpuTotal;
  TL.width = TL.leftPad + (hours * pxPerHour);
  TL.height = TL.headerH + gpuTotal * TL.laneH + 16;

  const now = new Date(DASH.now);
  TL.start = new Date(now);
  TL.start.setMinutes(0, 0, 0);
  TL.start.setHours(TL.start.getHours() - 1);
  TL.end = new Date(TL.start.getTime() + hours * 3600000);

  svg.setAttribute('width', TL.width);
  svg.setAttribute('height', TL.height);
  svg.innerHTML = '';

  const dark = isDark();

  drawRect(svg, 0, 0, TL.width, TL.height, { fill: dark ? '#0b1220' : '#ffffff' });

  const defs = svgEl('defs');
  const filter = svgEl('filter', { id: 'glow', x: '-50%', y: '-50%', width: '200%', height: '200%' });
  const blur = svgEl('feGaussianBlur', { stdDeviation: '3', result: 'coloredBlur' });
  filter.appendChild(blur);
  const merge = svgEl('feMerge');
  const mn1 = svgEl('feMergeNode', { in: 'coloredBlur' });
  const mn2 = svgEl('feMergeNode', { in: 'SourceGraphic' });
  merge.appendChild(mn1);
  merge.appendChild(mn2);
  filter.appendChild(merge);
  defs.appendChild(filter);
  svg.appendChild(defs);

  for (let h = 0; h <= hours; h++) {
    const x = TL.leftPad + h * pxPerHour;
    drawLine(svg, x, 0, x, TL.height, {
      stroke: dark ? '#334155' : '#94a3b8',
      'stroke-opacity': 0.2,
    });

    const t = new Date(TL.start.getTime() + h * 3600000);
    const label = `${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}`;
    drawText(svg, x + 4, 22, label, {
      fill: dark ? '#94a3b8' : '#64748b',
      'font-size': 12,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
  }

  for (let h = 0; h < hours; h++) {
    const x = TL.leftPad + (h + 0.5) * pxPerHour;
    drawLine(svg, x, TL.headerH, x, TL.height, {
      stroke: dark ? '#334155' : '#94a3b8',
      'stroke-opacity': 0.08,
    });
  }

  for (let i = 0; i < gpuTotal; i++) {
    const y = TL.headerH + i * TL.laneH;
    drawLine(svg, 0, y, TL.width, y, {
      stroke: dark ? '#334155' : '#94a3b8',
      'stroke-opacity': 0.15,
    });
    drawText(svg, 10, y + 28, `GPU ${i}`, {
      fill: dark ? '#cbd5e1' : '#334155',
      'font-size': 12,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
  }
  drawLine(svg, 0, TL.headerH + gpuTotal * TL.laneH, TL.width, TL.headerH + gpuTotal * TL.laneH, {
    stroke: dark ? '#334155' : '#94a3b8',
    'stroke-opacity': 0.15,
  });

  const nowDate = new Date(DASH.now);
  const leases = (DASH?.leases || [])
    .filter(l =>
      ['PLANNED', 'SUBMITTED', 'STARTING', 'RUNNING'].includes(l.state) ||
      (l.state === 'FAILED' && l.end_at && new Date(l.end_at) > nowDate)
    )
    .sort((a, b) => new Date(a.begin_at || a.created_at) - new Date(b.begin_at || b.created_at));

  for (const l of leases) {
    drawLeaseBlock(svg, l);
  }

  const nowX = dateToX(now);
  drawLine(svg, nowX, 0, nowX, TL.height, {
    stroke: '#0ea5e9',
    'stroke-width': 2,
    class: 'now-line-pulse',
  });

  drawText(svg, nowX + 4, 12, 'now', {
    fill: '#0ea5e9',
    'font-size': 10,
    'font-weight': 'bold',
    'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
  });

  const container = $('#timelineContainer');
  const scrollTarget = nowX - 200;
  if (scrollTarget > 0 && !dragState.active && !catalogDragState.active) {
    container.scrollLeft = scrollTarget;
  }
}

function drawLeaseBlock(svg, l) {
  const b = new Date(l.begin_at || l.created_at);
  const e = new Date(l.end_at);
  if (e <= TL.start || b >= TL.end) return;

  const x = dateToX(b);
  const w = dateToX(e) - x;
  const laneStart = l.lane_start ?? 0;
  const laneCount = l.lane_count ?? l.requested_gpus ?? 1;
  const y = TL.headerH + laneStart * TL.laneH + 4;
  const h = laneCount * TL.laneH - 8;

  const isRunning = l.state === 'RUNNING';
  const isSubmitted = l.state === 'SUBMITTED';
  const isStarting = l.state === 'STARTING';
  const isPlanned = l.state === 'PLANNED';
  const isFailed = l.state === 'FAILED';
  const isConflict = !!l.conflict;

  const dark = isDark();

  let fill, stroke, dash;
  if (isFailed) {
    fill = 'rgba(239, 68, 68, 0.15)';
    stroke = 'rgba(239, 68, 68, 0.7)';
    dash = '4 4';
  } else if (isConflict) {
    fill = 'rgba(239, 68, 68, 0.18)';
    stroke = 'rgba(239, 68, 68, 0.85)';
    dash = null;
  } else if (isRunning) {
    fill = 'rgba(16, 185, 129, 0.18)';
    stroke = 'rgba(16, 185, 129, 0.85)';
    dash = null;
  } else if (isSubmitted || isStarting) {
    fill = 'rgba(245, 158, 11, 0.18)';
    stroke = 'rgba(245, 158, 11, 0.85)';
    dash = null;
  } else {
    fill = 'rgba(14, 165, 233, 0.16)';
    stroke = 'rgba(14, 165, 233, 0.75)';
    dash = '6 4';
  }

  const blockW = Math.max(8, w);

  const g = drawGroup(svg, { 'data-lease-id': l.id, cursor: 'pointer' });

  const stateLabel = isFailed ? 'FAILED ✕' : `${l.state}${isConflict ? ' ⚠' : ''}`;
  const tooltipText = `${l.model} · ${l.requested_gpus} GPU${l.requested_gpus > 1 ? 's' : ''}\n${fmtHour(b)} → ${fmtHour(e)} · ${stateLabel}`;
  const titleEl = svgEl('title');
  titleEl.textContent = tooltipText;
  g.appendChild(titleEl);

  drawRect(g, x, y, blockW, h, {
    fill, stroke, 'stroke-width': 2,
    ...(dash ? { 'stroke-dasharray': dash } : {}),
    rx: 10,
    'data-lease-id': l.id,
  });

  if (isFailed && w > 20) {
    const clipId = `clip-failed-${l.id}`;
    const defsEl = svg.querySelector('defs');
    if (defsEl) {
      const clipPath = svgEl('clipPath', { id: clipId });
      clipPath.appendChild(svgEl('rect', { x, y, width: blockW, height: h, rx: 10 }));
      defsEl.appendChild(clipPath);

      const hatchGroup = drawGroup(g, { 'clip-path': `url(#${clipId})`, 'pointer-events': 'none' });
      const step = 16;
      for (let i = -Math.ceil(h / step); i < Math.ceil(w / step) + Math.ceil(h / step); i++) {
        drawLine(hatchGroup, x + i * step, y, x + i * step + h, y + h, {
          stroke: 'rgba(239, 68, 68, 0.15)',
          'stroke-width': 2,
        });
      }
    }
  }

  const textClipId = `clip-text-${l.id}`;
  const defsEl = svg.querySelector('defs');
  if (defsEl) {
    const textClip = svgEl('clipPath', { id: textClipId });
    textClip.appendChild(svgEl('rect', {
      x: x + 6,
      y: y,
      width: Math.max(0, blockW - 12),
      height: h,
      rx: 6,
    }));
    defsEl.appendChild(textClip);
  }

  const textGroup = drawGroup(g, {
    'clip-path': `url(#${textClipId})`,
    'pointer-events': 'none',
  });

  const labelColor = isFailed
    ? (dark ? '#fca5a5' : '#991b1b')
    : (dark ? '#e2e8f0' : '#0f172a');
  const subColor = isFailed
    ? (dark ? '#f87171' : '#b91c1c')
    : (dark ? '#94a3b8' : '#64748b');

  if (w >= 120) {
    const notesSnippet = l.notes ? ` · 📝 ${l.notes.substring(0, 30)}${l.notes.length > 30 ? '…' : ''}` : '';
    const label = `${l.model} · ${l.requested_gpus} GPU${l.requested_gpus > 1 ? 's' : ''}${notesSnippet}`;
    drawText(textGroup, x + 10, y + 18, label, {
      fill: labelColor,
      'font-size': 11,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });

    const timeLabel = `${fmtHour(b)} → ${fmtHour(e)} · ${stateLabel}`;
    drawText(textGroup, x + 10, y + 34, timeLabel, {
      fill: subColor,
      'font-size': 10,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
  } else if (w >= 60) {
    const label = `${l.model}`;
    drawText(textGroup, x + 8, y + 18, label, {
      fill: labelColor,
      'font-size': 11,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
    drawText(textGroup, x + 8, y + 32, stateLabel, {
      fill: subColor,
      'font-size': 9,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
  } else if (w >= 30) {
    drawText(textGroup, x + 6, y + h / 2 + 4, l.model, {
      fill: labelColor,
      'font-size': 10,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
  }

  const hoverRect = drawRect(g, x, y, blockW, h, {
    fill: 'transparent',
    rx: 10,
    'pointer-events': 'all',
    class: 'block-hover-target',
  });
  hoverRect.addEventListener('mouseenter', () => {
    hoverRect.setAttribute('fill', dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)');
  });
  hoverRect.addEventListener('mouseleave', () => {
    hoverRect.setAttribute('fill', 'transparent');
  });

  const handleW = 8;

  // Right handle — now includes STARTING
  if (['PLANNED', 'SUBMITTED', 'STARTING', 'RUNNING'].includes(l.state) && w > 20) {
    drawRect(g, x + blockW - handleW, y, handleW, h, {
      fill: 'transparent',
      class: 'resize-handle',
      'data-handle': 'right',
      'data-lease-id': l.id,
      cursor: 'ew-resize',
    });
    const cx = x + blockW - handleW / 2;
    for (let dy = -6; dy <= 6; dy += 6) {
      g.appendChild(svgEl('circle', {
        cx, cy: y + h / 2 + dy, r: 1.5,
        fill: dark ? '#64748b' : '#94a3b8',
        'pointer-events': 'none',
      }));
    }
  }

  if (l.state === 'PLANNED' && w > 20) {
    drawRect(g, x, y, handleW, h, {
      fill: 'transparent',
      class: 'resize-handle',
      'data-handle': 'left',
      'data-lease-id': l.id,
      cursor: 'ew-resize',
    });
    const cx = x + handleW / 2;
    for (let dy = -6; dy <= 6; dy += 6) {
      g.appendChild(svgEl('circle', {
        cx, cy: y + h / 2 + dy, r: 1.5,
        fill: dark ? '#64748b' : '#94a3b8',
        'pointer-events': 'none',
      }));
    }
  }

  if (isRunning && !isConflict) {
    drawRect(g, x, y, blockW, h, {
      fill: 'none',
      stroke: 'rgba(16, 185, 129, 0.3)',
      'stroke-width': 6,
      rx: 12,
      'pointer-events': 'none',
      filter: 'url(#glow)',
    });
  }
}

// ─── Table Rendering ────────────────────────────────────────────────────────
function renderTable() {
  const body = $('#leasesTableBody');

  const nowDate = new Date(DASH?.now || Date.now());
  const active = (DASH?.leases || [])
    .filter(l =>
      ['PLANNED', 'SUBMITTED', 'STARTING', 'RUNNING'].includes(l.state) ||
      (l.state === 'FAILED' && l.end_at && new Date(l.end_at) > nowDate)
    )
    .sort((a, b) => new Date(a.begin_at || a.created_at) - new Date(b.begin_at || b.created_at));

  const failedCount = active.filter(l => l.state === 'FAILED').length;
  const activeCount = active.length - failedCount;
  $('#bookingHint').textContent = `${activeCount} active / planned` + (failedCount > 0 ? ` · ${failedCount} failed` : '');

  body.innerHTML = active.map(l => {
    const b = new Date(l.begin_at || l.created_at);
    const e = new Date(l.end_at);
    const when = `${fmtTime(b)} → ${fmtTime(e)}`;
    const durationSec = (e - b) / 1000;

    const isFailed = l.state === 'FAILED';

    const statusBadge = isFailed
      ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-red-500/15 text-red-600 dark:text-red-300 border border-red-500/30">
           <span class="w-1.5 h-1.5 rounded-full bg-red-500 mr-1.5"></span>Failed
         </span>`
      : l.conflict
        ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-red-500/15 text-red-600 dark:text-red-300 border border-red-500/30">Conflict</span>`
        : l.state === 'RUNNING'
          ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/30">
               <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>Running
             </span>`
          : l.state === 'SUBMITTED'
            ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30">Submitted</span>`
            : l.state === 'STARTING'
              ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30">
                   <span class="w-1.5 h-1.5 rounded-full bg-amber-400 mr-1.5 animate-pulse"></span>Starting
                 </span>`
              : l.state === 'PLANNED'
                ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-sky-500/15 text-sky-700 dark:text-sky-300 border border-sky-500/30">Planned</span>`
                : `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-slate-500/15 text-slate-600 dark:text-slate-300 border border-slate-500/30">${escapeHtml(l.state)}</span>`;

    let actions = '';

    const logBtn = l.slurm_job_id
      ? `<button class="text-xs px-2.5 py-1.5 rounded-md bg-gray-100 dark:bg-slate-600/20 text-gray-700 dark:text-slate-300 border border-gray-300 dark:border-slate-500/30 hover:bg-gray-200 dark:hover:bg-slate-600/30 transition font-medium" onclick="openLogModalForLease(${l.id})" title="View Logs">📋 Logs</button>`
      : '';

    if (isFailed) {
      actions = `
        ${logBtn}
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-red-50 dark:bg-red-600/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-500/30 hover:bg-red-100 dark:hover:bg-red-600/30 transition font-medium" onclick="dismissFailedLease(${l.id})">Dismiss</button>
      `;
    } else if (l.state === 'PLANNED') {
      actions = `
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-sky-50 dark:bg-brand-600/20 text-sky-700 dark:text-brand-300 border border-sky-300 dark:border-brand-500/30 hover:bg-sky-100 dark:hover:bg-brand-600/30 transition font-medium" onclick="editLease(${l.id})">Edit</button>
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-emerald-50 dark:bg-emerald-600/20 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30 hover:bg-emerald-100 dark:hover:bg-emerald-600/30 transition font-medium" onclick="extendLease(${l.id}, 3600)">+1h</button>
        ${logBtn}
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-red-50 dark:bg-red-600/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-500/30 hover:bg-red-100 dark:hover:bg-red-600/30 transition font-medium" onclick="stopLease(${l.id})">Remove</button>
      `;
    } else if (l.state === 'STARTING') {
      actions = `
        ${logBtn}
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-red-50 dark:bg-red-600/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-500/30 hover:bg-red-100 dark:hover:bg-red-600/30 transition font-medium" onclick="stopLeaseNow(${l.id})">Stop</button>
      `;
    } else {
      actions = `
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-emerald-50 dark:bg-emerald-600/20 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30 hover:bg-emerald-100 dark:hover:bg-emerald-600/30 transition font-medium" onclick="extendLease(${l.id}, 3600)">+1h</button>
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-emerald-50 dark:bg-emerald-600/20 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30 hover:bg-emerald-100 dark:hover:bg-emerald-600/30 transition font-medium" onclick="extendLease(${l.id}, 7200)">+2h</button>
        ${logBtn}
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-amber-50 dark:bg-amber-600/20 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-500/30 hover:bg-amber-100 dark:hover:bg-amber-600/30 transition font-medium" onclick="shortenLeasePrompt(${l.id})">Shorten</button>
        <button class="text-xs px-2.5 py-1.5 rounded-md bg-red-50 dark:bg-red-600/20 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-500/30 hover:bg-red-100 dark:hover:bg-red-600/30 transition font-medium" onclick="stopLeaseNow(${l.id})">Stop</button>
      `;
    }

    const rowBg = isFailed ? 'bg-red-50/50 dark:bg-red-500/5' : (l.conflict ? 'bg-red-50/30 dark:bg-red-500/5' : '');

    return `
      <tr class="${rowBg} hover:bg-gray-50 dark:hover:bg-slate-800/50 transition">
        <td class="px-4 py-3 text-sm break-all ${isFailed ? 'text-red-700 dark:text-red-300' : ''}">
          <div class="font-medium">${escapeHtml(l.model)}</div>
          ${l.notes ? `<div class="text-xs text-slate-500 dark:text-slate-400 mt-0.5 italic">📝 ${escapeHtml(l.notes)}</div>` : ''}
        </td>
        <td class="px-4 py-3 text-sm">${statusBadge}</td>
        <td class="px-4 py-3 text-sm text-gray-500 dark:text-slate-400 font-mono">${when}</td>
        <td class="px-4 py-3 text-sm text-gray-500 dark:text-slate-400 font-mono">${l.requested_gpus}</td>
        <td class="px-4 py-3 text-sm text-right"><div class="flex justify-end gap-2 flex-wrap">${actions}</div></td>
      </tr>
    `;
  }).join('');
}

// ─── Global Actions (table buttons) ─────────────────────────────────────────
window.stopLease = async (id) => {
  try {
    await api(`/admin/leases/${id}`, { method: "DELETE" });
    toast('Booking removed', 'success');
    await refresh();
  } catch (e) {
    toast(e.message, 'error');
  }
};

window.stopLeaseNow = async (id) => {
  const lease = DASH?.leases?.find(l => l.id === id);
  const name = lease?.model || 'this model';
  if (!confirm(`Stop ${name} immediately? This will cancel the Slurm job.`)) return;
  try {
    await api(`/admin/leases/${id}/stop`, { method: "POST" });
    toast('Model stopped', 'success');
    await refresh();
  } catch (e) {
    toast(e.message, 'error');
  }
};

window.extendLease = async (id, seconds) => {
  try {
    await api(`/admin/leases/${id}/extend`, {
      method: "POST",
      body: JSON.stringify({ duration_seconds: seconds })
    });
    const label = seconds >= 3600 ? `${seconds / 3600}h` : `${seconds / 60}m`;
    toast(`Extended by ${label}`, 'success');
    await refresh();
  } catch (e) {
    toast(e.message, 'error');
  }
};

window.editLease = (id) => {
  const lease = DASH?.leases?.find(x => x.id === id);
  if (!lease) return;
  const b = new Date(lease.begin_at || lease.created_at);
  const e = new Date(lease.end_at);
  const hours = Math.max(1, Math.round((e - b) / 3600000));
  openModal({ model: lease.model, beginAt: b, durationHours: hours, leaseId: lease.id });
};

window.shortenLeasePrompt = async (id) => {
  const lease = DASH?.leases?.find(l => l.id === id);
  if (!lease) return;

  const now = new Date();
  const currentEnd = new Date(lease.end_at);
  const remainingMin = Math.round((currentEnd - now) / 60000);

  if (remainingMin <= 5) {
    toast('Booking ends too soon to shorten', 'info');
    return;
  }

  // Simple prompt: ask how many minutes from now to end
  const options = [];
  if (remainingMin > 30) options.push('30 minutes');
  if (remainingMin > 60) options.push('1 hour');
  if (remainingMin > 120) options.push('2 hours');

  const choice = prompt(
    `Current end: ${fmtTime(currentEnd)} (${remainingMin}m remaining)\n\n` +
    `Enter minutes from now to set new end time:\n` +
    `(Suggestions: ${options.join(', ') || 'N/A'})`,
    Math.min(60, Math.floor(remainingMin / 2)).toString()
  );

  if (!choice) return;
  const mins = parseInt(choice, 10);
  if (isNaN(mins) || mins < 1) {
    toast('Invalid number of minutes', 'error');
    return;
  }

  const newEnd = new Date(now.getTime() + mins * 60000);
  if (newEnd >= currentEnd) {
    toast('New end must be before current end. Use extend instead.', 'info');
    return;
  }

  try {
    await api(`/admin/leases/${id}/shorten`, {
      method: "POST",
      body: JSON.stringify({ new_end_at: newEnd.toISOString() })
    });
    toast(`Shortened to end at ${fmtTime(newEnd)}`, 'success');
    await refresh();
  } catch (e) {
    toast(e.message, 'error');
  }
};

window.dismissFailedLease = async (id) => {
  try {
    await api(`/admin/leases/${id}`, { method: "DELETE" });
    toast('Failed booking dismissed', 'success');
    await refresh();
  } catch (e) {
    toast(e.message, 'error');
  }
};

// ─── Controls & Init ────────────────────────────────────────────────────────
$('#refreshBtn').addEventListener('click', refresh);
$('#windowSelect').addEventListener('change', () => { if (DASH) renderTimeline(); });
$('#zoomSelect').addEventListener('change', () => { if (DASH) renderTimeline(); });
$('#searchInput').addEventListener('input', () => { if (DASH) renderCatalog(); });
$('#filterSelect').addEventListener('change', () => { if (DASH) renderCatalog(); });

$('#newBookingBtn').addEventListener('click', () => {
  openModal();
});

// Init
initDragHandlers();
refresh();
startAutoRefresh();

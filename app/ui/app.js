// ============================================================================
// Model Hub UI — Full overhaul with drag interactions, popovers, better UX
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

// Timeline geometry (recomputed on render)
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

// Convert SVG client coords to SVG user coords
function svgPoint(svg, clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX; pt.y = clientY;
  const ctm = svg.getScreenCTM();
  return ctm ? pt.matrixTransform(ctm.inverse()) : pt;
}

// Convert x position to Date
function xToDate(x) {
  const hoursFromStart = (x - TL.leftPad) / TL.pxPerHour;
  return new Date(TL.start.getTime() + hoursFromStart * 3600000);
}

// Convert Date to x position
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
    populateModalModels();
  } catch (e) {
    toast(e.message, 'error');
  }
}

// Auto-refresh
function startAutoRefresh() {
  if (REFRESH_INTERVAL) clearInterval(REFRESH_INTERVAL);
  REFRESH_INTERVAL = setInterval(() => {
    // Don't refresh during active drag
    if (dragState.active) return;
    refresh();
  }, 8000);
}

// ─── Catalog Rendering ──────────────────────────────────────────────────────
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

    const badge = isRunning
      ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
           <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>Running
         </span>`
      : `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-slate-500/15 text-slate-400 border border-slate-500/30">Idle</span>`;

    return `
      <div class="rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4 hover:shadow-md hover:border-brand-500/30 transition-all duration-150">
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <div class="font-semibold text-sm break-all">${escapeHtml(m.id)}</div>
            <div class="mt-1 text-xs text-slate-500 font-mono">${g} GPUs · TP ${tp}</div>
            ${notes ? `<div class="mt-1 text-xs text-slate-500 italic">${escapeHtml(notes)}</div>` : ''}
          </div>
          <div class="shrink-0">${badge}</div>
        </div>
        <div class="mt-3 flex gap-2">
          <button class="flex-1 px-3 py-2 text-xs rounded-lg bg-brand-600 text-white hover:bg-brand-500 transition font-medium"
            onclick="openNewBookingForModel('${escapeHtml(m.id)}')">
            <svg class="w-3.5 h-3.5 inline mr-1 -mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
            Schedule
          </button>
          ${isRunning ? `
            <button class="px-3 py-2 text-xs rounded-lg border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/10 transition"
              onclick="openChat('${escapeHtml(m.id)}')">Chat</button>
          ` : ''}
        </div>
      </div>
    `;
  }).join('');
}

window.openChat = (modelId) => {
  window.open(`/v1/chat/ui?model=${encodeURIComponent(modelId)}`, '_blank');
};

window.openNewBookingForModel = (modelId) => {
  openModal({ model: modelId });
};

// ─── Booking Modal ──────────────────────────────────────────────────────────
const modalBackdrop = $('#modalBackdrop');
const modalBeginAt = $('#modalBeginAt');
const modalDurationRange = $('#modalDurationRange');
const modalDurationLabel = $('#modalDurationLabel');

let modalState = {
  mode: 'create', // 'create' or 'edit'
  leaseId: null,
  selectedStartOffset: 0, // minutes from now, or 'custom' or 'tomorrow9'
  startDate: null,
  durationHours: 4,
};

function openModal({ model = null, beginAt = null, durationHours = 4, leaseId = null } = {}) {
  modalState.mode = leaseId ? 'edit' : 'create';
  modalState.leaseId = leaseId;
  modalState.durationHours = durationHours;

  $('#modalTitle').textContent = leaseId ? 'Edit Booking' : 'New Booking';
  $('#modalSubtitle').textContent = leaseId ? 'Modify a planned booking' : 'Schedule a model on the GPUs';
  $('#modalSaveText').textContent = leaseId ? 'Update Booking' : 'Create Booking';
  $('#modalError').classList.add('hidden');

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

// Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeModal();
    hideBlockPopover();
  }
});

function populateModalModels(selected = null) {
  if (!DASH) return;
  const sel = $('#modalModel');
  const models = DASH.models.slice().sort((a, b) => a.id.localeCompare(b.id));
  sel.innerHTML = models.map(m => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.id)}</option>`).join('');
  if (selected) sel.value = selected;
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

// Quick time buttons
$$('#quickTimeButtons .qt-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const offset = btn.dataset.offset;
    modalState.selectedStartOffset = offset;
    updateQuickTimeButtons(offset);

    if (offset === 'custom') {
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
      const beginAt = modalState.startDate ? modalState.startDate.toISOString() : null;
      await api("/admin/leases", {
        method: "POST",
        body: JSON.stringify({
          model,
          duration_seconds: durationHours * 3600,
          begin_at: beginAt,
        })
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

// ─── Block Popover ──────────────────────────────────────────────────────────
const blockPopover = $('#blockPopover');
let popoverLeaseId = null;

function showBlockPopover(lease, anchorX, anchorY) {
  popoverLeaseId = lease.id;

  const b = new Date(lease.begin_at || lease.created_at);
  const e = new Date(lease.end_at);

  $('#popoverModel').textContent = lease.model;
  $('#popoverTime').textContent = `${fmtTime(b)} → ${fmtTime(e)} (${lease.requested_gpus} GPUs)`;

  const stateColors = {
    RUNNING: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    SUBMITTED: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    PLANNED: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  };
  const colorClass = stateColors[lease.state] || 'bg-slate-500/15 text-slate-300 border-slate-500/30';
  $('#popoverStatus').innerHTML = `
    <span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs border ${colorClass}">
      ${lease.state}${lease.conflict ? ' · CONFLICT' : ''}
    </span>
  `;

  // Actions
  const actionsEl = $('#popoverActions');
  actionsEl.innerHTML = '';

  // Extend buttons (for RUNNING/SUBMITTED)
  if (['RUNNING', 'SUBMITTED', 'PLANNED'].includes(lease.state)) {
    const extendRow = document.createElement('div');
    extendRow.className = 'flex gap-2';
    [
      { label: '+1h', secs: 3600 },
      { label: '+2h', secs: 7200 },
      { label: '+4h', secs: 14400 },
    ].forEach(({ label, secs }) => {
      const btn = document.createElement('button');
      btn.className = 'flex-1 px-2 py-1.5 text-xs rounded-lg bg-emerald-600/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-600/30 transition font-medium';
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

  // Edit (PLANNED only)
  if (lease.state === 'PLANNED') {
    const editBtn = document.createElement('button');
    editBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-brand-600/20 text-brand-300 border border-brand-500/30 hover:bg-brand-600/30 transition font-medium';
    editBtn.textContent = 'Edit Booking';
    editBtn.addEventListener('click', () => {
      hideBlockPopover();
      const b = new Date(lease.begin_at || lease.created_at);
      const e = new Date(lease.end_at);
      const hours = Math.max(1, Math.round((e - b) / 3600000));
      openModal({ model: lease.model, beginAt: b, durationHours: hours, leaseId: lease.id });
    });
    actionsEl.appendChild(editBtn);
  }

  // Stop/Cancel
  const stopBtn = document.createElement('button');
  stopBtn.className = 'w-full px-3 py-1.5 text-xs rounded-lg bg-red-600/20 text-red-300 border border-red-500/30 hover:bg-red-600/30 transition font-medium';
  stopBtn.textContent = lease.state === 'PLANNED' ? 'Remove Booking' : 'Stop / Cancel';
  stopBtn.addEventListener('click', async () => {
    try {
      await api(`/admin/leases/${lease.id}`, { method: "DELETE" });
      toast('Booking removed', 'success');
      hideBlockPopover();
      await refresh();
    } catch (err) {
      toast(err.message, 'error');
    }
  });
  actionsEl.appendChild(stopBtn);

  // Position
  const popW = 288;
  const popH = actionsEl.childElementCount * 40 + 120; // rough estimate
  let left = anchorX + 8;
  let top = anchorY - 20;

  // Keep on screen
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

// Close popover on outside click
document.addEventListener('mousedown', (e) => {
  if (!blockPopover.classList.contains('hidden') && !blockPopover.contains(e.target)) {
    hideBlockPopover();
  }
});

// ─── Drag State Machine ─────────────────────────────────────────────────────
const dragState = {
  active: false,
  type: null, // 'create' | 'resize-left' | 'resize-right' | 'move'
  leaseId: null,
  lease: null,
  startX: 0,
  startY: 0,
  currentX: 0,
  origBegin: null,
  origEnd: null,
  ghostEl: null,
  snapLines: [],
  committed: false, // has drag moved enough to be intentional
};

const DRAG_THRESHOLD = 5; // px before drag activates
const SNAP_MINUTES = 15;

function snapDate(d) {
  const snapped = new Date(d);
  snapped.setMinutes(Math.round(snapped.getMinutes() / SNAP_MINUTES) * SNAP_MINUTES, 0, 0);
  return snapped;
}

function initDragHandlers() {
  const svg = $('#timelineSvg');
  const container = $('#timelineContainer');

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
    // For non-PLANNED, clicking body opens popover (handled separately)
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
    // Minimum 15 min
    if (ghostEnd - ghostBegin < 15 * 60000) {
      ghostEnd = new Date(ghostBegin.getTime() + 15 * 60000);
    }
    // Determine lane from Y position
    const clickedLane = Math.floor((pt.y - TL.headerH) / TL.laneH);
    // Default to first model's GPU count or 1
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

  // Check for conflicts visually
  const hasConflict = checkVisualConflict(ghostBegin, ghostEnd, laneCount, dragState.leaseId);

  const ghost = drawGroup(svg, { class: 'ghost-block' });
  drawRect(ghost, x, y, Math.max(8, w), h, {
    fill: hasConflict ? 'rgba(239, 68, 68, 0.25)' : 'rgba(14, 165, 233, 0.25)',
    stroke: hasConflict ? 'rgba(239, 68, 68, 0.8)' : 'rgba(14, 165, 233, 0.8)',
    'stroke-width': 2,
    'stroke-dasharray': '6 3',
    rx: 10,
  });

  // Label
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
    ['PLANNED', 'SUBMITTED', 'RUNNING'].includes(l.state) && l.id !== excludeLeaseId
  );

  // Simple check: at each moment in [begin, end], sum GPUs in use + gpusNeeded <= total
  // We check at discrete points (every 15 min) for performance
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
      // Open modal pre-filled with the dragged time range
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
        // RUNNING/SUBMITTED — use extend
        if (diffSeconds > 0) {
          await api(`/admin/leases/${lease.id}/extend`, {
            method: "POST",
            body: JSON.stringify({ duration_seconds: diffSeconds })
          });
          toast(`Extended by ${Math.round(diffSeconds / 60)}m`, 'success');
        } else {
          toast('Cannot shorten a running booking from the timeline', 'info');
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
  TL.start = new Date(now.getTime() - 30 * 60000);
  TL.end = new Date(TL.start.getTime() + hours * 3600000);

  svg.setAttribute('width', TL.width);
  svg.setAttribute('height', TL.height);
  svg.innerHTML = '';

  const dark = isDark();

  // Background
  drawRect(svg, 0, 0, TL.width, TL.height, { fill: dark ? '#0b1220' : '#ffffff' });

  // Time grid
  for (let h = 0; h <= hours; h++) {
    const x = TL.leftPad + h * pxPerHour;
    drawLine(svg, x, 0, x, TL.height, {
      stroke: dark ? '#334155' : '#94a3b8',
      'stroke-opacity': 0.2,
    });

    const t = new Date(TL.start.getTime() + h * 3600000);
    // Use local time display consistently
    const label = `${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}`;
    drawText(svg, x + 4, 22, label, {
      fill: dark ? '#94a3b8' : '#64748b',
      'font-size': 12,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });
  }

  // 30-min sub-grid
  for (let h = 0; h < hours; h++) {
    const x = TL.leftPad + (h + 0.5) * pxPerHour;
    drawLine(svg, x, TL.headerH, x, TL.height, {
      stroke: dark ? '#334155' : '#94a3b8',
      'stroke-opacity': 0.08,
    });
  }

  // Lane labels + lines
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
  // Bottom line
  drawLine(svg, 0, TL.headerH + gpuTotal * TL.laneH, TL.width, TL.headerH + gpuTotal * TL.laneH, {
    stroke: dark ? '#334155' : '#94a3b8',
    'stroke-opacity': 0.15,
  });

  // Draw blocks
  const leases = (DASH?.leases || [])
    .filter(l => ['PLANNED', 'SUBMITTED', 'RUNNING'].includes(l.state))
    .sort((a, b) => new Date(a.begin_at || a.created_at) - new Date(b.begin_at || b.created_at));

  for (const l of leases) {
    drawLeaseBlock(svg, l);
  }

  // Now marker (on top of blocks)
  const nowX = dateToX(now);
  drawLine(svg, nowX, 0, nowX, TL.height, {
    stroke: '#0ea5e9',
    'stroke-width': 2,
    class: 'now-line-pulse',
  });

  // Small "now" label
  drawText(svg, nowX + 4, 12, 'now', {
    fill: '#0ea5e9',
    'font-size': 10,
    'font-weight': 'bold',
    'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
  });

  // Auto-scroll to now on first render
  const container = $('#timelineContainer');
  const scrollTarget = nowX - 200;
  if (scrollTarget > 0 && !dragState.active) {
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
  const isPlanned = l.state === 'PLANNED';
  const isConflict = !!l.conflict;

  const dark = isDark();

  let fill, stroke, dash;
  if (isConflict) {
    fill = 'rgba(239, 68, 68, 0.18)';
    stroke = 'rgba(239, 68, 68, 0.85)';
    dash = null;
  } else if (isRunning) {
    fill = 'rgba(16, 185, 129, 0.18)';
    stroke = 'rgba(16, 185, 129, 0.85)';
    dash = null;
  } else if (isSubmitted) {
    fill = 'rgba(245, 158, 11, 0.18)';
    stroke = 'rgba(245, 158, 11, 0.85)';
    dash = null;
  } else {
    fill = 'rgba(14, 165, 233, 0.16)';
    stroke = 'rgba(14, 165, 233, 0.75)';
    dash = '6 4';
  }

  // Group for the block
  const g = drawGroup(svg, { 'data-lease-id': l.id, cursor: 'pointer' });

  // Main rect
  const blockRect = drawRect(g, x, y, Math.max(8, w), h, {
    fill, stroke, 'stroke-width': 2,
    ...(dash ? { 'stroke-dasharray': dash } : {}),
    rx: 10,
    'data-lease-id': l.id,
  });

  // Model label
  const label = `${l.model} · ${l.requested_gpus} GPU${l.requested_gpus > 1 ? 's' : ''}`;
  if (w > 60) {
    drawText(g, x + 10, y + 18, label, {
      fill: dark ? '#e2e8f0' : '#0f172a',
      'font-size': 11,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
      'pointer-events': 'none',
    });

    // Time range
    const timeLabel = `${fmtHour(b)} → ${fmtHour(e)} · ${l.state}${isConflict ? ' ⚠' : ''}`;
    drawText(g, x + 10, y + 34, timeLabel, {
      fill: dark ? '#94a3b8' : '#64748b',
      'font-size': 10,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
      'pointer-events': 'none',
    });
  }

  // Resize handles (visual indicators at left/right edges)
  const handleW = 8;

  // Right handle (always for resizable states)
  if (['PLANNED', 'SUBMITTED', 'RUNNING'].includes(l.state) && w > 20) {
    drawRect(g, x + Math.max(8, w) - handleW, y, handleW, h, {
      fill: 'transparent',
      class: 'resize-handle',
      'data-handle': 'right',
      'data-lease-id': l.id,
      cursor: 'ew-resize',
    });
    // Visual grip dots
    const cx = x + Math.max(8, w) - handleW / 2;
    for (let dy = -6; dy <= 6; dy += 6) {
      const dot = svgEl('circle', {
        cx, cy: y + h / 2 + dy, r: 1.5,
        fill: dark ? '#64748b' : '#94a3b8',
        'pointer-events': 'none',
      });
      g.appendChild(dot);
    }
  }

  // Left handle (only PLANNED)
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
      const dot = svgEl('circle', {
        cx, cy: y + h / 2 + dy, r: 1.5,
        fill: dark ? '#64748b' : '#94a3b8',
        'pointer-events': 'none',
      });
      g.appendChild(dot);
    }
  }

  // Running glow effect
  if (isRunning && !isConflict) {
    drawRect(g, x, y, Math.max(8, w), h, {
      fill: 'none',
      stroke: 'rgba(16, 185, 129, 0.3)',
      'stroke-width': 6,
      rx: 12,
      'pointer-events': 'none',
      filter: 'url(#glow)',
    });
  }
}

// ─── Table Rendering ─────────────────────────────────────────────────────────
function renderTable() {
  const body = $('#leasesTableBody');

  const active = (DASH?.leases || [])
    .filter(l => ['PLANNED', 'SUBMITTED', 'RUNNING'].includes(l.state))
    .sort((a, b) => new Date(a.begin_at || a.created_at) - new Date(b.begin_at || b.created_at));

  $('#bookingHint').textContent = `${active.length} active / planned`;

  body.innerHTML = active.map(l => {
    const b = new Date(l.begin_at || l.created_at);
    const e = new Date(l.end_at);
    const when = `${fmtTime(b)} → ${fmtTime(e)}`;

    const statusBadge = l.conflict
      ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-red-500/15 text-red-300 border border-red-500/30">Conflict</span>`
      : l.state === 'RUNNING'
        ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
             <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>Running
           </span>`
        : l.state === 'SUBMITTED'
          ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-amber-500/15 text-amber-300 border border-amber-500/30">Submitted</span>`
          : l.state === 'PLANNED'
            ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-sky-500/15 text-sky-300 border border-sky-500/30">Planned</span>`
            : `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-slate-500/15 text-slate-300 border border-slate-500/30">${escapeHtml(l.state)}</span>`;

    let actions = '';
    if (l.state === 'PLANNED') {
      actions = `
        <button class="text-xs px-2 py-1 rounded bg-brand-600/20 text-brand-300 border border-brand-500/30 hover:bg-brand-600/30 transition" onclick="editLease(${l.id})">Edit</button>
        <button class="text-xs px-2 py-1 rounded bg-emerald-600/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-600/30 transition" onclick="extendLease(${l.id}, 3600)">+1h</button>
        <button class="text-xs px-2 py-1 rounded bg-red-600/20 text-red-300 border border-red-500/30 hover:bg-red-600/30 transition" onclick="stopLease(${l.id})">Remove</button>
      `;
    } else {
      actions = `
        <button class="text-xs px-2 py-1 rounded bg-emerald-600/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-600/30 transition" onclick="extendLease(${l.id}, 3600)">+1h</button>
        <button class="text-xs px-2 py-1 rounded bg-emerald-600/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-600/30 transition" onclick="extendLease(${l.id}, 7200)">+2h</button>
        <button class="text-xs px-2 py-1 rounded bg-red-600/20 text-red-300 border border-red-500/30 hover:bg-red-600/30 transition" onclick="stopLease(${l.id})">Stop</button>
      `;
    }

    return `
      <tr class="${l.conflict ? 'bg-red-500/5' : ''} hover:bg-slate-50 dark:hover:bg-slate-800/50 transition">
        <td class="px-4 py-3 text-sm font-medium break-all">${escapeHtml(l.model)}</td>
        <td class="px-4 py-3 text-sm">${statusBadge}</td>
        <td class="px-4 py-3 text-sm text-slate-500 dark:text-slate-400 font-mono">${when}</td>
        <td class="px-4 py-3 text-sm text-slate-400 font-mono">${l.requested_gpus}</td>
        <td class="px-4 py-3 text-sm text-right"><div class="flex justify-end gap-2">${actions}</div></td>
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

// ─── Controls & Init ─────────────────────────────────────────────────────────
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
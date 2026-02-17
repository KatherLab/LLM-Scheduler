const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ---------------- Theme ----------------
const themeToggle = $('#themeToggle');
const iconSun = $('#iconSun');
const iconMoon = $('#iconMoon');

function updateTheme() {
  if (document.documentElement.classList.contains('dark')) {
    iconSun.classList.remove('hidden');
    iconMoon.classList.add('hidden');
  } else {
    iconSun.classList.add('hidden');
    iconMoon.classList.remove('hidden');
  }
}
themeToggle.addEventListener('click', () => {
  document.documentElement.classList.toggle('dark');
  updateTheme();
});
updateTheme();

// ---------------- API helper ----------------
async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const txt = await res.text();
  const data = txt ? JSON.parse(txt) : {};
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// ---------------- State ----------------
let DASH = null; // dashboard payload
let MODEL_MAP = new Map();
let SELECTED_LEASE_ID = null;

function parseDT(s) { return s ? new Date(s) : null; }
function fmtTime(d) {
  return d.toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}
function fmtHour(d) {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

// ---------------- Drawer (booking) ----------------
const drawer = $('#bookingDrawer');
const backdrop = $('#drawerBackdrop');

function openDrawer({ title = "New Booking", subtitle = "Schedule a model on the GPUs" } = {}) {
  $('#drawerTitle').textContent = title;
  $('#drawerSubtitle').textContent = subtitle;
  backdrop.classList.remove('hidden');
  drawer.classList.remove('translate-x-full');
}

function closeDrawer() {
  backdrop.classList.add('hidden');
  drawer.classList.add('translate-x-full');
  $('#drawerError').classList.add('hidden');
  $('#drawerError').textContent = '';
  SELECTED_LEASE_ID = null;
}

$('#drawerClose').addEventListener('click', closeDrawer);
$('#drawerCancel').addEventListener('click', closeDrawer);
backdrop.addEventListener('click', closeDrawer);

$('#drawerStartMode').addEventListener('change', () => {
  const mode = $('#drawerStartMode').value;
  if (mode === 'time') $('#drawerBeginAt').classList.remove('hidden');
  else $('#drawerBeginAt').classList.add('hidden');
});

function drawerError(msg) {
  $('#drawerError').textContent = msg;
  $('#drawerError').classList.remove('hidden');
}

// ---------------- UI: catalog rendering ----------------
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

  $('#modelCount').textContent = `${models.length} models`;

  const list = $('#modelsList');
  list.innerHTML = models.map(m => {
    const meta = m.meta || {};
    const g = meta.gpus ?? '?';
    const tp = meta.tensor_parallel_size ?? '?';
    const notes = meta.notes || '';

    const isRunning = m.ready;
    const badge = isRunning
      ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">Running</span>`
      : `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-slate-500/15 text-slate-300 border border-slate-500/30">Not running</span>`;

    return `
      <div class="rounded-xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4 hover:shadow-sm transition">
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <div class="font-semibold text-sm break-all">${m.id}</div>
            <div class="mt-1 text-xs text-slate-500 font-mono">Default: ${g} GPUs · TP ${tp}</div>
            ${notes ? `<div class="mt-1 text-xs text-slate-500">${notes}</div>` : ``}
          </div>
          <div class="shrink-0">${badge}</div>
        </div>

        <div class="mt-3 flex gap-2">
          <button class="flex-1 px-3 py-2 text-xs rounded-md bg-brand-600 text-white hover:bg-brand-500 transition"
            onclick="openNewBookingForModel('${escapeHtml(m.id)}')">
            Start / Schedule
          </button>
          ${isRunning ? `<button class="px-3 py-2 text-xs rounded-md border border-emerald-500/30 text-emerald-200 hover:bg-emerald-500/10 transition"
            onclick="openChat('${escapeHtml(m.id)}')">Chat</button>` : ``}
        </div>
      </div>
    `;
  }).join('');
}

function escapeHtml(s) {
  return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}

window.openChat = (modelId) => {
  window.open(`/v1/chat/ui?model=${encodeURIComponent(modelId)}`, '_blank');
};

window.openNewBookingForModel = (modelId) => {
  fillDrawerForNew(modelId);
  openDrawer({ title: "New Booking", subtitle: "Start now or schedule later" });
};

// ---------------- Drawer fill ----------------
function fillDrawerModelOptions(selected) {
  const sel = $('#drawerModel');
  sel.innerHTML = DASH.models
    .slice()
    .sort((a, b) => a.id.localeCompare(b.id))
    .map(m => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.id)}</option>`).join('');
  sel.value = selected || DASH.models[0]?.id || '';
  sel.dispatchEvent(new Event('change'));
}

$('#drawerModel').addEventListener('change', () => {
  const id = $('#drawerModel').value;
  const m = MODEL_MAP.get(id);
  if (!m) return;
  const meta = m.meta || {};
  $('#drawerModelMeta').textContent = `Default needs ${meta.gpus} GPUs · TP ${meta.tensor_parallel_size}`;
  // prefill defaults if this is a “new booking”
  if (!SELECTED_LEASE_ID) {
    $('#drawerGpus').value = meta.gpus || 1;
    $('#drawerDuration').value = 4;
  }
});

function fillDrawerForNew(modelId, beginAtDate = null) {
  SELECTED_LEASE_ID = null;
  fillDrawerModelOptions(modelId);
  $('#drawerStartMode').value = beginAtDate ? 'time' : 'now';
  $('#drawerStartMode').dispatchEvent(new Event('change'));
  if (beginAtDate) {
    $('#drawerBeginAt').value = toLocalDTInput(beginAtDate);
  } else {
    $('#drawerBeginAt').value = '';
  }
}

function fillDrawerForEdit(lease) {
  SELECTED_LEASE_ID = lease.id;
  fillDrawerModelOptions(lease.model);
  $('#drawerModel').value = lease.model;
  $('#drawerGpus').value = lease.requested_gpus;
  const b = lease.begin_at ? new Date(lease.begin_at) : new Date(lease.created_at);
  const e = lease.end_at ? new Date(lease.end_at) : new Date(b.getTime() + 3600*1000);
  const hours = Math.max(1, Math.round((e - b) / 3600000));
  $('#drawerDuration').value = hours;

  // editing planned uses begin_at
  $('#drawerStartMode').value = 'time';
  $('#drawerStartMode').dispatchEvent(new Event('change'));
  $('#drawerBeginAt').value = toLocalDTInput(b);
}

function toLocalDTInput(date) {
  // datetime-local wants "YYYY-MM-DDTHH:MM"
  const pad = (n) => String(n).padStart(2, '0');
  const y = date.getFullYear();
  const m = pad(date.getMonth() + 1);
  const d = pad(date.getDate());
  const hh = pad(date.getHours());
  const mm = pad(date.getMinutes());
  return `${y}-${m}-${d}T${hh}:${mm}`;
}

// ---------------- Save drawer ----------------
$('#drawerSave').addEventListener('click', async () => {
  try {
    $('#drawerError').classList.add('hidden');

    const model = $('#drawerModel').value;
    const durationHours = parseInt($('#drawerDuration').value || '1', 10);

    if (!model) return drawerError("Please choose a model.");
    if (!Number.isFinite(gpus) || gpus < 1) return drawerError("GPUs must be >= 1.");
    if (!Number.isFinite(durationHours) || durationHours < 1) return drawerError("Duration must be >= 1 hour.");

    if (!SELECTED_LEASE_ID) {
      // create
      const mode = $('#drawerStartMode').value;
      let begin_at = null;
      if (mode === 'time') {
        const v = $('#drawerBeginAt').value;
        if (!v) return drawerError("Please choose a start time.");
        begin_at = new Date(v).toISOString();
      }
      await api("/admin/leases", {
        method: "POST",
        body: JSON.stringify({
          model,
          duration_seconds: durationHours * 3600,
          begin_at
        })
      });
    } else {
      // edit planned booking (PATCH)
      const v = $('#drawerBeginAt').value;
      if (!v) return drawerError("Please choose a start time.");
      const begin = new Date(v);
      const end = new Date(begin.getTime() + durationHours * 3600 * 1000);

      await api(`/admin/leases/${SELECTED_LEASE_ID}`, {
        method: "PATCH",
        body: JSON.stringify({
          begin_at: begin.toISOString(),
          end_at: end.toISOString(),
        })
      });
    }

    closeDrawer();
    await refresh();
  } catch (e) {
    drawerError(e.message);
  }
});

// ---------------- Timeline rendering ----------------
function renderTimeline() {
  const svg = $('#timelineSvg');
  const hours = parseInt($('#windowSelect').value, 10);
  const pxPerHour = parseInt($('#zoomSelect').value, 10);
  const gpuTotal = DASH.total_gpus || 8;

  const headerH = 34;
  const laneH = 44;
  const leftPad = 70; // for GPU labels
  const width = leftPad + (hours * pxPerHour);
  const height = headerH + gpuTotal * laneH + 16;

  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.innerHTML = '';

  const now = new Date(DASH.now);
  const start = new Date(now.getTime() - 30 * 60000);
  const end = new Date(start.getTime() + hours * 3600000);

  // background
  drawRect(svg, 0, 0, width, height, { fill: isDark() ? '#0b1220' : '#ffffff' });

  // time grid
  for (let h = 0; h <= hours; h++) {
    const x = leftPad + h * pxPerHour;
    drawLine(svg, x, 0, x, height, { stroke: isDark() ? '#334155' : '#94a3b8', 'stroke-opacity': 0.2 });

    const t = new Date(start.getTime() + h * 3600000);
    drawText(svg, x + 4, 22, `${t.getHours()}:00`, { fill: isDark() ? '#94a3b8' : '#64748b', 'font-size': 12, 'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace' });
  }

  // lane labels + lane lines
  for (let i = 0; i < gpuTotal; i++) {
    const y = headerH + i * laneH;
    drawLine(svg, 0, y, width, y, { stroke: isDark() ? '#334155' : '#94a3b8', 'stroke-opacity': 0.15 });
    drawText(svg, 10, y + 28, `GPU ${i}`, { fill: isDark() ? '#cbd5e1' : '#334155', 'font-size': 12, 'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace' });
  }

  // now marker
  const nowX = leftPad + ((now - start) / 3600000) * pxPerHour;
  drawLine(svg, nowX, 0, nowX, height, { stroke: '#0ea5e9', 'stroke-width': 2, 'stroke-opacity': 0.9 });

  // click-to-schedule empty space
  svg.addEventListener('click', (ev) => {
    // ignore clicks on blocks (they stop propagation)
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const loc = pt.matrixTransform(ctm.inverse());
    if (loc.x < leftPad || loc.y < headerH) return;

    // snap to 15min increments
    const hoursFromStart = (loc.x - leftPad) / pxPerHour;
    const ms = hoursFromStart * 3600000;
    const clicked = new Date(start.getTime() + ms);
    clicked.setMinutes(Math.round(clicked.getMinutes() / 15) * 15, 0, 0);

    fillDrawerForNew(DASH.models[0]?.id || '', clicked);
    openDrawer({ title: "New Booking", subtitle: `Start at ${fmtTime(clicked)}` });
  }, { once: false });

  // draw blocks (planned/submitted/running)
  const leases = DASH.leases
    .filter(l => ['PLANNED','SUBMITTED','RUNNING'].includes(l.state))
    .slice()
    .sort((a, b) => new Date(a.begin_at || a.created_at) - new Date(b.begin_at || b.created_at));

  for (const l of leases) {
    const b = new Date(l.begin_at || l.created_at);
    const e = new Date(l.end_at);
    if (e <= start || b >= end) continue;

    const x = leftPad + ((b - start) / 3600000) * pxPerHour;
    const w = ((e - b) / 3600000) * pxPerHour;

    const laneStart = (l.lane_start ?? 0);
    const laneCount = (l.lane_count ?? l.requested_gpus ?? 1);

    const y = headerH + laneStart * laneH + 6;
    const h = laneCount * laneH - 12;

    const isRunning = (l.state === 'RUNNING');
    const isPlanned = (l.state === 'PLANNED');
    const isConflict = !!l.conflict;

    const fill = isConflict ? 'rgba(239, 68, 68, 0.18)'
      : isRunning ? 'rgba(16, 185, 129, 0.18)'
      : 'rgba(14, 165, 233, 0.16)';

    const stroke = isConflict ? 'rgba(239, 68, 68, 0.85)'
      : isRunning ? 'rgba(16, 185, 129, 0.85)'
      : 'rgba(14, 165, 233, 0.75)';

    const dash = isPlanned ? '6 4' : null;

    const block = drawRect(svg, x, y, Math.max(6, w), h, {
      fill, stroke, 'stroke-width': 2,
      ...(dash ? { 'stroke-dasharray': dash } : {}),
      rx: 10,
      cursor: 'pointer'
    });

    // label
    const label = `${l.model} · ${l.requested_gpus} GPU${l.requested_gpus > 1 ? 's' : ''}`;
    const labelText = drawText(svg, x + 10, y + 18, label, {
      fill: isDark() ? '#e2e8f0' : '#0f172a',
      'font-size': 11,
      'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
      'pointer-events': 'none'
    });

    // secondary line
    drawText(svg, x + 10, y + 34, `${fmtHour(b)} → ${fmtHour(e)} · ${l.state}${isConflict ? ' (CONFLICT)' : ''}`, {
      fill: isDark() ? '#cbd5e1' : '#334155',
      'font-size': 10,
      'pointer-events': 'none'
    });

    // click actions
    block.addEventListener('click', (ev) => {
      ev.stopPropagation();
      openLeaseActions(l);
    });
  }
}

// Basic SVG helpers
function drawRect(svg, x, y, w, h, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  el.setAttribute('x', x);
  el.setAttribute('y', y);
  el.setAttribute('width', w);
  el.setAttribute('height', h);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  svg.appendChild(el);
  return el;
}
function drawLine(svg, x1, y1, x2, y2, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
  el.setAttribute('x1', x1); el.setAttribute('y1', y1);
  el.setAttribute('x2', x2); el.setAttribute('y2', y2);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  svg.appendChild(el);
  return el;
}
function drawText(svg, x, y, text, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
  el.setAttribute('x', x);
  el.setAttribute('y', y);
  el.textContent = text;
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  svg.appendChild(el);
  return el;
}
function isDark() {
  return document.documentElement.classList.contains('dark');
}

// ---------------- Jobs table ----------------
function renderTable() {
  const body = $('#leasesTableBody');

  const active = DASH.leases
    .filter(l => ['PLANNED','SUBMITTED','RUNNING'].includes(l.state))
    .slice()
    .sort((a, b) => new Date(a.begin_at || a.created_at) - new Date(b.begin_at || b.created_at));

  $('#bookingHint').textContent = `${active.length} active / planned`;

  body.innerHTML = active.map(l => {
    const b = new Date(l.begin_at || l.created_at);
    const e = new Date(l.end_at);
    const when = `${fmtTime(b)} → ${fmtTime(e)}`;
    const statusBadge = l.conflict
      ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-red-500/15 text-red-200 border border-red-500/30">Conflict</span>`
      : l.state === 'RUNNING'
        ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-emerald-500/15 text-emerald-200 border border-emerald-500/30">Running</span>`
        : l.state === 'PLANNED'
          ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-sky-500/15 text-sky-200 border border-sky-500/30">Planned</span>`
          : `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-slate-500/15 text-slate-200 border border-slate-500/30">${l.state}</span>`;

    const actions = l.state === 'PLANNED'
      ? `
        <button class="text-brand-300 hover:text-brand-200" onclick="editLease(${l.id})">Edit</button>
        <span class="mx-2 text-slate-600">·</span>
        <button class="text-emerald-300 hover:text-emerald-200" onclick="extendLease(${l.id}, 3600)">+1h</button>
        <span class="mx-2 text-slate-600">·</span>
        <button class="text-red-300 hover:text-red-200" onclick="stopLease(${l.id})">Remove</button>
      `
      : `
        <button class="text-emerald-300 hover:text-emerald-200" onclick="extendLease(${l.id}, 3600)">+1h</button>
        <span class="mx-2 text-slate-600">·</span>
        <button class="text-red-300 hover:text-red-200" onclick="stopLease(${l.id})">Stop</button>
      `;

    return `
      <tr class="${l.conflict ? 'bg-red-500/5' : ''}">
        <td class="px-4 py-3 text-sm font-medium break-all">${escapeHtml(l.model)}</td>
        <td class="px-4 py-3 text-sm">${statusBadge}</td>
        <td class="px-4 py-3 text-sm text-slate-500">${when}</td>
        <td class="px-4 py-3 text-sm text-slate-300 font-mono">${l.requested_gpus}</td>
        <td class="px-4 py-3 text-sm text-right whitespace-nowrap">${actions}</td>
      </tr>
    `;
  }).join('');
}

window.stopLease = async (id) => {
  if (!confirm("Remove/Stop this booking?")) return;
  await api(`/admin/leases/${id}`, { method: "DELETE" });
  await refresh();
};

window.extendLease = async (id, seconds) => {
  await api(`/admin/leases/${id}/extend`, {
    method: "POST",
    body: JSON.stringify({ duration_seconds: seconds })
  });
  await refresh();
};

window.editLease = (id) => {
  const lease = DASH.leases.find(x => x.id === id);
  if (!lease) return;
  fillDrawerForEdit(lease);
  openDrawer({ title: "Edit Booking", subtitle: "Move or resize a planned booking" });
};

// ---------------- Block actions (simple prompt-based) ----------------
function openLeaseActions(lease) {
  const b = new Date(lease.begin_at || lease.created_at);
  const e = new Date(lease.end_at);

  let msg = `${lease.model}\n\n${fmtTime(b)} → ${fmtTime(e)}\n${lease.requested_gpus} GPU(s)\nStatus: ${lease.state}${lease.conflict ? ' (CONFLICT)' : ''}\n\nChoose an action:\n- OK: Extend by 1 hour\n- Cancel: More actions`;
  if (confirm(msg)) {
    extendLease(lease.id, 3600);
    return;
  }

  // more actions
  const actions = [];
  if (lease.state === 'PLANNED') actions.push('edit');
  actions.push('stop');

  const choice = prompt(`Type one: ${actions.join(', ')}`);
  if (!choice) return;
  if (choice.toLowerCase() === 'edit' && lease.state === 'PLANNED') editLease(lease.id);
  if (choice.toLowerCase() === 'stop') stopLease(lease.id);
}

// ---------------- Refresh ----------------
async function refresh() {
  try {
    DASH = await api('/admin/dashboard');
    MODEL_MAP = new Map(DASH.models.map(m => [m.id, m]));
    $('#subtitle').textContent = `Single node · ${DASH.total_gpus} GPUs · Server time ${fmtTime(new Date(DASH.now))}`;

    renderCatalog();
    renderTimeline();
    renderTable();

    // fill drawer select
    if (!$('#drawerModel').options.length) {
      fillDrawerModelOptions(DASH.models[0]?.id || '');
    }
  } catch (e) {
    alert(e.message);
  }
}

// ---------------- Controls ----------------
$('#refreshBtn').addEventListener('click', refresh);
$('#windowSelect').addEventListener('change', refresh);
$('#zoomSelect').addEventListener('change', refresh);
$('#searchInput').addEventListener('input', renderCatalog);
$('#filterSelect').addEventListener('change', renderCatalog);

$('#newBookingBtn').addEventListener('click', () => {
  fillDrawerForNew(DASH?.models?.[0]?.id || '');
  openDrawer({ title: "New Booking", subtitle: "Start now or schedule later" });
});

// init
refresh();

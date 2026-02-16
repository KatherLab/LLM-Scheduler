const $ = (sel) => document.querySelector(sel);

function fmtTs(d){
  const pad = (n)=> String(n).padStart(2,'0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function fmtDur(seconds){
  const h = Math.floor(seconds/3600);
  const m = Math.floor((seconds%3600)/60);
  if(h === 0) return `${m}m`;
  if(m === 0) return `${h}h`;
  return `${h}h${m}m`;
}
function isoToLocalInput(iso){
  // for datetime-local
  const d = new Date(iso);
  const pad = (n)=> String(n).padStart(2,'0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function localInputToIso(value){
  if(!value) return null;
  // treat as local time
  const d = new Date(value);
  return d.toISOString().slice(0,19);
}

async function api(path, opts={}){
  const res = await fetch(path, {
    headers: {"Content-Type":"application/json"},
    ...opts
  });
  const txt = await res.text();
  let data = null;
  try{ data = txt ? JSON.parse(txt) : null; }catch{ data = {raw: txt}; }
  if(!res.ok){
    const msg = data?.detail || data?.raw || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function badgeState(state){
  const s = (state||"").toUpperCase();
  if(s === "READY") return `<span class="badge ok">READY</span>`;
  if(s === "FAILED") return `<span class="badge no">FAILED</span>`;
  if(s === "STARTING") return `<span class="badge">STARTING</span>`;
  return `<span class="badge">${s}</span>`;
}

function leaseStateBadge(state){
  const s = (state||"").toUpperCase();
  if(s === "SUBMITTED" || s === "RUNNING") return `<span class="badge ok">${s}</span>`;
  if(s === "CANCELED" || s === "FAILED") return `<span class="badge no">${s}</span>`;
  return `<span class="badge">${s}</span>`;
}

function secondsBetween(a,b){ return Math.max(0, (b.getTime()-a.getTime())/1000); }

function filterLeases(leases, mode){
  const now = new Date();
  if(mode === "running"){
    return leases.filter(l => {
      const begin = l.begin_at ? new Date(l.begin_at) : new Date(l.created_at);
      const end = l.end_at ? new Date(l.end_at) : new Date(begin.getTime()+3600*1000);
      return begin <= now && now <= end && (l.state !== "CANCELED");
    });
  }
  if(mode === "scheduled"){
    return leases.filter(l => {
      const begin = l.begin_at ? new Date(l.begin_at) : new Date(l.created_at);
      const now = new Date();
      return begin > now && (l.state !== "CANCELED");
    });
  }
  return leases;
}

function buildModelCards(models, leases){
  const wrap = $("#modelsList");
  wrap.innerHTML = "";

  const startDialog = $("#startDialog");
  const modelSel = $("#startModel");
  modelSel.innerHTML = "";
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.id;
    modelSel.appendChild(opt);
  });

  models.forEach(m => {
    const running = leases.some(l => l.model === m.id && l.state !== "CANCELED");
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `
      <div class="itemTop">
        <div class="name">${m.id}</div>
        <div>${m.ready ? '<span class="badge ok">READY</span>' : (running ? '<span class="badge">SCHEDULED</span>' : '<span class="badge no">STOPPED</span>')}</div>
      </div>
      <div class="meta">
        <span class="badge">gpus:${m.meta?.gpus ?? "?"}</span>
        <span class="badge">tp:${m.meta?.tensor_parallel_size ?? "?"}</span>
        ${m.meta?.notes ? `<span class="badge">${m.meta.notes}</span>` : ""}
      </div>
      <div style="margin-top:10px;display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn small primary" data-start="${m.id}">Start / Schedule</button>
      </div>
    `;
    el.querySelector("[data-start]").addEventListener("click", () => {
      $("#startModel").value = m.id;
      $("#startOwner").value = "";
      $("#startBegin").value = "";
      $("#startDuration").value = "6";
      $("#startGpus").value = "";
      $("#startTp").value = "";
      $("#startError").textContent = "";
      startDialog.showModal();
    });
    wrap.appendChild(el);
  });

  $("#startForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#startError").textContent = "";
    try{
      const model = $("#startModel").value;
      const owner = $("#startOwner").value.trim() || null;
      const beginVal = $("#startBegin").value;
      const begin_at = beginVal ? localInputToIso(beginVal) : null;
      const durHours = parseFloat($("#startDuration").value || "6");
      const duration_seconds = Math.max(60, Math.round(durHours * 3600));

      const gpus = $("#startGpus").value ? parseInt($("#startGpus").value,10) : null;
      const tensor_parallel_size = $("#startTp").value ? parseInt($("#startTp").value,10) : null;

      await api("/admin/leases", {
        method:"POST",
        body: JSON.stringify({model, owner, begin_at, duration_seconds, gpus, tensor_parallel_size})
      });
      $("#startDialog").close();
      await refreshAll();
    }catch(err){
      $("#startError").textContent = String(err.message || err);
    }
  }, {once:true});
}

function buildLeaseList(leases){
  const wrap = $("#leasesList");
  wrap.innerHTML = "";

  leases.forEach(l => {
    const begin = l.begin_at ? new Date(l.begin_at) : new Date(l.created_at);
    const end = l.end_at ? new Date(l.end_at) : null;
    const now = new Date();
    const duration = end ? secondsBetween(begin,end) : 0;
    const remaining = end ? Math.max(0, secondsBetween(now,end)) : 0;

    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `
      <div class="itemTop">
        <div class="itemTitle">${l.model} ${l.owner ? `<span class="badge">${l.owner}</span>` : ""}</div>
        <div>${leaseStateBadge(l.state)}</div>
      </div>
      <div class="itemSub">
        <span class="kv">lease:${l.id}</span>
        <span class="kv">job:${l.slurm_job_id ?? "-"}</span>
        <span class="kv">gpus:${l.requested_gpus}</span>
        <span class="kv">tp:${l.requested_tp}</span>
        <span class="kv">begin:${fmtTs(begin)}</span>
        <span class="kv">end:${end ? fmtTs(end) : "-"}</span>
        ${end ? `<span class="kv">remain:${fmtDur(remaining)}</span>` : ""}
      </div>
      <div style="margin-top:10px;display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn small" data-extend="${l.id}">Edit duration</button>
        <button class="btn small danger" data-cancel="${l.id}">Unload</button>
      </div>
    `;

    el.querySelector(`[data-cancel="${l.id}"]`).addEventListener("click", async () => {
      if(!confirm(`Unload ${l.model} (lease ${l.id})?`)) return;
      await api(`/admin/leases/${l.id}`, {method:"DELETE"});
      await refreshAll();
    });

    el.querySelector(`[data-extend="${l.id}"]`).addEventListener("click", async () => {
      const hours = prompt("New duration (hours). This updates the Slurm TimeLimit for the job.", String(Math.max(1, Math.round(duration/3600 || 6))));
      if(hours === null) return;
      const dur = Math.max(0.05, parseFloat(hours));
      const duration_seconds = Math.round(dur * 3600);
      await api(`/admin/leases/${l.id}/extend`, {method:"POST", body: JSON.stringify({duration_seconds})});
      await refreshAll();
    });

    wrap.appendChild(el);
  });
}

function packIntoGpuLanes(leases, gpuTotal, start, end){
  // Returns array lanes[gpuIndex] = array of blocks assigned to that lane.
  // We do not know physical GPU IDs -> we visualize N "lanes" and pack blocks greedily.
  const lanes = Array.from({length: gpuTotal}, () => []);

  const normalized = leases.map(l => {
    const b = l.begin_at ? new Date(l.begin_at) : new Date(l.created_at);
    const e = l.end_at ? new Date(l.end_at) : new Date(b.getTime() + 3600*1000);
    return {...l, _begin: b, _end: e};
  }).filter(l => l._end > start && l._begin < end && l.state !== "CANCELED")
    .sort((a,b)=> a._begin - b._begin);

  function laneFree(lane, b, e){
    // lane is array of blocks with _begin/_end
    return lane.every(x => (e <= x._begin) || (b >= x._end));
  }

  for(const lease of normalized){
    const need = Math.min(gpuTotal, Math.max(1, lease.requested_gpus));
    const chosen = [];
    for(let i=0;i<gpuTotal && chosen.length<need;i++){
      if(laneFree(lanes[i], lease._begin, lease._end)){
        chosen.push(i);
      }
    }
    // If can't fit visually, just stack on lane 0..need-1 (still shows overlap)
    if(chosen.length < need){
      chosen.length = 0;
      for(let i=0;i<need;i++) chosen.push(i);
    }
    // Put one identical block into each chosen lane
    for(const idx of chosen){
      lanes[idx].push(lease);
    }
  }
  return lanes;
}

function drawTimeline(leases){
  const gpuTotal = 8; // HGX H200 8 GPUs (can make this configurable later)
  const hours = parseInt($("#windowSelect").value,10);
  const mode = $("#showSelect").value;

  const now = new Date();
  const start = new Date(now.getTime() - 60*60*1000); // include 1h in past
  const end = new Date(now.getTime() + hours*60*60*1000);

  const filtered = filterLeases(leases, mode);
  const lanes = packIntoGpuLanes(filtered, gpuTotal, start, end);

  const svg = $("#timelineSvg");
  const W = 1200;
  const laneH = 40;
  const headerH = 50;
  const leftPad = 70;
  const rightPad = 20;
  const H = headerH + laneH*gpuTotal + 30;

  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.innerHTML = "";

  const x0 = leftPad, x1 = W - rightPad;
  const t0 = start.getTime(), t1 = end.getTime();
  const tx = (t) => x0 + ( (t - t0) / (t1 - t0) ) * (x1-x0);

  // background grid + labels
  const mk = (tag, attrs={}) => {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for(const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  };

  svg.appendChild(mk("rect", {x:"0", y:"0", width:String(W), height:String(H), fill:"transparent"}));

  // time ticks every hour
  for(let h=0; h<=hours+1; h++){
    const t = new Date(now.getTime() + (h-1)*3600*1000);
    const x = tx(t.getTime());
    svg.appendChild(mk("line", {x1:String(x), y1:"10", x2:String(x), y2:String(H-10), stroke:"rgba(255,255,255,0.08)"}));
    if(h % 2 === 0){
      const label = mk("text", {x:String(x+2), y:"25", fill:"rgba(231,238,252,0.65)", "font-size":"11", "font-family":"ui-monospace"});
      label.textContent = `${String(t.getHours()).padStart(2,'0')}:00`;
      svg.appendChild(label);
    }
  }

  // now marker
  const nowX = tx(now.getTime());
  svg.appendChild(mk("line", {x1:String(nowX), y1:"10", x2:String(nowX), y2:String(H-10), stroke:"rgba(110,168,255,0.65)", "stroke-width":"2"}));

  // lane labels + separators
  for(let i=0;i<gpuTotal;i++){
    const y = headerH + i*laneH;
    svg.appendChild(mk("line", {x1:String(x0), y1:String(y), x2:String(x1), y2:String(y), stroke:"rgba(255,255,255,0.08)"}));
    const label = mk("text", {x:"16", y:String(y+26), fill:"rgba(231,238,252,0.65)", "font-size":"12", "font-family":"ui-monospace"});
    label.textContent = `GPU ${i}`;
    svg.appendChild(label);
  }
  svg.appendChild(mk("line", {x1:String(x0), y1:String(headerH + laneH*gpuTotal), x2:String(x1), y2:String(headerH + laneH*gpuTotal), stroke:"rgba(255,255,255,0.08)"}));

  // blocks
  function colorFor(model){
    // deterministic hash -> hue-like selection using rgba variations
    let h = 0;
    for(let i=0;i<model.length;i++) h = (h*31 + model.charCodeAt(i)) >>> 0;
    const a = 0.18 + (h % 60)/300; // 0.18..0.38
    // use blue/green palette only (still looks consistent)
    const isBlue = (h % 2) === 0;
    const fill = isBlue ? `rgba(110,168,255,${a})` : `rgba(110,255,179,${a})`;
    const stroke = isBlue ? `rgba(110,168,255,0.55)` : `rgba(110,255,179,0.55)`;
    return {fill, stroke};
  }

  for(let lane=0; lane<gpuTotal; lane++){
    const y = headerH + lane*laneH + 6;
    for(const l of lanes[lane]){
      const b = l._begin, e = l._end;
      const x = Math.max(x0, tx(b.getTime()));
      const w = Math.max(2, Math.min(x1, tx(e.getTime())) - x);
      const {fill, stroke} = colorFor(l.model);
      const rect = mk("rect", {x:String(x), y:String(y), width:String(w), height:String(laneH-12), rx:"10", fill, stroke, "stroke-width":"1"});
      svg.appendChild(rect);

      const txt = mk("text", {x:String(x+8), y:String(y+18), fill:"rgba(231,238,252,0.85)", "font-size":"11", "font-family":"ui-monospace"});
      const short = l.model.length > 18 ? l.model.slice(0,18)+"…" : l.model;
      txt.textContent = short;
      svg.appendChild(txt);

      const txt2 = mk("text", {x:String(x+8), y:String(y+34), fill:"rgba(231,238,252,0.55)", "font-size":"10", "font-family":"ui-monospace"});
      const dur = Math.round(secondsBetween(b,e)/60);
      txt2.textContent = `${l.requested_gpus} GPU • ${dur}m`;
      svg.appendChild(txt2);

      rect.addEventListener?.("click", ()=>{});
    }
  }
}

async function refreshAll(){
  const [models, leases] = await Promise.all([
    api("/v1/models"),
    api("/admin/leases"),
  ]);

  buildModelCards(models.data, leases);
  buildLeaseList(leases);
  drawTimeline(leases);
}

document.addEventListener("DOMContentLoaded", () => {
  $("#refreshBtn").addEventListener("click", refreshAll);
  $("#windowSelect").addEventListener("change", refreshAll);
  $("#showSelect").addEventListener("change", refreshAll);
  refreshAll().catch(err => console.error(err));
});

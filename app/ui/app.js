const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// Theme Toggle
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

// API Helper
async function api(path, opts = {}) {
    try {
        const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
        const txt = await res.text();
        const data = txt ? JSON.parse(txt) : {};
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
    } catch (e) {
        alert(e.message);
        throw e;
    }
}

// Global State
let selectedModel = null;
let durationHours = 1;

// Modal Logic
function openLaunchModal(modelId) {
    selectedModel = modelId;
    $('#modalModelName').textContent = modelId;
    $('#launchError').classList.add('hidden');
    $('#launchModal').classList.remove('hidden');
    setDuration(1);
}

function closeModal() {
    $('#launchModal').classList.add('hidden');
}

function setDuration(h) {
    durationHours = h;
    $$('.dur-btn').forEach(b => {
        if (parseInt(b.textContent) === h) {
            b.classList.add('bg-brand-50', 'dark:bg-brand-900', 'border-brand-500', 'text-brand-700', 'dark:text-brand-100');
        } else {
            b.classList.remove('bg-brand-50', 'dark:bg-brand-900', 'border-brand-500', 'text-brand-700', 'dark:text-brand-100');
        }
    });
}

async function submitLaunch() {
    const when = $('#launchWhen').value;
    const begin_at = when === 'now' ? null : null; // Logic for 'queue' would require finding next gap, kept simple for now
    
    try {
        await api("/admin/leases", {
            method: "POST",
            body: JSON.stringify({
                model: selectedModel,
                duration_seconds: durationHours * 3600,
                begin_at: begin_at
            })
        });
        closeModal();
        refresh();
    } catch (e) {
        $('#launchError').textContent = e.message;
        $('#launchError').classList.remove('hidden');
    }
}

// Data Handling
async function refresh() {
    const [models, leases] = await Promise.all([
        api("/v1/models"),
        api("/admin/leases")
    ]);

    renderGrid(models.data, leases);
    renderTable(leases);
    renderTimeline(leases);
}

function renderGrid(models, leases) {
    const grid = $('#modelsGrid');
    grid.innerHTML = models.map(m => {
        // Is it running?
        const activeLease = leases.find(l => l.model === m.id && l.state === "RUNNING");
        const status = activeLease ? 
            `<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">Running</span>` : 
            `<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-slate-700 dark:text-gray-300">Stopped</span>`;
        
        const borderClass = activeLease ? "border-brand-500 ring-1 ring-brand-500" : "border-gray-200 dark:border-slate-700";

        return `
        <div class="bg-white dark:bg-slate-800 rounded-xl border ${borderClass} shadow-sm p-6 flex flex-col justify-between transition hover:shadow-md">
            <div>
                <div class="flex justify-between items-start">
                    <h3 class="text-lg font-bold text-gray-900 dark:text-white break-all">${m.id}</h3>
                </div>
                <div class="mt-2 space-y-2">
                   ${status}
                   <div class="text-xs text-gray-500 font-mono">GPUs: ${m.meta?.gpus} | TP: ${m.meta?.tensor_parallel_size}</div>
                </div>
                ${m.meta?.notes ? `<p class="mt-2 text-sm text-gray-600 dark:text-gray-400">${m.meta.notes}</p>` : ''}
            </div>
            <div class="mt-6">
                ${activeLease ? 
                    `<button onclick="window.open('/v1/chat/ui?model=${m.id}', '_blank')" class="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700">Open Chat</button>` :
                    `<button onclick="openLaunchModal('${m.id}')" class="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-brand-600 hover:bg-brand-700">Launch</button>`
                }
            </div>
        </div>
        `;
    }).join('');
}

function renderTable(leases) {
    const active = leases.filter(l => ["STARTING", "RUNNING", "SUBMITTED"].includes(l.state));
    $('#leasesTableBody').innerHTML = active.map(l => {
        const end = new Date(l.end_at);
        const now = new Date();
        const minsLeft = Math.round((end - now) / 60000);
        
        return `
        <tr>
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">${l.model}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${l.state}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">${minsLeft} mins</td>
            <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <button onclick="stopLease(${l.id})" class="text-red-600 hover:text-red-900 dark:hover:text-red-400">Stop</button>
            </td>
        </tr>
        `;
    }).join('');
}

async function stopLease(id) {
    if(confirm("Stop this model immediately?")) {
        await api(`/admin/leases/${id}`, {method: "DELETE"});
        refresh();
    }
}

// Timeline Drawing
function renderTimeline(leases) {
    const svg = $('#timelineSvg');
    const hours = parseInt($('#windowSelect').value);
    const gpuTotal = 8; 
    
    // Config
    const headerH = 30;
    const laneH = 40;
    const pxPerHour = 100; // Wide spacing
    const width = pxPerHour * hours;
    const height = headerH + (gpuTotal * laneH) + 20;
    
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    svg.innerHTML = '';
    
    const now = new Date();
    const start = new Date(now.getTime() - (1000 * 60 * 30)); // Start 30 mins ago
    
    // 1. Grid & Time Labels
    for (let h = 0; h < hours; h++) {
        const x = h * pxPerHour;
        const time = new Date(start.getTime() + (h * 3600000));
        
        // Line
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x);
        line.setAttribute("y1", 0);
        line.setAttribute("x2", x);
        line.setAttribute("y2", height);
        line.setAttribute("stroke", "#334155"); // slate-700
        line.setAttribute("stroke-opacity", "0.2");
        svg.appendChild(line);
        
        // Text
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", x + 5);
        text.setAttribute("y", 20);
        text.setAttribute("fill", "#94a3b8"); // slate-400
        text.setAttribute("font-size", "12");
        text.setAttribute("font-family", "monospace");
        text.textContent = `${time.getHours()}:00`;
        svg.appendChild(text);
    }
    
    // 2. Current Time Marker
    const nowOffset = (now - start) / 3600000 * pxPerHour;
    const nowLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    nowLine.setAttribute("x1", nowOffset);
    nowLine.setAttribute("y1", 0);
    nowLine.setAttribute("x2", nowOffset);
    nowLine.setAttribute("y2", height);
    nowLine.setAttribute("stroke", "#0ea5e9"); // brand-500
    nowLine.setAttribute("stroke-width", "2");
    svg.appendChild(nowLine);
    
    // 3. Simple Pack Logic (Visual only)
    const sorted = leases.filter(l => l.state !== 'CANCELED').sort((a,b) => new Date(a.begin_at || a.created_at) - new Date(b.begin_at || b.created_at));
    
    // Simulate lanes (just for visual vertical stacking, not physical GPU mapping)
    const lanes = Array(gpuTotal).fill(0); 
    
    sorted.forEach(l => {
        const begin = new Date(l.begin_at || l.created_at);
        const end = l.end_at ? new Date(l.end_at) : new Date(begin.getTime() + 3600000);
        
        if (end < start) return;
        
        const x = Math.max(0, (begin - start) / 3600000 * pxPerHour);
        const w = (end - begin) / 3600000 * pxPerHour;
        const gpus = l.requested_gpus || 1;
        
        // Find fit
        let laneIdx = -1;
        for(let i=0; i<=gpuTotal-gpus; i++) {
            // Check if lanes i to i+gpus are free at time x
            // Simplified: We just stack them sequentially for this demo to ensure they appear
            if (lanes[i] < (begin.getTime())) {
                laneIdx = i;
                break;
            }
        }
        if (laneIdx === -1) laneIdx = 0; // Fallback
        
        // Update availability
        for(let k=0; k<gpus; k++) lanes[laneIdx+k] = end.getTime();
        
        const y = headerH + (laneIdx * laneH) + 2;
        const h = (laneH * gpus) - 4;
        
        // Draw Rect
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", x);
        rect.setAttribute("y", y);
        rect.setAttribute("width", Math.max(w, 2));
        rect.setAttribute("height", h);
        rect.setAttribute("rx", 4);
        
        // Color based on state
        if (l.state === 'RUNNING') {
            rect.setAttribute("fill", "rgba(16, 185, 129, 0.2)"); // Green
            rect.setAttribute("stroke", "rgba(16, 185, 129, 0.8)");
        } else {
             rect.setAttribute("fill", "rgba(14, 165, 233, 0.2)"); // Blue
             rect.setAttribute("stroke", "rgba(14, 165, 233, 0.8)");
        }
        
        svg.appendChild(rect);
        
        // Label
        if (w > 30) {
            const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
            lbl.setAttribute("x", x + 5);
            lbl.setAttribute("y", y + 15);
            lbl.setAttribute("fill", "#e2e8f0");
            lbl.setAttribute("font-size", "10");
            lbl.textContent = l.model.substring(0, Math.floor(w/8));
            svg.appendChild(lbl);
        }
    });
}

// Init
$('#refreshBtn').addEventListener('click', refresh);
$('#windowSelect').addEventListener('change', refresh);
refresh();

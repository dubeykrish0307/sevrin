// HUD side panels. They stay hidden until SEVRIN opens one (a <<panel:x>>
// tag in a reply) or something relevant happens, like a new reminder or a
// stored fact, and slide away again on a timer.
//
// TODO: only the schedule and memory panels show real data. Alerts, the
// system meters and telemetry latency are still placeholders.

const tasksEl = document.getElementById('hudTasks');
const metersEl = document.getElementById('hudMeters');
const alertsEl = document.getElementById('hudAlerts');
const telemetryEl = document.getElementById('hudTelemetry');
const memoryEl = document.getElementById('hudMemory');
const memCountEl = document.getElementById('memCount');

const PANEL_DURATIONS = {
  schedule: 40000,
  memory: 9000,       // just a flash
  system: 22000,
  alerts: 22000,
  telemetry: 22000,
};
const GRACE_ON_NEW_TURN_MS = 6000;
const MAX_MEM_EVENTS = 6;
const EMPTY_SCHEDULE = [{ name: 'NOTHING SCHEDULED', progress: 0, eta: 0 }];

const panelTimers = {};
const bootTime = Date.now();

let tasks = EMPTY_SCHEDULE;
let alerts = [
  { level: 'warn', text: 'Standup in 25 min' },
  { level: 'info', text: 'Build finished — 2 warnings' },
  { level: 'ok', text: 'Backups verified' },
];
let meters = [
  { label: 'CPU', value: 0.34 },
  { label: 'MEM', value: 0.51 },
  { label: 'NET', value: 0.18 },
];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function bar(trackClass, fraction) {
  const track = el('div', trackClass);
  const fill = el('i');
  fill.style.width = `${Math.round(fraction * 100)}%`;
  track.appendChild(fill);
  return track;
}

const pad2 = (n) => String(n).padStart(2, '0');

function formatEta(sec) {
  if (sec <= 0) return 'DUE';
  if (sec >= 86400) return `in ${Math.floor(sec / 86400)}d`;
  if (sec >= 3600) return `in ${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `in ${m}m ${pad2(s)}s` : `in ${s}s`;
}

function formatUptime() {
  const s = Math.floor((Date.now() - bootTime) / 1000);
  return `${pad2(Math.floor(s / 3600))}:${pad2(Math.floor((s % 3600) / 60))}:${pad2(s % 60)}`;
}

// --- panels ---------------------------------------------------------------

function renderTasks() {
  tasksEl.replaceChildren(...tasks.map((task) => {
    const row = el('div', 'hud-task-row');
    row.append(el('span', '', task.name), el('span', 'hud-task-eta', formatEta(task.eta)));
    const wrap = el('div', 'hud-task');
    wrap.append(row, bar('hud-bar', task.progress));
    return wrap;
  }));
}

function renderMeters() {
  metersEl.replaceChildren(...meters.map((m) => {
    const label = el('div', 'hud-meter-label');
    label.append(el('span', '', m.label), el('span', '', `${Math.round(m.value * 100)}%`));
    const wrap = el('div', 'hud-meter');
    wrap.append(label, bar('hud-meter-track', m.value));
    return wrap;
  }));
}

function renderAlerts() {
  alertsEl.replaceChildren(...alerts.slice(0, 4).map((a) => {
    const row = el('div', 'hud-alert');
    row.append(el('span', '', a.text), el('span', `tag ${a.level}`, a.level.toUpperCase()));
    return row;
  }));
}

function renderTelemetry(state) {
  const rows = [
    ['STATE', state.toUpperCase()],
    ['LATENCY', `${(0.9 + Math.random() * 0.4).toFixed(2)}s`],   // placeholder, not measured
    ['MODEL', 'HAIKU-4.5'],
    ['VOICE', 'ELEVEN-FLASH'],
    ['UPTIME', formatUptime()],
  ];
  telemetryEl.replaceChildren(...rows.map(([k, v]) => {
    const row = el('div', 'hud-tel-row');
    row.append(el('span', '', k), el('span', '', v));
    return row;
  }));
}

export function renderSchedules(rows) {
  if (!rows || !rows.length) {
    tasks = EMPTY_SCHEDULE;
  } else {
    const now = Date.now() / 1000;
    tasks = rows.map((r) => ({
      name: (r.recurring ? '↻ ' : '') + r.subject.toUpperCase().slice(0, 26),
      progress: 1,
      eta: Math.max(0, Math.round(r.due - now)),
    }));
  }
  renderTasks();
}

export function setTasks(data) { tasks = data; renderTasks(); }
export function setAlerts(data) { alerts = data; renderAlerts(); }
export function setMeters(data) { meters = data; renderMeters(); }

export function setMemoryCount(n) { memCountEl.textContent = n; }

// one candidate fact's outcome from the memory pipeline (stored, rejected, ...)
export function pushMemoryEvent(ev) {
  const outcome = ev.outcome || 'info';
  const text = el('span', 'txt', ev.fact || ev.detail || '');
  text.appendChild(el('span', 'stage', `${(ev.stage || '').toUpperCase()} · ${outcome.toUpperCase()}`));
  const row = el('div', `mem-event ${outcome}`);
  row.append(el('span', 'dot'), text);
  memoryEl.prepend(row);
  while (memoryEl.children.length > MAX_MEM_EVENTS) memoryEl.removeChild(memoryEl.lastChild);
}

// --- showing and hiding -----------------------------------------------------

export function showPanel(name) {
  const panel = document.querySelector(`.panel-${name}`);
  if (!panel) return;
  panel.classList.remove('hidden-panel');
  clearTimeout(panelTimers[name]);
  panelTimers[name] = setTimeout(() => hidePanel(name), PANEL_DURATIONS[name] || 20000);
}

export function hidePanel(name) {
  const panel = document.querySelector(`.panel-${name}`);
  if (!panel) return;
  panel.classList.add('hidden-panel');
  clearTimeout(panelTimers[name]);
}

// A new question cuts open panels down to a short grace period. If the
// reply opens the same panel again, showPanel resets its full duration.
export function shortenOpenPanels() {
  Object.keys(PANEL_DURATIONS).forEach((name) => {
    const panel = document.querySelector(`.panel-${name}`);
    if (panel && !panel.classList.contains('hidden-panel')) {
      clearTimeout(panelTimers[name]);
      panelTimers[name] = setTimeout(() => hidePanel(name), GRACE_ON_NEW_TURN_MS);
    }
  });
}

export function startHud(getState) {
  setInterval(() => {
    // schedule ETAs count down locally between backend updates
    tasks.forEach((t) => { if (t.eta > 0) t.eta = Math.max(0, t.eta - 1); });
    meters.forEach((m) => {
      m.value = Math.max(0.05, Math.min(0.97, m.value + (Math.random() - 0.5) * 0.06));
    });
    renderTasks();
    renderMeters();
    renderTelemetry(getState());
  }, 1000);

  renderTasks();
  renderMeters();
  renderAlerts();
  renderTelemetry(getState());
}

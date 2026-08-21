import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const WS_URL = 'ws://localhost:8765';

let ws = null;
let currentState = 'idle';
let amplitude = 0;
let smoothedAmplitude = 0;

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const stateLabel = document.getElementById('stateLabel');
const textInput = document.getElementById('textInput');
const speechLayer = document.getElementById('speechLayer');
const clockEl = document.getElementById('clock');

// ==================================================================
// HUD PANELS
// Currently driven by a local mock model so the display is alive and
// believable. It's deliberately structured as DATA (the arrays below),
// not hardcoded markup — so when SEVRIN gains real task/alert awareness,
// the backend just sends these same shapes over the WebSocket and the
// panels light up for real. See handleMessage's 'hud_*' cases.
// ==================================================================

const hudTasksEl = document.getElementById('hudTasks');
const hudMetersEl = document.getElementById('hudMeters');
const hudAlertsEl = document.getElementById('hudAlerts');
const hudTelemetryEl = document.getElementById('hudTelemetry');
const hudMemoryEl = document.getElementById('hudMemory');
const memCountEl = document.getElementById('memCount');

let TASKS = [
  { name: 'INDEXING PROJECT FILES', progress: 0.62, eta: 214 },
  { name: 'CALENDAR SYNC', progress: 0.88, eta: 47 },
  { name: 'VOICE MODEL WARM', progress: 1.0, eta: 0 },
];

let ALERTS = [
  { level: 'warn', text: 'Standup in 25 min' },
  { level: 'info', text: 'Build finished — 2 warnings' },
  { level: 'ok', text: 'Backups verified' },
];

let METERS = [
  { label: 'CPU', value: 0.34 },
  { label: 'MEM', value: 0.51 },
  { label: 'NET', value: 0.18 },
];

function fmtEta(sec) {
  if (sec <= 0) return 'COMPLETE';
  const m = Math.floor(sec / 60), s = sec % 60;
  return m > 0 ? `ETA ${m}m ${String(s).padStart(2, '0')}s` : `ETA ${s}s`;
}

function renderTasks() {
  hudTasksEl.innerHTML = '';
  TASKS.forEach((task) => {
    const wrap = document.createElement('div');
    wrap.className = 'hud-task';
    const row = document.createElement('div');
    row.className = 'hud-task-row';
    const name = document.createElement('span');
    name.textContent = task.name;
    const eta = document.createElement('span');
    eta.className = 'hud-task-eta';
    eta.textContent = fmtEta(task.eta);
    row.append(name, eta);
    const bar = document.createElement('div');
    bar.className = 'hud-bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.round(task.progress * 100)}%`;
    bar.appendChild(fill);
    wrap.append(row, bar);
    hudTasksEl.appendChild(wrap);
  });
}

function renderMeters() {
  hudMetersEl.innerHTML = '';
  METERS.forEach((m) => {
    const wrap = document.createElement('div');
    wrap.className = 'hud-meter';
    const label = document.createElement('div');
    label.className = 'hud-meter-label';
    const l = document.createElement('span'); l.textContent = m.label;
    const v = document.createElement('span'); v.textContent = `${Math.round(m.value * 100)}%`;
    label.append(l, v);
    const track = document.createElement('div');
    track.className = 'hud-meter-track';
    const fill = document.createElement('i');
    fill.style.width = `${Math.round(m.value * 100)}%`;
    track.appendChild(fill);
    wrap.append(label, track);
    hudMetersEl.appendChild(wrap);
  });
}

function renderAlerts() {
  hudAlertsEl.innerHTML = '';
  ALERTS.slice(0, 4).forEach((a) => {
    const row = document.createElement('div');
    row.className = 'hud-alert';
    const tag = document.createElement('span');
    tag.className = `tag ${a.level}`;
    tag.textContent = a.level.toUpperCase();
    const txt = document.createElement('span');
    txt.textContent = a.text;
    row.append(txt, tag);
    hudAlertsEl.appendChild(row);
  });
}

function renderTelemetry() {
  const rows = [
    ['STATE', currentState.toUpperCase()],
    ['LATENCY', `${(0.9 + Math.random() * 0.4).toFixed(2)}s`],
    ['MODEL', 'HAIKU-4.5'],
    ['VOICE', 'ELEVEN-FLASH'],
    ['UPTIME', fmtUptime()],
  ];
  hudTelemetryEl.innerHTML = '';
  rows.forEach(([k, v]) => {
    const row = document.createElement('div');
    row.className = 'hud-tel-row';
    const a = document.createElement('span'); a.textContent = k;
    const b = document.createElement('span'); b.textContent = v;
    row.append(a, b);
    hudTelemetryEl.appendChild(row);
  });
}

const bootTime = Date.now();
function fmtUptime() {
  const s = Math.floor((Date.now() - bootTime) / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}

// tick the mock data so the HUD feels live
setInterval(() => {
  TASKS.forEach((t) => {
    if (t.eta > 0) {
      t.eta = Math.max(0, t.eta - 1);
      t.progress = Math.min(1, t.progress + 0.002 + Math.random() * 0.003);
    }
  });
  METERS.forEach((m) => {
    const drift = (Math.random() - 0.5) * 0.06;
    m.value = Math.max(0.05, Math.min(0.97, m.value + drift));
  });
  renderTasks(); renderMeters(); renderTelemetry();
}, 1000);

setInterval(() => {
  clockEl.textContent = new Date().toLocaleTimeString('en-GB');
}, 1000);

renderTasks(); renderMeters(); renderAlerts(); renderTelemetry();
clockEl.textContent = new Date().toLocaleTimeString('en-GB');

// ==================================================================
// MEMORY PANEL — shows facts being extracted and verified live, so memory
// formation is visible rather than invisible magic. Each event is one
// candidate fact's outcome from the multi-layer verification pipeline.
// ==================================================================

const MAX_MEM_EVENTS = 6;

function pushMemoryEvent(ev) {
  const outcome = ev.outcome || 'info';
  const row = document.createElement('div');
  row.className = `mem-event ${outcome}`;

  const dot = document.createElement('span');
  dot.className = 'dot';

  const txt = document.createElement('span');
  txt.className = 'txt';
  txt.textContent = ev.fact || ev.detail || '';

  const stage = document.createElement('span');
  stage.className = 'stage';
  stage.textContent = `${(ev.stage || '').toUpperCase()} · ${outcome.toUpperCase()}`;
  txt.appendChild(stage);

  row.append(dot, txt);
  hudMemoryEl.prepend(row);
  while (hudMemoryEl.children.length > MAX_MEM_EVENTS) {
    hudMemoryEl.removeChild(hudMemoryEl.lastChild);
  }
}

// ==================================================================
// FLOATING SPEECH — no chat box. Lines fade in under the reactor, at
// most a few at a time, then fade away on their own.
// ==================================================================

const MAX_LINES = 3;
const LINE_LIFETIME_MS = 9000;
let activeAssistantLine = null;

function pushLine(role, text) {
  const el = document.createElement('div');
  el.className = `speech-line ${role}`;
  el.textContent = text;
  speechLayer.appendChild(el);

  while (speechLayer.children.length > MAX_LINES) {
    speechLayer.removeChild(speechLayer.firstChild);
  }

  // auto-fade after a while
  el._timer = setTimeout(() => fadeOut(el), LINE_LIFETIME_MS);
  return el;
}

function fadeOut(el) {
  if (!el || !el.isConnected) return;
  el.classList.add('fading');
  setTimeout(() => { if (el.isConnected) el.remove(); }, 800);
}

function refreshLineTimer(el) {
  if (!el) return;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => fadeOut(el), LINE_LIFETIME_MS);
}

// ==================================================================
// WebSocket
// ==================================================================

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { statusDot.classList.add('connected'); statusText.textContent = 'ONLINE'; };
  ws.onclose = () => {
    statusDot.classList.remove('connected');
    statusText.textContent = 'OFFLINE';
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => { let m; try { m = JSON.parse(e.data); } catch (x) { return; } handleMessage(m); };
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'state': setState(msg.value); break;
    case 'transcript': pushLine('user', msg.text); break;
    case 'transcript_start':
      activeAssistantLine = pushLine('assistant', '');
      pickAnswerHue();
      break;
    case 'transcript_delta':
      if (activeAssistantLine) {
        activeAssistantLine.textContent += (activeAssistantLine.textContent ? ' ' : '') + msg.text;
        refreshLineTimer(activeAssistantLine);
      }
      break;
    case 'transcript_end':
      activeAssistantLine = null;
      break;
    case 'amplitude': amplitude = msg.value; break;
    // --- future: real data from the backend drops straight in here ---
    case 'hud_tasks':     TASKS = msg.data;  renderTasks();  break;
    case 'hud_alerts':    ALERTS = msg.data; renderAlerts(); break;
    case 'hud_meters':    METERS = msg.data; renderMeters(); break;
    case 'interrupted':   handleInterrupted(); break;
    case 'memory_event':  pushMemoryEvent(msg.data); break;
    case 'memory_stats':  memCountEl.textContent = msg.data.active_facts; break;
    case 'error': console.error('[backend error]', msg.message); break;
  }
}

function handleInterrupted() {
  // visual acknowledgement that he was cut off — the current assistant line
  // stops growing and dims, so it's clear that reply was abandoned
  if (activeAssistantLine) {
    activeAssistantLine.style.opacity = '0.4';
    activeAssistantLine = null;
  }
  stateLabel.textContent = 'INTERRUPTED';
}

function setState(state) {
  const wasIdle = currentState === 'idle';
  currentState = state;
  stateLabel.textContent = state.toUpperCase();
  if (wasIdle && state === 'listening') triggerWakeFlash();
}

textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && textInput.value.trim()) {
    const text = textInput.value.trim();
    pushLine('user', text);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'text_input', text }));
    textInput.value = '';
  }
});

// ==================================================================
// ARC REACTOR — precise, machined, graphic. Concentric mechanical rings,
// arc segments, tick detail, a hard bright core, viewed straight-on like
// the references. Built from crisp geometry (not fuzzy noise), glowing via
// bloom. Faces the camera flat so it reads as an engineered interface.
//
// Square-box bug fix: bloom on a transparent canvas was compositing its
// full render-target rectangle over the page. Fixed by (1) keeping the
// scene background null, (2) using OutputPass, and (3) a radial vignette
// sprite so the glow falls off to true transparent-black at the edges
// instead of a hard rectangle.
// ==================================================================

const container = document.getElementById('orb-container');
let width = container.clientWidth || 400;
let height = container.clientHeight || 400;

const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
// pulled back to 8.2 so the wider outer-arc swings (which can reach radius
// ~3.3 at peak volume) stay fully in frame instead of clipping off-screen
camera.position.set(0, 0, 9.0);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, premultipliedAlpha: false });
renderer.setClearColor(0x000000, 0);
renderer.setSize(width, height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.5)); // higher DPR = crisper edges
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
container.appendChild(renderer.domElement);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
// A subtle, premium sheen — not the blown-out haze from before. Low
// strength + high threshold means only the brightest edges pick up a
// gentle glow, the way real Apple UI highlights do. (Was fully off; a
// tasteful amount reads more premium than dead-flat.)
const bloom = new UnrealBloomPass(new THREE.Vector2(width, height), 0.28, 0.5, 0.6);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// the whole reactor lives in a group facing the camera
const reactor = new THREE.Group();
scene.add(reactor);

const C_HOT = 0xdff4ff;
const C_BLUE = 0x2e9eff;
const C_CYAN = 0x6ee7ff;

function ringMat(color, opacity) {
  return new THREE.MeshBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
}

// --- flat concentric rings (thin torus, faced flat) ---
function flatRing(radius, tube, color, opacity) {
  const m = new THREE.Mesh(new THREE.TorusGeometry(radius, tube, 12, 200), ringMat(color, opacity));
  return m;
}
const rings = [];
[
  [0.52, 0.016, C_HOT, 0.9],
  [0.78, 0.008, C_CYAN, 0.6],
  [1.05, 0.014, C_BLUE, 0.75],
  [1.5, 0.008, C_CYAN, 0.55],
].forEach(([r, t, c, o]) => { const ring = flatRing(r, t, c, o); reactor.add(ring); rings.push(ring); });

// --- arc segments (partial rings) — the mechanical reactor look ---
function arcSegment(radius, tube, color, opacity, start, len) {
  // torus geometry from higher segment count for smoother curves
  const geo = new THREE.TorusGeometry(radius, tube, 12, 128, len);
  const m = new THREE.Mesh(geo, ringMat(color, opacity));
  m.rotation.z = start;
  m.userData.baseRadius = radius;   // remembered so we can push it outward on speech
  m.userData.baseTube = tube;
  m.userData.color = color;
  m.userData.curPush = 0;            // current outward push, smoothed each frame
  m.userData.kick = 0;               // (legacy) impulse; unused by the gentle eased flare
  m.userData.target = 0;             // eased flare target for the outer ring
  return m;
}
const arcGroup1 = new THREE.Group();
const arcGroup2 = new THREE.Group();
// inner ring of 6 thick arc segments (the reactor's coil housings)
for (let i = 0; i < 6; i++) {
  const start = (i / 6) * Math.PI * 2 + 0.12;
  arcGroup1.add(arcSegment(1.65, 0.05, C_BLUE, 0.85, start, (Math.PI * 2 / 6) - 0.24));
}
// outer ring of many finer arc segments — these are the ones that flare
// out during speech. More of them, more compressed (tighter gaps) for a
// denser, higher-detail machined look.
const OUTER_ARCS = 12;
for (let i = 0; i < OUTER_ARCS; i++) {
  const start = (i / OUTER_ARCS) * Math.PI * 2 + 0.04;
  const seg = arcSegment(2.2, 0.012, C_CYAN, 0.7, start, (Math.PI * 2 / OUTER_ARCS) - 0.09);
  arcGroup2.add(seg);
}
reactor.add(arcGroup1, arcGroup2);

// --- radial tick marks between rings (engraved detail) ---
const TICK = 60;
const ticks = new THREE.InstancedMesh(
  new THREE.BoxGeometry(0.09, 0.02, 0.001),
  ringMat(C_CYAN, 0.55),
  TICK
);
const td = new THREE.Object3D();
for (let i = 0; i < TICK; i++) {
  const a = (i / TICK) * Math.PI * 2, r = 1.28;
  td.position.set(Math.cos(a) * r, Math.sin(a) * r, 0);
  td.rotation.z = a;
  td.updateMatrix();
  ticks.setMatrixAt(i, td.matrix);
}
reactor.add(ticks);

// second, denser ring of finer ticks further out — more machined detail
const TICK2 = 120;
const ticks2 = new THREE.InstancedMesh(
  new THREE.BoxGeometry(0.05, 0.012, 0.001),
  ringMat(C_BLUE, 0.4),
  TICK2
);
for (let i = 0; i < TICK2; i++) {
  const a = (i / TICK2) * Math.PI * 2, r = 1.9;
  td.position.set(Math.cos(a) * r, Math.sin(a) * r, 0);
  td.rotation.z = a;
  td.updateMatrix();
  ticks2.setMatrixAt(i, td.matrix);
}
reactor.add(ticks2);

// --- the core: a bright hard disc with a soft hot center ---
const coreDisc = new THREE.Mesh(
  new THREE.CircleGeometry(0.4, 96),
  new THREE.MeshBasicMaterial({ color: C_HOT, transparent: true, opacity: 1.0, blending: THREE.AdditiveBlending, depthWrite: false })
);
reactor.add(coreDisc);

// hot center via a radial-gradient sprite (fades to transparent — no hard edge)
function makeGlowTexture() {
  const s = 256;
  const cv = document.createElement('canvas');
  cv.width = cv.height = s;
  const ctx = cv.getContext('2d');
  const g = ctx.createRadialGradient(s/2, s/2, 0, s/2, s/2, s/2);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.25, 'rgba(200,240,255,0.9)');
  g.addColorStop(0.55, 'rgba(46,158,255,0.35)');
  g.addColorStop(1, 'rgba(46,158,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  const tex = new THREE.CanvasTexture(cv);
  return tex;
}
const glowTex = makeGlowTexture();
const coreGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: 0xffffff, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false }));
coreGlow.scale.set(1.6, 1.6, 1);
reactor.add(coreGlow);

// modest soft halo behind the reactor. NOTE: an earlier version scaled
// this huge (7.5) and it filled the container as a lit blue rectangle.
// Kept small here; the CSS radial mask on #orb-container does the real
// edge-fade work so no box can appear.
const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: 0x1c6bb0, transparent: true, opacity: 0.18, blending: THREE.AdditiveBlending, depthWrite: false }));
halo.scale.set(4.2, 4.2, 1);
halo.position.z = -0.5;
reactor.add(halo);

// palette / behavior per state
const PALETTE = {
  idle:      { bright: 0.4,  bloom: 0.35, spin: 0.06, halo: 0.2,  core: 0.45 },
  listening: { bright: 0.8,  bloom: 0.5,  spin: 0.3,  halo: 0.35, core: 0.85 },
  thinking:  { bright: 0.9,  bloom: 0.6,  spin: 0.7,  halo: 0.4,  core: 0.9 },
  speaking:  { bright: 1.0,  bloom: 0.7,  spin: 0.45, halo: 0.45, core: 1.0 },
};

let flashStrength = 0;
function triggerWakeFlash() { flashStrength = 1.0; }

// Per-answer core hue. Each new answer settles the core into one of a
// curated, tasteful set — deep blue, azure, cyan, teal-white, cool violet.
// Kept in a tight premium range so it reads as intentional accent shifts,
// never a rainbow. The core eases toward answerHue; on speech it drifts
// very subtly around it.
const ANSWER_HUES = [0x2e9eff, 0x3ad0ff, 0x6ee7ff, 0x4f8bff, 0x8a9bff, 0x36c8d8];
const answerColor = new THREE.Color(0x6ee7ff);   // current target
let answerHueIdx = 2;
function pickAnswerHue() {
  // step to a different hue than last time
  let next = answerHueIdx;
  while (next === answerHueIdx) next = Math.floor(Math.random() * ANSWER_HUES.length);
  answerHueIdx = next;
  answerColor.setHex(ANSWER_HUES[next]);
}

let t = 0;
let curBright = 0.5;
let hotAngle = 0;   // (unused now that flare is syllable-triggered; kept harmless)
let lastAmp = 0;         // previous frame amplitude, for spike detection
let spikeCooldown = 0;   // refractory frames so one syllable fires one arc
let ampAvg = 0;          // running average amplitude, for adaptive spike threshold
let baseTick = 0;        // stagger counter for the continuous base flare
const _tmpColor = new THREE.Color();

function animate() {
  requestAnimationFrame(animate);
  t += 0.016;
  smoothedAmplitude += (amplitude - smoothedAmplitude) * 0.25;
  flashStrength = Math.max(0, flashStrength - 0.02);

  const p = PALETTE[currentState] || PALETTE.idle;
  curBright += (p.bright + flashStrength * 0.7 - curBright) * 0.08;
  // bloom stays off (glow removed per request)

  // counter-rotating layers — the mechanical arc-reactor motion.
  // Outer arcs spin only slowly so the travelling voice-flare (below) reads
  // as moving around a near-stationary ring rather than chasing spin.
  // ROTATION: everything in the middle turns — the inner coil ring, both
  // tick rings, and the concentric rings — each at its own speed and
  // direction for that layered mechanical feel. What stays STILL: the
  // outermost arcs (arcGroup2, the flare ring — it needs a fixed frame for
  // the travelling flare to read against) and the core (the stable anchor).
  const spin = p.spin;
  arcGroup1.rotation.z += 0.012 * spin;
  ticks.rotation.z += 0.006 * spin;
  ticks2.rotation.z -= 0.009 * spin;
  rings[1].rotation.z += 0.008 * spin;
  rings[2].rotation.z -= 0.005 * spin;
  rings[3].rotation.z += 0.003 * spin;
  // rings[0] (innermost) and arcGroup2 (outermost arcs) intentionally do NOT spin

  // amplitude + state drive brightness of everything
  const energy = curBright + smoothedAmplitude * 0.8;
  rings.forEach((r) => { r.material.opacity = Math.min(1, r.material.opacity + (energy * 0.7 - r.material.opacity) * 0.1); });

  // PERSONALITY: instead of the whole ring breathing at once, a "hot spot"
  // travels around the reactor. Only the arc(s) nearest the hot spot flare
  // OUT far; the rest sit at rest. The spot's travel speed and how hard it
  // flares both scale with your voice, so louder speech = bigger, faster
  // travelling flares. This is the "one or two shoot out, then another"
  // motion rather than a uniform pulse.
  //
  // hotAngle sweeps around; it drifts faster when you're louder, and also
  // jumps a little on amplitude spikes so syllables feel like they "land"
  // at different points on the ring.
  // GENTLE OUTER-ARC FLARE. Only the true OUTERMOST ring (arcGroup2, radius
  // 2.2) moves — it eases outward smoothly and eases back, like an
  // animation, never a shake. The inner coil ring (arcGroup1) does NOT flare
  // at all here; it only rotates (handled in the rotation block above).
  //
  // While speaking, a random outer arc is gently selected to swell out,
  // sized by the current amplitude. Selection is staggered and slow so it
  // reads as calm, deliberate motion — one arc eases out, eases back, then
  // another — tied to his voice level, not a periodic sweep.
  const ampNow = smoothedAmplitude;
  const speaking = currentState === 'speaking' && ampNow > 0.02;

  baseTick -= 1;
  if (speaking && baseTick <= 0) {
    // WIDER motion: push range roughly doubled so arcs travel visibly far.
    const idx = Math.floor(Math.random() * arcGroup2.children.length);
    arcGroup2.children[idx].userData.target = 0.28 + ampNow * 0.85;
    // sometimes a neighbouring arc swells too (smaller) — makes the motion
    // feel organic and connected rather than one isolated arc at a time
    if (Math.random() < 0.55) {
      const dir = Math.random() < 0.5 ? 1 : -1;
      const nb = (idx + dir + arcGroup2.children.length) % arcGroup2.children.length;
      const n = arcGroup2.children[nb];
      n.userData.target = Math.max(n.userData.target, 0.14 + ampNow * 0.45);
    }
    baseTick = 8 + Math.floor(Math.random() * 9);
  }

  arcGroup2.children.forEach((a, i) => {
    a.userData.target *= 0.955;
    // per-arc breathing: a slow sine unique to each arc, so even at rest the
    // ring has subtle life and the flares never look mechanically identical
    const breathe = (Math.sin(t * 0.7 + i * 1.7) * 0.5 + 0.5) * 0.03;
    const goal = a.userData.target + breathe;
    a.userData.curPush += (goal - a.userData.curPush) * 0.07;   // slow, smooth ease
    a.scale.setScalar(1 + a.userData.curPush / a.userData.baseRadius);
    a.material.opacity = Math.min(1, 0.5 + energy * 0.25 + a.userData.curPush * 0.5);
  });

  // inner coil ring: brightness only, no outward push — it just rotates
  arcGroup1.children.forEach((a) => {
    a.material.opacity = Math.min(1, 0.55 + energy * 0.3 + ampNow * 0.2);
  });

  const corePulse = (currentState === 'speaking')
    ? p.core + smoothedAmplitude * 0.8
    : p.core + Math.sin(t * (currentState === 'idle' ? 0.8 : 2.5)) * 0.12;
  coreDisc.material.opacity = Math.min(1, corePulse);
  const cs = 1 + smoothedAmplitude * 0.4 + flashStrength * 0.3;
  coreDisc.scale.setScalar(cs);
  // the innermost ball takes on the per-answer hue (eased, so it glides).
  // A tiny speech-driven drift keeps it alive while talking.
  const drift = 0.06 * Math.sin(t * 1.7) * smoothedAmplitude;
  _tmpColor.copy(answerColor).offsetHSL(drift, 0, 0);
  coreDisc.material.color.lerp(_tmpColor, 0.04);
  coreGlow.material.color.lerp(_tmpColor, 0.04);
  coreGlow.material.opacity = Math.min(1, (p.core + smoothedAmplitude * 0.6) * 0.7);
  coreGlow.scale.setScalar(1.6 * (1 + smoothedAmplitude * 0.25));

  halo.material.opacity = Math.min(0.5, p.halo * 0.6 + smoothedAmplitude * 0.15 + flashStrength * 0.15);

  // very slight 3D tilt so it has physical depth without breaking the flat HUD read
  reactor.rotation.x = Math.sin(t * 0.2) * 0.06;
  reactor.rotation.y = Math.sin(t * 0.15) * 0.08;

  composer.render();
}

window.addEventListener('resize', () => {
  width = container.clientWidth || 400;
  height = container.clientHeight || 400;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  composer.setSize(width, height);
  bloom.setSize(width, height);
});

animate();
connect();

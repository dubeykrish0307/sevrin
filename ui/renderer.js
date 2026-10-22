// SEVRIN desktop app. Connects to the Python backend over a local WebSocket,
// shows the conversation as a few floating lines under the orb, and starts
// and stops the backend through preload.js.
import * as hud from './hud.js';
import { createReactor } from './reactor.js';

const WS_URL = 'ws://localhost:8765';
const MAX_LINES = 3;
const LINE_LIFETIME_MS = 9000;

const $ = (id) => document.getElementById(id);
const statusDot = $('statusDot');
const statusText = $('statusText');
const stateLabel = $('stateLabel');
const textInput = $('textInput');
const speechLayer = $('speechLayer');
const clockEl = $('clock');
const backendBtn = $('backendBtn');
const backendState = $('backendState');
const logEl = $('hudLog');
const logFileEl = $('logFile');
const openLogsBtn = $('openLogs');

const reactor = createReactor($('orb-container'));

let ws = null;
let currentState = 'idle';
let backendRunning = false;
let activeLine = null;          // the assistant line currently being filled in
let nextLineKind = null;        // set by a 'proactive' event: the next line wasn't a reply

// --- backend control and log --------------------------------------------------
// window.sevrin comes from preload.js and is missing if the page is opened
// outside Electron.

function appendLog(line) {
  const row = document.createElement('div');
  if (/error|Traceback|failed/i.test(line)) row.className = 'err';
  else if (/confirmed|websocket server|using ECAPA/i.test(line)) row.className = 'ok';
  row.textContent = line;
  logEl.appendChild(row);
  while (logEl.children.length > 200) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}

function setBackendStatus(status) {
  backendRunning = status === 'running' || status === 'starting';
  backendState.textContent = status;
  backendBtn.classList.toggle('running', status === 'running');
  backendBtn.classList.toggle('starting', status === 'starting');
  backendBtn.textContent = status === 'running' ? 'STOP BACKEND'
    : status === 'starting' ? 'STARTING…'
    : 'START BACKEND';
}

function refreshLogName() {
  window.sevrin.currentLog().then((name) => { logFileEl.textContent = name || ''; });
}

if (window.sevrin) {
  window.sevrin.onBackendLog(appendLog);
  window.sevrin.onBackendStatus(setBackendStatus);
  window.sevrin.onBackendStatus(() => setTimeout(refreshLogName, 200));
  window.sevrin.getBackendStatus().then(setBackendStatus);
  refreshLogName();

  backendBtn.addEventListener('click', () => {
    if (backendRunning) window.sevrin.stopBackend();
    else window.sevrin.startBackend();
  });
  openLogsBtn.addEventListener('click', () => window.sevrin.revealLogs());
} else {
  backendBtn.textContent = 'N/A';
  backendBtn.disabled = true;
  backendState.textContent = 'no electron';
}

// --- floating speech lines ------------------------------------------------------

function fadeOut(line) {
  if (!line || !line.isConnected) return;
  line.classList.add('fading');
  setTimeout(() => { if (line.isConnected) line.remove(); }, 800);
}

function restartFade(line) {
  clearTimeout(line._timer);
  line._timer = setTimeout(() => fadeOut(line), LINE_LIFETIME_MS);
}

function pushLine(role, text) {
  if (role === 'user') hud.shortenOpenPanels();
  const line = document.createElement('div');
  line.className = `speech-line ${role}`;
  line.textContent = text;
  speechLayer.appendChild(line);
  while (speechLayer.children.length > MAX_LINES) speechLayer.removeChild(speechLayer.firstChild);
  restartFade(line);
  return line;
}

// --- backend events -------------------------------------------------------------

function setState(state) {
  const wasIdle = currentState === 'idle';
  currentState = state;
  reactor.setState(state);
  stateLabel.textContent = state.toUpperCase();
  if (wasIdle && state === 'listening') reactor.flash();
}

function startAssistantLine() {
  activeLine = pushLine('assistant', '');
  if (nextLineKind) {
    // spoken unprompted, so mark it; otherwise it looks like an answer to nothing
    activeLine.classList.add('proactive');
    activeLine.dataset.kind = nextLineKind;
    nextLineKind = null;
  }
  reactor.nextHue();
}

function appendToAssistantLine(text) {
  if (!activeLine) return;
  activeLine.textContent += (activeLine.textContent ? ' ' : '') + text;
  restartFade(activeLine);
}

function markInterrupted() {
  // the cut-off reply stops growing and dims
  if (activeLine) {
    activeLine.style.opacity = '0.4';
    activeLine = null;
  }
  stateLabel.textContent = 'INTERRUPTED';
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'state': setState(msg.value); break;
    case 'transcript': pushLine('user', msg.text); break;
    case 'transcript_start': startAssistantLine(); break;
    case 'transcript_delta': appendToAssistantLine(msg.text); break;
    case 'transcript_end': activeLine = null; break;
    case 'amplitude': reactor.setAmplitude(msg.value); break;
    case 'interrupted': markInterrupted(); break;
    case 'proactive': nextLineKind = msg.kind || 'proactive'; break;
    case 'ui_panel':
      if (msg.action === 'hide') hud.hidePanel(msg.panel);
      else hud.showPanel(msg.panel);
      break;
    case 'schedules': hud.renderSchedules(msg.data); break;
    case 'memory_event': hud.pushMemoryEvent(msg.data); break;
    case 'memory_stats': hud.setMemoryCount(msg.data.active_facts); break;
    // not sent by the backend yet; the panels are ready for them
    case 'hud_tasks': hud.setTasks(msg.data); break;
    case 'hud_alerts': hud.setAlerts(msg.data); break;
    case 'hud_meters': hud.setMeters(msg.data); break;
    case 'error': console.error('[backend error]', msg.message); break;
    default: break;
  }
}

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    statusDot.classList.add('connected');
    statusText.textContent = 'ONLINE';
  };
  ws.onclose = () => {
    statusDot.classList.remove('connected');
    statusText.textContent = 'OFFLINE';
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      return;
    }
    handleMessage(msg);
  };
}

textInput.addEventListener('keydown', (e) => {
  const text = textInput.value.trim();
  if (e.key !== 'Enter' || !text) return;
  pushLine('user', text);
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'text_input', text }));
  textInput.value = '';
});

function tickClock() {
  clockEl.textContent = new Date().toLocaleTimeString('en-GB');
}

tickClock();
setInterval(tickClock, 1000);
hud.startHud(() => currentState);
connect();

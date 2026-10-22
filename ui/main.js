// Electron main process: the window, starting/stopping the Python backend,
// per-session log files, and hot reload while developing.
const { app, BrowserWindow, ipcMain, shell } = require('electron');
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const PROJECT_ROOT = path.join(__dirname, '..');
const LOG_DIR = path.join(PROJECT_ROOT, 'logs');
const BACKEND_PORT = 8765;

let win = null;
let backendProc = null;
let backendStatus = 'stopped';
let logStream = null;
let currentLogPath = null;

// --- session logs -----------------------------------------------------------
// Each backend run gets its own numbered file, logs/Test N.log, so a whole
// session can be shared as a file instead of copied out of a terminal.

function nextLogPath() {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  let max = 0;
  for (const f of fs.readdirSync(LOG_DIR)) {
    const m = /^Test (\d+)\.log$/.exec(f);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return path.join(LOG_DIR, `Test ${max + 1}.log`);
}

function openLogFile() {
  try {
    currentLogPath = nextLogPath();
    logStream = fs.createWriteStream(currentLogPath, { flags: 'a' });
    logStream.write(
      '=== SEVRIN session log ===\n' +
      `started : ${new Date().toISOString()}\n` +
      `file    : ${path.basename(currentLogPath)}\n` +
      `${'='.repeat(40)}\n`,
    );
    return path.basename(currentLogPath);
  } catch (e) {
    console.warn('could not open log file:', e.message);
    logStream = null;
    return null;
  }
}

function closeLogFile() {
  if (!logStream) return;
  logStream.write(`\n=== session ended ${new Date().toISOString()} ===\n`);
  logStream.end();
  logStream = null;
}

// progress bars from model downloads leave ANSI escapes that clutter the log
const ANSI = /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g;

function stripAnsi(s) {
  return s.replace(ANSI, '').replace(/\[\d*[A-Z]/g, '').trim();
}

function writeLog(line) {
  if (!logStream) return;
  const clean = stripAnsi(line);
  if (!clean) return;
  const time = new Date().toISOString().slice(11, 23);
  logStream.write(`[${time}] ${clean}\n`);
}

// --- backend process ----------------------------------------------------------

function pythonPath() {
  // prefer the project venv, so it works even if Electron wasn't started from it
  for (const candidate of ['python3', 'python'].map((p) => path.join(PROJECT_ROOT, 'venv', 'bin', p))) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return 'python3';
}

function setStatus(status) {
  backendStatus = status;
  if (win && !win.isDestroyed()) win.webContents.send('backend:status', status);
}

function sendLog(line) {
  writeLog(line);
  if (win && !win.isDestroyed()) win.webContents.send('backend:log', line);
}

// A backend left running from a terminal keeps the port, and a new one would
// die with "address already in use". Kill whatever holds it first.
function freePort(port) {
  try {
    const pids = execSync(`lsof -ti:${port} || true`, { encoding: 'utf8' })
      .split('\n').map((x) => x.trim()).filter(Boolean);
    for (const pid of pids) {
      if (parseInt(pid, 10) === process.pid) continue;
      try {
        execSync(`kill -9 ${pid}`);
        sendLog(`[ui] freed port ${port} (killed stale process ${pid})`);
      } catch {
        // already gone
      }
    }
    return pids.length;
  } catch {
    return 0;
  }
}

function startBackend() {
  if (backendProc) return { ok: true, already: true };

  const py = pythonPath();
  const logName = openLogFile();
  setStatus('starting');
  if (logName) sendLog(`[ui] logging this session to logs/${logName}`);
  freePort(BACKEND_PORT);
  sendLog(`[ui] starting backend with ${py}`);

  backendProc = spawn(py, ['backend/service.py'], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },   // stream logs live
  });

  backendProc.stdout.on('data', (d) => {
    const text = d.toString();
    text.split('\n').filter(Boolean).forEach(sendLog);
    if (text.includes('websocket server on')) setStatus('running');
  });
  backendProc.stderr.on('data', (d) => {
    d.toString().split('\n').filter(Boolean).forEach(sendLog);
  });
  backendProc.on('exit', (code) => {
    sendLog(`[ui] backend exited (code ${code})`);
    backendProc = null;
    setStatus('stopped');
    closeLogFile();
  });
  backendProc.on('error', (err) => {
    sendLog(`[ui] failed to start backend: ${err.message}`);
    backendProc = null;
    setStatus('error');
  });

  return { ok: true };
}

function stopBackend() {
  if (!backendProc) return { ok: true, already: true };
  sendLog('[ui] stopping backend');
  const proc = backendProc;
  proc.kill('SIGTERM');
  // SIGTERM can hang if it's in the middle of audio
  setTimeout(() => { try { proc.kill('SIGKILL'); } catch { /* gone */ } }, 2500);
  backendProc = null;
  setStatus('stopped');
  return { ok: true };
}

function restartBackend() {
  stopBackend();
  setTimeout(startBackend, 600);   // give the mic a moment to be released
  return { ok: true };
}

ipcMain.handle('backend:start', () => startBackend());
ipcMain.handle('backend:stop', () => stopBackend());
ipcMain.handle('backend:restart', () => restartBackend());
ipcMain.handle('backend:status', () => backendStatus);
ipcMain.handle('logs:current', () => (currentLogPath ? path.basename(currentLogPath) : null));
ipcMain.handle('logs:reveal', () => {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  shell.openPath(LOG_DIR);
  return true;
});

// --- hot reload -----------------------------------------------------------------
// Editing a file under ui/ reloads the window; editing any .py restarts the
// backend. Debounced because editors fire several events per save.

function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

function watch(dir, options, onChange) {
  if (!fs.existsSync(dir)) return;
  try {
    fs.watch(dir, options, (_evt, filename) => { if (filename) onChange(filename); });
  } catch (e) {
    console.warn(`watching ${dir} failed:`, e.message);
  }
}

function watchForChanges() {
  const reloadWindow = debounce(() => {
    if (win && !win.isDestroyed()) {
      sendLog('[ui] reloading window (ui file changed)');
      win.reload();
    }
  }, 250);

  const restartForPython = debounce((file) => {
    if (!backendProc) return;
    sendLog(`[ui] ${file} changed, restarting backend`);
    restartBackend();
  }, 600);

  // main.js and preload.js only load at startup, so reloading wouldn't help
  watch(__dirname, { recursive: false }, (f) => {
    if (/\.(js|html|css)$/.test(f) && f !== 'main.js' && f !== 'preload.js') reloadWindow();
  });
  for (const dir of ['backend', 'brain', 'voice']) {
    watch(path.join(PROJECT_ROOT, dir), { recursive: true }, (f) => {
      if (f.endsWith('.py')) restartForPython(f);
    });
  }
  // config.py and main.py at the root
  watch(PROJECT_ROOT, { recursive: false }, (f) => {
    if (f.endsWith('.py')) restartForPython(f);
  });
}

// ---------------------------------------------------------------------------------

function createWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 780,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: '#000000',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });
  win.loadFile('index.html');
}

app.whenReady().then(() => {
  createWindow();
  watchForChanges();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// never leave an orphaned python process behind
app.on('before-quit', () => { stopBackend(); });
app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') app.quit();
});

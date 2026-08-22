const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const PROJECT_ROOT = path.join(__dirname, '..');

let win = null;
let backendProc = null;
let backendStatus = 'stopped';
let logStream = null;
let currentLogPath = null;

// ---------------------------------------------------------------------------
// Session logs
// Every backend run gets its own numbered file in logs/ — "Test 1.log",
// "Test 2.log", and so on — so a whole session can be handed over as a file
// instead of scraped out of a terminal buffer. Numbering continues from
// whatever's already there rather than restarting at 1.
// ---------------------------------------------------------------------------

const LOG_DIR = path.join(PROJECT_ROOT, 'logs');

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
    const header =
      `=== SEVRIN session log ===\n` +
      `started : ${new Date().toISOString()}\n` +
      `file    : ${path.basename(currentLogPath)}\n` +
      `${'='.repeat(40)}\n`;
    logStream.write(header);
    return path.basename(currentLogPath);
  } catch (e) {
    console.warn('could not open log file:', e.message);
    logStream = null;
    return null;
  }
}

function closeLogFile() {
  if (logStream) {
    logStream.write(`\n=== session ended ${new Date().toISOString()} ===\n`);
    logStream.end();
    logStream = null;
  }
}

function writeLog(line) {
  if (!logStream) return;
  // timestamp each line so timing issues (latency, races) are diagnosable
  const t = new Date().toISOString().substr(11, 12);
  logStream.write(`[${t}] ${line}\n`);
}

// ---------------------------------------------------------------------------
// Backend process control
// ---------------------------------------------------------------------------

function pythonPath() {
  // Prefer the project venv so the backend gets the right dependencies even
  // when Electron was launched from a shell without the venv activated.
  const candidates = [
    path.join(PROJECT_ROOT, 'venv', 'bin', 'python3'),
    path.join(PROJECT_ROOT, 'venv', 'bin', 'python'),
    'python3',
  ];
  for (const c of candidates) {
    if (c === 'python3' || fs.existsSync(c)) return c;
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

function freePort(port) {
  // A backend left running from an earlier terminal session keeps holding
  // 8765, and the new one dies instantly with "address already in use".
  // Clear it first so the button always works instead of silently failing.
  try {
    const { execSync } = require('child_process');
    const pids = execSync(`lsof -ti:${port} || true`, { encoding: 'utf8' })
      .split('\n').map((x) => x.trim()).filter(Boolean);
    for (const pid of pids) {
      // never kill ourselves
      if (parseInt(pid, 10) === process.pid) continue;
      try {
        execSync(`kill -9 ${pid}`);
        sendLog(`[ui] freed port ${port} (killed stale process ${pid})`);
      } catch (e) { /* already gone */ }
    }
    return pids.length;
  } catch (e) {
    return 0;
  }
}

function startBackend() {
  if (backendProc) return { ok: true, already: true };

  const py = pythonPath();
  const logName = openLogFile();
  setStatus('starting');
  if (logName) sendLog(`[ui] logging this session to logs/${logName}`);
  freePort(8765);
  sendLog(`[ui] starting backend with ${py}`);

  backendProc = spawn(py, ['backend/service.py'], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },  // so logs stream live
  });

  backendProc.stdout.on('data', (d) => {
    const text = d.toString();
    text.split('\n').filter(Boolean).forEach(sendLog);
    if (text.includes('websocket server on')) setStatus('running');
  });
  backendProc.stderr.on('data', (d) => {
    d.toString().split('\n').filter(Boolean).forEach((l) => sendLog(l));
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
  backendProc.kill('SIGTERM');
  // SIGTERM can hang if the backend is mid-audio; force it shortly after
  const proc = backendProc;
  setTimeout(() => { try { proc.kill('SIGKILL'); } catch (e) {} }, 2500);
  backendProc = null;
  setStatus('stopped');
  return { ok: true };
}

function restartBackend() {
  stopBackend();
  setTimeout(startBackend, 600);   // let the mic device be released first
  return { ok: true };
}

ipcMain.handle('backend:start', () => startBackend());
ipcMain.handle('backend:stop', () => stopBackend());
ipcMain.handle('backend:restart', () => restartBackend());
ipcMain.handle('backend:status', () => backendStatus);
ipcMain.handle('logs:current', () => (currentLogPath ? path.basename(currentLogPath) : null));
ipcMain.handle('logs:reveal', () => {
  const { shell } = require('electron');
  fs.mkdirSync(LOG_DIR, { recursive: true });
  // open the folder so the file can be grabbed and shared directly
  shell.openPath(LOG_DIR);
  return true;
});

// ---------------------------------------------------------------------------
// Hot reload — edit code, see it live, no manual restarts
//   * changing anything under ui/  reloads the window
//   * changing any .py          restarts the backend process
// Both are debounced, because editors emit several events per save.
// ---------------------------------------------------------------------------

function watchForChanges() {
  const debounce = (fn, ms) => {
    let t = null;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  };

  // --- UI files -> reload the renderer ---
  const reloadUI = debounce(() => {
    if (win && !win.isDestroyed()) {
      sendLog('[ui] reloading window (ui file changed)');
      win.reload();
    }
  }, 250);

  try {
    fs.watch(__dirname, { recursive: false }, (_evt, filename) => {
      if (!filename) return;
      if (/\.(js|html|css)$/.test(filename) && filename !== 'main.js' && filename !== 'preload.js') {
        reloadUI();
      }
    });
  } catch (e) {
    console.warn('ui watch failed:', e.message);
  }

  // --- Python files -> restart the backend ---
  const reloadBackend = debounce((file) => {
    if (!backendProc) return;   // nothing running; nothing to restart
    sendLog(`[ui] ${file} changed — restarting backend`);
    restartBackend();
  }, 600);

  ['backend', 'brain', 'voice', 'connections'].forEach((dir) => {
    const full = path.join(PROJECT_ROOT, dir);
    if (!fs.existsSync(full)) return;
    try {
      fs.watch(full, { recursive: true }, (_evt, filename) => {
        if (filename && filename.endsWith('.py')) reloadBackend(filename);
      });
    } catch (e) {
      console.warn(`watch ${dir} failed:`, e.message);
    }
  });

  // config.py sits at the root, watched individually
  try {
    fs.watch(PROJECT_ROOT, { recursive: false }, (_evt, filename) => {
      if (filename && filename.endsWith('.py')) reloadBackend(filename);
    });
  } catch (e) {
    console.warn('root watch failed:', e.message);
  }
}

// ---------------------------------------------------------------------------

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

// make sure we never leave an orphaned python process behind
app.on('before-quit', () => { stopBackend(); });
app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') app.quit();
});

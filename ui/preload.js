const { contextBridge, ipcRenderer } = require('electron');

// Bridge between the sandboxed page and the Electron main process. The page
// can't spawn processes itself (contextIsolation is on, which we want), so
// backend control goes through here.
contextBridge.exposeInMainWorld('sevrin', {
  startBackend: () => ipcRenderer.invoke('backend:start'),
  stopBackend: () => ipcRenderer.invoke('backend:stop'),
  restartBackend: () => ipcRenderer.invoke('backend:restart'),
  getBackendStatus: () => ipcRenderer.invoke('backend:status'),
  onBackendLog: (cb) => ipcRenderer.on('backend:log', (_e, line) => cb(line)),
  onBackendStatus: (cb) => ipcRenderer.on('backend:status', (_e, status) => cb(status)),
  currentLog: () => ipcRenderer.invoke('logs:current'),
  revealLogs: () => ipcRenderer.invoke('logs:reveal'),
});

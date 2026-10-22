const { contextBridge, ipcRenderer } = require('electron');

// The page runs with contextIsolation and can't spawn processes, so backend
// control and log access go through the main process.
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

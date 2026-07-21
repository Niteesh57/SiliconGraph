const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("siliconGraph", {
  listDevices: () => ipcRenderer.invoke("devices:list"),
  startCompilation: (request) => ipcRenderer.invoke("compile:start", request),
  huggingFace: {
    getStatus: () => ipcRenderer.invoke("huggingface:status"),
    saveToken: (token) => ipcRenderer.invoke("huggingface:save-token", token),
    clearToken: () => ipcRenderer.invoke("huggingface:clear-token"),
    search: (query) => ipcRenderer.invoke("huggingface:search", query),
    listLocal: () => ipcRenderer.invoke("huggingface:list-local"),
    download: (request) => ipcRenderer.invoke("huggingface:download", request),
    cancelDownload: (id) => ipcRenderer.invoke("huggingface:cancel-download", id),
    onDownloadProgress: (listener) => {
      const handler = (_event, message) => listener(message);
      ipcRenderer.on("huggingface:download-progress", handler);
      return () => ipcRenderer.removeListener("huggingface:download-progress", handler);
    }
  },
  onCompileOutput: (listener) => {
    const handler = (_event, message) => listener(message);
    ipcRenderer.on("compile:output", handler);
    return () => ipcRenderer.removeListener("compile:output", handler);
  },
  onCompileFinished: (listener) => {
    const handler = (_event, message) => listener(message);
    ipcRenderer.on("compile:finished", handler);
    return () => ipcRenderer.removeListener("compile:finished", handler);
  }
});

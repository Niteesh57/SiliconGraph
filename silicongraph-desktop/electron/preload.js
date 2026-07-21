const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("siliconGraph", {
  listDevices: () => ipcRenderer.invoke("devices:list"),
  listProfiles: () => ipcRenderer.invoke("profiles:list"),
  startCompilation: (request) => ipcRenderer.invoke("compile:start", request),
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

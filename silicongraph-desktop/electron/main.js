const { app, BrowserWindow, ipcMain, safeStorage } = require("electron");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const { buildDynamicProfile, listDevices } = require("./adb");

const projectRoot = path.resolve(__dirname, "..", "..");
const activeJobs = new Map();
const activeDownloads = new Map();
const localModelsDirectory = path.join(projectRoot, "models");
const huggingFaceSettingsPath = path.join(app.getPath("userData"), "huggingface-settings.json");

function readHuggingFaceSettings() {
  try {
    const settings = JSON.parse(fs.readFileSync(huggingFaceSettingsPath, "utf8"));
    if (!settings.token || !safeStorage.isEncryptionAvailable()) return { token: "" };
    return { token: safeStorage.decryptString(Buffer.from(settings.token, "base64")) };
  } catch {
    return { token: "" };
  }
}

async function saveHuggingFaceSettings(settings) {
  if (!safeStorage.isEncryptionAvailable()) throw new Error("Windows secure credential storage is unavailable, so the Hugging Face token cannot be saved safely.");
  await fsp.mkdir(path.dirname(huggingFaceSettingsPath), { recursive: true });
  const encrypted = safeStorage.encryptString(String(settings.token || "")).toString("base64");
  // The renderer never receives the token. Electron encrypts this value with
  // the current Windows user's credential store before it reaches disk.
  await fsp.writeFile(huggingFaceSettingsPath, `${JSON.stringify({ token: encrypted })}\n`, { mode: 0o600 });
}

function huggingFaceHeaders() {
  const { token } = readHuggingFaceSettings();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function validateRepositoryId(value) {
  const repoId = String(value || "").trim().replace(/^https?:\/\/huggingface\.co\//i, "").replace(/\/$/, "");
  if (!/^[\w.-]+\/[\w.-]+$/.test(repoId)) throw new Error("Enter a Hugging Face repository ID in the form organization/model-name.");
  return repoId;
}

function repositoryDirectory(repoId) {
  const [owner, name] = validateRepositoryId(repoId).split("/");
  return path.join(localModelsDirectory, `${owner}--${name}`);
}

function isWithinDirectory(parent, child) {
  const relative = path.relative(parent, child);
  return relative && !relative.startsWith("..") && !path.isAbsolute(relative);
}

async function hfJson(url) {
  const response = await fetch(url, { headers: huggingFaceHeaders() });
  if (!response.ok) {
    if (response.status === 401 || response.status === 403) throw new Error("Hugging Face rejected this request. Add a token with read access in Hugging Face setup.");
    if (response.status === 404) throw new Error("Model not found, or your token does not have access to it.");
    throw new Error(`Hugging Face request failed (${response.status}).`);
  }
  return response.json();
}

function normalizeHubModel(model) {
  return {
    id: model.id || model.modelId,
    author: model.author || "",
    downloads: Number(model.downloads || 0),
    likes: Number(model.likes || 0),
    lastModified: model.lastModified || null,
    private: Boolean(model.private),
    gated: Boolean(model.gated),
    pipelineTag: model.pipeline_tag || ""
  };
}

function selectCompilerFiles(repositoryFiles) {
  const rootFiles = repositoryFiles.filter((file) => file.rfilename && !file.rfilename.includes("/"));
  const safeTensors = rootFiles.filter((file) => /(?:^|_)model.*\.safetensors$/i.test(file.rfilename) || /^model-\d+-of-\d+\.safetensors$/i.test(file.rfilename));
  const binaryWeights = rootFiles.filter((file) => /^(?:pytorch_)?model.*\.bin$/i.test(file.rfilename));
  const weights = safeTensors.length ? safeTensors : binaryWeights;
  const supportFiles = rootFiles.filter((file) =>
    /^(?:config|generation_config|tokenizer_config|special_tokens_map|preprocessor_config|processor_config|chat_template)\.json$/i.test(file.rfilename) ||
    /^(?:tokenizer|vocab|merges)\.(?:json|txt|model)$/i.test(file.rfilename) ||
    /\.index\.json$/i.test(file.rfilename)
  );
  const selected = [...new Map([...supportFiles, ...weights].map((file) => [file.rfilename, file])).values()];
  if (!weights.length) throw new Error("This repository does not contain root-level Safetensors or PyTorch model weights that SiliconGraph can compile.");
  return selected;
}

async function searchHuggingFaceModels(_event, query) {
  const search = String(query || "").trim();
  const params = new URLSearchParams({ limit: "18", full: "true", sort: "downloads", direction: "-1" });
  if (search) params.set("search", search);
  const models = await hfJson(`https://huggingface.co/api/models?${params}`);
  return models.map(normalizeHubModel);
}

async function listLocalModels() {
  const discovered = new Map();
  const addModel = (model) => discovered.set(model.path.toLowerCase(), model);
  await fsp.mkdir(localModelsDirectory, { recursive: true });
  const downloaded = await fsp.readdir(localModelsDirectory, { withFileTypes: true });
  for (const entry of downloaded) {
    if (!entry.isDirectory()) continue;
    const modelPath = path.join(localModelsDirectory, entry.name);
    if (!fs.existsSync(path.join(modelPath, "config.json"))) continue;
    const stat = await fsp.stat(modelPath);
    addModel({ name: entry.name.replace("--", "/"), path: modelPath, source: "SiliconGraph downloads", updatedAt: stat.mtimeMs });
  }

  const cacheRoots = [
    process.env.HF_HUB_CACHE,
    process.env.HUGGINGFACE_HUB_CACHE,
    process.env.HF_HOME && path.join(process.env.HF_HOME, "hub"),
    process.env.USERPROFILE && path.join(process.env.USERPROFILE, ".cache", "huggingface", "hub")
  ].filter(Boolean);
  for (const cacheRoot of new Set(cacheRoots)) {
    try {
      const entries = await fsp.readdir(cacheRoot, { withFileTypes: true });
      for (const entry of entries.filter((item) => item.isDirectory() && item.name.startsWith("models--"))) {
        try {
          const snapshots = path.join(cacheRoot, entry.name, "snapshots");
          const revisions = await fsp.readdir(snapshots, { withFileTypes: true });
          const revision = revisions.find((item) => item.isDirectory());
          if (!revision) continue;
          const modelPath = path.join(snapshots, revision.name);
          const stat = await fsp.stat(modelPath);
          addModel({ name: entry.name.slice(8).replace(/--/g, "/"), path: modelPath, source: "Hugging Face cache", updatedAt: stat.mtimeMs });
        } catch { /* Skip incomplete cache entries without hiding other models. */ }
      }
    } catch { /* The cache is optional, and may not exist yet. */ }
  }
  return [...discovered.values()].sort((a, b) => b.updatedAt - a.updatedAt);
}

async function downloadHuggingFaceModel(event, request) {
  const repoId = validateRepositoryId(request?.repoId);
  // The Hub route requires the namespace slash to remain a path separator;
  // encodeURIComponent(repoId) turns it into %2F and the API returns 400.
  // validateRepositoryId above limits the value to safe URL path characters.
  const model = await hfJson(`https://huggingface.co/api/models/${repoId}?blobs=true`);
  const repositoryFiles = (model.siblings || [])
    .filter((file) => file.rfilename && !file.rfilename.startsWith(".") && !file.rfilename.split("/").includes(".."));
  const files = selectCompilerFiles(repositoryFiles);
  if (!files.length) throw new Error("This repository does not expose downloadable files.");

  const destination = repositoryDirectory(repoId);
  const window = BrowserWindow.fromWebContents(event.sender);
  const downloadId = crypto.randomUUID();
  const abortController = new AbortController();
  activeDownloads.set(downloadId, abortController);
  const totalBytes = files.reduce((total, file) => total + Number(file.lfs?.size || file.size || 0), 0);
  let receivedBytes = 0;
  let completedFiles = 0;
  const sendProgress = (status, extra = {}) => sendJobEvent(window, "huggingface:download-progress", {
    downloadId, repoId, status, totalBytes, receivedBytes, completedFiles, totalFiles: files.length, ...extra
  });

  try {
    await fsp.mkdir(destination, { recursive: true });
    sendProgress("starting");
    for (const file of files) {
      const output = path.resolve(destination, file.rfilename);
      if (!isWithinDirectory(destination, output)) throw new Error("A repository file path was unsafe and was rejected.");
      await fsp.mkdir(path.dirname(output), { recursive: true });
      const filePath = file.rfilename.split("/").map(encodeURIComponent).join("/");
      const response = await fetch(`https://huggingface.co/${repoId}/resolve/main/${filePath}?download=true`, {
        headers: huggingFaceHeaders(), signal: abortController.signal, redirect: "follow"
      });
      if (!response.ok || !response.body) throw new Error(`Could not download ${file.rfilename} (${response.status}).`);
      const temporary = `${output}.partial`;
      const handle = await fsp.open(temporary, "w");
      try {
        for await (const chunk of response.body) {
          await handle.write(chunk);
          receivedBytes += chunk.length;
          sendProgress("downloading", { currentFile: file.rfilename });
        }
      } finally { await handle.close(); }
      await fsp.rename(temporary, output);
      completedFiles += 1;
      sendProgress("downloading", { currentFile: file.rfilename });
    }
    await fsp.writeFile(path.join(destination, ".silicongraph-model.json"), `${JSON.stringify({ repoId, downloadedAt: new Date().toISOString() })}\n`);
    sendProgress("completed", { path: destination });
    return { downloadId, repoId, path: destination };
  } catch (error) {
    sendProgress(abortController.signal.aborted ? "cancelled" : "failed", { error: error.message });
    throw error;
  } finally {
    activeDownloads.delete(downloadId);
  }
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1120,
    minHeight: 720,
    backgroundColor: "#08111f",
    title: "SiliconGraph Device Compiler",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  window.loadFile(path.join(__dirname, "..", "src", "index.html"));
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
}

function safeOutputName(modelName) {
  return modelName
    .trim()
    .split(/[\\/]/)
    .filter(Boolean)
    .pop()
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "model";
}

function sendJobEvent(window, channel, payload) {
  if (!window.isDestroyed()) window.webContents.send(channel, payload);
}

async function startCompilation(event, request) {
  const model = String(request?.model || "").trim();
  const deviceSerial = String(request?.deviceSerial || "").trim();
  const quantization = String(request?.quantization || "int8").trim();
  const contextLengths = String(request?.contextLengths || "512,2048").trim();
  const outputDirectory = path.resolve(String(request?.outputDirectory || path.join(projectRoot, "output")));

  if (!model) throw new Error("Enter a Hugging Face model ID, model URL, or local model directory.");
  if (!deviceSerial) throw new Error("Connect and select an Android device before compiling.");
  if (!/^(\d+)(,\d+)*$/.test(contextLengths)) {
    throw new Error("Context lengths must be a comma-separated list of positive integers.");
  }

  const liveDevice = (await listDevices()).find((device) =>
    !device.unavailable && device.serial === deviceSerial
  );
  if (!liveDevice) {
    throw new Error("The selected device is no longer connected or authorized through ADB.");
  }
  await fsp.mkdir(outputDirectory, { recursive: true });
  const outputPath = path.join(
    outputDirectory,
    `${safeOutputName(model)}-${safeOutputName(liveDevice.model)}-${safeOutputName(liveDevice.socModel)}.armpack`
  );
  const profileDirectory = path.join(outputDirectory, "profiles");
  await fsp.mkdir(profileDirectory, { recursive: true });
  const deviceProfilePath = path.join(profileDirectory, `${safeOutputName(deviceSerial)}.json`);
  await fsp.writeFile(deviceProfilePath, `${JSON.stringify(buildDynamicProfile(liveDevice), null, 2)}\n`);
  const pythonExecutable = process.env.PYTHON || "python";
  const args = [
    "-m", "armcc.cli", "compile",
    "--model", model,
    // The C++ cost model needs a stable enum target. The generated profile is
    // the source of truth and replaces static target-profile data.
    "--targets", "generic_arm64",
    "--quantization", quantization,
    "--context-lengths", contextLengths,
    "--output", outputPath
  ];
  args.push("--device-profile", deviceProfilePath);
  const pythonPath = [path.join(projectRoot, "python"), process.env.PYTHONPATH]
    .filter(Boolean)
    .join(path.delimiter);
  const huggingFaceToken = readHuggingFaceSettings().token;
  const jobId = crypto.randomUUID();
  const window = BrowserWindow.fromWebContents(event.sender);
  const child = spawn(pythonExecutable, args, {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONPATH: pythonPath,
      PYTHONUTF8: process.env.PYTHONUTF8 || "1",
      ...(huggingFaceToken ? { HF_TOKEN: huggingFaceToken, HUGGING_FACE_HUB_TOKEN: huggingFaceToken } : {})
    },
    windowsHide: true,
    shell: false
  });

  activeJobs.set(jobId, child);
  sendJobEvent(window, "compile:output", {
    jobId,
    stream: "system",
    text: `Generated live profile: ${deviceProfilePath}\nStarting: ${pythonExecutable} ${args.join(" ")}\n`
  });
  child.stdout.on("data", (data) => sendJobEvent(window, "compile:output", {
    jobId, stream: "stdout", text: data.toString()
  }));
  child.stderr.on("data", (data) => sendJobEvent(window, "compile:output", {
    jobId, stream: "stderr", text: data.toString()
  }));
  child.on("error", (error) => sendJobEvent(window, "compile:output", {
    jobId, stream: "stderr", text: `${error.message}\n`
  }));
  child.on("close", (exitCode, signal) => {
    activeJobs.delete(jobId);
    const packageCreated = exitCode === 0 && fs.existsSync(outputPath);
    sendJobEvent(window, "compile:finished", {
      jobId, exitCode, signal, outputPath, packageCreated
    });
  });

  return { jobId, outputPath };
}

app.whenReady().then(() => {
  ipcMain.handle("devices:list", () => listDevices());
  ipcMain.handle("compile:start", startCompilation);
  ipcMain.handle("huggingface:status", () => ({ configured: Boolean(readHuggingFaceSettings().token), localDirectory: localModelsDirectory }));
  ipcMain.handle("huggingface:save-token", async (_event, token) => {
    const trimmed = String(token || "").trim();
    if (!trimmed) throw new Error("Paste a Hugging Face access token before saving.");
    await saveHuggingFaceSettings({ token: trimmed });
    return { configured: true };
  });
  ipcMain.handle("huggingface:clear-token", async () => {
    await fsp.rm(huggingFaceSettingsPath, { force: true });
    return { configured: false };
  });
  ipcMain.handle("huggingface:search", searchHuggingFaceModels);
  ipcMain.handle("huggingface:list-local", listLocalModels);
  ipcMain.handle("huggingface:download", downloadHuggingFaceModel);
  ipcMain.handle("huggingface:cancel-download", (_event, downloadId) => activeDownloads.get(downloadId)?.abort());
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  for (const child of activeJobs.values()) child.kill();
});

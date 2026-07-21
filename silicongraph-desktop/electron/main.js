const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const { listDevices } = require("./adb");

const projectRoot = path.resolve(__dirname, "..", "..");
const profilesDir = path.join(projectRoot, "device_profiles");
const activeJobs = new Map();

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

function readProfiles() {
  return fs.readdirSync(profilesDir)
    .filter((file) => file.endsWith(".json"))
    .map((file) => {
      const source = JSON.parse(fs.readFileSync(path.join(profilesDir, file), "utf8"));
      return {
        id: source.soc_id,
        name: source.name,
        vendor: source.vendor,
        cpu: source.cpu || {},
        gpu: source.gpu || {},
        npu: source.npu || {}
      };
    })
    .sort((left, right) => left.name.localeCompare(right.name));
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
  const profileId = String(request?.profileId || "").trim();
  const quantization = String(request?.quantization || "int8").trim();
  const contextLengths = String(request?.contextLengths || "512,2048").trim();
  const outputDirectory = path.resolve(String(request?.outputDirectory || path.join(projectRoot, "output")));
  const profiles = readProfiles();

  if (!model) throw new Error("Enter a HuggingFace model ID or local model directory.");
  if (!deviceSerial) throw new Error("Connect and select an Android device before compiling.");
  if (!profiles.some((profile) => profile.id === profileId)) {
    throw new Error("Select a valid SiliconGraph device profile.");
  }
  if (!/^(\d+)(,\d+)*$/.test(contextLengths)) {
    throw new Error("Context lengths must be a comma-separated list of positive integers.");
  }

  await fsp.mkdir(outputDirectory, { recursive: true });
  const outputPath = path.join(outputDirectory, `${safeOutputName(model)}-${profileId}.armpack`);
  const pythonExecutable = process.env.PYTHON || "python";
  const args = [
    "-m", "armcc.cli", "compile",
    "--model", model,
    "--targets", profileId,
    "--quantization", quantization,
    "--context-lengths", contextLengths,
    "--output", outputPath
  ];
  const pythonPath = [path.join(projectRoot, "python"), process.env.PYTHONPATH]
    .filter(Boolean)
    .join(path.delimiter);
  const jobId = crypto.randomUUID();
  const window = BrowserWindow.fromWebContents(event.sender);
  const child = spawn(pythonExecutable, args, {
    cwd: projectRoot,
    env: { ...process.env, PYTHONPATH: pythonPath },
    windowsHide: true,
    shell: false
  });

  activeJobs.set(jobId, child);
  sendJobEvent(window, "compile:output", {
    jobId,
    stream: "system",
    text: `Starting: ${pythonExecutable} ${args.join(" ")}\n`
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
  child.on("close", (exitCode) => {
    activeJobs.delete(jobId);
    sendJobEvent(window, "compile:finished", { jobId, exitCode, outputPath });
  });

  return { jobId, outputPath };
}

app.whenReady().then(() => {
  ipcMain.handle("devices:list", () => listDevices());
  ipcMain.handle("profiles:list", () => readProfiles());
  ipcMain.handle("compile:start", startCompilation);
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

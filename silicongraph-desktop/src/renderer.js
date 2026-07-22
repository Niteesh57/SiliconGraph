const state = {
  devices: [],
  selectedDevice: null,
  activeJob: null,
  scanning: false,
  hub: { configured: false, localDirectory: "", activeDownload: null }
};

const elements = {
  scanButton: document.querySelector("#scan-button"),
  deviceList: document.querySelector("#device-list"),
  deviceCount: document.querySelector("#device-count"),
  deviceHelp: document.querySelector("#device-help"),
  connectionState: document.querySelector("#connection-state"),
  deviceSummary: document.querySelector("#device-summary"),
  targetDeviceName: document.querySelector("#target-device-name"),
  profileStatus: document.querySelector("#profile-status"),
  profileDetails: document.querySelector("#profile-details"),
  compileForm: document.querySelector("#compile-form"),
  compileButton: document.querySelector("#compile-button"),
  compileLog: document.querySelector("#compile-log"),
  jobStatus: document.querySelector("#job-status"),
  hubSetupButton: document.querySelector("#hub-setup-button"),
  hubModal: document.querySelector("#hub-modal"),
  hubCloseButton: document.querySelector("#hub-close-button"),
  hubTokenInput: document.querySelector("#hub-token-input"),
  hubSaveToken: document.querySelector("#hub-save-token"),
  hubClearToken: document.querySelector("#hub-clear-token"),
  hubTokenStatus: document.querySelector("#hub-token-status"),
  hubTokenHelp: document.querySelector("#hub-token-help"),
  hubSearchInput: document.querySelector("#hub-search-input"),
  hubSearchButton: document.querySelector("#hub-search-button"),
  hubResults: document.querySelector("#hub-results"),
  hubRefreshLocal: document.querySelector("#hub-refresh-local"),
  hubLocalPath: document.querySelector("#hub-local-path"),
  hubLocalModels: document.querySelector("#hub-local-models"),
  hubDownloadStatus: document.querySelector("#hub-download-status"),
  hubDownloadTitle: document.querySelector("#hub-download-title"),
  hubDownloadBar: document.querySelector("#hub-download-bar"),
  hubDownloadDetail: document.querySelector("#hub-download-detail"),
  hubCancelDownload: document.querySelector("#hub-cancel-download")
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;"
  }[character]));
}

function setConnection(connected, text) {
  elements.connectionState.classList.toggle("connected", connected);
  elements.connectionState.lastElementChild.textContent = text;
}

function appendLog(text) {
  elements.compileLog.textContent += text;
  elements.compileLog.scrollTop = elements.compileLog.scrollHeight;
}

function setJobStatus(text, kind = "") {
  elements.jobStatus.textContent = text;
  elements.jobStatus.className = `pill ${kind}`;
}

function formatMb(value) {
  return value === null || value === undefined ? "Unavailable" : `${Number(value).toLocaleString()} MB`;
}

function formatTemperature(value) {
  return value === null || value === undefined ? null : `${Number(value).toFixed(1)} C`;
}

function formatFrequency(mhz) {
  return mhz ? `${(Number(mhz) / 1000).toFixed(3)} GHz` : "frequency unavailable";
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!value) return "Size unavailable";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}

function setHubTokenStatus(configured) {
  state.hub.configured = configured;
  elements.hubTokenStatus.textContent = configured ? "Connected" : "Not configured";
  elements.hubTokenStatus.classList.toggle("active", configured);
  elements.hubClearToken.hidden = !configured;
}

async function refreshLocalModels() {
  elements.hubRefreshLocal.disabled = true;
  try {
    const models = await window.siliconGraph.huggingFace.listLocal();
    if (!models.length) {
      elements.hubLocalModels.innerHTML = '<div class="empty-state">No local Hugging Face models found. Download one from the search above, or use a model cached by Python.</div>';
      return;
    }
    elements.hubLocalModels.innerHTML = models.map((model) => `<div class="hub-model-row"><div><strong>${escapeHtml(model.name)}</strong><small>${escapeHtml(model.source)} · updated ${escapeHtml(new Date(model.updatedAt).toLocaleDateString())}</small><small>${escapeHtml(model.path)}</small></div><button class="button secondary compact use-local-model" type="button" data-path="${escapeHtml(model.path)}">Use for compile</button></div>`).join("");
    for (const button of elements.hubLocalModels.querySelectorAll(".use-local-model")) {
      button.addEventListener("click", () => {
        document.querySelector("#model-input").value = button.dataset.path;
        closeHubModal();
      });
    }
  } catch (error) {
    elements.hubLocalModels.innerHTML = `<div class="empty-state">Could not inspect local models: ${escapeHtml(error.message)}</div>`;
  } finally {
    elements.hubRefreshLocal.disabled = false;
  }
}

function renderHubResults(models) {
  if (!models.length) {
    elements.hubResults.innerHTML = '<div class="empty-state">No matching public models were found.</div>';
    return;
  }
  elements.hubResults.innerHTML = models.map((model) => {
    const badges = [model.gated && "Gated", model.private && "Private"].filter(Boolean).map((label) => `<span class="hub-badge">${label}</span>`).join("");
    const details = [model.pipelineTag, `${model.downloads.toLocaleString()} downloads`, `${model.likes.toLocaleString()} likes`].filter(Boolean).join(" · ");
    return `<div class="hub-model-row"><div><strong>${escapeHtml(model.id)} <span class="hub-badges">${badges}</span></strong><small>${escapeHtml(details)}</small></div><button class="button secondary compact hub-download-button" type="button" data-repo-id="${escapeHtml(model.id)}">Download</button></div>`;
  }).join("");
  for (const button of elements.hubResults.querySelectorAll(".hub-download-button")) {
    button.addEventListener("click", () => beginModelDownload(button.dataset.repoId));
  }
}

async function searchHubModels() {
  const query = elements.hubSearchInput.value.trim();
  elements.hubSearchButton.disabled = true;
  elements.hubSearchButton.textContent = "Searching…";
  elements.hubResults.innerHTML = '<div class="empty-state">Searching Hugging Face…</div>';
  try {
    renderHubResults(await window.siliconGraph.huggingFace.search(query));
  } catch (error) {
    elements.hubResults.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  } finally {
    elements.hubSearchButton.disabled = false;
    elements.hubSearchButton.textContent = "Search";
  }
}

async function beginModelDownload(repoId) {
  if (state.hub.activeDownload) return;
  state.hub.activeDownload = "pending";
  elements.hubDownloadStatus.hidden = false;
  elements.hubDownloadTitle.textContent = `Preparing ${repoId}`;
  elements.hubDownloadDetail.textContent = "Getting the compiler-ready files and their total size…";
  elements.hubDownloadBar.style.width = "0%";
  for (const button of elements.hubResults.querySelectorAll(".hub-download-button")) button.disabled = true;
  try {
    await window.siliconGraph.huggingFace.download({ repoId });
  } catch (error) {
    // The main process also sends a detailed failed progress event. This catch
    // covers errors discovered before that event can be created.
    state.hub.activeDownload = null;
    elements.hubDownloadTitle.textContent = "Download failed";
    elements.hubDownloadDetail.textContent = error.message;
    for (const button of elements.hubResults.querySelectorAll(".hub-download-button")) button.disabled = false;
  }
}

function closeHubModal() {
  elements.hubModal.hidden = true;
  elements.hubSetupButton.focus();
}

async function openHubModal() {
  elements.hubModal.hidden = false;
  try {
    const status = await window.siliconGraph.huggingFace.getStatus();
    setHubTokenStatus(status.configured);
    state.hub.localDirectory = status.localDirectory;
    elements.hubLocalPath.textContent = `SiliconGraph downloads: ${status.localDirectory}`;
    await refreshLocalModels();
  } catch (error) {
    elements.hubTokenHelp.textContent = error.message;
  }
  elements.hubTokenInput.focus();
}

function profileFacts(profile) {
  if (!profile) return "";
  const cpu = profile.cpu || {};
  const gpu = profile.gpu || {};
  const clusters = (cpu.clusters || [])
    .map((cluster) => `${cluster.cores.length} x ${formatFrequency(cluster.mhz)}`)
    .join(" | ") || "Frequency access unavailable";
  const isa = [cpu.has_neon && "NEON", cpu.has_dotprod && "DotProd", cpu.has_i8mm && "i8mm", cpu.has_sve && "SVE", cpu.has_sve2 && "SVE2"]
    .filter(Boolean).join(" | ") || "Not exposed";
  const cacheProblem = profile.unavailable_fields?.cache_topology;
  const caches = cacheProblem
    ? `Unavailable: ${cacheProblem}`
    : `L1 ${profile.l1_cache_kb} KB | L2 ${profile.l2_cache_kb} KB | L3 ${profile.l3_cache_kb} KB`;
  return [
    ["CPU", `${cpu.physical_core_count ?? "Unavailable"} physical | ${cpu.online_core_count ?? "Unavailable"} online | ${cpu.adb_visible_core_count ?? "Unavailable"} ADB visible`],
    ["Clusters", clusters],
    ["ISA", isa],
    ["GPU", gpu.name || "Not detected"],
    ["Vulkan", gpu.supports_vulkan ? "Candidate — benchmark must beat CPU" : "Not available"],
    ["Runtime", "CPU + Vulkan; free RAM selects graph/KV budget; direct answers only below 50% battery"],
    ["Caches", caches]
  ].map(([name, value]) => `<div class="profile-detail"><b>${escapeHtml(name)}</b>${escapeHtml(value)}</div>`).join("");
}

function renderProfile() {
  const device = state.selectedDevice;
  const profile = device?.dynamicProfile;
  elements.targetDeviceName.textContent = device
    ? `${device.name} - ${device.socModel}`
    : "Select a connected Android device";
  elements.profileDetails.innerHTML = profileFacts(profile);
  elements.profileStatus.textContent = device ? "Live device profile" : "No device selected";
  elements.profileStatus.classList.toggle("active", Boolean(profile));
}

function thermalSummary(thermals) {
  const values = [];
  if (thermals?.status !== null && thermals?.status !== undefined) values.push(`status ${thermals.status}`);
  const cpu = formatTemperature(thermals?.maxCpuC);
  const gpu = formatTemperature(thermals?.gpuC);
  const soc = formatTemperature(thermals?.socC);
  if (cpu) values.push(`CPU ${cpu}`);
  if (gpu) values.push(`GPU ${gpu}`);
  if (soc) values.push(`SoC ${soc}`);
  return values.join(" | ") || "Unavailable";
}

function renderDeviceSummary(device) {
  if (!device) {
    elements.deviceSummary.className = "device-summary empty-state";
    elements.deviceSummary.textContent = "Select a detected device to inspect its Android, CPU, memory, thermal, GPU, and SoC information.";
    return;
  }
  const battery = device.battery?.level === null || device.battery?.level === undefined
    ? "Unavailable"
    : `${device.battery.level}%${formatTemperature(device.battery.temperatureC) ? ` | ${formatTemperature(device.battery.temperatureC)}` : ""}${device.battery.voltageMv ? ` | ${device.battery.voltageMv} mV` : ""}`;
  const parts = (device.cpuParts || []).map((item) => `${item.part} x ${item.cores}`).join(" | ") || "Unavailable";
  const facts = [
    ["Device", device.name],
    ["Android", `${device.androidVersion} (API ${device.apiLevel})`],
    ["SoC", `${device.socModel} (${device.boardPlatform})`],
    ["CPU cores", `${device.physicalCoreCount ?? "Unavailable"} physical | ${device.onlineCoreCount ?? "Unavailable"} online | ${device.adbVisibleCoreCount ?? "Unavailable"} ADB visible`],
    ["CPU parts", parts],
    ["Memory", `${formatMb(device.memory?.totalMb)} total | ${formatMb(device.memory?.availableMb)} available | ${formatMb(device.memory?.swapTotalMb)} swap`],
    ["Storage", `${formatMb(device.storage?.totalMb)} total | ${formatMb(device.storage?.availableMb)} available`],
    ["Battery", battery],
    ["Thermal", thermalSummary(device.thermals)],
    ["GPU", device.gpu?.renderer || device.gpu?.hardware || "Unavailable"],
    ["Vulkan", device.gpu?.vulkanHardware || "Not exposed"],
    ["Security patch", device.securityPatch],
    ["Kernel", device.kernelVersion],
    ["ABI", device.abi],
    ["Serial", device.serial]
  ];
  elements.deviceSummary.className = "device-summary";
  elements.deviceSummary.innerHTML = facts.map(([label, value]) => `<div class="fact"><small>${escapeHtml(label)}</small><span>${escapeHtml(value)}</span></div>`).join("");
}

function selectDevice(serial) {
  state.selectedDevice = state.devices.find((device) => !device.unavailable && device.serial === serial) || null;
  elements.compileButton.disabled = !state.selectedDevice || Boolean(state.activeJob);
  renderDevices();
  renderDeviceSummary(state.selectedDevice);
  renderProfile();
}

function renderDevices() {
  elements.deviceCount.textContent = String(state.devices.length);
  if (!state.devices.length) {
    elements.deviceList.innerHTML = '<div class="empty-state">No authorized Android devices found.</div>';
    return;
  }
  elements.deviceList.innerHTML = state.devices.map((device) => {
    const selected = device.serial === state.selectedDevice?.serial ? "selected" : "";
    const subtitle = device.unavailable
      ? `Status: ${device.status}`
      : `${device.socModel} | ${device.physicalCoreCount ?? device.coreCount ?? "Unavailable"} physical cores`;
    return `<button class="device-card ${selected}" data-serial="${escapeHtml(device.serial)}"><strong>${escapeHtml(device.name)}</strong><span>${escapeHtml(subtitle)}</span><span>${escapeHtml(device.serial)}</span></button>`;
  }).join("");
  for (const card of elements.deviceList.querySelectorAll(".device-card")) {
    card.addEventListener("click", () => selectDevice(card.dataset.serial));
  }
}

async function scanDevices() {
  if (state.scanning) return;
  state.scanning = true;
  const selectedSerial = state.selectedDevice?.serial;
  elements.scanButton.disabled = true;
  elements.scanButton.textContent = "Scanning...";
  try {
    state.devices = await window.siliconGraph.listDevices();
    const authorized = state.devices.filter((device) => !device.unavailable);
    state.selectedDevice = authorized.find((device) => device.serial === selectedSerial)
      || (authorized.length === 1 ? authorized[0] : null);
    elements.compileButton.disabled = !state.selectedDevice || Boolean(state.activeJob);
    renderDevices();
    renderDeviceSummary(state.selectedDevice);
    renderProfile();
    const ready = authorized.length;
    setConnection(ready > 0, ready ? `${ready} Android device${ready === 1 ? "" : "s"} detected` : "No authorized Android device");
  } catch (error) {
    state.devices = [];
    state.selectedDevice = null;
    elements.compileButton.disabled = true;
    renderDevices();
    renderDeviceSummary(null);
    renderProfile();
    setConnection(false, error.message);
    elements.deviceHelp.textContent = error.message;
  } finally {
    state.scanning = false;
    elements.scanButton.disabled = false;
    elements.scanButton.textContent = "Scan connected devices";
  }
}

elements.scanButton.addEventListener("click", scanDevices);
elements.hubSetupButton.addEventListener("click", openHubModal);
elements.hubCloseButton.addEventListener("click", closeHubModal);
elements.hubModal.addEventListener("click", (event) => {
  if (event.target === elements.hubModal) closeHubModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.hubModal.hidden) closeHubModal();
});
elements.hubSaveToken.addEventListener("click", async () => {
  const token = elements.hubTokenInput.value.trim();
  elements.hubSaveToken.disabled = true;
  try {
    const status = await window.siliconGraph.huggingFace.saveToken(token);
    setHubTokenStatus(status.configured);
    elements.hubTokenInput.value = "";
    elements.hubTokenHelp.textContent = "Token saved with Windows credential encryption. It will also authorize model compilation.";
  } catch (error) {
    elements.hubTokenHelp.textContent = error.message;
  } finally {
    elements.hubSaveToken.disabled = false;
  }
});
elements.hubClearToken.addEventListener("click", async () => {
  await window.siliconGraph.huggingFace.clearToken();
  setHubTokenStatus(false);
  elements.hubTokenHelp.textContent = "The saved Hugging Face token was removed.";
});
elements.hubSearchButton.addEventListener("click", searchHubModels);
elements.hubSearchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); searchHubModels(); }
});
elements.hubRefreshLocal.addEventListener("click", refreshLocalModels);
elements.hubCancelDownload.addEventListener("click", () => {
  if (state.hub.activeDownload && state.hub.activeDownload !== "pending") window.siliconGraph.huggingFace.cancelDownload(state.hub.activeDownload);
});
elements.compileForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.selectedDevice) return;
  const request = {
    deviceSerial: state.selectedDevice.serial,
    model: document.querySelector("#model-input").value,
    quantization: document.querySelector("#quantization-select").value,
    contextLengths: document.querySelector("#context-input").value,
    outputDirectory: document.querySelector("#output-input").value
  };
  elements.compileButton.disabled = true;
  elements.compileLog.textContent = "";
  try {
    const started = await window.siliconGraph.startCompilation(request);
    state.activeJob = started.jobId;
    setJobStatus("Running", "running");
  } catch (error) {
    appendLog(`Unable to start compilation: ${error.message}\n`);
    setJobStatus("Failed", "failed");
    elements.compileButton.disabled = false;
  }
});

window.siliconGraph.onCompileOutput((message) => {
  if (message.jobId === state.activeJob) appendLog(message.text);
});
window.siliconGraph.onCompileFinished((message) => {
  if (message.jobId !== state.activeJob) return;
  const success = message.exitCode === 0 && message.packageCreated;
  const failureReason = message.signal
    ? `terminated by signal ${message.signal}`
    : message.exitCode === 0 && !message.packageCreated
      ? "completed without creating an .armpack file"
    : message.exitCode === null || message.exitCode === undefined
      ? "ended without an exit code"
      : `ended with exit code ${message.exitCode}`;
  appendLog(success ? `\nPackage ready: ${message.outputPath}\n` : `\nCompilation ${failureReason}.\n`);
  setJobStatus(success ? "Completed" : "Failed", success ? "success" : "failed");
  state.activeJob = null;
  elements.compileButton.disabled = !state.selectedDevice;
});

window.siliconGraph.huggingFace.onDownloadProgress((message) => {
  state.hub.activeDownload = message.status === "completed" || message.status === "failed" || message.status === "cancelled"
    ? null : message.downloadId;
  elements.hubDownloadStatus.hidden = false;
  const percent = message.totalBytes > 0 ? Math.min(100, (message.receivedBytes / message.totalBytes) * 100) : 0;
  elements.hubDownloadBar.style.width = `${percent}%`;
  if (message.status === "completed") {
    elements.hubDownloadTitle.textContent = `${message.repoId} downloaded`;
    elements.hubDownloadDetail.textContent = `${formatBytes(message.receivedBytes)} downloaded to ${message.path}. It is ready to use for compilation.`;
    elements.hubDownloadBar.style.width = "100%";
    refreshLocalModels();
  } else if (message.status === "failed" || message.status === "cancelled") {
    elements.hubDownloadTitle.textContent = message.status === "cancelled" ? "Download cancelled" : "Download failed";
    elements.hubDownloadDetail.textContent = message.error || "The download did not finish.";
  } else {
    const progress = message.totalBytes > 0
      ? `${percent.toFixed(1)}% · ${formatBytes(message.receivedBytes)} of ${formatBytes(message.totalBytes)}`
      : `${formatBytes(message.receivedBytes)} downloaded`;
    elements.hubDownloadTitle.textContent = `Downloading ${message.repoId}`;
    elements.hubDownloadDetail.textContent = `${progress} · ${message.completedFiles}/${message.totalFiles} files${message.currentFile ? ` · ${message.currentFile}` : ""}`;
  }
  for (const button of elements.hubResults.querySelectorAll(".hub-download-button")) button.disabled = Boolean(state.hub.activeDownload);
});

scanDevices();
window.setInterval(scanDevices, 5_000);

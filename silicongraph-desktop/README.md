# SiliconGraph Desktop

Windows Electron application for compiling a model against a USB-connected
Android device profile.

## What it does

1. Runs `adb devices -l` to find authorized Android phones connected over USB.
2. Reads Android properties, SoC information, core count, memory, ABI, and
   battery level through `adb shell`.
3. Builds a live per-phone profile from its ADB-exposed CPU topology,
   frequencies, ISA features, memory, SoC, GPU, battery, and Android facts.
4. Uses the selected connected device as the only target, writing its live
   profile beside the output package and inside the `.armpack` as
   `device_profile.json`.
5. Starts the repository's Python `armcc compile` workflow using the live
   device profile, precision, context buckets, and output directory.
6. Streams compiler output into the application.

The Model field accepts a Hugging Face repository ID, a browser URL for a
Hugging Face model repository, or a local model directory. Browser URLs are
converted to their repository IDs before loading.

## Prerequisites

- Windows with Node.js 22 or later.
- Android Platform Tools (`adb`) on `PATH`, or set `ADB_PATH` to `adb.exe`.
- A phone with Developer options and USB debugging enabled; authorize the PC
  when Android asks.
- The root project’s Python and C++ compiler dependencies/build artifacts.

## Run

```powershell
cd silicongraph-desktop
npm install
npm start
```

Set `PYTHON` when the compiler should use a specific Python executable:

```powershell
$env:PYTHON = "C:\path\to\python.exe"
npm start
```

The default output directory is `<repository-root>\output`.

Some Android builds restrict cache, memory-bandwidth, or accelerator-driver
details from unprivileged ADB. SiliconGraph marks those values as unavailable
or estimated rather than inventing them; the planned benchmark harness supplies
the missing measured data for a production profile.

## Security boundary

The renderer has no Node.js access. It uses a small preload API for device
listing and compilation. The main process starts `adb` and
Python using argument arrays (`shell: false`), rather than constructing shell
commands from form input.

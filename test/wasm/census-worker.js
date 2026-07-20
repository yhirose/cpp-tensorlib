// Same device-acquisition shape as worker.js: JS does the async part and hands
// wasm a ready device, so every C++ entry point stays synchronous.
import createCensus from "./census.js";

let log = "";

async function main() {
  if (!navigator.gpu) {
    postMessage({ type: "fail", msg: "navigator.gpu missing (no WebGPU in this worker)" });
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    postMessage({ type: "fail", msg: "requestAdapter returned null" });
    return;
  }
  // A census runs far longer than the test suite and its numbers are only
  // meaningful if nothing was silently dropped, so surface validation errors
  // the same way — one invalid dispatch discards its whole command buffer.
  const device = await adapter.requestDevice();
  const gpuErrors = [];
  device.onuncapturederror = (e) => {
    if (gpuErrors.length < 5) gpuErrors.push(String(e.error.message).split("\n")[0]);
  };
  device.lost.then((info) => gpuErrors.push("DEVICE LOST: " + info.message));
  const info = adapter.info ?? {};

  const mod = await createCensus({
    preinitializedWebGPUDevice: device,
    print: (s) => { log += s + "\n"; postMessage({ type: "line", line: s }); },
    printErr: (s) => { log += s + "\n"; postMessage({ type: "line", line: s }); },
  });

  postMessage({
    type: "ready",
    adapter: `${info.vendor ?? "?"} / ${info.architecture ?? "?"} / ${info.description ?? ""}`,
  });

  onmessage = async () => {
    log = "";
    const t0 = performance.now();
    let ok = 0;
    try {
      ok = await mod.ccall("run_census", "number", [], [], { async: true });
    } catch (err) {
      log += "exception: " + String((err && err.stack) || err) + "\n";
    }
    if (gpuErrors.length) log = "WEBGPU ERRORS:\n" + gpuErrors.join("\n") + "\n\n" + log;
    postMessage({ type: "done", ok: !!ok, log, ms: performance.now() - t0 });
  };
}

main().catch((e) => postMessage({ type: "fail", msg: String((e && e.stack) || e) }));

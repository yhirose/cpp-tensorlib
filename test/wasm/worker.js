// Acquire the WebGPU device here, in JS, and hand it to wasm fully formed.
// That keeps every C++-side entry point synchronous — the async surface is
// this file, not the library.
import createTests from "./tests.js";

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
  const device = await adapter.requestDevice();

  // Surface WebGPU validation errors. Without this they are only warnings in
  // the worker console, and a single invalid dispatch silently discards the
  // whole command buffer it was batched into — which reads as "every op
  // returned zeros", pointing nowhere near the actual mistake.
  const gpuErrors = [];
  device.onuncapturederror = (e) => {
    if (gpuErrors.length < 5) gpuErrors.push(String(e.error.message).split("\n")[0]);
  };
  device.lost.then((info) => gpuErrors.push("DEVICE LOST: " + info.message));
  const info = adapter.info ?? {};

  // doctest writes its report to stdout; capture it rather than let it vanish
  // into the worker console.
  const mod = await createTests({
    preinitializedWebGPUDevice: device,
    print: (s) => { log += s + "\n"; },
    printErr: (s) => { log += s + "\n"; },
  });

  postMessage({
    type: "ready",
    adapter: `${info.vendor ?? "?"} / ${info.architecture ?? "?"} / ${info.description ?? ""}`,
  });

  const MODES = { cpu: 0, gpu: 1, auto: 2 };
  onmessage = async (e) => {
    const gpuMode = MODES[e.data] ?? 0;
    log = "";
    const t0 = performance.now();
    let ok = 0;
    try {
      // JSPI makes this export return a promise; the GPU waits beneath it
      // suspend. Nothing in the C++ call chain is written as async.
      ok = await mod.ccall("run_tests", "number", ["number"], [gpuMode], { async: true });
    } catch (err) {
      log += "exception: " + String(err && err.stack || err) + "\n";
    }
    if (gpuErrors.length) log = "WEBGPU ERRORS:\n" + gpuErrors.join("\n") + "\n\n" + log;
    postMessage({ type: "done", mode: e.data, ok: !!ok, log, ms: performance.now() - t0 });
  };
}

main().catch((e) => postMessage({ type: "fail", msg: String(e && e.stack || e) }));

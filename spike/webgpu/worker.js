// Acquire the WebGPU device here, in JS, and hand it to wasm fully formed.
// That keeps every C++-side entry point synchronous -- the async surface is
// this file, not the library.
import createSpike from "./spike.js";

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
  const info = adapter.info ?? {};

  const mod = await createSpike({ preinitializedWebGPUDevice: device });

  postMessage({
    type: "ready",
    adapter: `${info.vendor ?? "?"} / ${info.architecture ?? "?"} / ${info.description ?? ""}`,
  });

  // Under JSPI these exports return promises; under Asyncify they return
  // values directly. `await` handles both, so the same worker drives either
  // build without change.
  onmessage = async () => {
    const t0 = performance.now();
    const okInit = await mod.ccall("wgpu_spike_init", "number", [], [], { async: true });
    const okBench = okInit
      ? await mod.ccall("wgpu_spike_bench", "number", [], [], { async: true })
      : 0;
    postMessage({
      type: "done",
      ok: !!okBench,
      log: mod.UTF8ToString(mod._wgpu_spike_log()),
      ms: performance.now() - t0,
    });
  };
}

main().catch((e) => postMessage({ type: "fail", msg: String(e && e.stack || e) }));

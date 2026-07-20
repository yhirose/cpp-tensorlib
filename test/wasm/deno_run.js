// Run the WebGPU test suite under Deno instead of a browser.
//
// The backend ships to browsers, but what CI needs to verify is the WGSL and
// the C++ — not Chrome. Deno exposes navigator.gpu natively (wgpu) and its V8
// has JSPI, so it can host this wasm with no browser, no display and no
// puppeteer. That matters because headless Chrome on a hosted runner refuses
// to expose WebGPU at all, even given a working software Vulkan driver; Deno
// has none of Chrome's GPU sandboxing and blocklist machinery in the way.
//
// This mirrors test/wasm/worker.js: acquire the device here and hand it to
// wasm fully formed, so the C++ side stays synchronous.
//
//   deno run --allow-all test/wasm/deno_run.js [path/to/tests.js]

const modPath = Deno.args[0] ?? new URL("./site/tests.js", import.meta.url).href;

if (!navigator.gpu) {
  console.log("[deno] NO WEBGPU — navigator.gpu missing");
  Deno.exit(2);
}
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  console.log("[deno] NO ADAPTER — requestAdapter returned null");
  Deno.exit(2);
}
const device = await adapter.requestDevice();
console.log("[deno] adapter:", JSON.stringify(adapter.info ?? {}));

// One invalid dispatch discards the whole command buffer it batched into, which
// presents as every op returning zeros. Surface that rather than debug blind.
const gpuErrors = [];
device.addEventListener?.("uncapturederror", (e) => gpuErrors.push(String(e.error?.message ?? e)));

const createTests = (await import(modPath)).default;
let out = "";
const mod = await createTests({
  preinitializedWebGPUDevice: device,
  print: (s) => { out += s + "\n"; console.log(s); },
  printErr: (s) => { out += s + "\n"; console.log(s); },
});

// gpu then auto, as the page does: gpu proves the kernels match the oracle,
// auto proves the size-based routing in types.h does too.
let ok = true;
for (const [name, mode] of [["gpu", 1], ["auto", 2]]) {
  out = "";
  const t0 = performance.now();
  const r = await mod.ccall("run_tests", "number", ["number"], [mode], { async: true });
  console.log(`[deno] ${name} ${r ? "OK" : "FAILED"} in ${Math.round(performance.now() - t0)}ms`);
  ok = ok && !!r;
}
if (gpuErrors.length) {
  console.log("[deno] WEBGPU ERRORS:\n" + gpuErrors.join("\n"));
  ok = false;
}
console.log("[deno] " + (ok ? "OK" : "FAILED"));
Deno.exit(ok ? 0 : 1);

// Stands in for an Emscripten module (src/wasm/czechspeech-*.js) in the unit tests of src/engine-wasm.js, which
// import()s its module from a URL. The test puts the module factory into globalThis.czechspeechFakeModule first; it
// receives what loadWasmEngine() passes to createModule (the czechspeech* callbacks, print, onAbort...).
export default async function createModule(overrides) {
  return globalThis.czechspeechFakeModule(overrides);
}

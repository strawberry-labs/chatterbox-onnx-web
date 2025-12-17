// Backwards-compatible re-export of the WebGPU TTS engine.
// Consumers that previously imported from `tts-engine-onnx.js`
// now receive the fully featured implementation.
export { ChatterboxTTSEngine, audioArrayToWav } from './tts-engine-complete.js';

# Chatterbox TTS Audio Corruption Issue - Debug Summary

## Problem Statement

The JavaScript/ONNX Runtime Web implementation of Chatterbox TTS produces **garbled and repeated audio at the end** of generated speech, despite correctly generating speech tokens.

### Example
**Input text:**
```
[chuckle] The JavaScript implementation produces garbled audio at the end of generated speech. [dramatic] This is caused by incorrect speech token processing that doesn't match the Python reference implementation.
```

**Expected output audio:**
Clean speech with the full sentence ending naturally at "...reference implementation."

**Actual output audio:**
```
"The JavaScript implementation produces garbled audio at the end of generated speech. 9 to 1. This is caused by incorrect speech token processing that doesn't match the Python reference. That doesn't match the Python reference. FDN."
```

**Issues observed:**
1. Missing emotion tags ([chuckle] and [dramatic] not audible)
2. Strange inserted text ("9 to 1")
3. Repetition of ending phrase ("doesn't match the Python reference" appears twice)
4. Garbled ending ("FDN")

---

## System Architecture

### Models (4 ONNX models):
1. **speech_encoder** - Encodes voice audio sample → speaker embeddings/features
2. **embed_tokens** - Converts token IDs → embeddings
3. **language_model** - Autoregressive LM with KV cache (24 layers)
4. **conditional_decoder** - Converts speech tokens → audio waveform

### Token Ranges:
- **Text tokens**: 0-50256 (GPT-2 vocabulary)
- **Emotion tags**: 50257-50275 (added tokens: [chuckle], [dramatic], etc.)
- **Speech tokens**: 0-6560 (valid speech codes for decoder)
- **Special tokens**:
  - START_SPEECH_TOKEN = 6561
  - STOP_SPEECH_TOKEN = 6562
  - SILENCE_TOKEN = 4299

### Generation Flow:
```
Text → Tokenizer → Text Embeddings
                          ↓
Voice Audio → Speech Encoder → Speaker Embeddings + Prompt Token
                          ↓
            [Concat Embeddings]
                          ↓
            Language Model (autoregressive)
                          ↓
            Speech Tokens (0-6560)
                          ↓
            Conditional Decoder
                          ↓
            Audio Waveform
```

---

## Current Implementation Status

### Token Generation (Language Model Phase)
✅ **Working correctly** based on console logs:
- Generated 380 tokens total
- Starts with START_SPEECH_TOKEN (6561)
- Ends with STOP_SPEECH_TOKEN (6562) at step 378
- Filtered to 378 valid speech tokens (< 6561)
- No obvious token corruption in generation

### Execution Providers
- **speech_encoder**: WASM (forced, due to AveragePool op incompatibility with WebGPU)
- **embed_tokens**: WebGPU + WASM fallback
- **language_model**: WebGPU + WASM fallback
- **conditional_decoder**: WASM (forced, to avoid boundary issues)

### Recent Changes Applied
1. ✅ Token filtering: Changed from `slice(1, -1)` to `filter(token => token < 6561)` to match Python
2. ✅ WASM backend forced for conditional_decoder
3. ✅ Aggressive KV cache copying (element-by-element) to prevent buffer reuse issues
4. ✅ Audio normalization (scales if peak > 0.99)
5. ✅ Spike detection in last 100ms of audio
6. ✅ 20ms cosine fade-out

---

## Console Logs from Failing Generation

### Key Observations:

**Text Tokenization:**
```
Token count: 40 → 33 (after emotion tag replacement)
⚠ No emotion tags detected, attempting manual fix...
✓ Replaced sequence at position 18: [5 tokens] → 50262 ([dramatic])
✓ Replaced sequence at position 0: [4 tokens] → 50274 ([chuckle])
Token IDs to embed: Array(33)
Max token ID: 50274, Min token ID: 13
```

**Speech Encoder:**
```
Voice audio: 247680 samples (10.32 seconds)
Speaker embeddings statistics: Mean: -6.8637e-9, Absolute mean: 8.0573e-1
Non-zero values (>1e-6): 192 / 192 ✅ (embeddings look valid)
```

**Token Generation:**
```
Stop token generated at step 378 ✅
Total generated tokens: 380
First token: 6561 (START), Last token: 6562 (STOP)
Valid speech tokens: 378
```

**Decoder Output:**
```
Total sequence: 631 tokens (250 prompt + 378 speech + 3 silence)
Raw audio samples: 365760 (15.24 seconds)
Peak amplitude: 0.6781 (no normalization needed)
```

**NO spike detected, NO trimming occurred**

---

## Hypotheses for Root Cause

### Hypothesis 1: Emotion Tags Not Affecting Generation ❌
**Status**: Emotion tags ARE being tokenized and embedded correctly (50274, 50262 present in embeddings). However, the model may not be responding to them as expected. This doesn't explain the garbled ending though.

### Hypothesis 2: KV Cache Corruption During Generation 🤔
**Status**: Despite aggressive element-by-element copying, WebGPU language model might still have buffer reuse issues. However:
- Stop token was generated correctly
- Token sequence looks valid
- No obvious repetition in generated tokens (we need to check actual token values)

### Hypothesis 3: Decoder Producing Wrong Audio for Correct Tokens ⚠️
**Status**: Most likely. The speech tokens (0-6560) are being generated correctly, but:
- The conditional_decoder might be producing wrong audio
- WASM backend might have different behavior than Python ONNX Runtime
- The decoder might be sensitive to exact input format/values

### Hypothesis 4: Prompt Token or Speaker Embeddings Issue 🤔
**Status**: If the voice sample encoding is incorrect, the decoder will produce audio in the wrong voice/style, but this doesn't explain repetition.

### Hypothesis 5: Token Repetition at End (Model Loop) 🎯
**Status**: NEEDS VERIFICATION. The phrase "doesn't match the Python reference" repeats, suggesting the language model might be generating repeated speech tokens at the end. Need to check:
- Are the last 20-30 speech tokens showing repetition patterns?
- Is the repetition penalty working correctly?

---

## Data Needed for Root Cause Analysis

### Critical Missing Information:

1. **Actual last 10 speech token values**
   - Console shows `Last 5: Array(5)` but not the actual numbers
   - Need: Actual token IDs (e.g., [2234, 5432, 2234, 2234, 1234])

2. **Token frequency in last 20 tokens**
   - Are certain tokens repeating?
   - Pattern: {2234: 5, 5432: 2, ...}

3. **Max consecutive repeats**
   - Is the model stuck generating the same token repeatedly?

4. **Execution provider actually used**
   - During model loading, does it confirm "Using: wasm" for decoder?
   - Check: `console.log` in loadModelWithExternalData function

5. **Comparison with Python implementation**
   - Run same text through Python version
   - Compare:
     - Generated speech token IDs
     - Audio duration
     - Presence of repetition

---

## Code Locations

### Token Generation Loop
**File**: `js/tts-engine-complete.js`
**Lines**: 744-916
- Implements autoregressive generation with KV cache
- Uses greedy sampling (argmax, not temperature sampling despite parameter)
- Applies repetition penalty via RepetitionPenaltyLogitsProcessor

### Token Post-Processing
**File**: `js/tts-engine-complete.js`
**Lines**: 918-962
- Filters tokens: `generateTokens.filter(token => token < 6561)`
- Current debug analysis code at lines 936-958

### Decoder Invocation
**File**: `js/tts-engine-complete.js`
**Lines**: 970-987
- Constructs final sequence: prompt + speech tokens + 3 silence
- Runs conditional_decoder with speaker embeddings

### Audio Post-Processing
**File**: `js/tts-engine-complete.js`
**Lines**: 988-1052
- Spike detection (100ms window, 5× RMS threshold)
- Cosine fade-out (20ms)
- Normalization (if peak > 0.99)

---

## Key Questions to Answer

1. **Are the speech tokens actually correct?**
   - Do the last 20 tokens show repetition patterns?
   - Are there any invalid token IDs (>6560 that slipped through)?

2. **Is the language model generating correctly?**
   - Is repetition penalty actually working? (penalty=1.2)
   - Is the KV cache causing the model to loop?

3. **Is the decoder correct?**
   - Does WASM decoder produce same output as Python for same token sequence?
   - Are speaker embeddings/features being passed correctly?

4. **Is this a post-processing issue?**
   - Does the raw decoder output already have the problem?
   - Is normalization/fade-out causing artifacts?

---

## Suggested Next Steps

1. **Add detailed token logging** - See actual token values in last 20-30 positions
2. **Test with Python** - Run exact same input through Python reference
3. **Test decoder in isolation** - Feed known-good token sequence to decoder
4. **Disable KV cache aggressive copying** - See if that changes behavior
5. **Try FP16 models instead of FP32** - Different precision might behave differently
6. **Check if repetition penalty is working** - Log penalty application results

---

## Reference Links

- **Hugging Face Model**: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX
- **Python Reference**: https://github.com/resemble-ai/chatterbox (inferred)
- **ONNX Runtime Web Docs**: https://onnxruntime.ai/docs/api/js/

---

## Environment

- **Browser**: (user should specify)
- **WebGPU Support**: Yes (confirmed in logs)
- **Model Version**: FP32 (non-quantized)
- **Sample Rate**: 24kHz
- **Voice Sample**: 10.32 seconds, 247680 samples

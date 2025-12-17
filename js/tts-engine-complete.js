// Complete Chatterbox-Turbo TTS Engine using ONNX Runtime Web with WebGPU

import * as ort from 'onnxruntime-web';
import { AutoTokenizer } from '@xenova/transformers';

// Configuration constants from config.json
const MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX";
const BASE_MODEL_ID = "ResembleAI/chatterbox-turbo"; // Base model for tokenizer
const SAMPLE_RATE = 24000;
const START_SPEECH_TOKEN = 6561;
const STOP_SPEECH_TOKEN = 6562;
const SILENCE_TOKEN = 4299;
const NUM_LAYERS = 24;
const NUM_KV_HEADS = 16;
const HEAD_DIM = 64;
const HIDDEN_SIZE = 1024;
const MODEL_CACHE_NAME = 'chatterbox-model-cache-v1';
const CACHE_AVAILABLE = typeof caches !== 'undefined';

async function fetchArrayBufferWithCache(url) {
    if (!CACHE_AVAILABLE) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status}`);
        }
        return { buffer: await response.arrayBuffer(), fromCache: false };
    }

    const cache = await caches.open(MODEL_CACHE_NAME);
    const cachedResponse = await cache.match(url);
    if (cachedResponse) {
        console.log(`Cache hit for ${url}`);
        return { buffer: await cachedResponse.arrayBuffer(), fromCache: true };
    }

    const networkResponse = await fetch(url);
    if (!networkResponse.ok) {
        throw new Error(`Failed to fetch ${url}: ${networkResponse.status}`);
    }

    try {
        await cache.put(url, networkResponse.clone());
    } catch (cacheError) {
        console.warn(`Unable to cache ${url}:`, cacheError);
    }

    return { buffer: await networkResponse.arrayBuffer(), fromCache: false };
}

async function fetchJSONFromRepo(repoId, filename, progressCallback, progressValue) {
    const url = `https://huggingface.co/${repoId}/resolve/main/${filename}`;
    progressCallback?.({ status: 'loading', message: `Downloading ${filename}`, progress: progressValue });
    const { buffer, fromCache } = await fetchArrayBufferWithCache(url);
    if (fromCache) {
        progressCallback?.({ status: 'loading', message: `Using cached ${filename}`, progress: progressValue });
    }
    const decoder = new TextDecoder('utf-8');
    return JSON.parse(decoder.decode(buffer));
}

async function loadTokenizerForRepo(repoId, progressCallback) {
    // Load tokenizer.json directly and parse it properly
    try {
        progressCallback?.({ status: 'loading', message: 'Loading Chatterbox tokenizer...', progress: 5 });
        console.log(`Loading tokenizer from ${repoId}...`);

        // Fetch tokenizer.json directly
        const tokenizerUrl = `https://huggingface.co/${repoId}/resolve/main/tokenizer.json`;
        const configUrl = `https://huggingface.co/${repoId}/resolve/main/tokenizer_config.json`;

        console.log('Fetching tokenizer files...');
        const [tokenizerResponse, configResponse] = await Promise.all([
            fetch(tokenizerUrl),
            fetch(configUrl)
        ]);

        if (!tokenizerResponse.ok || !configResponse.ok) {
            throw new Error('Failed to fetch tokenizer files');
        }

        const tokenizerData = await tokenizerResponse.json();
        const tokenizerConfig = await configResponse.json();

        console.log('✓ Tokenizer files loaded');
        console.log('  Tokenizer type:', tokenizerConfig.tokenizer_class);
        console.log('  Model max length:', tokenizerConfig.model_max_length);

        // Extract vocabulary including emotion tags
        // Emotion tags are in added_tokens, not model.vocab!
        const baseVocab = tokenizerData.model.vocab;
        const addedTokens = tokenizerData.added_tokens || [];

        console.log('  Base vocabulary size:', Object.keys(baseVocab).length);
        console.log('  Added tokens:', addedTokens.length);

        // Merge base vocabulary with added tokens
        const fullVocab = { ...baseVocab };
        for (const addedToken of addedTokens) {
            fullVocab[addedToken.content] = addedToken.id;
        }

        const vocabSize = Object.keys(fullVocab).length;
        console.log('  Total vocabulary size:', vocabSize);

        // Verify emotion tags are in the vocabulary
        console.log('  Checking emotion tags in vocabulary:');
        const emotionTags = {
            '[chuckle]': 50274,
            '[laugh]': 50275,
            '[sigh]': 50268,
            '[gasp]': 50273,
            '[angry]': 50257,
            '[happy]': 50265
        };

        let allTagsPresent = true;
        for (const [tag, expectedId] of Object.entries(emotionTags)) {
            const actualId = fullVocab[tag];
            if (actualId === expectedId) {
                console.log(`    ✓ ${tag} = ${actualId}`);
            } else {
                console.warn(`    ✗ ${tag} expected ${expectedId}, got ${actualId}`);
                allTagsPresent = false;
            }
        }

        if (!allTagsPresent) {
            throw new Error('Emotion tags not found in vocabulary');
        }

        // Update the tokenizer data with merged vocabulary
        tokenizerData.model.vocab = fullVocab;

        // Try to create tokenizer using the PreTrainedTokenizer class directly
        // Get the tokenizer class from Transformers.js
        const TokenizerClass = AutoTokenizer.TOKENIZER_CLASS_MAPPING['GPT2Tokenizer'] ||
                               AutoTokenizer.TOKENIZER_CLASS_MAPPING['PreTrainedTokenizer'];

        if (!TokenizerClass) {
            throw new Error('Could not find GPT2Tokenizer class');
        }

        console.log('  Creating tokenizer instance with Chatterbox vocabulary...');

        // Create tokenizer with the Chatterbox data
        const tokenizer = new TokenizerClass(tokenizerData, tokenizerConfig);

        // Test emotion tag tokenization
        const testText = '[chuckle]';
        try {
            const testTokens = tokenizer(testText, { add_special_tokens: false });
            const testIds = Array.from(testTokens.input_ids.data).map(id => Number(id));
            console.log(`  Test tokenization of "${testText}":`, testIds);

            if (testIds.length === 1 && testIds[0] === 50274) {
                console.log('  ✓ Emotion tags working perfectly with native tokenizer!');
                tokenizer.vocab_size = vocabSize;
                return tokenizer;
            } else {
                console.warn('  ⚠ Emotion tags not working correctly, got:', testIds);
                throw new Error('Emotion tag tokenization failed');
            }
        } catch (testError) {
            console.warn('  Test tokenization failed:', testError.message);
            throw testError;
        }

    } catch (err) {
        console.warn(`Failed to load tokenizer from ${repoId}:`, err.message);
        console.log('Falling back to GPT-2 + manual emotion tags');

        // Fallback: Use GPT-2 and manually add emotion tags
        const tokenizer = await AutoTokenizer.from_pretrained('Xenova/gpt2', { legacy: true });

        // GPT-2 has vocab_size 50257, we need to extend to 6563
        // First add the 19 emotion/paralinguistic tags (50257-50275)
        const emotionTags = [
            '[angry]', '[fear]', '[surprised]', '[whispering]', '[advertisement]',
            '[dramatic]', '[narration]', '[crying]', '[happy]', '[sarcastic]',
            '[clear throat]', '[sigh]', '[shush]', '[cough]', '[groan]',
            '[sniff]', '[gasp]', '[chuckle]', '[laugh]'
        ];

        if (tokenizer.add_tokens) {
            // Add emotion tags at positions 50257-50275
            tokenizer.add_tokens(emotionTags);
            console.log(`Added ${emotionTags.length} emotion tags to tokenizer`);

            // Then add remaining tokens to reach vocab_size 6563
            const currentSize = tokenizer.vocab_size || 50257 + emotionTags.length;
            const missing = 6563 - currentSize;
            if (missing > 0) {
                const extraTokens = Array.from({ length: missing }, (_, i) => `<extra_${i}>`);
                tokenizer.add_tokens(extraTokens);
                console.log(`Added ${missing} extra tokens to reach vocab_size 6563`);
            }
        }

        tokenizer.vocab_size = 6563;
        console.log('Final tokenizer vocab_size:', tokenizer.vocab_size);

        // Verify emotion tags work
        const testTokens = tokenizer('[chuckle]', { add_special_tokens: false });
        console.log('Test tokenization of "[chuckle]":', Array.from(testTokens.input_ids.data));

        return tokenizer;
    }
}

// Configure ONNX Runtime
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'warning';

// Repetition Penalty Logits Processor
class RepetitionPenaltyLogitsProcessor {
    constructor(penalty) {
        if (penalty <= 0) {
            throw new Error('Penalty must be positive');
        }
        this.penalty = penalty;
    }

    process(inputIds, logits) {
        // inputIds: [batch_size, seq_len]
        // logits: Float32Array of [vocab_size]

        const vocabSize = logits.length;
        const processedLogits = new Float32Array(logits);

        // Apply penalty to tokens that have been generated
        for (let i = 0; i < inputIds.length; i++) {
            const tokenId = inputIds[i];
            if (tokenId >= 0 && tokenId < vocabSize) {
                const score = processedLogits[tokenId];
                processedLogits[tokenId] = score < 0
                    ? score * this.penalty
                    : score / this.penalty;
            }
        }

        return processedLogits;
    }
}

export class ChatterboxTTSEngine {
    constructor() {
        this.tokenizer = null;
        this.speechEncoderSession = null;
        this.embedTokensSession = null;
        this.languageModelSession = null;
        this.conditionalDecoderSession = null;
        this.isInitialized = false;
        this.isGenerating = false;
    }

    async initialize(progressCallback, useQuantized = false) {
        if (this.isInitialized) {
            progressCallback?.({ status: 'ready', message: 'Already initialized' });
            return;
        }

        try {
            progressCallback?.({ status: 'loading', message: 'Loading tokenizer...', progress: 0 });

            // Load tokenizer, preferring the actual ONNX export
            const tokenizerCandidates = [MODEL_ID, BASE_MODEL_ID];
            let tokenizerError = null;
            for (const candidate of tokenizerCandidates) {
                if (!candidate) continue;
                try {
                    this.tokenizer = await loadTokenizerForRepo(candidate, progressCallback);
                    console.log(`✓ Tokenizer loaded from ${candidate}`);
                    break;
                } catch (error) {
                    tokenizerError = error;
                    console.warn(`Tokenizer load failed for ${candidate}:`, error.message);
                }
            }

            if (!this.tokenizer) {
                throw new Error(`Failed to load tokenizer: ${tokenizerError?.message || 'unknown error'}`);
            }

            // Determine model suffix based on quantization preference
            // Note: FP32 models have NO suffix, quantized models have _fp16, _q4, etc.
            const modelSuffix = useQuantized ? '_fp16' : '';
            const baseUrl = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx`;

            // Session options - Use WebGPU where possible, WASM as fallback
            let executionProviders = ['webgpu', 'wasm'];

            if (!navigator.gpu) {
                console.warn('WebGPU not available, using WASM');
                executionProviders = ['wasm'];
            } else {
                console.log('✓ WebGPU available');
            }

            const sessionOptions = {
                executionProviders: executionProviders,
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true,
            };

            // Speech encoder MUST use pure WASM (no WebGPU) due to AveragePool ceil() issue
            // WebGPU/JSEP doesn't support AveragePool with ceil() in shape computation
            const speechEncoderOptions = {
                executionProviders: ['wasm'], // Force WASM only, no WebGPU fallback
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true,
                enableGraphCapture: false, // Disable graph capture to avoid WebGPU usage
            };

            // Helper function to load model with external data
            const loadModelWithExternalData = async (modelName, message, progress, customOptions = null) => {
                progressCallback?.({ status: 'loading', message, progress });

                const modelUrl = `${baseUrl}/${modelName}${modelSuffix}.onnx`;
                const dataUrl = `${baseUrl}/${modelName}${modelSuffix}.onnx_data`;

                console.log(`Fetching ${modelName}...`);
                console.log(`  Model: ${modelUrl}`);
                console.log(`  Data: ${dataUrl}`);

                const [modelResult, dataResult] = await Promise.all([
                    fetchArrayBufferWithCache(modelUrl),
                    fetchArrayBufferWithCache(dataUrl)
                ]);

                const cacheSuffix = modelResult.fromCache && dataResult.fromCache
                    ? ' (cached)'
                    : (modelResult.fromCache || dataResult.fromCache) ? ' (partial cache)' : '';

                if (cacheSuffix) {
                    progressCallback?.({ status: 'loading', message: `${message}${cacheSuffix}`, progress });
                }

                console.log(`  Model size: ${(modelResult.buffer.byteLength / 1024 / 1024).toFixed(2)} MB`);
                console.log(`  Data size: ${(dataResult.buffer.byteLength / 1024 / 1024).toFixed(2)} MB`);
                console.log(`  Source: ${modelResult.fromCache && dataResult.fromCache ? 'cache' : (modelResult.fromCache || dataResult.fromCache) ? 'cache/network mix' : 'network'}`);

                const options = customOptions || sessionOptions;
                console.log(`  Using: ${options.executionProviders.join(', ')}`);

                // Create session with external data
                return await ort.InferenceSession.create(
                    modelResult.buffer,
                    {
                        ...options,
                        externalData: [
                            {
                                data: dataResult.buffer,
                                path: `${modelName}${modelSuffix}.onnx_data`
                            }
                        ]
                    }
                );
            };

            // Load speech encoder (MUST use pure WASM - WebGPU not supported)
            console.log('Loading speech encoder with WASM backend (WebGPU not compatible)...');
            this.speechEncoderSession = await loadModelWithExternalData(
                'speech_encoder',
                'Loading speech encoder (WASM)...',
                20,
                speechEncoderOptions  // Force pure WASM
            );

            // Load embed tokens
            this.embedTokensSession = await loadModelWithExternalData(
                'embed_tokens',
                'Loading text embedder...',
                40
            );

            // Load language model (largest model)
            this.languageModelSession = await loadModelWithExternalData(
                'language_model',
                'Loading language model (this may take a while)...',
                50
            );

            // Load conditional decoder
            this.conditionalDecoderSession = await loadModelWithExternalData(
                'conditional_decoder',
                'Loading audio decoder...',
                80
            );

            this.isInitialized = true;
            progressCallback?.({ status: 'ready', message: 'All models loaded successfully!', progress: 100 });

            console.log('TTS Engine initialized successfully');
            console.log('Speech Encoder:');
            console.log('  Inputs:', this.speechEncoderSession.inputNames);
            console.log('  Outputs:', this.speechEncoderSession.outputNames);
            console.log('  Execution providers:', this.speechEncoderSession.executionProviders || 'unknown');
            console.log('Language Model:');
            console.log('  Inputs:', this.languageModelSession.inputNames);
            console.log('  Outputs:', this.languageModelSession.outputNames);
            console.log('  Execution providers:', this.languageModelSession.executionProviders || 'unknown');
            console.log('Conditional Decoder:');
            console.log('  Inputs:', this.conditionalDecoderSession.inputNames);
            console.log('  Outputs:', this.conditionalDecoderSession.outputNames);
            console.log('  Execution providers:', this.conditionalDecoderSession.executionProviders || 'unknown');

        } catch (error) {
            console.error('Error initializing TTS engine:', error);
            throw new Error(`Failed to initialize TTS engine: ${error.message}`);
        }
    }

    async generate(text, audioData, options = {}) {
        if (!this.isInitialized) {
            throw new Error('TTS engine not initialized. Call initialize() first.');
        }

        if (this.isGenerating) {
            throw new Error('Generation already in progress');
        }

        this.isGenerating = true;

        const {
            maxNewTokens = 1024,
            repetitionPenalty = 1.2,
            temperature = 0.2,
            progressCallback = null
        } = options;

        try {
            console.log('TTS Engine Generate Called:');
            console.log('  Text length:', text.length, 'characters');
            console.log('  Voice audio length:', audioData.length, 'samples');
            console.log('  Voice audio duration:', (audioData.length / SAMPLE_RATE).toFixed(2), 'seconds');
            console.log('  Parameters: temperature =', temperature, ', repetition penalty =', repetitionPenalty);

            // Step 1: Tokenize text
            progressCallback?.({ status: 'processing', message: 'Tokenizing text...' });

            const textInputs = await this.tokenizer(text);
            // Convert BigInt to Number for easier handling
            let inputIds = Array.from(textInputs.input_ids.data).map(id => Number(id));

            console.log('Text tokenization:');
            console.log('  Input text:', text);
            console.log('  Text length:', text.length, 'characters');
            console.log('  Token IDs:', inputIds);
            console.log('  Token count:', inputIds.length);
            console.log('  Last 5 tokens:', inputIds.slice(-5));

            // Check if emotion tags are being tokenized (they should be in range 50257-50275)
            const emotionTokens = inputIds.filter(id => id >= 50257 && id <= 50275);
            if (emotionTokens.length > 0) {
                console.log('  ✓ Native tokenizer - Found', emotionTokens.length, 'emotion tag tokens:', emotionTokens);
                // Emotion tags are already properly tokenized, no manual insertion needed!
            } else {
                console.log('  ⚠ No emotion tags detected, attempting manual fix...');

                // Manual emotion tag mapping (tag -> token ID)
                const emotionTagMap = {
                    '[angry]': 50257, '[fear]': 50258, '[surprised]': 50259,
                    '[whispering]': 50260, '[advertisement]': 50261, '[dramatic]': 50262,
                    '[narration]': 50263, '[crying]': 50264, '[happy]': 50265,
                    '[sarcastic]': 50266, '[clear throat]': 50267, '[sigh]': 50268,
                    '[shush]': 50269, '[cough]': 50270, '[groan]': 50271,
                    '[sniff]': 50272, '[gasp]': 50273, '[chuckle]': 50274, '[laugh]': 50275
                };

                // Re-tokenize by replacing emotion tags in the original text
                let processedText = text;
                const emotionTagsFound = [];
                for (const [tag, tokenId] of Object.entries(emotionTagMap)) {
                    if (text.includes(tag)) {
                        emotionTagsFound.push(tag);
                        // Replace tag with a unique placeholder
                        processedText = processedText.replace(tag, `<|EMOTION_${tokenId}|>`);
                    }
                }

                if (emotionTagsFound.length > 0) {
                    console.log('  Found emotion tags in text:', emotionTagsFound);
                    console.log('  Attempting smarter token replacement...');

                    // Better approach: tokenize each emotion tag with and without leading space
                    // GPT-2 tokenizer is context-sensitive, so "[tag]" and " [tag]" tokenize differently
                    const emotionTagSequences = {};
                    for (const tag of emotionTagsFound) {
                        // Try without leading space
                        const tagTokens = await this.tokenizer(tag, { add_special_tokens: false });
                        const seqWithoutSpace = Array.from(tagTokens.input_ids.data).map(id => Number(id));

                        // Try with leading space (more common in context)
                        const tagTokensWithSpace = await this.tokenizer(' ' + tag, { add_special_tokens: false });
                        const seqWithSpace = Array.from(tagTokensWithSpace.input_ids.data).map(id => Number(id));

                        emotionTagSequences[tag] = {
                            withoutSpace: seqWithoutSpace,
                            withSpace: seqWithSpace
                        };
                        console.log(`  ${tag} tokenizes to:`, seqWithoutSpace);
                        console.log(`  ${tag} with leading space tokenizes to:`, seqWithSpace);
                    }

                    // Now find and replace these sequences in the original inputIds
                    // FIXED: Replace ALL occurrences, not just first one per variant
                    let newInputIds = [...inputIds];
                    let totalReplacements = 0;
                    for (const tag of emotionTagsFound) {
                        const sequences = emotionTagSequences[tag];
                        const replacementToken = emotionTagMap[tag];
                        let replacementsForTag = 0;

                        // Try both variants so tags inside sentences are also caught
                        for (const variant of [sequences.withSpace, sequences.withoutSpace]) {
                            if (!variant || variant.length === 0) continue;

                            // Keep scanning until NO MORE matches are found
                            let foundAny = true;
                            while (foundAny) {
                                foundAny = false;

                                // Find every occurrence of the variant sequence
                                for (let i = 0; i <= newInputIds.length - variant.length; i++) {
                                    let match = true;
                                    for (let j = 0; j < variant.length; j++) {
                                        if (newInputIds[i + j] !== variant[j]) {
                                            match = false;
                                            break;
                                        }
                                    }

                                    if (match) {
                                        // Replace the sequence with the single emotion token
                                        newInputIds.splice(i, variant.length, replacementToken);
                                        replacementsForTag++;
                                        totalReplacements++;
                                        console.log(`  ✓ Replaced sequence at position ${i}:`, variant, '→', replacementToken);
                                        foundAny = true;
                                        break; // Restart scan after modification
                                    }
                                }
                            }
                        }

                        if (replacementsForTag === 0) {
                            console.warn(`  ⚠ Could not find tokenizer sequence for ${tag}, leaving text tokens as-is`);
                        } else {
                            console.log(`  → Replaced ${replacementsForTag} occurrence(s) of ${tag}`);
                        }
                    }

                    if (totalReplacements === 0) {
                        console.warn('  ⚠ Emotion tags detected but sequences were not replaced. Check tokenizer behavior.');
                    } else {
                        inputIds = newInputIds;
                        console.log('  Token replacement complete');
                        console.log('  Total replacements:', totalReplacements);
                        console.log('  New token IDs:', inputIds);
                        console.log('  New token count:', inputIds.length);
                    }
                }
            }

            // Step 2: Process voice audio with speech encoder
            progressCallback?.({ status: 'processing', message: 'Encoding voice sample...' });

            // Check audio data statistics (avoid spread operator on large arrays)
            let audioMin = audioData[0];
            let audioMax = audioData[0];
            let audioSum = 0;
            let audioSumSquared = 0;
            for (let i = 0; i < audioData.length; i++) {
                const sample = audioData[i];
                if (sample < audioMin) audioMin = sample;
                if (sample > audioMax) audioMax = sample;
                audioSum += sample;
                audioSumSquared += sample * sample;
            }
            const audioMean = audioSum / audioData.length;
            const audioVariance = (audioSumSquared / audioData.length) - (audioMean * audioMean);
            const audioRMS = Math.sqrt(audioSumSquared / audioData.length);

            console.log('🎤 [TTS] Input voice audio fingerprint:');
            console.log('  Length:', audioData.length, 'samples');
            console.log('  Duration:', (audioData.length / SAMPLE_RATE).toFixed(2), 'seconds');
            console.log('  Min:', audioMin.toFixed(4), 'Max:', audioMax.toFixed(4));
            console.log('  Mean:', audioMean.toFixed(6));
            console.log('  RMS:', audioRMS.toFixed(6));
            console.log('  Variance:', audioVariance.toFixed(6));
            console.log('  First 10 samples:', Array.from(audioData.slice(0, 10).map(v => v.toFixed(4))));

            const audioTensor = new ort.Tensor('float32', audioData, [1, audioData.length]);

            const speechEncoderOutputs = await this.speechEncoderSession.run({
                audio_values: audioTensor
            });

            // Use correct output names from the model
            const condEmb = speechEncoderOutputs.audio_features;  // not cond_emb
            const promptToken = speechEncoderOutputs.audio_tokens;  // not prompt_token
            const speakerEmbeddings = speechEncoderOutputs.speaker_embeddings;
            const speakerFeatures = speechEncoderOutputs.speaker_features;

            console.log('Speech encoder outputs received:');
            console.log('  Conditional embedding shape:', condEmb?.dims);
            console.log('  Prompt token shape:', promptToken?.dims);
            console.log('  Speaker embeddings shape:', speakerEmbeddings?.dims);
            console.log('  Speaker features shape:', speakerFeatures?.dims);

            // Check if speaker embeddings have reasonable values
            if (speakerEmbeddings) {
                const embData = Array.from(speakerEmbeddings.data.slice(0, 10));
                console.log('  First 10 speaker embedding values:', embData);

                let embSum = 0;
                let nonZeroCount = 0;
                let absSum = 0;
                for (let i = 0; i < speakerEmbeddings.data.length; i++) {
                    const val = speakerEmbeddings.data[i];
                    embSum += val;
                    absSum += Math.abs(val);
                    if (Math.abs(val) > 1e-6) nonZeroCount++;
                }
                const embMean = embSum / speakerEmbeddings.data.length;
                const embAbsMean = absSum / speakerEmbeddings.data.length;

                console.log('  Speaker embeddings statistics:');
                console.log('    Mean:', embMean.toExponential(4));
                console.log('    Absolute mean:', embAbsMean.toExponential(4));
                console.log('    Non-zero values (>1e-6):', nonZeroCount, '/', speakerEmbeddings.data.length);

                if (embAbsMean < 1e-5) {
                    console.warn('  ⚠️ WARNING: Speaker embeddings are nearly zero!');
                    console.warn('     This will prevent voice cloning from working correctly.');
                    console.warn('     The model may not be extracting speaker characteristics properly.');
                }
            }

            // Check speaker features as well
            if (speakerFeatures) {
                let featSum = 0;
                let featAbsSum = 0;
                for (let i = 0; i < speakerFeatures.data.length; i++) {
                    const val = speakerFeatures.data[i];
                    featSum += val;
                    featAbsSum += Math.abs(val);
                }
                const featMean = featSum / speakerFeatures.data.length;
                const featAbsMean = featAbsSum / speakerFeatures.data.length;

                console.log('  Speaker features statistics:');
                console.log('    Mean:', featMean.toExponential(4));
                console.log('    Absolute mean:', featAbsMean.toExponential(4));
                console.log('    First 10 values:', Array.from(speakerFeatures.data.slice(0, 10)));
            }

            // Step 3: Get text embeddings
            progressCallback?.({ status: 'processing', message: 'Embedding text...' });

            console.log('About to embed text tokens:');
            console.log('  Token IDs to embed:', inputIds);
            console.log('  Max token ID:', Math.max(...inputIds));
            console.log('  Min token ID:', Math.min(...inputIds));

            const inputIdsTensor = new ort.Tensor(
                'int64',
                BigInt64Array.from(inputIds.map(id => BigInt(id))),
                [1, inputIds.length]
            );

            const embedOutputs = await this.embedTokensSession.run({
                input_ids: inputIdsTensor
            });

            const textEmbeds = embedOutputs.inputs_embeds;
            console.log('Text embeddings shape:', textEmbeds.dims);

            // Check if embeddings look reasonable
            const embedData = textEmbeds.data;
            let embedSum = 0;
            for (let i = 0; i < Math.min(1024, embedData.length); i++) {
                embedSum += Math.abs(embedData[i]);
            }
            console.log('First text embedding absolute sum:', embedSum.toFixed(2), '(should be >0)');

            // Step 4: Concatenate conditional embedding with text embeddings
            console.log('\nChecking concatenation:');
            console.log('  condEmb shape:', condEmb.dims);
            console.log('  textEmbeds shape:', textEmbeds.dims);

            // Check condEmb statistics
            let condSum = 0;
            for (let i = 0; i < Math.min(1024, condEmb.data.length); i++) {
                condSum += Math.abs(condEmb.data[i]);
            }
            console.log('  First condEmb absolute sum:', condSum.toFixed(2));

            const inputsEmbeds = this.concatenateTensors(condEmb, textEmbeds);
            const totalSeqLen = inputsEmbeds.dims[1];

            console.log('  Concatenated embeddings shape:', inputsEmbeds.dims);

            // Verify concatenation worked
            let concatSum = 0;
            for (let i = 0; i < Math.min(1024, inputsEmbeds.data.length); i++) {
                concatSum += Math.abs(inputsEmbeds.data[i]);
            }
            console.log('  First concatenated embedding absolute sum:', concatSum.toFixed(2));

            // Step 5: Initialize generation
            const generationStartTime = Date.now();
            progressCallback?.({
                status: 'generating',
                message: 'Generating speech tokens...',
                elapsed: 0
            });

            const repetitionProcessor = new RepetitionPenaltyLogitsProcessor(repetitionPenalty);
            const generateTokens = [START_SPEECH_TOKEN];

            // Initialize KV cache
            const pastKeyValues = this.initializeKVCache(1);

            // Create attention mask and position IDs
            let attentionMask = new ort.Tensor(
                'int64',
                new BigInt64Array(totalSeqLen).fill(1n),
                [1, totalSeqLen]
            );

            let positionIds = new ort.Tensor(
                'int64',
                BigInt64Array.from(Array.from({ length: totalSeqLen }, (_, i) => BigInt(i))),
                [1, totalSeqLen]
            );

            // Step 6: Generation loop
            let currentInputsEmbeds = inputsEmbeds;

            for (let step = 0; step < maxNewTokens; step++) {
                // Prepare inputs for language model
                const lmInputs = {
                    inputs_embeds: currentInputsEmbeds,
                    attention_mask: attentionMask,
                    position_ids: positionIds,
                    ...pastKeyValues
                };

                // Log iteration details for first few steps
                if (step < 3) {
                    console.log(`\n[Step ${step}]`);
                    console.log('  inputs_embeds shape:', currentInputsEmbeds.dims);
                    console.log('  inputs_embeds dtype:', currentInputsEmbeds.type);
                    console.log('  attention_mask shape:', attentionMask.dims);
                    console.log('  attention_mask dtype:', attentionMask.type);
                    console.log('  position_ids shape:', positionIds.dims);
                    console.log('  position_ids dtype:', positionIds.type);
                    console.log('  position_ids values:', Array.from(positionIds.data.slice(0, 10), Number));

                    // Check embeddings have valid values
                    if (step === 0) {
                        let embedSum = 0;
                        const embedSlice = currentInputsEmbeds.data.slice(0, 1024);
                        for (let i = 0; i < embedSlice.length; i++) {
                            embedSum += Math.abs(embedSlice[i]);
                        }
                        console.log('  First embedding absolute sum:', embedSum.toFixed(2));
                    }
                }

                // Run language model
                const lmOutputs = await this.languageModelSession.run(lmInputs);

                // Extract logits for the last token
                const logits = lmOutputs.logits;
                const logitsData = logits.data;
                const vocabSize = logits.dims[logits.dims.length - 1];
                const lastTokenLogits = new Float32Array(
                    logitsData.slice(logitsData.length - vocabSize)
                );

                if (step < 3) {
                    console.log('  logits shape:', logits.dims);
                    console.log('  vocabSize:', vocabSize);
                }

                // Apply repetition penalty
                const processedLogits = repetitionProcessor.process(
                    generateTokens,
                    lastTokenLogits
                );

                // Sample next token with temperature
                const nextTokenId = this.sampleToken(processedLogits, temperature);
                generateTokens.push(nextTokenId);

                if (step < 3 || step === maxNewTokens - 1) {
                    console.log('  generated token:', nextTokenId);
                    console.log('  generateTokens so far:', generateTokens.slice(0, 10), '...');
                }

                // Check for stop token
                if (nextTokenId === STOP_SPEECH_TOKEN) {
                    console.log('Stop token generated at step', step);
                    break;
                }

                // Update progress every 10 steps
                if (step % 10 === 0) {
                    const elapsed = ((Date.now() - generationStartTime) / 1000).toFixed(1);
                    progressCallback?.({
                        status: 'generating',
                        message: 'Generating speech tokens...',
                        elapsed: elapsed
                    });
                }

                // Update KV cache for next iteration
                // CRITICAL: Deep copy to prevent WebGPU buffer reuse corruption
                // Based on ONNX Runtime Web docs: "User must ensure buffer is valid during inference"
                // We must copy GPU data to CPU before the next session.run() reuses buffers
                for (let i = 0; i < NUM_LAYERS; i++) {
                    const presentKey = lmOutputs[`present.${i}.key`];
                    const presentValue = lmOutputs[`present.${i}.value`];

                    if (presentKey) {
                        // AGGRESSIVE COPY: Force GPU->CPU transfer with .slice()
                        const srcData = presentKey.data;
                        const keyData = presentKey.type === 'float16'
                            ? new Uint16Array(srcData.length)
                            : new Float32Array(srcData.length);

                        // Element-by-element copy to guarantee data transfer
                        for (let j = 0; j < srcData.length; j++) {
                            keyData[j] = srcData[j];
                        }

                        pastKeyValues[`past_key_values.${i}.key`] = new ort.Tensor(
                            presentKey.type,
                            keyData,
                            [...presentKey.dims]  // Clone dimensions array too
                        );
                    }

                    if (presentValue) {
                        // AGGRESSIVE COPY: Force GPU->CPU transfer with .slice()
                        const srcData = presentValue.data;
                        const valueData = presentValue.type === 'float16'
                            ? new Uint16Array(srcData.length)
                            : new Float32Array(srcData.length);

                        // Element-by-element copy to guarantee data transfer
                        for (let j = 0; j < srcData.length; j++) {
                            valueData[j] = srcData[j];
                        }

                        pastKeyValues[`past_key_values.${i}.value`] = new ort.Tensor(
                            presentValue.type,
                            valueData,
                            [...presentValue.dims]  // Clone dimensions array too
                        );
                    }
                }

                // For next iteration, we only need the embedding of the newly generated token
                // Get embedding for the next token
                const nextTokenTensor = new ort.Tensor(
                    'int64',
                    BigInt64Array.from([BigInt(nextTokenId)]),
                    [1, 1]
                );

                const nextEmbedOutputs = await this.embedTokensSession.run({
                    input_ids: nextTokenTensor
                });

                // CRITICAL: Clone embeddings too - same buffer reuse issue
                const embedSrc = nextEmbedOutputs.inputs_embeds.data;
                const embedData = new Float32Array(embedSrc.length);
                for (let j = 0; j < embedSrc.length; j++) {
                    embedData[j] = embedSrc[j];
                }
                currentInputsEmbeds = new ort.Tensor(
                    'float32',
                    embedData,
                    [...nextEmbedOutputs.inputs_embeds.dims]
                );

                // Update attention mask and position IDs
                // CRITICAL: Concatenate with existing mask, don't create new one
                const oldMaskData = attentionMask.data;
                const newMaskData = new BigInt64Array(oldMaskData.length + 1);
                newMaskData.set(oldMaskData, 0);
                newMaskData[oldMaskData.length] = 1n;
                attentionMask = new ort.Tensor('int64', newMaskData, [1, newMaskData.length]);

                const lastPos = Number(positionIds.data[positionIds.data.length - 1]);
                positionIds = new ort.Tensor(
                    'int64',
                    BigInt64Array.from([BigInt(lastPos + 1)]),
                    [1, 1]
                );
            }

            // Remove start token (index 0) and last token (index -1) like Python: [:, 1:-1]
            // Python always removes both first and last, regardless of STOP_SPEECH_TOKEN
            console.log('Speech token generation complete:');
            console.log('  Total generated tokens:', generateTokens.length);
            console.log('  First token (should be START):', generateTokens[0], '(expected:', START_SPEECH_TOKEN, ')');
            console.log('  Last token:', generateTokens[generateTokens.length - 1]);
            console.log('  Second-to-last token:', generateTokens[generateTokens.length - 2]);

            const speechTokens = generateTokens.slice(1, -1);

            console.log('After slicing [1:-1]:');
            console.log('  Speech tokens count:', speechTokens.length);
            console.log('  First speech token:', speechTokens[0]);
            console.log('  Last 10 speech tokens:', speechTokens.slice(-10));
            console.log('  Last speech token:', speechTokens[speechTokens.length - 1]);

            // HACK: Add padding silence tokens at start and end to prevent corruption
            // We'll trim the corresponding audio samples later
            const PADDING_SILENCE_TOKENS = 5; // ~0.5 seconds of silence padding at each end
            const paddingStart = new Array(PADDING_SILENCE_TOKENS).fill(SILENCE_TOKEN);
            const paddingEnd = new Array(PADDING_SILENCE_TOKENS).fill(SILENCE_TOKEN);
            const paddedSpeechTokens = [...paddingStart, ...speechTokens, ...paddingEnd];

            console.log('Applied padding hack:');
            console.log('  Original speech tokens:', speechTokens.length);
            console.log('  Padded speech tokens:', paddedSpeechTokens.length);
            console.log('  Padding:', PADDING_SILENCE_TOKENS, 'silence tokens at start and', PADDING_SILENCE_TOKENS, 'at end');

            // Step 7: Decode to audio
            progressCallback?.({ status: 'decoding', message: 'Decoding to audio waveform...' });

            // Concatenate prompt token, generated tokens, and silence
            const promptTokenData = promptToken?.data
                ? Array.from(promptToken.data, x => Number(x))
                : [];

            console.log('Token sequence composition:');
            console.log('  Prompt tokens:', promptTokenData.length);
            console.log('  Padded speech tokens:', paddedSpeechTokens.length);
            if (promptTokenData.length > 0) {
                console.log('  First 5 prompt tokens:', promptTokenData.slice(0, 5));
            }
            if (paddedSpeechTokens.length > 0) {
                console.log('  First 5 padded tokens:', paddedSpeechTokens.slice(0, 5));
            }

            const silenceTokens = new Array(3).fill(SILENCE_TOKEN);
            const finalTokens = [...promptTokenData, ...paddedSpeechTokens, ...silenceTokens];

            console.log('  Silence tokens:', silenceTokens.length);
            console.log('  Total sequence length:', finalTokens.length);

            const speechTokensTensor = new ort.Tensor(
                'int64',
                BigInt64Array.from(finalTokens.map(t => BigInt(t))),
                [1, finalTokens.length]
            );

            const decoderOutputs = await this.conditionalDecoderSession.run({
                speech_tokens: speechTokensTensor,
                speaker_embeddings: speakerEmbeddings,
                speaker_features: speakerFeatures
            });

            console.log('Decoder output keys:', Object.keys(decoderOutputs));

            // The output might be named differently - check all possible names
            const audioOutput = decoderOutputs.audio || decoderOutputs.wav || decoderOutputs.output || decoderOutputs[Object.keys(decoderOutputs)[0]];

            if (!audioOutput) {
                throw new Error(`No audio output found. Available outputs: ${Object.keys(decoderOutputs).join(', ')}`);
            }

            console.log('Audio output shape:', audioOutput.dims);
            const audioArray = Float32Array.from(audioOutput.data);

            console.log('Generated audio samples (with padding):', audioArray.length);

            // HACK: Trim padding from start and end
            // Each silence token generates approximately 100 samples (24kHz / 240 tokens per second)
            // We added PADDING_SILENCE_TOKENS at start and end, so trim accordingly
            const SAMPLES_PER_TOKEN = 100; // Approximate
            const trimSamplesStart = PADDING_SILENCE_TOKENS * SAMPLES_PER_TOKEN;
            const trimSamplesEnd = PADDING_SILENCE_TOKENS * SAMPLES_PER_TOKEN;

            // Ensure we don't trim more than available
            const startIdx = Math.min(trimSamplesStart, Math.floor(audioArray.length * 0.1));
            const endIdx = Math.max(audioArray.length - trimSamplesEnd, Math.floor(audioArray.length * 0.9));

            const trimmedAudioArray = audioArray.slice(startIdx, endIdx);

            console.log('Trimming padding hack:');
            console.log('  Original length:', audioArray.length, 'samples');
            console.log('  Trimmed from start:', startIdx, 'samples');
            console.log('  Trimmed from end:', audioArray.length - endIdx, 'samples');
            console.log('  Final length:', trimmedAudioArray.length, 'samples');
            console.log('  Duration:', (trimmedAudioArray.length / SAMPLE_RATE).toFixed(2), 'seconds');

            progressCallback?.({
                status: 'complete',
                message: 'Audio generated successfully!',
                progress: 100
            });

            return {
                audio: trimmedAudioArray,
                sampleRate: SAMPLE_RATE,
                duration: trimmedAudioArray.length / SAMPLE_RATE
            };

        } catch (error) {
            console.error('Error during generation:', error);
            console.error('Error stack:', error.stack);
            throw new Error(`Generation failed: ${error.message}`);
        } finally {
            this.isGenerating = false;
        }
    }

    // Helper: Initialize KV cache for all layers
    initializeKVCache(batchSize) {
        const cache = {};
        for (let i = 0; i < NUM_LAYERS; i++) {
            // Each layer has a key and value cache
            // Initial shape: [batch_size, num_heads, 0, head_dim]
            const keyName = `past_key_values.${i}.key`;
            const valueName = `past_key_values.${i}.value`;
            const keyMeta = this.languageModelSession.inputMetadata?.[keyName];
            const valueMeta = this.languageModelSession.inputMetadata?.[valueName];

            const keyDtype = keyMeta?.type || 'float32';
            const valueDtype = valueMeta?.type || 'float32';

            const KeyArrayCtor = keyDtype === 'float16' ? Uint16Array : Float32Array;
            const ValueArrayCtor = valueDtype === 'float16' ? Uint16Array : Float32Array;

            cache[`past_key_values.${i}.key`] = new ort.Tensor(
                keyDtype,
                new KeyArrayCtor(0),
                [batchSize, NUM_KV_HEADS, 0, HEAD_DIM]
            );

            cache[`past_key_values.${i}.value`] = new ort.Tensor(
                valueDtype,
                new ValueArrayCtor(0),
                [batchSize, NUM_KV_HEADS, 0, HEAD_DIM]
            );
        }

        return cache;
    }

    // Helper: Concatenate two tensors along dimension 1 (sequence length)
    concatenateTensors(tensor1, tensor2) {
        // Both tensors should have shape [batch_size, seq_len, hidden_size]
        const batch1 = tensor1.dims[0];
        const seq1 = tensor1.dims[1];
        const hidden1 = tensor1.dims[2];

        const batch2 = tensor2.dims[0];
        const seq2 = tensor2.dims[1];
        const hidden2 = tensor2.dims[2];

        if (batch1 !== batch2 || hidden1 !== hidden2) {
            throw new Error('Tensor dimensions must match for concatenation');
        }

        const newSeqLen = seq1 + seq2;
        const totalSize = batch1 * newSeqLen * hidden1;
        const concatenated = new Float32Array(totalSize);

        // Copy first tensor
        const data1 = tensor1.data;
        const data2 = tensor2.data;

        let offset = 0;
        for (let b = 0; b < batch1; b++) {
            // Copy seq1 from tensor1
            const start1 = b * seq1 * hidden1;
            concatenated.set(
                data1.slice(start1, start1 + seq1 * hidden1),
                offset
            );
            offset += seq1 * hidden1;

            // Copy seq2 from tensor2
            const start2 = b * seq2 * hidden2;
            concatenated.set(
                data2.slice(start2, start2 + seq2 * hidden2),
                offset
            );
            offset += seq2 * hidden2;
        }

        return new ort.Tensor('float32', concatenated, [batch1, newSeqLen, hidden1]);
    }

    // Helper: Sample token with temperature
    sampleToken(logits, temperature = 1.0) {
        // Python uses greedy decoding (argmax), not sampling
        // Find the index with the maximum logit value
        let maxIdx = 0;
        let maxVal = logits[0];

        for (let i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    // Cleanup
    dispose() {
        if (this.speechEncoderSession) this.speechEncoderSession.release();
        if (this.embedTokensSession) this.embedTokensSession.release();
        if (this.languageModelSession) this.languageModelSession.release();
        if (this.conditionalDecoderSession) this.conditionalDecoderSession.release();

        this.isInitialized = false;
        this.isGenerating = false;
    }
}

// Utility: Convert Float32Array to WAV blob
export function audioArrayToWav(audioData, sampleRate) {
    const numChannels = 1;
    const bitDepth = 16;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const dataLength = audioData.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    // Write WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataLength, true);

    // Write audio data
    let offset = 44;
    for (let i = 0; i < audioData.length; i++) {
        const sample = Math.max(-1, Math.min(1, audioData[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

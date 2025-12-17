// Audio recording and processing utilities

const SAMPLE_RATE = 24000;
const MAX_RECORDING_TIME = 30000; // 30 seconds in milliseconds
const MIN_RECORDING_TIME = 1000; // 1 second

export class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.startTime = null;
        this.recordingTimer = null;
    }

    async startRecording() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: SAMPLE_RATE,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            // Start recording
            this.mediaRecorder.start();
            this.startTime = Date.now();

            // Auto-stop after max recording time
            this.recordingTimer = setTimeout(() => {
                if (this.isRecording()) {
                    this.stopRecording();
                }
            }, MAX_RECORDING_TIME);

            return true;
        } catch (error) {
            console.error('Error starting recording:', error);
            throw new Error('Failed to access microphone. Please grant microphone permissions.');
        }
    }

    async stopRecording() {
        return new Promise((resolve, reject) => {
            if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
                reject(new Error('No active recording'));
                return;
            }

            const recordingDuration = Date.now() - this.startTime;

            if (recordingDuration < MIN_RECORDING_TIME) {
                reject(new Error('Recording too short. Minimum 1 second required.'));
                this.cleanup();
                return;
            }

            this.mediaRecorder.onstop = async () => {
                try {
                    // Create blob from recorded chunks
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

                    // Convert to AudioBuffer
                    const audioBuffer = await this.blobToAudioBuffer(audioBlob);

                    // Resample to 24kHz mono if needed
                    const resampledBuffer = await this.resampleAudio(audioBuffer, SAMPLE_RATE);

                    // Convert to WAV blob for storage
                    const wavBlob = await this.audioBufferToWav(resampledBuffer);

                    this.cleanup();
                    resolve({
                        blob: wavBlob,
                        duration: recordingDuration / 1000,
                        sampleRate: SAMPLE_RATE
                    });
                } catch (error) {
                    console.error('Error processing recording:', error);
                    this.cleanup();
                    reject(error);
                }
            };

            this.mediaRecorder.stop();

            if (this.recordingTimer) {
                clearTimeout(this.recordingTimer);
            }
        });
    }

    isRecording() {
        return this.mediaRecorder && this.mediaRecorder.state === 'recording';
    }

    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.recordingTimer) {
            clearTimeout(this.recordingTimer);
            this.recordingTimer = null;
        }
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.startTime = null;
    }

    async blobToAudioBuffer(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        const audioContext = new AudioContextClass();
        try {
            return await audioContext.decodeAudioData(arrayBuffer);
        } finally {
            if (typeof audioContext.close === 'function') {
                await audioContext.close();
            }
        }
    }

    async resampleAudio(audioBuffer, targetSampleRate) {
        // If already at target sample rate and mono, return as is
        if (audioBuffer.sampleRate === targetSampleRate && audioBuffer.numberOfChannels === 1) {
            return audioBuffer;
        }

        const offlineContext = new OfflineAudioContext(
            1, // mono
            audioBuffer.duration * targetSampleRate,
            targetSampleRate
        );

        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();

        const renderedBuffer = await offlineContext.startRendering();
        if (typeof offlineContext.close === 'function') {
            await offlineContext.close();
        }
        return renderedBuffer;
    }

    async audioBufferToWav(audioBuffer) {
        const numberOfChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numberOfChannels * bytesPerSample;

        const data = audioBuffer.getChannelData(0);
        const dataLength = data.length * bytesPerSample;
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
        view.setUint32(16, 16, true); // fmt chunk size
        view.setUint16(20, format, true);
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, dataLength, true);

        // Write audio data
        const volume = 0.8;
        let offset = 44;
        for (let i = 0; i < data.length; i++) {
            const sample = Math.max(-1, Math.min(1, data[i] * volume));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    getRecordingDuration() {
        if (!this.startTime) return 0;
        return Math.floor((Date.now() - this.startTime) / 1000);
    }
}

// Audio loading utilities
export async function loadAudioFromFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    if (typeof audioContext.close === 'function') {
        await audioContext.close();
    }

    // Resample to 24kHz mono
    const recorder = new AudioRecorder();
    const resampledBuffer = await recorder.resampleAudio(audioBuffer, SAMPLE_RATE);
    const wavBlob = await recorder.audioBufferToWav(resampledBuffer);

    return {
        blob: wavBlob,
        duration: resampledBuffer.duration,
        sampleRate: SAMPLE_RATE
    };
}

// Convert blob to Float32Array for model input
export async function blobToFloat32Array(blob) {
    console.log('🎵 [Audio] Converting blob to Float32Array:', {
        blobSize: blob.size,
        blobType: blob.type
    });

    const arrayBuffer = await blob.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();

    try {
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        console.log('  Original audio format:', {
            sampleRate: audioBuffer.sampleRate,
            channels: audioBuffer.numberOfChannels,
            duration: audioBuffer.duration.toFixed(2) + 's'
        });

        // CRITICAL: Always resample to 24kHz mono for the model
        let finalBuffer = audioBuffer;
        if (audioBuffer.sampleRate !== SAMPLE_RATE || audioBuffer.numberOfChannels !== 1) {
            console.log(`  ⚠ Resampling from ${audioBuffer.sampleRate}Hz to ${SAMPLE_RATE}Hz...`);

            const offlineContext = new OfflineAudioContext(
                1, // mono
                Math.ceil(audioBuffer.duration * SAMPLE_RATE),
                SAMPLE_RATE
            );

            const source = offlineContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(offlineContext.destination);
            source.start();

            finalBuffer = await offlineContext.startRendering();
            if (typeof offlineContext.close === 'function') {
                await offlineContext.close();
            }

            console.log(`  ✓ Resampled to ${SAMPLE_RATE}Hz`);
        }

        const audioData = finalBuffer.getChannelData(0);

        // Calculate simple statistics to identify this audio
        let sum = 0;
        let sumSquared = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i];
            sumSquared += audioData[i] * audioData[i];
        }
        const mean = sum / audioData.length;
        const variance = (sumSquared / audioData.length) - (mean * mean);
        const rms = Math.sqrt(sumSquared / audioData.length);

        console.log('✓ [Audio] Conversion complete:', {
            sampleCount: audioData.length,
            duration: finalBuffer.duration.toFixed(2) + 's',
            sampleRate: finalBuffer.sampleRate,
            channels: finalBuffer.numberOfChannels,
            statistics: {
                mean: mean.toFixed(6),
                rms: rms.toFixed(6),
                variance: variance.toFixed(6),
                first10: Array.from(audioData.slice(0, 10).map(v => v.toFixed(4)))
            }
        });

        return audioData;
    } finally {
        if (typeof audioContext.close === 'function') {
            await audioContext.close();
        }
    }
}

// Create waveform visualization
export function drawWaveform(canvas, audioData) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#1a1f2e';
    ctx.fillRect(0, 0, width, height);

    // Draw waveform
    ctx.strokeStyle = '#5b7cff';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const step = Math.ceil(audioData.length / width);
    const amp = height / 2;

    for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;

        for (let j = 0; j < step; j++) {
            const datum = audioData[(i * step) + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
        }

        const y1 = (1 + min) * amp;
        const y2 = (1 + max) * amp;

        ctx.moveTo(i, y1);
        ctx.lineTo(i, y2);
    }

    ctx.stroke();
}

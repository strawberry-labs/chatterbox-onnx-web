// Main application controller

import { db } from './db.js';
import { AudioRecorder, loadAudioFromFile, blobToFloat32Array, drawWaveform } from './audio.js';
import { ChatterboxTTSEngine, audioArrayToWav } from './tts-engine-complete.js';

const escapeHtml = (value = '') => String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const EMOTION_TAGS = [
    'angry', 'fear', 'surprised', 'whispering', 'advertisement', 'dramatic',
    'narration', 'crying', 'happy', 'sarcastic', 'clear throat', 'sigh',
    'shush', 'cough', 'groan', 'sniff', 'gasp', 'chuckle', 'laugh'
];

const escapeRegExp = (str = '') => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const emotionTagLookup = new Set(EMOTION_TAGS.map(tag => `[${tag.toLowerCase()}]`));
const emotionTagPattern = new RegExp(`\\[(?:${EMOTION_TAGS.map(escapeRegExp).join('|')})\\]`, 'gi');

let refreshTextHighlights = () => {};

function buildHighlightedHtml(text = '') {
    if (!text) return '';

    const segments = [];
    let lastIndex = 0;

    text.replace(emotionTagPattern, (match, offset) => {
        if (offset > lastIndex) {
            segments.push({ text: text.slice(lastIndex, offset), isTag: false });
        }
        segments.push({ text: match, isTag: true });
        lastIndex = offset + match.length;
    });

    if (lastIndex < text.length) {
        segments.push({ text: text.slice(lastIndex), isTag: false });
    }

    return segments.map(segment => {
        const safe = escapeHtml(segment.text);
        return segment.isTag
            ? `<mark class="tag-highlight">${safe}</mark>`
            : safe;
    }).join('');
}

function initializeTextHighlighter() {
    const textarea = document.getElementById('text-input');
    const highlights = document.getElementById('text-input-highlights');
    if (!textarea || !highlights) {
        refreshTextHighlights = () => {};
        return;
    }

    const syncScroll = () => {
        highlights.scrollTop = textarea.scrollTop;
        highlights.scrollLeft = textarea.scrollLeft;
    };

    refreshTextHighlights = () => {
        highlights.innerHTML = buildHighlightedHtml(textarea.value || '');
        syncScroll();
    };

    textarea.addEventListener('input', refreshTextHighlights);
    textarea.addEventListener('scroll', syncScroll);
    textarea.addEventListener('keydown', (event) => handleTagDeletion(event, textarea));

    refreshTextHighlights();
}

function handleTagDeletion(event, textarea) {
    if (event.key !== 'Backspace' || textarea.selectionStart !== textarea.selectionEnd) {
        return;
    }

    const cursor = textarea.selectionStart;
    const value = textarea.value;

    if (cursor === 0 || value[cursor - 1] !== ']') {
        return;
    }

    const start = value.lastIndexOf('[', cursor - 1);
    if (start === -1) {
        return;
    }

    const candidate = value.slice(start, cursor).toLowerCase();
    if (!emotionTagLookup.has(candidate)) {
        return;
    }

    event.preventDefault();
    const before = value.slice(0, start);
    const after = value.slice(cursor);
    textarea.value = before + after;
    textarea.selectionStart = textarea.selectionEnd = start;
    refreshTextHighlights();
}

function cleanupHistoryAudioUrls() {
    state.historyAudioUrls.forEach(url => URL.revokeObjectURL(url));
    state.historyAudioUrls = [];
}

function cleanupVoicePreview() {
    if (state.activeVoicePreview?.audio) {
        state.activeVoicePreview.audio.pause();
    }
    if (state.activeVoicePreview?.url) {
        URL.revokeObjectURL(state.activeVoicePreview.url);
    }
    state.activeVoicePreview = null;
}

function getOutputAudioUrl(blob) {
    if (state.currentAudioUrl) {
        URL.revokeObjectURL(state.currentAudioUrl);
    }
    const url = URL.createObjectURL(blob);
    state.currentAudioUrl = url;
    return url;
}

const modelProgressElements = {
    container: document.getElementById('model-progress'),
    text: document.getElementById('model-progress-text'),
    bar: document.getElementById('model-progress-fill')
};

function updateModelProgress(progress = {}) {
    const { message, progress: percent, status } = progress;
    if (modelProgressElements.text && (message || status)) {
        let displayMessage = message;
        if (!displayMessage) {
            if (status === 'ready') {
                displayMessage = 'Models ready';
            } else if (status === 'error') {
                displayMessage = 'Model load failed';
            } else if (status === 'loading') {
                displayMessage = 'Loading models...';
            }
        }
        if (displayMessage) {
            modelProgressElements.text.textContent = displayMessage;
        }
    }

    if (modelProgressElements.container && status) {
        modelProgressElements.container.dataset.status = status;

        // Add breathing glow effect when loading
        if (status === 'loading') {
            modelProgressElements.container.classList.add('loading');
        } else {
            modelProgressElements.container.classList.remove('loading');
        }
    }

    let normalized = undefined;
    if (typeof percent === 'number' && Number.isFinite(percent)) {
        normalized = Math.max(0, Math.min(100, percent));
    } else if (status === 'ready') {
        normalized = 100;
    } else if (status === 'error') {
        normalized = 0;
    }

    if (modelProgressElements.bar && normalized !== undefined) {
        modelProgressElements.bar.style.width = `${normalized}%`;
    }
}

updateModelProgress({ status: 'idle', message: 'Idle', progress: 0 });

// Global state
const state = {
    currentPage: 'home',
    selectedVoice: null,
    voices: [],
    history: [],
    isGenerating: false,
    ttsEngine: new ChatterboxTTSEngine(),
    audioRecorder: new AudioRecorder(),
    recordedAudio: null,
    previewAudioUrl: null,
    temperature: 0.20,
    repetitionPenalty: 1.20,
    voiceFilter: '',
    currentAudioUrl: null,
    activeVoicePreview: null,
    historyAudioUrls: []
};

// Initialize application
async function init() {
    try {
        // Initialize database
        await db.init();

        // Load voices and history
        await loadVoices();
        await loadHistory();

        // Set up event listeners
        setupNavigation();
        setupHomePageEvents();
        setupVoiceLibraryEvents();
        setupHistoryPageEvents();
        setupModalEvents();

        // Initialize TTS engine
        console.log('Initializing TTS engine...');
        updateModelProgress({ status: 'loading', message: 'Initializing models...', progress: 5 });
        await state.ttsEngine.initialize((progress) => {
            updateModelProgress(progress);
        });
        updateModelProgress({ status: 'ready', message: 'Models ready', progress: 100 });
        console.log('TTS engine ready!');

        // Create default voice if none exists
        if (state.voices.length === 0) {
            await createDefaultVoice();
        }

    } catch (error) {
        console.error('Error initializing application:', error);
        updateModelProgress({ status: 'error', message: 'Initialization failed', progress: 0 });
        alert('Failed to initialize application. Please refresh the page.');
    }
}

// Create a default voice for testing
async function createDefaultVoice() {
    const defaultVoice = {
        name: 'Ethan',
        description: 'Male - Clear and assertive',
        audioBlob: null, // Would need actual audio data
        duration: 0,
        isDefault: true
    };

    const id = await db.addVoice(defaultVoice);
    defaultVoice.id = id;
    state.voices.push(defaultVoice);
    state.selectedVoice = defaultVoice;

    updateVoiceSelector();
    renderVoiceLibrary();
}

// Navigation
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            navigateTo(page);
        });
    });

    // Footer buttons
    document.getElementById('view-model-btn').addEventListener('click', () => {
        window.open('https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX', '_blank');
    });

    document.getElementById('source-code-btn').addEventListener('click', () => {
        window.open('https://github.com/resemble-ai/chatterbox', '_blank');
    });
}

function navigateTo(page) {
    // Update navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.page === page) {
            item.classList.add('active');
        }
    });

    // Update pages
    document.querySelectorAll('.page').forEach(p => {
        p.classList.remove('active');
    });

    document.getElementById(`${page}-page`).classList.add('active');
    state.currentPage = page;

    // Load page-specific data
    if (page === 'library') {
        renderVoiceLibrary();
    } else if (page === 'history') {
        renderHistory();
    }
}

// Home page events
function setupHomePageEvents() {
    const textInput = document.getElementById('text-input');
    initializeTextHighlighter();

    // Voice selector
    document.getElementById('voice-selector').addEventListener('change', (e) => {
        const voiceId = parseInt(e.target.value);
        state.selectedVoice = state.voices.find(v => v.id === voiceId);
        console.log('🎯 [UI] Voice selected from dropdown:', {
            id: state.selectedVoice?.id,
            name: state.selectedVoice?.name,
            hasAudio: !!state.selectedVoice?.audioBlob,
            audioBlobSize: state.selectedVoice?.audioBlob?.size || 0,
            audioBlobType: state.selectedVoice?.audioBlob?.type || 'unknown'
        });
    });

    // Create voice button
    document.getElementById('create-voice-btn').addEventListener('click', () => {
        showModal();
    });

    // Text input emotion tags
    document.querySelectorAll('.tag-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tag = `[${btn.textContent.trim()}]`;
            const cursorPos = textInput.selectionStart;
            const textBefore = textInput.value.substring(0, cursorPos);
            const textAfter = textInput.value.substring(cursorPos);
            textInput.value = textBefore + tag + ' ' + textAfter;
            textInput.focus();
            textInput.selectionStart = textInput.selectionEnd = cursorPos + tag.length + 1;
            refreshTextHighlights();
        });
    });

    // Temperature slider
    const tempSlider = document.getElementById('temperature-slider');
    const tempValue = document.getElementById('temp-value');

    tempSlider.addEventListener('input', (e) => {
        state.temperature = parseFloat(e.target.value);
        tempValue.textContent = state.temperature.toFixed(2);
    });

    // Repetition penalty slider
    const repSlider = document.getElementById('repetition-slider');
    const repValue = document.getElementById('rep-value');

    repSlider.addEventListener('input', (e) => {
        state.repetitionPenalty = parseFloat(e.target.value);
        repValue.textContent = state.repetitionPenalty.toFixed(2);
    });

    // Generate button
    document.getElementById('generate-btn').addEventListener('click', handleGenerate);

    // Download button
    document.getElementById('download-btn').addEventListener('click', handleDownload);
}

// Generate audio
async function handleGenerate() {
    const text = document.getElementById('text-input').value.trim();

    if (!text) {
        alert('Please enter some text to generate.');
        return;
    }

    if (!state.selectedVoice) {
        alert('Please select a voice.');
        return;
    }

    if (state.isGenerating) {
        return;
    }

    // Log selected voice details
    console.log('=== STARTING GENERATION ===');
    console.log('Selected Voice:', {
        id: state.selectedVoice.id,
        name: state.selectedVoice.name,
        description: state.selectedVoice.description,
        hasAudioBlob: !!state.selectedVoice.audioBlob,
        audioBlobSize: state.selectedVoice.audioBlob?.size || 0,
        audioBlobType: state.selectedVoice.audioBlob?.type || 'unknown'
    });
    console.log('Generation Parameters:', {
        temperature: state.temperature,
        repetitionPenalty: state.repetitionPenalty,
        maxNewTokens: 1024
    });

    state.isGenerating = true;

    // Show loading section
    document.getElementById('output-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';

    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = true;

    try {
        // Get voice audio data
        let voiceAudioData;

        if (state.selectedVoice.audioBlob) {
            console.log('Converting voice audio blob to Float32Array...');
            voiceAudioData = await blobToFloat32Array(state.selectedVoice.audioBlob);
            console.log('✓ Voice audio loaded successfully');
        } else {
            throw new Error('Selected voice has no audio data');
        }

        // Update loading status
        const updateStatus = (progress) => {
            const statusEl = document.getElementById('loading-status');
            if (progress.elapsed !== undefined) {
                statusEl.textContent = `${progress.message}\n${progress.elapsed}s elapsed`;
            } else {
                statusEl.textContent = progress.message || progress;
            }
        };

        // Generate audio
        const result = await state.ttsEngine.generate(text, voiceAudioData, {
            maxNewTokens: 1024,
            repetitionPenalty: state.repetitionPenalty,
            temperature: state.temperature,
            progressCallback: updateStatus
        });

        // Convert to WAV blob
        const audioBlob = audioArrayToWav(result.audio, result.sampleRate);

        // Save to history
        const historyEntry = {
            text,
            voiceName: state.selectedVoice.name,
            audioBlob,
            temperature: state.temperature,
            repetitionPenalty: state.repetitionPenalty,
            duration: result.duration
        };

        const historyId = await db.addHistory(historyEntry);
        historyEntry.id = historyId;
        state.history.unshift(historyEntry);
        renderHistory();

        // Display audio
        displayGeneratedAudio(audioBlob, result.audio);

    } catch (error) {
        console.error('Error generating audio:', error);
        alert(`Failed to generate audio: ${error.message}`);
        document.getElementById('loading-section').style.display = 'none';
    } finally {
        state.isGenerating = false;
        generateBtn.disabled = false;
    }
}

// Display generated audio
function displayGeneratedAudio(audioBlob, audioData) {
    document.getElementById('loading-section').style.display = 'none';
    document.getElementById('output-section').style.display = 'block';

    // Set up audio player
    const audioPlayer = document.getElementById('audio-player');
    const audioUrl = getOutputAudioUrl(audioBlob);
    audioPlayer.src = audioUrl;

    // Draw waveform
    const waveformEl = document.getElementById('waveform');
    waveformEl.innerHTML = '<canvas id="waveform-canvas" width="1000" height="100"></canvas>';
    const canvas = document.getElementById('waveform-canvas');
    drawWaveform(canvas, audioData);

    // Store current audio for download
    state.currentAudioBlob = audioBlob;
}

// Download audio
function handleDownload() {
    if (!state.currentAudioBlob) {
        return;
    }

    const url = URL.createObjectURL(state.currentAudioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chatterbox-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Voice Library
function setupVoiceLibraryEvents() {
    document.getElementById('add-voice-btn').addEventListener('click', () => {
        showModal();
    });

    document.getElementById('voice-search').addEventListener('input', (e) => {
        filterVoices(e.target.value);
    });
}

async function loadVoices() {
    state.voices = await db.getAllVoices();
    updateVoiceSelector();
    renderVoiceLibrary();

    if (state.voices.length > 0 && !state.selectedVoice) {
        state.selectedVoice = state.voices[0];
        document.getElementById('voice-selector').value = state.selectedVoice.id;
    }
}

function updateVoiceSelector() {
    const selector = document.getElementById('voice-selector');

    console.log('🔄 [UI] Updating voice selector with', state.voices.length, 'voices:');
    state.voices.forEach(voice => {
        console.log(`  - ID ${voice.id}: "${voice.name}" (${voice.audioBlob ? voice.audioBlob.size + ' bytes' : 'no audio'})`);
    });

    selector.innerHTML = state.voices.map(voice => `
        <option value="${voice.id}">${escapeHtml(voice.name)}</option>
    `).join('');

    if (state.selectedVoice) {
        selector.value = state.selectedVoice.id;
        console.log('  Selected voice:', state.selectedVoice.name, '(ID:', state.selectedVoice.id + ')');
    }
}

function renderVoiceLibrary() {
    const grid = document.getElementById('voice-grid');
    const count = document.getElementById('voice-count');

    const filter = state.voiceFilter.toLowerCase();
    const voicesToRender = filter
        ? state.voices.filter(voice =>
            voice.name?.toLowerCase().includes(filter) ||
            (voice.description && voice.description.toLowerCase().includes(filter))
        )
        : state.voices;

    count.textContent = `${voicesToRender.length} voice${voicesToRender.length !== 1 ? 's' : ''}${state.voiceFilter ? ' (filtered)' : ''}`;

    if (voicesToRender.length === 0) {
        const message = state.voiceFilter
            ? 'No voices match your search'
            : 'Click "Add Voice" to create your first voice clone';
        grid.innerHTML = `
            <div class="empty-state">
                <h3>${state.voiceFilter ? 'No matches' : 'No voices yet'}</h3>
                <p>${message}</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = voicesToRender.map(voice => `
        <div class="voice-card ${state.selectedVoice?.id === voice.id ? 'selected' : ''}" data-voice-id="${voice.id}">
            <div class="voice-card-header">
                <div class="voice-avatar">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="currentColor"/>
                    </svg>
                </div>
                <div class="voice-info">
                    <div class="voice-name">${escapeHtml(voice.name)}</div>
                    <div class="voice-desc">${escapeHtml(voice.description || 'No description')}</div>
                </div>
                <div class="voice-selected-indicator">
                    ${state.selectedVoice?.id === voice.id ? '✓' : ''}
                </div>
            </div>
            <div class="voice-actions">
                <button class="voice-action-btn play-voice" data-voice-id="${voice.id}">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M5 3L13 8L5 13V3Z" fill="currentColor"/>
                    </svg>
                    Play
                </button>
                <button class="voice-action-btn delete-voice" data-voice-id="${voice.id}">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M4 6V14H12V6H4ZM6 3H10L11 4H13V6H3V4H5L6 3Z" fill="currentColor"/>
                    </svg>
                    Delete
                </button>
            </div>
        </div>
    `).join('');

    // Add event listeners
    document.querySelectorAll('.voice-card').forEach(card => {
        card.addEventListener('click', (e) => {
            if (!e.target.closest('.voice-action-btn')) {
                const voiceId = parseInt(card.dataset.voiceId);
                selectVoice(voiceId);
            }
        });
    });

    document.querySelectorAll('.play-voice').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const voiceId = parseInt(btn.dataset.voiceId);
            playVoice(voiceId);
        });
    });

    document.querySelectorAll('.delete-voice').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const voiceId = parseInt(btn.dataset.voiceId);

            if (confirm('Are you sure you want to delete this voice?')) {
                await db.deleteVoice(voiceId);
                state.voices = state.voices.filter(v => v.id !== voiceId);

                if (state.selectedVoice?.id === voiceId) {
                    state.selectedVoice = state.voices[0] || null;
                }
                if (state.activeVoicePreview?.voiceId === voiceId) {
                    cleanupVoicePreview();
                }

                updateVoiceSelector();
                renderVoiceLibrary();
            }
        });
    });
}

function selectVoice(voiceId) {
    state.selectedVoice = state.voices.find(v => v.id === voiceId);
    document.getElementById('voice-selector').value = voiceId;
    renderVoiceLibrary();
    navigateTo('home');
}

function playVoice(voiceId) {
    const voice = state.voices.find(v => v.id === voiceId);

    if (voice && voice.audioBlob) {
        cleanupVoicePreview();
        const url = URL.createObjectURL(voice.audioBlob);
        const audio = new Audio(url);
        state.activeVoicePreview = { audio, url, voiceId };

        const releaseUrl = () => {
            if (state.activeVoicePreview?.url === url) {
                state.activeVoicePreview = null;
            }
            URL.revokeObjectURL(url);
        };

        audio.addEventListener('ended', releaseUrl, { once: true });
        audio.addEventListener('error', releaseUrl, { once: true });

        audio.play().catch(error => {
            console.error('Error playing voice preview:', error);
            releaseUrl();
        });
    }
}

function filterVoices(query) {
    state.voiceFilter = query.trim();
    renderVoiceLibrary();
}

// History
function setupHistoryPageEvents() {
    // Event listeners set up in renderHistory
}

async function loadHistory() {
    state.history = await db.getAllHistory();
}

function renderHistory() {
    const historyList = document.getElementById('history-list');
    cleanupHistoryAudioUrls();

    if (state.history.length === 0) {
        historyList.innerHTML = `
            <div class="empty-state">
                <h3>No history yet</h3>
                <p>Generated audio will appear here</p>
            </div>
        `;
        state.historyAudioUrls = [];
        return;
    }

    const audioUrls = [];
    historyList.innerHTML = state.history.map(entry => {
        const safeText = escapeHtml(entry.text || '').replace(/\n/g, '<br>');
        const safeVoiceName = escapeHtml(entry.voiceName || 'Unknown voice');
        const audioUrl = URL.createObjectURL(entry.audioBlob);
        audioUrls.push(audioUrl);
        return `
        <div class="history-entry" data-history-id="${entry.id}">
            <div class="history-header">
                <div class="history-icon">
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="2" fill="none"/>
                        <path d="M10 6V10L13 13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </div>
                <div class="history-meta">
                    <div class="history-date">${new Date(entry.timestamp).toLocaleString()}</div>
                    <div class="history-title">History Entry</div>
                </div>
                <button class="history-delete" data-history-id="${entry.id}">
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <path d="M5 7V17H15V7H5ZM7 4H13L14 5H17V7H3V5H6L7 4Z" fill="currentColor"/>
                    </svg>
                </button>
            </div>
            <div class="history-text">${safeText}</div>
            <div class="history-details">
                <div class="history-detail-item">
                    <div class="history-detail-label">Voice</div>
                    <div class="history-detail-value">${safeVoiceName}</div>
                </div>
                <div class="history-detail-item">
                    <div class="history-detail-label">Temperature</div>
                    <div class="history-detail-value">${entry.temperature.toFixed(2)}</div>
                </div>
                <div class="history-detail-item">
                    <div class="history-detail-label">Repetition Penalty</div>
                    <div class="history-detail-value">${entry.repetitionPenalty.toFixed(2)}</div>
                </div>
            </div>
            <audio controls class="preview-audio" src="${audioUrl}"></audio>
        </div>
    `;
    }).join('');
    state.historyAudioUrls = audioUrls;

    // Add delete event listeners
    document.querySelectorAll('.history-delete').forEach(btn => {
        btn.addEventListener('click', async () => {
            const historyId = parseInt(btn.dataset.historyId);

            if (confirm('Are you sure you want to delete this entry?')) {
                await db.deleteHistory(historyId);
                state.history = state.history.filter(h => h.id !== historyId);
                renderHistory();
            }
        });
    });
}

async function startRecordingFlow() {
    if (state.audioRecorder.isRecording()) {
        console.warn('Recording already in progress');
        return;
    }

    hidePreview({ resetForm: false });

    try {
        state.recordedAudio = null;
        await state.audioRecorder.startRecording();
        showRecordingState();
    } catch (error) {
        alert(error.message);
    }
}

// Modal
function setupModalEvents() {
    const modal = document.getElementById('voice-upload-modal');
    const closeBtn = document.getElementById('close-modal-btn');
    const uploadFileBtn = document.getElementById('upload-file-btn');
    const fileInput = document.getElementById('file-input');
    const recordBtn = document.getElementById('record-btn');
    const stopRecordingBtn = document.getElementById('stop-recording-btn');
    const reRecordBtn = document.getElementById('re-record-btn');
    const cancelVoiceBtn = document.getElementById('cancel-voice-btn');
    const saveVoiceBtn = document.getElementById('save-voice-btn');

    closeBtn.addEventListener('click', hideModal);

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            hideModal();
        }
    });

    uploadFileBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];

        if (file) {
            try {
                const audioData = await loadAudioFromFile(file);
                state.recordedAudio = audioData;
                showPreview(audioData.blob);
            } catch (error) {
                alert(`Error loading audio file: ${error.message}`);
            }
        }
    });

    recordBtn.addEventListener('click', () => {
        startRecordingFlow();
    });

    stopRecordingBtn.addEventListener('click', async () => {
        try {
            const audioData = await state.audioRecorder.stopRecording();
            state.recordedAudio = audioData;
            showPreview(audioData.blob);
        } catch (error) {
            alert(error.message);
            hideRecordingState();
        }
    });

    if (reRecordBtn) {
        reRecordBtn.addEventListener('click', () => {
            startRecordingFlow();
        });
    }

    cancelVoiceBtn.addEventListener('click', () => {
        hidePreview({ resetForm: true });
        hideRecordingState();
        state.audioRecorder.cleanup();
        state.recordedAudio = null;
    });

    saveVoiceBtn.addEventListener('click', async () => {
        const name = document.getElementById('voice-name-input').value.trim();
        const description = document.getElementById('voice-desc-input').value.trim();

        if (!name) {
            alert('Please enter a voice name');
            return;
        }

        if (!state.recordedAudio) {
            alert('No audio recorded');
            return;
        }

        try {
            console.log('💾 [UI] Saving new voice:', {
                name,
                description,
                blobSize: state.recordedAudio.blob.size,
                blobType: state.recordedAudio.blob.type,
                duration: state.recordedAudio.duration
            });

            const voice = {
                name,
                description,
                audioBlob: state.recordedAudio.blob,
                duration: state.recordedAudio.duration
            };

            const id = await db.addVoice(voice);
            voice.id = id;
            state.voices.push(voice);

            console.log('✓ [UI] Voice saved with ID:', id);

            // Auto-select if this is the first/only voice
            if (!state.selectedVoice || state.voices.length === 1) {
                state.selectedVoice = voice;
                console.log('✓ [UI] Auto-selected voice:', voice.name);
            }

            updateVoiceSelector();
            renderVoiceLibrary();

            hideModal();
            alert('Voice saved successfully!');

        } catch (error) {
            alert(`Error saving voice: ${error.message}`);
        }
    });
}

function showModal() {
    const modal = document.getElementById('voice-upload-modal');
    modal.classList.add('active');
    state.recordedAudio = null;
    hideRecordingState();
    hidePreview({ resetForm: true });
    state.audioRecorder.cleanup();
}

function hideModal() {
    document.getElementById('voice-upload-modal').classList.remove('active');
    hideRecordingState();
    hidePreview({ resetForm: true });
    state.audioRecorder.cleanup();
    state.recordedAudio = null;
}

function showRecordingState() {
    document.getElementById('recording-state').style.display = 'block';

    const timer = setInterval(() => {
        const duration = state.audioRecorder.getRecordingDuration();
        const minutes = Math.floor(duration / 60);
        const seconds = duration % 60;
        document.getElementById('recording-time').textContent = `${minutes}:${String(seconds).padStart(2, '0')}`;

        if (!state.audioRecorder.isRecording()) {
            clearInterval(timer);
        }
    }, 1000);
}

function hideRecordingState() {
    document.getElementById('recording-state').style.display = 'none';
    document.getElementById('recording-time').textContent = '0:00';
}

function showPreview(audioBlob) {
    hideRecordingState();

    if (state.previewAudioUrl) {
        URL.revokeObjectURL(state.previewAudioUrl);
        state.previewAudioUrl = null;
    }

    const previewAudio = document.getElementById('preview-audio');
    const objectUrl = URL.createObjectURL(audioBlob);
    previewAudio.src = objectUrl;
    previewAudio.load();
    state.previewAudioUrl = objectUrl;

    document.getElementById('preview-state').style.display = 'block';
}

function hidePreview({ resetForm = false } = {}) {
    document.getElementById('preview-state').style.display = 'none';
    const previewAudio = document.getElementById('preview-audio');
    if (previewAudio) {
        previewAudio.pause();
        previewAudio.removeAttribute('src');
        previewAudio.load();
    }
    if (state.previewAudioUrl) {
        URL.revokeObjectURL(state.previewAudioUrl);
        state.previewAudioUrl = null;
    }
    if (resetForm) {
        document.getElementById('voice-name-input').value = '';
        document.getElementById('voice-desc-input').value = '';
    }
}

window.addEventListener('beforeunload', () => {
    if (state.currentAudioUrl) {
        URL.revokeObjectURL(state.currentAudioUrl);
        state.currentAudioUrl = null;
    }
    if (state.previewAudioUrl) {
        URL.revokeObjectURL(state.previewAudioUrl);
        state.previewAudioUrl = null;
    }
    cleanupVoicePreview();
    cleanupHistoryAudioUrls();
    state.audioRecorder.cleanup();
});

// Start the application
init().catch(console.error);

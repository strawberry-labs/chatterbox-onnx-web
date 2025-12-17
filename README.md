# ChatterboxWeb - Chatterbox-Turbo TTS in Browser

**In-browser voice cloning with Chatterbox Turbo powered by Transformers.js and WebGPU**

![ChatterboxWeb Banner](https://storage.googleapis.com/chatterbox-demo-samples/turbo/turbo-banner.jpg)

## Live Demo

**Try it now:** [https://chatterbox-turbo.b-cdn.net/](https://chatterbox-turbo.b-cdn.net/)

## Overview

ChatterboxWeb is a browser-based implementation of [Resemble AI's Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) text-to-speech model. It runs 100% locally in your browser using WebGPU acceleration for high-performance inference.

### Features

- **Zero-shot voice cloning** - Clone any voice from a short audio sample (1-60 seconds)
- **Default voices included** - Start generating speech immediately with pre-loaded voice samples
- **Paralinguistic tags** - Add emotions and sounds like `[laugh]`, `[chuckle]`, `[sigh]`, etc.
- **Voice library management** - Save and organize multiple voice profiles locally
- **Generation history** - Keep track of all your generated audio
- **100% local processing** - All voice data and audio stays in your browser (IndexedDB)
- **WebGPU acceleration** - Fast inference using your GPU
- **No server required** - Everything runs client-side except model weight downloads

## Quick Start

### Prerequisites

- **Node.js 18+** and npm
- **Modern browser with WebGPU**: Chrome 113+, Edge 113+, or Chrome Canary
- **At least 4GB RAM** available
- **Good internet connection** for initial model download (~350MB for FP32, ~175MB for FP16)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd chatterbox-turbo-onnx
```

2. **Install dependencies**

```bash
npm install
```

3. **Start the development server**

```bash
npm run dev
```

The development server will automatically open in your browser at `http://localhost:5173`

**Important**: Open browser DevTools (F12) to see initialization progress and debug any issues.

### First Time Setup

On first launch, the application will:

1. Download the model weights (~350MB for FP32, ~175MB for FP16) from HuggingFace
2. Initialize the ONNX Runtime with WebGPU backend (or fallback to WASM)
3. Set up IndexedDB for local storage

**Note:** The initial model download may take 2-5 minutes depending on your internet speed. Models are cached in your browser, so subsequent loads will be instant.

### Enable WebGPU (Required for Good Performance)

1. Open Chrome and navigate to `chrome://flags`
2. Search for "WebGPU"
3. Enable "Unsafe WebGPU" flag
4. Restart browser
5. Verify at `chrome://gpu` - should show "WebGPU: Hardware accelerated"

**Alternative**: Use Chrome Canary which has WebGPU enabled by default.

## Usage

### 1. Create a Voice Profile

Click "Create new voice" or "Add Voice" and either:

- **Upload an audio file** - Any format, will be automatically converted to 24kHz mono WAV
- **Record with microphone** - Record 1-60 seconds of speech

Give your voice a name and description, then save it.

### 2. Generate Speech

1. Select a voice from the dropdown (or use the default voices provided)
2. Enter your text in the text box
3. (Optional) Add emotion tags like `[laugh]`, `[chuckle]`, `[sigh]` by clicking the emotion tag buttons below the text area
4. Adjust temperature and repetition penalty if desired:
   - **Temperature**: Controls randomness and variation in speech (0.0-2.0, default 0.20)
   - **Repetition Penalty**: Reduces repetitive speech patterns (1.0-2.0, default 1.20)
5. Click "Generate" to create the audio
6. Play, download, or regenerate with different settings

### 3. Manage Your Voices

Go to "Voice Library" to:

- View all saved voices
- Play voice samples
- Delete voices you no longer need
- Search through your voice collection

### 4. View Generation History

Go to "History" to:

- Review all previously generated audio
- Re-listen to past generations
- See the parameters used
- Delete old entries

## Available Emotion Tags

The following paralinguistic tags are supported. Simply click the tag buttons in the UI or type them directly into your text:

**Emotions:** `[angry]`, `[fear]`, `[surprised]`, `[happy]`, `[crying]`, `[dramatic]`, `[sarcastic]`

**Sounds:** `[laugh]`, `[chuckle]`, `[cough]`, `[groan]`, `[sniff]`, `[gasp]`, `[sigh]`, `[shush]`, `[clear throat]`

**Styles:** `[whispering]`, `[narration]`, `[advertisement]`

### Example Usage

```
Oh, that's hilarious! [chuckle] Um anyway, how are you doing today?

[whispering] I can't believe this is happening. [gasp] What should we do?

[dramatic] The time has come... [sigh] for us to make our final decision.
```

## Project Structure

```
chatterbox-turbo-onnx/
├── index.html                    # Main HTML structure and UI layout
├── package.json                  # Dependencies and npm scripts
├── vite.config.js                # Vite configuration with CORS headers
├── styles/
│   └── main.css                 # Complete UI styling and theme
├── js/
│   ├── main.js                  # Main application controller and UI logic
│   ├── db.js                    # IndexedDB wrapper for voices and history
│   ├── audio.js                 # Audio recording and processing utilities
│   └── tts-engine-complete.js   # ONNX Runtime TTS engine with WebGPU
├── public/
│   └── voices/                  # Default voice samples
│       ├── lucy.wav             # Default female voice
│       └── stewie.wav           # Default male voice
└── README.md                    # This file
```

## Technical Details

### Architecture

The application uses a multi-model ONNX architecture:

1. **Speech Encoder** - Encodes reference audio into speaker embeddings
2. **Embed Tokens** - Converts text tokens to embeddings
3. **Language Model** - Generates speech tokens autoregressively with KV caching
4. **Conditional Decoder** - Decodes speech tokens to 24kHz audio waveform

### Technologies Used

- **ONNX Runtime Web** - For running ONNX models with WebGPU
- **Transformers.js** - For tokenization and text processing
- **Vite** - Fast development server and build tool
- **IndexedDB** - Local storage for voices and history
- **Web Audio API** - Audio recording and processing

### Browser Compatibility

**Recommended:**
- Chrome/Edge 113+ (full WebGPU support)
- Safari Technology Preview (experimental WebGPU)

**Requirements:**
- WebGPU enabled (check `chrome://flags`)
- Microphone access (for recording voices)
- At least 4GB RAM
- Modern ES modules support

## Building for Production

```bash
npm run build
```

This creates an optimized build in the `dist/` directory that you can deploy to any static hosting service.

## Deployment

Since everything runs client-side, you can deploy to any static hosting service. The app requires specific CORS headers for WebGPU support (configured in `vite.config.js`):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

**Deployment Options:**

- **Bunny CDN**: See live demo at [https://chatterbox-turbo.b-cdn.net/](https://chatterbox-turbo.b-cdn.net/)
- **Netlify**: Drag and drop the `dist/` folder (configure headers in `netlify.toml`)
- **Vercel**: Connect your git repository (configure headers in `vercel.json`)
- **GitHub Pages**: Deploy the `dist/` folder to gh-pages branch
- **AWS S3 + CloudFront**: Upload the `dist/` folder (configure CloudFront headers)

**Important**: Ensure your hosting provider supports the required CORS headers for WebGPU to work properly.

## Limitations & Known Issues

1. **Model Download Size**: Initial download is ~350MB
2. **WebGPU Requirement**: Older browsers won't work
3. **Memory Usage**: Requires significant RAM for model inference
4. **Generation Speed**: Slower than cloud-based solutions, but fully private
5. **Voice Quality**: Dependent on reference audio quality and length

## Troubleshooting

### "WebGPU not supported" error

1. Make sure you're using Chrome 113+ or Edge 113+
2. Check `chrome://flags` and enable "Unsafe WebGPU"
3. Update your graphics drivers

### Models fail to download

1. Check your internet connection
2. Try clearing browser cache
3. Check HuggingFace availability

### Audio generation fails

1. Ensure you have enough RAM available
2. Try reducing max tokens or using a shorter reference audio
3. Check browser console for detailed error messages

### Microphone not working

1. Grant microphone permissions to your browser
2. Check your system's microphone settings
3. Try a different browser

## Credits & Acknowledgements

This project is based on [Resemble AI's Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) model.

Original acknowledgements from Resemble AI:
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

## License

MIT License - See the original [Chatterbox repository](https://github.com/resemble-ai/chatterbox) for details.

## Support & Community

- **Original Model**: [Chatterbox-Turbo on HuggingFace](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
- **Demo Samples**: [Listen Here](https://resemble-ai.github.io/chatterbox_turbo_demopage/)
- **Resemble AI Discord**: [Join](https://discord.gg/rJq9cRJBJ6)
- **Resemble AI Website**: [resemble.ai](https://resemble.ai)

## Citation

```bibtex
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```

## Disclaimer

This is an unofficial browser port of Chatterbox-Turbo. Use responsibly and ethically. Do not use this tool to create misleading or harmful content, impersonate others without consent, or violate any laws or regulations.

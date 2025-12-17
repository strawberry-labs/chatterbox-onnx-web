import { defineConfig } from 'vite';

export default defineConfig({
    server: {
        port: 5173,
        open: true,
        allowedHosts: [
            'localhost',
            '.trycloudflare.com',  // Allow all Cloudflare tunnel subdomains
            '.ngrok.io'            // Also allow ngrok if needed
        ],
        headers: {
            // Required for ONNX Runtime Web with WebGPU
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp'
        }
    },
    build: {
        target: 'esnext',
        minify: 'terser',
        rollupOptions: {
            output: {
                manualChunks: {
                    'vendor': ['@xenova/transformers', 'onnxruntime-web']
                }
            }
        }
    },
    optimizeDeps: {
        include: ['@xenova/transformers', 'onnxruntime-web']
    },
    worker: {
        format: 'es'
    }
});

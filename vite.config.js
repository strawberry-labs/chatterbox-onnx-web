import { defineConfig } from 'vite';

export default defineConfig({
    server: {
        port: 5173,
        open: true,
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

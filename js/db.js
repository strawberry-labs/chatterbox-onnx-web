// IndexedDB wrapper for local storage of voices and history

const DB_NAME = 'ChatterboxDB';
const DB_VERSION = 1;
const VOICES_STORE = 'voices';
const HISTORY_STORE = 'history';

class Database {
    constructor() {
        this.db = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;

                // Create voices store
                if (!db.objectStoreNames.contains(VOICES_STORE)) {
                    const voicesStore = db.createObjectStore(VOICES_STORE, {
                        keyPath: 'id',
                        autoIncrement: true
                    });
                    voicesStore.createIndex('name', 'name', { unique: false });
                    voicesStore.createIndex('createdAt', 'createdAt', { unique: false });
                }

                // Create history store
                if (!db.objectStoreNames.contains(HISTORY_STORE)) {
                    const historyStore = db.createObjectStore(HISTORY_STORE, {
                        keyPath: 'id',
                        autoIncrement: true
                    });
                    historyStore.createIndex('timestamp', 'timestamp', { unique: false });
                }
            };
        });
    }

    // Voice operations
    async addVoice(voice) {
        const transaction = this.db.transaction([VOICES_STORE], 'readwrite');
        const store = transaction.objectStore(VOICES_STORE);

        const voiceData = {
            ...voice,
            createdAt: Date.now()
        };

        return new Promise((resolve, reject) => {
            const request = store.add(voiceData);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getVoice(id) {
        const transaction = this.db.transaction([VOICES_STORE], 'readonly');
        const store = transaction.objectStore(VOICES_STORE);

        return new Promise((resolve, reject) => {
            const request = store.get(id);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllVoices() {
        const transaction = this.db.transaction([VOICES_STORE], 'readonly');
        const store = transaction.objectStore(VOICES_STORE);

        return new Promise((resolve, reject) => {
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async updateVoice(id, updates) {
        const transaction = this.db.transaction([VOICES_STORE], 'readwrite');
        const store = transaction.objectStore(VOICES_STORE);

        return new Promise((resolve, reject) => {
            const getRequest = store.get(id);

            getRequest.onsuccess = () => {
                const voice = getRequest.result;
                const updatedVoice = { ...voice, ...updates };

                const updateRequest = store.put(updatedVoice);
                updateRequest.onsuccess = () => resolve(updateRequest.result);
                updateRequest.onerror = () => reject(updateRequest.error);
            };

            getRequest.onerror = () => reject(getRequest.error);
        });
    }

    async deleteVoice(id) {
        const transaction = this.db.transaction([VOICES_STORE], 'readwrite');
        const store = transaction.objectStore(VOICES_STORE);

        return new Promise((resolve, reject) => {
            const request = store.delete(id);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    // History operations
    async addHistory(entry) {
        const transaction = this.db.transaction([HISTORY_STORE], 'readwrite');
        const store = transaction.objectStore(HISTORY_STORE);

        const historyData = {
            ...entry,
            timestamp: Date.now()
        };

        return new Promise((resolve, reject) => {
            const request = store.add(historyData);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getHistory(id) {
        const transaction = this.db.transaction([HISTORY_STORE], 'readonly');
        const store = transaction.objectStore(HISTORY_STORE);

        return new Promise((resolve, reject) => {
            const request = store.get(id);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllHistory() {
        const transaction = this.db.transaction([HISTORY_STORE], 'readonly');
        const store = transaction.objectStore(HISTORY_STORE);
        const index = store.index('timestamp');

        return new Promise((resolve, reject) => {
            const request = index.openCursor(null, 'prev'); // Get newest first
            const results = [];

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    results.push(cursor.value);
                    cursor.continue();
                } else {
                    resolve(results);
                }
            };

            request.onerror = () => reject(request.error);
        });
    }

    async deleteHistory(id) {
        const transaction = this.db.transaction([HISTORY_STORE], 'readwrite');
        const store = transaction.objectStore(HISTORY_STORE);

        return new Promise((resolve, reject) => {
            const request = store.delete(id);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async clearHistory() {
        const transaction = this.db.transaction([HISTORY_STORE], 'readwrite');
        const store = transaction.objectStore(HISTORY_STORE);

        return new Promise((resolve, reject) => {
            const request = store.clear();
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
}

// Export singleton instance
export const db = new Database();

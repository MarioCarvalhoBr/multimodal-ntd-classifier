const { createApp } = Vue;

createApp({
    data() {
        return {
            file: null,
            imagePreview: null,
            availableModels: [
                "microsoft/resnet-50",
                "google/efficientnet-b3",
                "google/vit-base-patch16-224",
                "openai/clip-vit-base-patch16",
                "google/siglip-base-patch16-224",
            ],
            selectedModel: "microsoft/resnet-50",
            isLoading: false,
            predictions: [],
            showDiseaseModal: false,
            showAbout: false,
            showImageModal: false,
            imageZoom: 1.0,
            selectedDiseaseInfo: null,
            apiUrl: "http://127.0.0.1:8000",

            // i18n
            currentLang: localStorage.getItem('cadtn_lang') || 'en',
            supportedLangs: SUPPORTED_LANGS,
        };
    },

    methods: {
        // ── i18n ────────────────────────────────────────────────────────────
        t(key) {
            return t(this.currentLang, key);
        },
        setLang(code) {
            this.currentLang = code;
            localStorage.setItem('cadtn_lang', code);
            document.documentElement.lang = code;
        },
        getDiseaseInfo(classId) {
            return getDiseaseTranslation(this.currentLang, classId);
        },

        // ── Upload ───────────────────────────────────────────────────────────
        handleFileUpload(event) {
            const selected = event.target.files[0];
            if (!selected) return;
            this.file = selected;
            this.predictions = [];
            const reader = new FileReader();
            reader.onload = (e) => { this.imagePreview = e.target.result; };
            reader.readAsDataURL(selected);
        },

        // ── Inference ────────────────────────────────────────────────────────
        async submitAnalysis() {
            if (!this.file) return;
            this.isLoading = true;
            this.predictions = [];

            const formData = new FormData();
            formData.append("file", this.file);
            formData.append("model_name", this.selectedModel);

            try {
                const response = await fetch(`${this.apiUrl}/predict`, {
                    method: "POST",
                    body: formData,
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                this.predictions = data.predictions;
            } catch (error) {
                console.error("API error:", error);
                alert("Could not connect to the server. Make sure the FastAPI backend is running.");
            } finally {
                this.isLoading = false;
            }
        },

        // ── Helpers ──────────────────────────────────────────────────────────
        formatClassName(rawName) {
            const info = this.getDiseaseInfo(rawName);
            if (info) return info.name;
            // Fallback: strip prefix and capitalise
            return rawName
                .replace('microscopy_parasite_', '')
                .replace('clinical_', '')
                .replace('microscopy_', '')
                .replace(/\b\w/g, c => c.toUpperCase());
        },
        openDiseaseModal(classId) {
            this.selectedDiseaseInfo = this.getDiseaseInfo(classId);
            if (this.selectedDiseaseInfo) this.showDiseaseModal = true;
        },

        // ── Lightbox zoom ────────────────────────────────────────────────────
        openImageModal() {
            this.imageZoom = 1.0;
            this.showImageModal = true;
        },
        closeImageModal() {
            this.showImageModal = false;
            this.imageZoom = 1.0;
        },
        zoomIn()    { if (this.imageZoom < 4.0) this.imageZoom = Math.round((this.imageZoom + 0.5) * 10) / 10; },
        zoomOut()   { if (this.imageZoom > 1.0) this.imageZoom = Math.round((this.imageZoom - 0.5) * 10) / 10; },
        resetZoom() { this.imageZoom = 1.0; },
    },
}).mount('#app');

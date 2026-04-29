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
                "google/siglip-base-patch16-224"
            ],
            selectedModel: "microsoft/resnet-50",
            isLoading: false,
            predictions: [],
            diseaseDatabase: [],
            showDiseaseModal: false,
            showAbout: false,
            selectedDiseaseInfo: null,
            // Certifique-se de que a porta aqui seja a mesma do settings.api_port
            apiUrl: "http://127.0.0.1:8000" 
        }
    },
    mounted() {
        fetch('./data/parasite.json')
            .then(response => response.json())
            .then(data => {
                this.diseaseDatabase = data;
                console.log("Banco de dados médico carregado:", this.diseaseDatabase.length, "doenças.");
            })
            .catch(error => console.error("Erro ao carregar parasite.json:", error));
    },
    methods: {
        handleFileUpload(event) {
            const selectedFile = event.target.files[0];
            if (!selectedFile) return;
            
            this.file = selectedFile;
            this.predictions = []; 
            
            const reader = new FileReader();
            reader.onload = (e) => {
                this.imagePreview = e.target.result;
            };
            reader.readAsDataURL(selectedFile);
        },
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
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Erro HTTP: ${response.status}`);
                }

                const data = await response.json();
                this.predictions = data.predictions;
            } catch (error) {
                console.error("Erro na API:", error);
                alert("Erro ao conectar com o servidor. Verifique se o FastAPI está rodando.");
            } finally {
                this.isLoading = false;
            }
        },
        formatClassName(rawName) {
            let name = rawName.replace('microscopy_parasite_', '')
                              .replace('clinical_', '')
                              .replace('microscopy_', '');
            return name.charAt(0).toUpperCase() + name.slice(1);
        },
        getDiseaseInfo(classId) {
            return this.diseaseDatabase.find(d => d.id === classId);
        },
        openDiseaseModal(classId) {
            this.selectedDiseaseInfo = this.getDiseaseInfo(classId);
            if (this.selectedDiseaseInfo) {
                this.showDiseaseModal = true;
            }
        }
    }
}).mount('#app');
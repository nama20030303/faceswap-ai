class FaceSwapApp {
    constructor() {
        this.sourceImage = null;
        this.targetImage = null;
        this.initEventListeners();
        this.checkServerStatus();
    }

    initEventListeners() {
        // Обработчики для выбора файлов
        document.getElementById('sourceInput').addEventListener('change', (e) => {
            this.handleImageUpload(e.target.files[0], 'sourcePreview');
            this.sourceImage = e.target.files[0];
            this.checkReady();
        });

        document.getElementById('targetInput').addEventListener('change', (e) => {
            this.handleImageUpload(e.target.files[0], 'targetPreview');
            this.targetImage = e.target.files[0];
            this.checkReady();
        });

        // Обработчик для кнопки обработки
        document.getElementById('processBtn').addEventListener('click', () => {
            this.processFaceSwap();
        });
    }

    async checkServerStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.models_loaded) {
                document.getElementById('status').className = 'status ready';
                document.getElementById('status').innerHTML = '✅ AI модели загружены! Готов к работе.';
            } else {
                document.getElementById('status').className = 'status loading';
                document.getElementById('status').innerHTML = '⏳ AI модели загружаются... Подождите 2-3 минуты';
                // Повторная проверка через 10 секунд
                setTimeout(() => this.checkServerStatus(), 10000);
            }
        } catch (error) {
            document.getElementById('status').className = 'status error';
            document.getElementById('status').innerHTML = '❌ Ошибка подключения к серверу';
        }
    }

    handleImageUpload(file, previewId) {
        if (!file.type.startsWith('image/')) {
            alert('Пожалуйста, загрузите изображение (JPEG, PNG)');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById(previewId);
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    checkReady() {
        const btn = document.getElementById('processBtn');
        btn.disabled = !(this.sourceImage && this.targetImage);
    }

    async processFaceSwap() {
        const loading = document.getElementById('loading');
        const resultDiv = document.getElementById('result');
        const processBtn = document.getElementById('processBtn');
        
        // Показываем индикатор загрузки
        loading.style.display = 'block';
        resultDiv.style.display = 'none';
        processBtn.disabled = true;

        const formData = new FormData();
        formData.append('source', this.sourceImage);
        formData.append('target', this.targetImage);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Ошибка обработки');
            }

            if (data.success) {
                this.showResult(data.image);
            }
            
        } catch (error) {
            console.error('Error:', error);
            alert('Ошибка: ' + error.message);
        } finally {
            loading.style.display = 'none';
            processBtn.disabled = false;
        }
    }

    showResult(imageData) {
        const resultImage = document.getElementById('resultImage');
        const resultDiv = document.getElementById('result');
        const downloadLink = document.getElementById('downloadLink');
        
        resultImage.src = imageData;
        downloadLink.href = imageData;
        downloadLink.download = 'faceswap_result.jpg';
        resultDiv.style.display = 'block';
        
        // Прокручиваем к результату
        resultDiv.scrollIntoView({ behavior: 'smooth' });
    }
}

// Запуск приложения после загрузки страницы
document.addEventListener('DOMContentLoaded', () => {
    new FaceSwapApp();
});

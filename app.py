from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import base64
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем Flask приложение с исправлением для Codespaces
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Исправление для GitHub Codespaces
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.config['SERVER_NAME'] = None

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Глобальные переменные для моделей
face_app = None
swapper = None

def initialize_models():
    global face_app, swapper
    logger.info("🔄 Загрузка AI моделей...")
    try:
        # Инициализация детектора лиц
        face_app = FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU режим
        logger.info("✅ Детектор лиц загружен")
        
        # Загрузка модели для замены лиц
        swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
        logger.info("✅ Модель замены лиц загружена")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки моделей: {e}")
        raise

# Инициализация моделей при старте
try:
    initialize_models()
    models_loaded = True
except Exception as e:
    logger.error(f"❌ Не удалось загрузить модели: {e}")
    models_loaded = False

class FaceSwapper:
    def __init__(self):
        self.face_app = face_app
        self.swapper = swapper

    def extract_face(self, image):
        """Извлечение лица из изображения"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_app.get(rgb_image)
        
        if len(faces) == 0:
            raise Exception("❌ Лицо не найдено на изображении")
        return faces[0]

    def swap_faces(self, source_img, target_img):
        """Замена лиц между изображениями"""
        if self.swapper is None:
            raise Exception("Модель замены лиц не загружена")

        # Извлекаем исходное лицо
        source_face = self.extract_face(source_img)
        
        # Конвертируем целевое изображение
        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        # Находим лица в целевом изображении
        target_faces = self.face_app.get(target_rgb)
        if len(target_faces) == 0:
            raise Exception("❌ Лицо не найдено в целевом изображении")

        # Заменяем первое найденное лицо
        result_image = self.swapper.get(target_rgb, target_faces[0], source_face, paste_back=True)
        return cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

# Создаем экземпляр сваппера если модели загружены
face_swapper = FaceSwapper() if models_loaded else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Проверка статуса сервера"""
    return jsonify({
        'status': 'healthy' if models_loaded else 'loading',
        'models_loaded': models_loaded,
        'message': 'Готов к работе!' if models_loaded else 'Модели загружаются...'
    })

@app.route('/process', methods=['POST'])
def process():
    """Обработка замены лиц"""
    try:
        if face_swapper is None:
            return jsonify({'error': 'Модели AI еще загружаются. Подождите 2-3 минуты...'}), 503

        if 'source' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Необходимо загрузить оба изображения'}), 400

        source_file = request.files['source']
        target_file = request.files['target']

        # Проверка файлов
        if source_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'Файлы не выбраны'}), 400

        # Чтение изображений
        source_bytes = source_file.read()
        target_bytes = target_file.read()

        # Конвертируем в numpy массивы
        nparr_source = np.frombuffer(source_bytes, np.uint8)
        nparr_target = np.frombuffer(target_bytes, np.uint8)

        source_img = cv2.imdecode(nparr_source, cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(nparr_target, cv2.IMREAD_COLOR)

        if source_img is None or target_img is None:
            return jsonify({'error': 'Неверный формат изображений'}), 400

        # Замена лиц
        result_img = face_swapper.swap_faces(source_img, target_img)

        # Кодирование результата в base64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{result_base64}'
        })

    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Запуск Flask приложения...")
    # Исправление для Codespaces
    app.run(host='0.0.0.0', port=5000, debug=False, ssl_context='adhoc')

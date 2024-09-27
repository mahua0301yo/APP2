import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import base64
import io
from flask import Flask, request, jsonify
import streamlit.components.v1 as components

# 初始化 Flask 應用
app = Flask(__name__)

# 載入 YOLO 模型
model = YOLO('yolov8n-pose.pt')

# 定義 Streamlit 頁面標題
st.title("YOLO 即時鏡頭物件偵測")

# 嵌入前端 HTML + JavaScript 獲取攝像頭影像流
html_code = """
<video id="video" width="100%" autoplay></video>
<canvas id="canvas" style="display:none;"></canvas>
<img id="result" style="max-width: 100%; margin-top: 20px;"/>

<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const resultImage = document.getElementById('result');

    // 訪問使用者的攝像頭
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;

            // 每隔100毫秒捕捉影像幀並發送到後端
            setInterval(() => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                // 將影像轉換為 base64
                const imageData = canvas.toDataURL('image/jpeg');

                // 發送影像幀到後端
                fetch('/process_frame', {
                    method: 'POST',
                    body: JSON.stringify({ image: imageData }),
                    headers: { 'Content-Type': 'application/json' }
                })
                .then(response => response.json())
                .then(data => {
                    // 更新顯示偵測後的影像
                    resultImage.src = 'data:image/jpeg;base64,' + data.result_image;
                })
                .catch(error => console.error('錯誤：', error));
            }, 100); // 每100毫秒傳送一次影像
        })
        .catch(err => {
            console.error('Error accessing the camera: ', err);
        });
</script>
"""

# 使用 Streamlit 嵌入 HTML
components.html(html_code, height=600)

# 後端 Flask 處理影像的路由
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data['image']

    # 解析 Base64 影像
    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # 將影像轉換為 OpenCV 格式 (RGB -> BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 使用 YOLO 模型進行偵測
    results = model(image)

    # 在影像上繪製偵測結果
    result_img = np.squeeze(results.render())

    # 將處理後的影像轉換為 Base64 格式返回
    _, buffer = cv2.imencode('.jpg', result_img)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'result_image': result_image})

# 讓 Flask 應用運行
if __name__ == '__main__':
    app.run()

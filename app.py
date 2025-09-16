from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载YOLO模型
model = YOLO(r"D:\python\pythonProject\BeeDetection\runs\train2\weights\best.pt")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': '没有上传图片'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'})

    image_name = file.filename

    try:
        # 读取图片
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'message': '无法读取图片'})

        # 记录原始图片尺寸
        height, width = image.shape[:2]
        image_size = f"{width}x{height}"

        # 使用模型进行检测
        results = model(image)

        # 处理检测结果
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box[:6]
                    # 绘制边界框和标签
                    label = f"{int(cls_id)} {conf:.2f}"
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 保存检测信息
                    detections.append({
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'confidence': float(conf),
                        'class_id': int(cls_id)
                    })

        # 添加检测数量标签
        object_count = len(detections)
        count_label = f"Bee count: {object_count}"
        cv2.putText(image, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        _, encoded_image = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(encoded_image).decode('utf-8')

        # 返回结果
        return jsonify({
            'success': True,
            'object_count': object_count,
            'detections': detections,
            'image': image_base64,
            'image_size': image_size,
            'image_name': image_name
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'处理过程中发生错误: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)


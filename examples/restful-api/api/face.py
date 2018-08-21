import io
import base64
import requests
import face_recognition

from flask import Flask, request, jsonify
from api import API_FACE_PATH


app = Flask(__name__)


@app.route(API_FACE_PATH+'detect', methods=['POST'])
def detect():
    """
    curl --data-binary "@images/zxc1.jpeg" -H "Content-Type: application/octet-stream" -X POST http://localhost:5000/face/api/v1.0/detect
    :return:
    """
    image_data = None
    if request.headers['Content-Type'] == 'application/octet-stream':
        image_data = request.data
    elif request.json or 'url' in request.json:
        url = request.json['url']
        response = requests.get(url, stream=True)
        image_data = response.content

    if not image_data:
        return jsonify({'message': 'Not image data.'}), 400

    face_info = detect_faces(image_data)
    if not face_info:
        return jsonify({'message': 'Not found face.'}), 400

    return jsonify(face_info), 201


def detect_faces(image_data):
    img_data = face_recognition.load_image_file(io.BytesIO(image_data))
    face_locations = face_recognition.face_locations(img_data)
    face_landmarks_list = face_recognition.face_landmarks(img_data, face_locations)

    items = []
    for location, landmarks in zip(face_locations, face_landmarks_list):
        item = {'faceRectangle': location, 'faceLandmarks': landmarks}
        items.append(item)

    return items


@app.route('/')
def index():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)

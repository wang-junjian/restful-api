import io
import os
import base64
import requests
import face_recognition
import numpy as np

from flask import Flask, request, jsonify
from api import API_FACE_PATH


app = Flask(__name__)

FACE_DATABASE = 'faces.db.txt'


@app.route(API_FACE_PATH + 'faces', methods=['POST'])
def create_face():
    """
    curl -i -H "Content-Type: application/json" -X POST -d '{"faceId":"ZhouXingChi", "faceFeature":"AAAAwC7Gub8AAACgUXW3PwAAAEBo1qM/AAAAYIENsb8AAAAg/I2rvwAAAGD5ap+/AAAAgCnXtL8AAAAAIxzCvwAAAEDuFcE/AAAAoGpquL8AAAAAq3XMPwAAAEA6eqy/AAAAQKIC0b8AAABA9kLAvwAAAGBS6q2/AAAAoMEGxT8AAABA+H3FvwAAAMAjksC/AAAAAKPUsr8AAADABAKnvwAAAIBj3rE/AAAAAHEBoj8AAACAKcCCPwAAAACMfoI/AAAAIOQZtr8AAACANhLSvwAAAMCoe7q/AAAAgMEXq78AAABA8vtxvwAAAMAhBam/AAAAwKhQrb8AAABAM7qSPwAAAKCF5sS/AAAAwAcGsr8AAACgsqiuPwAAAEDJwrg/AAAAIHhIpL8AAADAFtKhvwAAAEAsm8U/AAAAYCpoob8AAAAAIGfJvwAAAEDhJ6Y/AAAAgKrJvD8AAADgSg/NPwAAAMBqwco/AAAAAAGJsz8AAABgAgehvwAAAKAxBci/AAAA4I2hwj8AAAAAKrC2vwAAAIDUVLU/AAAAwFfrxj8AAAAAgXTAPwAAAEB02LY/AAAAgAT/uT8AAABgW47AvwAAAGBjOaQ/AAAAQP12wD8AAADAJJivvwAAAIB5vHg/AAAA4GabuT8AAADgmyuvvwAAAMC6jpA/AAAA4KHls78AAACgWjnKPwAAAABeG2C/AAAAgNQyvb8AAADgsyjHvwAAAOBVUMM/AAAA4MVyuL8AAADA+a3BvwAAAADXGLo/AAAAYOLExL8AAACgrhTBvwAAAKCvg9S/AAAAIPBVgT8AAAAAeD7ZPwAAAKAvlbg/AAAAwFzlzb8AAAAgZkeZPwAAAEDY75W/AAAAIEBPor8AAAAANrPFPwAAAGDcL8A/AAAAgCjZp78AAACAPHGZvwAAAIADCL6/AAAAAPN7pr8AAACAMujPPwAAAAASgJy/AAAA4LWgs78AAAAAedbEPwAAAAD9tZ0/AAAAIKftsz8AAABgLV2pPwAAAICTjoM/AAAAgEsNtb8AAADAqqCKPwAAAEBEksi/AAAAwExws78AAABAjXGsPwAAAADRdKK/AAAAoB1jpb8AAABAVe2/PwAAACD5Mba/AAAAwDjmuz8AAAAAYbOwPwAAAICU36k/AAAAgAMAlL8AAADAzeiaPwAAAADYSL6/AAAAAPEvm78AAAAAPve/PwAAAEAT7Mu/AAAAIKoWzz8AAABg3gO9PwAAAIAgBbo/AAAAgOnXrT8AAABA64zCPwAAAKB+WLo/AAAAQLtwnr8AAAAAuYiUvwAAAKCb0cS/AAAAAPUwtb8AAABgI9O5PwAAAACr4Hk/AAAA4PVKxj8AAADgXpSdPw=="}' http://localhost:5000/face/api/v1.0/faces
    curl -i -H "Content-Type: application/json" -X POST -d '{"faceId":"ZhouRunFa", "faceFeature":"AAAA4Kvqwr8AAACA8I61PwAAAAA0Yna/AAAAAAcnpr8AAABgjb6xvwAAAIAyl7i/AAAAoMuHtb8AAADAen61vwAAAMDt7bA/AAAAQEAes78AAACAwSXBPwAAAECXOLS/AAAAwKBfyL8AAACAxFG2vwAAACCB76y/AAAAQCeWvz8AAAAgfnrQvwAAAMD3Trm/AAAAwDO3mr8AAABAIzqzvwAAACDFu7s/AAAAALFVdz8AAADgXYewvwAAAADkM5y/AAAAQPeuub8AAABgrE7VvwAAAEAqsL+/AAAAILMsqL8AAADgx4moPwAAAEAGk6K/AAAAAOxBoj8AAACAhUaovwAAAGA2us2/AAAAwFWSvr8AAACA0DKnPwAAAAALX6M/AAAAAFEFaD8AAAAgt4KmvwAAAKAcwLo/AAAAgGQVnD8AAACAADXDvwAAAEAhl5G/AAAAgF7IuT8AAAAAAtPQPwAAAIBdpsM/AAAAgO7jvD8AAADA1BiivwAAAMAC+bW/AAAAYD/7vz8AAAAgOOHFvwAAAEAKn6A/AAAAQGofxD8AAAAAPLtvvwAAAADnyr0/AAAAAHaomT8AAAAAr3apvwAAAMDmV7I/AAAAgGJVwz8AAABARBDHvwAAAEBjD54/AAAAgHbYvj8AAADAKhSwvwAAACAtNK2/AAAAwNB8o78AAADgo1nDPwAAAICFtKo/AAAAwOLUsr8AAABgYYXLvwAAAOCpQMI/AAAA4IbExL8AAACggPS6vwAAAEB/xnc/AAAAwNprnb8AAACgAgLBvwAAAKDlNdO/AAAAwLMgkT8AAACg3H/WPwAAAABoXsI/AAAAADwzvL8AAABgMEqiPwAAACD+orG/AAAAQCBXpr8AAACgOkTIPwAAAABEmbA/AAAAAFR3cD8AAADAGu2cPwAAAEBnY66/AAAAALmMfz8AAABACsPIPwAAAIAteJI/AAAA4Fnvob8AAAAgo7PHPwAAAIBzz4O/AAAAIIaRlT8AAAAAFe2qPwAAAADRa7A/AAAAoMoswb8AAABAriiVPwAAAAD08L+/AAAAAOlUs78AAACARErCPwAAAKCWZ6W/AAAAAOSmYz8AAADAbLDBPwAAAMDHj7+/AAAAIJSnxT8AAACgJumXPwAAAIAm36U/AAAAgEWFdr8AAABgztefvwAAAIDnD7S/AAAAALdkgL8AAAAAOKa2PwAAACASJsm/AAAAAJxozT8AAACAYGfIPwAAAADb4Zo/AAAAAD9yuz8AAABAfpi7PwAAAKCNJYU/AAAAoAVMkr8AAAAgzZywvwAAACDrjMm/AAAAgEtyvr8AAAAApSlhPwAAAGARVpg/AAAAwKRErj8AAABAfD6Svw=="}' http://localhost:5000/face/api/v1.0/faces
    :return:
    """

    if not request.json:
        return jsonify({'message': 'Not argument.'}), 400

    if 'faceId' not in request.json:
        return jsonify({'message': 'Not argument faceId.'}), 400

    if 'faceFeature' not in request.json:
        return jsonify({'message': 'Not argument faceFeature.'}), 400

    face_id = request.json['faceId']
    face_feature = request.json['faceFeature']

    id_features = read_faces()

    id_features[face_id] = face_feature
    write_faces(id_features)

    return jsonify({'faceId': face_id, 'faceFeature': face_feature}), 201


def read_faces():
    id_features = {}

    if not os.path.exists(FACE_DATABASE):
        return id_features

    with open(FACE_DATABASE, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            items = line.split()
            if items and len(items) == 2:
                id = items[0]
                feature = items[1]
                id_features[id] = feature

    return id_features


def write_faces(id_features):
    with open(FACE_DATABASE, 'w') as f:
        for id, feature in id_features.items():
            f.write('{} {}\n'.format(id, feature))


@app.route(API_FACE_PATH + 'faces/<string:face_id>', methods=['DELETE'])
def delete_face(face_id):
    """
    curl -X "DELETE" http://localhost:5000/face/api/v1.0/faces/ZhouXingChi
    :param face_id:
    :return:
    """

    if not face_id:
        return jsonify({'message': 'Not argument faceId.'}), 400

    id_features = read_faces()
    if face_id in id_features:
        id_features.pop(face_id, None)
        write_faces(id_features)
    else:
        return jsonify({'message': 'faceId: {} not exist'.format(face_id)}), 204

    return jsonify({'result': True})


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


@app.route(API_FACE_PATH + 'feature', methods=['POST'])
def get_feature():
    """
    curl --data-binary "@images/zxc1.jpeg" -H "Content-Type: application/octet-stream" -X POST http://localhost:5000/face/api/v1.0/feature
    curl --data-binary "@images/zrf1.jpeg" -H "Content-Type: application/octet-stream" -X POST http://localhost:5000/face/api/v1.0/feature
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

    face_feature = get_face_feature(image_data)
    if face_feature is None or not face_feature.all():
        return jsonify({'message': 'Not found face.'}), 400

    face_feature_base64 = base64.b64encode(face_feature).decode()
    return jsonify({'feature': face_feature_base64}), 201


def get_face_feature(image_data):
    img_data = face_recognition.load_image_file(io.BytesIO(image_data))
    face_encodings = face_recognition.face_encodings(img_data)
    if face_encodings:
        return face_encodings[0]

    return None


@app.route(API_FACE_PATH + 'identify', methods=['POST'])
def get_face_id():
    """
    curl --data-binary "@images/zxc2.jpeg" -H "Content-Type: application/octet-stream" -X POST http://localhost:5000/face/api/v1.0/identify
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

    face_feature = get_face_feature(image_data)
    if face_feature is None or not face_feature.all():
        return jsonify({'message': 'Face not recognition.'}), 204

    id_features = read_faces()
    for id in id_features:
        id_features[id] = np.fromstring(base64.b64decode(id_features[id]))

    face_id = identify_face_id(face_feature, id_features)

    return jsonify({'faceId': face_id}), 201


def identify_face_id(face_encoding, id_features):
    face_ids = []
    face_encodings = []
    for id, encoding in id_features.items():
        face_ids.append(id)
        face_encodings.append(encoding)

    results = face_recognition.compare_faces(face_encodings, face_encoding, tolerance=0.55)

    for index, result in enumerate(results):
        if result:
            return face_ids[index]

    return None


@app.route('/')
def index():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)

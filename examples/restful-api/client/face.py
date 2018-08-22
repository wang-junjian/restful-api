import http.client
import requests
import os

from client import API_URL


MAX_RETRIES = 30


# def detect_by_url():
#     headers = {
#         # Request headers
#         'Content-Type': 'application/json',
#     }
#
#     body = '{"url": "http://n.sinaimg.cn/sinacn/w699h417/20180226/110c-fyrwsqi3120349.jpg"}'
#
#     try:
#         conn = http.client.HTTPConnection(API_URL)
#         conn.request("POST",
#                      "/face/api/v1.0/detect",
#                      body,
#                      headers)
#
#         response = conn.getresponse()
#         data = response.read()
#         face_info = data.decode()
#
#         print(face_info)
#
#         conn.close()
#     except Exception as e:
#         print("[Errno {0}] {1}".format(e.errno, e.strerror))

def detect_by_url(url):
    response = requests.post(url=API_URL + "/face/api/v1.0/detect",
                             data='{"url": "%s"}' % url,
                             headers={'Content-Type': 'application/json'})
    print('detect_by_url ', response.json())
    return response.json()


def detect_by_file(filename):
    data = open(filename, 'rb').read()
    response = requests.post(url=API_URL + "/face/api/v1.0/detect",
                             data=data,
                             headers={'Content-Type': 'application/octet-stream'})
    print('detect_by_file ', response.json())
    return response.json()


def feature_by_file(filename):
    # data = open(filename, 'rb').read()
    # response = requests.post(url=API_URL + "/face/api/v1.0/feature",
    #                          data=data,
    #                          headers={'Content-Type': 'application/octet-stream'})
    # print('feature_by_file ', response.json())
    # return response.json()
    print('-----', filename)
    data = open(filename, 'rb').read()

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    response = session.post(url=API_URL + "/face/api/v1.0/feature",
                            data=data,
                            headers={'Content-Type': 'application/octet-stream'})
    print('feature_by_file ', response.json())
    return response.json()


def identify_by_file(filename):
    """
    使用session 修复错误：requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response',))
    :param filename:
    :return:
    """
    # data = open(filename, 'rb').read()
    # response = requests.post(url=API_URL + "/face/api/v1.0/identify",
    #                          data=data,
    #                          headers={'Content-Type': 'application/octet-stream'})
    # print('identify_by_file ', response.json())
    # return response.json()
    data = open(filename, 'rb').read()

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    print('$ ', filename)
    response = session.post(url=API_URL + "/face/api/v1.0/identify",
                            data=data,
                            headers={'Content-Type': 'application/octet-stream'})
    print('identify_by_file ', response.json())
    return response.json()


def create_face(face_id, face_feature):
    # response = requests.post(url=API_URL + "/face/api/v1.0/faces",
    #                          data='{"faceId":"%s", "faceFeature":"%s"}' % (face_id, face_feature),
    #                          headers={'Content-Type': 'application/json'})
    # print('create_face ', response.json())
    # return response.json()
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    response = session.post(url=API_URL + "/face/api/v1.0/faces",
                            data='{"faceId":"%s", "faceFeature":"%s"}' % (face_id, face_feature),
                            headers={'Content-Type': 'application/json'})
    print('identify_by_file ', response.json())
    return response.json()


def init():
    face_dataset = get_face_dataset()
    create_face_db(face_dataset)
    identify_face(face_dataset)


def get_face_dataset():
    faces_dir = '../images/faces'

    faces_path = []

    for face_dir in os.listdir(faces_dir):
        path = os.path.join(faces_dir, face_dir)
        if os.path.isdir(path):
            faces_path.append(path)

    face_dataset = {}
    for face_path in faces_path:
        files = []
        for file in os.listdir(face_path):
            if file == '.DS_Store':
                continue
            files.append(os.path.join(face_path, file))

        name = os.path.basename(face_path)
        face_dataset[name] = files

    return face_dataset


def create_face_db(face_dataset):
    for name, face_files in face_dataset.items():
        filename = face_files[0]
        feature = feature_by_file(filename)
        if feature and 'feature' in feature:
            create_face(name, feature['feature'])


def identify_face(face_dataset):
    error_num = 0
    count = 0
    for name, face_files in face_dataset.items():
        # 移除第一张图片
        face_files = face_files[1:]
        for filename in face_files:
            count += 1
            identify_result = identify_by_file(filename)
            face_id = identify_result['faceId']
            if name != face_id:
                error_num += 1
                print('{}/{}应该是{}, 识别为{}, file{}'.format(error_num, count, name, face_id, filename))


if __name__ == '__main__':
    # detect_by_url('http://n.sinaimg.cn/sinacn/w699h417/20180226/110c-fyrwsqi3120349.jpg')
    # detect_by_file('../images/zxc1.jpeg')
    # feature_by_file('../images/zxc1.jpeg')
    # identify_by_file('../images/zxc2.jpeg')
    # feature = feature_by_file('../images/zxc1.jpeg')
    # if feature:
    #     create_face('zxc', feature['feature'])

    init()

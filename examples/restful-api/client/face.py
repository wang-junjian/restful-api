import http.client
import requests

from client import API_URL


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
    data = open(filename, 'rb').read()
    response = requests.post(url=API_URL + "/face/api/v1.0/feature",
                             data=data,
                             headers={'Content-Type': 'application/octet-stream'})
    print('feature_by_file ', response.json())
    return response.json()


def create_face(face_id, face_feature):
    response = requests.post(url=API_URL + "/face/api/v1.0/faces",
                             data='{"faceId":"%s", "faceFeature":"%s"}' % (face_id, face_feature),
                             headers={'Content-Type': 'application/json'})
    print('create_face ', response.json())
    return response.json()


if __name__ == '__main__':
    detect_by_url('http://n.sinaimg.cn/sinacn/w699h417/20180226/110c-fyrwsqi3120349.jpg')
    detect_by_file('../images/zxc1.jpeg')
    feature_by_file('../images/zxc1.jpeg')

    feature = feature_by_file('../images/zxc1.jpeg')
    if feature:
        create_face('zxc', feature['feature'])

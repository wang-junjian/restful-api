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

def detect_by_url():
    response = requests.post(url=API_URL + "/face/api/v1.0/detect",
                             data='{"url": "http://n.sinaimg.cn/sinacn/w699h417/20180226/110c-fyrwsqi3120349.jpg"}',
                             headers={'Content-Type': 'application/json'})
    print(response.json())


def detect_by_image_data():
    data = open('../images/zxc1.jpeg', 'rb').read()
    response = requests.post(url=API_URL + "/face/api/v1.0/detect",
                             data=data,
                             headers={'Content-Type': 'application/octet-stream'})
    print(response.json())


detect_by_url()
detect_by_image_data()

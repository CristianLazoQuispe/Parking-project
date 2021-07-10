from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:8080'
test_url = addr + '/api/test'


response = requests.post(test_url)
# decode response
print(response)
print(response.json())
# expected output: {u'message': u'image received. size=124x124'}
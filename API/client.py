from __future__ import print_function
import requests

addr = 'http://localhost:8080'
test_url = addr + '/api/test'


response = requests.post(test_url)

print(response)
print(response.json())

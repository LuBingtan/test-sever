import httplib   
import json  
import cv2 as cv
import base64
import numpy as np
image = cv.imread("test.jpg")
image_bytes = base64.b64encode(image)
test_data = {
    "name":"test.jpg",
    "height":image.shape[0],
    "width":image.shape[1],
    "channel":image.shape[2],
    "bytes":image_bytes,
}  
requrl = ""  
headerdata = {"Content-type": "application/json"}  
  
conn = httplib.HTTPConnection("172.16.30.2",7777)  
  
conn.request('POST',requrl,json.dumps(test_data),headerdata)   
  
response = conn.getresponse()  
  
res= response.read()  
rst = json.loads(res)
print len(rst)
for t in rst:
    print t
    if 'ocr' in t:
        print t['ocr']

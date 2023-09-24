import json
import cv2

def lambda_handler(event, context):
    request = json.loads(event['Body'])
    image_url = request['image_url']
    image = cv2.imread(image_url, 1)
    resized = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
    # do something with resized image
    return "Success"
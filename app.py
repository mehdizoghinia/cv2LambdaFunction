import boto3
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import json

s3_client = boto3.client('s3')
sagemaker_runtime_client = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    # Retrieve bucket name and file key from the S3 event
    print('event-', event)
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    
    # Fetch the image from S3
    s3_response_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    image_object = s3_response_object['Body'].read()

    # Load image with PIL to get dimensions
    image = Image.open(BytesIO(image_object))
    width, height = image.size
    print(f"Image dimensions: Width={width} x Height={height}")

    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Preprocess the image
    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    # Convert the float tensor to uint8
    resized_uint8 = tf.cast(resized, tf.uint8)
    # Invoke the SageMaker endpoint
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName='facedetection',
        ContentType='application/x-image',
        Body=resized_uint8.numpy()
    )

    yhat = np.frombuffer(response['Body'].read(), np.float32).reshape(1, -1)  # Adjust as necessary

    # Get bounding box coordinates
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(image_np, 
                      tuple(np.multiply(sample_coords[:2], [width, height]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [width, height]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(image_np, 
                      tuple(np.add(np.multiply(sample_coords[:2], [width, height]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [width, height]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(image_np, 'face', tuple(np.add(np.multiply(sample_coords[:2], [width, height]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    # Save image with bounding box back to S3
    _, buffer = cv2.imencode('.jpg', image_np)
    file_name = file_key.replace("raw-images/", "")
    s3_client.put_object(Bucket=bucket_name, Key=f'output-images/{file_name}', Body=buffer.tobytes())

    # Delete the image from the original folder
    s3_client.delete_object(Bucket=bucket_name, Key=file_key)

    return {
        'statusCode': 200,
        'body': json.dumps('Image processed successfully!')
    }

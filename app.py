import json
import boto3
from PIL import Image
from io import BytesIO
import cv2
s3_client = boto3.client('s3')

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

    # Process the image (your custom code can go here)
    # ...

    # Copy the image to another folder (e.g., 'output-images/') within the same bucket
    file_name = file_key.replace("raw-images/", "")
    print('file_name  ', file_name)
    copy_source = {'Bucket': bucket_name, 'Key': file_key}
    s3_client.copy_object(Bucket=bucket_name, CopySource=copy_source, Key=f'output-images/{file_name}')

    # Delete the image from the original folder
    s3_client.delete_object(Bucket=bucket_name, Key=file_key)

    return {
        'statusCode': 200,
        'body': json.dumps('Image processed successfully!')
    }

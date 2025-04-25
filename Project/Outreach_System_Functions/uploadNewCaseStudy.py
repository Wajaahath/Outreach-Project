############ IMPORTS ######################
from docx import Document
from botocore.exceptions import BotoCoreError, ClientError
import boto3

###### CONFIGURE AWS & S3 INFO ####################################
s3_bucket = "project-outreach-system"
s3_prefix = "Project Case Study Outreach System/Data/Case Studies"

###### MODULE FUNCTIONS ###########################################
def uploadCaseStudy(content, filename):
    """
    Creates a .docx file from the given content and uploads it to an S3 bucket.

    The function creates a Word document using the `python-docx` library, writes each line of content
    as a new paragraph, saves it locally with the specified filename, and uploads it to the given S3 bucket
    and prefix using the `boto3` client.

    :param content: The text content to be written to the .docx file. Each line will become a new paragraph.
    :type content: str
    :param filename: The base filename (without extension) to use for the local file and the S3 object.
    :type filename: str

    :raises BotoCoreError: If a low-level error occurs in the boto3 library while communicating with AWS.
    :raises ClientError: If a client-level error occurs, such as authentication or permission issues.

    :return: None
    """
    # Step 1: Create the .docx file
    object_key = f'{s3_prefix}/{filename}.docx'
    doc = Document()

    for line in content.strip().split("\n"):
        doc.add_paragraph(line)

    doc.save(filename)

    # Step 2: Upload to S3
    try:
        s3 = boto3.client('s3')

        with open(filename, 'rb') as data:
            s3.upload_fileobj(data, s3_bucket, object_key)

        print(f"Uploaded '{filename}' to s3://{s3_bucket}/{object_key}")

    except (BotoCoreError, ClientError) as e:
        print(f"Failed to upload to s3:\n {e}")

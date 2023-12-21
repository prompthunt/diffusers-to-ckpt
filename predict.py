import subprocess
import os
import boto3
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def predict(
        self,
        model_url: str = Input(description="URL of the Diffusers model zip file"),
        s3_bucket: str = Input(
            default=None, description="S3 bucket to store the converted model"
        ),
        s3_key: str = Input(
            default=None, description="S3 key to store the converted model"
        ),
        aws_access_key_id: str = Input(default=None, description="AWS access key ID"),
        aws_secret_access_key: str = Input(
            default=None, description="AWS secret access key"
        ),
        aws_region: str = Input(default=None, description="AWS region"),
    ) -> str:
        """Run a single prediction on the model"""

        # Define the local file paths
        zip_file_path = "/tmp/model.zip"
        extracted_folder_path = "/tmp/extracted_model"

        # Download the model using wget
        subprocess.run(["wget", model_url, "-O", zip_file_path], check=True)

        # Create a directory to extract the model
        os.makedirs(extracted_folder_path, exist_ok=True)

        # Unzip the model
        subprocess.run(
            ["unzip", zip_file_path, "-d", extracted_folder_path], check=True
        )

        # Define paths for the converted model
        converted_model_path = "/tmp/converted_model.ckpt"

        # Run the conversion script
        subprocess.run(
            [
                "python",
                "convert_diffusers_to_original_stable_diffusion.py",
                "--model_path",
                extracted_folder_path,
                "--checkpoint_path",
                converted_model_path,
                "--half",
            ],
            check=True,
        )

        # Check if S3 parameters are provided
        if (
            s3_bucket
            and s3_key
            and aws_access_key_id
            and aws_secret_access_key
            and aws_region
        ):
            # Create an S3 client with provided credentials
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region,
            )

            # Upload the file to the specified bucket
            s3_client.upload_file(converted_model_path, s3_bucket, s3_key)

            # Construct the URL of the uploaded file
            s3_url = f"https://{s3_bucket}.s3.{aws_region}.amazonaws.com/{s3_key}"

            return s3_url

        else:
            # If S3 parameters are not provided, return local path
            return converted_model_path

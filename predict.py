# flake8: noqa E501
import subprocess
import os
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def predict(
        self, model_url: str = Input(description="URL of the Diffusers model zip file")
    ) -> Path:
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

        # Assume the conversion script is in the same directory and callable
        # Define paths for the converted model and model to convert
        model_to_convert_path = os.path.join(extracted_folder_path, "model.ckpt")
        converted_model_path = "/tmp/converted_model.ckpt"

        # Run the conversion script
        subprocess.run(
            [
                "python",
                "convert_diffusers_to_original_stable_diffusion.py",
                "--model_path",
                model_to_convert_path,
                "--checkpoint_path",
                converted_model_path,
                "--half",
            ],
            check=True,
        )

        # Return the path to the converted model
        return Path(converted_model_path)

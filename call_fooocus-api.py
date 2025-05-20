"""Calling Fooocs-API

Todo:
    * cmd interface
    * command line options
      * list-models
      * list-styles
      * image generation with specified models and styles
        * by names
        * by regex filer
"""
import env
import requests

models = requests.get(url=f"http://{env.host}:8888/v1/engines/all-models")
print(models.json())

styles = requests.get(url=f"http://{env.host}:8888/v1/engines/styles")
print(styles.json())

# Define the API endpoint URL
url = f"http://{env.host}:8888/v1/generation/text-to-image"

# Define the request JSON data
payload = {
    "prompt": "1girl sitting on the ground",
    'base_model_name': 'sdxl\\anime\\animaPencilXL_v310.safetensors',
    'style_selections': ['Fooocus V2', 'Fooocus Semi Realistic', 'Fooocus Masterpiece']
}

# Send the POST request with JSON data
response = requests.post(url, json=payload)

# Check for errors
if response.status_code != 200:
    raise Exception(f"API request failed with status code {response.status_code}")

# Parse the JSON response
response_json = response.json()

# Extract the image URL
image_url = response_json[0]["url"]
print(response_json)

# Replace localhost with cinematic in the URL
image_url = image_url.replace("localhost", env.host)

# Extract the file name from the URL
file_name = image_url.split("/")[-1]

# Save the image to a file with the extracted file name
image_data = requests.get(image_url).content
with open(file_name, "wb") as f:
    f.write(image_data)

# Print a success message
print(f"Image generated successfully as {file_name}.")

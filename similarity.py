import sys
import torch
import safetensors.torch
from diffusers import AutoencoderKL
from transformers import AutoProcessor, CLIPModel
from PIL import Image
import numpy as np

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Function to preprocess and encode images
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))  # Resize the image as VAE expects a specific size
    image_tensor = torch.tensor(np.array(image) / 127.5 - 1.0).permute(2, 0, 1).float().to(device)
    return image_tensor.unsqueeze(0)

# Define command line arguments for input images
if __name__ == '__main__':
    image1 = sys.argv[2]
    image2 = sys.argv[3]
    # Load the VAE model from .safetensors file
    vae_path = sys.argv[1]
    vae = AutoencoderKL.from_single_file(vae_path, safetensors=True).to(device)

    # Preprocess the images
    img1_tensor = preprocess_image(image1)
    img2_tensor = preprocess_image(image2)

    # Encode the images to latent space using VAE
    with torch.no_grad():
        z1 = vae.encode(img1_tensor).latent_dist.sample()
        z2 = vae.encode(img2_tensor).latent_dist.sample()

    print(z1.shape, z2.shape, z1.min(), z1.max(), z2.min(), z2.max())
    # Calculate the Euclidean distance between the two latent representations
    distance = torch.norm(z1 - z2)

    # Calculate the similarity score between 0 and 1 based on the distance
    similarity_score = (1 - distance) / 2

    # Calculate cosine similarity between the two latent representations
    cos_sim = torch.nn.functional.cosine_similarity(z1.view(-1), z2.view(-1),dim=0)

    # Convert cosine similarity to a score between 0 and 1
    similarity_score_cos = (cos_sim + 1) / 2

    # Write the similarity score to a text file
    print('Euclidean distance in latent space between Image 1 and Image 2: {0:.3f}\n'.format(distance.item()))
    print('Cosine similarity between Image 1 and Image 2: {0:.3f}\n'.format(similarity_score_cos.item()))

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs1 = processor(images=Image.open(image1), return_tensors="pt")
    image_features1 = model.get_image_features(**inputs1)

    inputs2 = processor(images=Image.open(image2), return_tensors="pt")
    image_features2 = model.get_image_features(**inputs2)

    print(image_features1.shape, image_features2.shape)

    distance = torch.norm(image_features1 - image_features2)
    cos_sim = torch.nn.functional.cosine_similarity(image_features1.view(-1), image_features2.view(-1),dim=0)
    similarity_score_cos = (cos_sim + 1) / 2
    print('Euclidean distance in latent space between Image 1 and Image 2: {0:.3f}\n'.format(distance.item()))
    print('Cosine similarity between Image 1 and Image 2: {0:.3f}\n'.format(similarity_score_cos.item()))

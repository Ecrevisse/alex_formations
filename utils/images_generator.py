# import replicate
import openai
from openai import OpenAI
import os

from dotenv import load_dotenv
import ast
import urllib.request
import json

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def generate_image_openai(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return image_url


########################
## images generation ###
########################


def save_img(img_url, file_path):
    # prend en entrée un url d'une image et l'enregistre dans file_path
    urllib.request.urlretrieve(img_url, file_path)

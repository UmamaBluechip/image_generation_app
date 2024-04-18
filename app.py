import os
import io
import base64
#from langchain_together import Together
from diffusers import StableDiffusionPipeline
import torch
from flask import Flask, render_template, request

app = Flask(__name__)

model_id = "CompVis/stable-diffusion-v1-4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

#pipe = Together (
#    model="stabilityai/stable-diffusion-xl-base-1.0",
#    together_api_key="48515099b0ed4e22e56da54e50feb4adfaaa901a444b0c34bb33c66abe7b2c61"
#)

@app.route('/', methods=['GET', 'POST'])
def generate_image():
    if request.method == 'POST':
        prompt = request.form['prompt']

        try:
            image = pipe(prompt=prompt).images[0]

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{image_base64}"

            return render_template('index.html', image_url=image_data_uri)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

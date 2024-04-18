import os
import io
import base64
from diffusers import StableDiffusionPipeline
import torch
from flask import Flask, render_template, request

app = Flask(__name__)

model_id = "CompVis/stable-diffusion-v1-4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
pipe.enable_attention_slicing()


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

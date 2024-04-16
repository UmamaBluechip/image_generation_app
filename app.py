import os
import io
import base64
import torch
from diffusers import DiffusionPipeline
from flask import Flask, render_template, request

app = Flask(__name__)

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")  

@app.route('/', methods=['GET', 'POST'])
def generate_image():
    if request.method == 'POST':
        prompt = request.form['prompt']

        try:
            image = pipe(prompt).images[0] 

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

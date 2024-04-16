import os
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
            image.save("generated_image.png")

            return render_template('index.html', filename="generated_image.png")

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

import os
import io
import base64
from langchain_together import Together
from flask import Flask, render_template, request

app = Flask(__name__)


pipe = Together (
    model="stabilityai/stable-diffusion-xl-base-1.0",
    together_api_key="48515099b0ed4e22e56da54e50feb4adfaaa901a444b0c34bb33c66abe7b2c61"
)

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

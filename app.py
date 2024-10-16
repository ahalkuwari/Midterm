
import gradio as gr
from fastai.vision.all import *

# Load the trained model
model_path = '/content/drive/MyDrive/colab_models/Midterm_Project_model.pkl'
learn = load_learner(model_path)

# Define the prediction function
def predict_image(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}

# Set up the Gradio interface
interface = gr.Interface(fn=predict_image, inputs="image", outputs="label")

# Launch the Gradio app
interface.launch(share=True)

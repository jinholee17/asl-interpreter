import streamlit as st
import sift_code.model
import torch
from sift_code.main import projSceneRecBoW
from cnn.models.model import ASLCNN 
from cnn.models.pretrained import get_pretrained_resnet 
from cnn.preprocess import get_transforms
from cnn.utils.model_utils import load_model

from sift_code.student import predict_single_image
from PIL import Image
from torchvision import datasets
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
MOMENTUM = 0.01
IMG_SIZE = (128, 128) 
PREPROCESS_SAMPLE_SIZE = 400
MAX_WEIGHTS_NUM = 5
BATCH_SIZE = 32
NUM_CLASSES = 28
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"




class ST_APP():

    def __init__(self):
        st.title("ASL Interpreter")
        if "model_type" not in st.session_state:
            st.session_state["model_type"] = "Traditional CV Algorithms" 
        st.session_state["model_type"] = st.radio(
            "Select a model", 
            options=["Traditional CV Algorithms", "Neural Network"],
            index=["Traditional CV Algorithms", "Neural Network"].index(st.session_state["model_type"])  # Keep previous selection
        )
        if st.session_state["model_type"] == "Traditional CV Algorithms":
            if st.session_state.get('vocab') is None:
                predictions, model, vocab = projSceneRecBoW('../data/')
                st.session_state["vocab"] = vocab

    def prediction(self):
            data_camera = st.camera_input("Take a picture of an ASL sign to get our classification!")
            if data_camera is not None:
                data = data_camera
            else:
                st.write("Please input data!")
                return

            if data is not None:
                if st.session_state["model_type"] == "Traditional CV Algorithms" and st.session_state.get("vocab") is not None:
                    prediction = predict_single_image(data, st.session_state["vocab"])
                    st.markdown(f"<h1 style='text-align: center; color: green;'>Predicted Letter: {prediction}</h1>", unsafe_allow_html=True)
                    st.success(f"🎉 Predicted Letter: {prediction}")
                elif st.session_state["model_type"] == "Neural Network":
            
                    num_classes = 28
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    testmodel = load_model(ASLCNN, "cnn/asl_model_custom.pth", num_classes)
                    testmodel = testmodel.to(device)
                    testmodel.eval()

                    transform = get_transforms()

                    image = Image.open(data_camera).convert("RGB") 
                    image = transform(image).unsqueeze(0)
                    input_tensor = image.to(device)

                    with torch.no_grad():
                        output = testmodel(input_tensor) 
                        _, predicted_class = torch.max(output, 1) 

                    print(f"Predicted class index: {predicted_class.item()}")


                    train_dir = TEST_DIR
                    train_dataset = datasets.ImageFolder(root=train_dir)
                    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

                    prediction = (idx_to_class.get(predicted_class.item()))
                    st.markdown(f"<h1 style='text-align: center; color: green;'>Predicted Letter: {prediction}</h1>", unsafe_allow_html=True)
                    st.success(f"🎉 Predicted Letter: {prediction}")
                    


                    

if __name__ == '__main__':
    app = ST_APP()
    app.prediction()
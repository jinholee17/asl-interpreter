# ASL INTERPRETER

OVERVIEW:
- An ASL interpeter program where you can both input/capture an image and our program can
detect and translate which letter/sign was given using different CV approaches.
- Our main research objective was seeing whether traditional CV approaches (e.g SVM) or CNN’s were more accurate at identifying letters/phrases in ASL.

INSTALLATION/SETUP:
- Install streamlit: pip install streamlit
- Install torch (we use python 3.9.19):  pip install torch torchvision 
- Change all data filepaths to be absolute filepath (weird bug wasn't allowing us to code this)
    - these filepaths are in helpers and sift_code/main.py
- Download data (too big for Github) https://drive.google.com/drive/folders/1Ye_Az1KMMXi9NrHQp25vgNUyr-K7Y-RZ?usp=drive_link
- Download pre-trained weights from the repo

HOW TO RUN:
- streamlit run main.py

DEMO:
- watch /DEMO.mp4 or at the link online:
https://drive.google.com/file/d/1yn_1dTN4pfM59jGQ23mw0tLCFhJHFPQd/view?usp=sharing

CREDITS:
- Jinho Lee
- Natalie King
- Domingo Viesca
- Sylvie Watts

Libraries/datasets:
- ASL Alphabet Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
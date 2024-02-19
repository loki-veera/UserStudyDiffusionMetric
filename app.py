import glob
import streamlit as st
import numpy as np

from PIL import Image
from copy import deepcopy
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
# from streamlit_image_select import image_select

st.title("User Study - Diffusion Metric")
# st.subheader("Task: Find the realistic image")
st.subheader(""" 
    Instruction:
    Given image pairs, :red[***select the image that looks realistic***].
""")

if 'count' not in st.session_state:
    st.session_state.count = 0

if 'rn' not in st.session_state:
    ctx = get_script_run_ctx()
    session_id = ctx.session_id
    # st.text(ctx.session_id)
    st.session_state["rn"] = 0

ddpm_images = sorted(glob.glob("./images/DDPM/*.jpg"))
ddim_images = sorted(glob.glob("./images/DDIM/*.jpg"))
ddgan_images = sorted(glob.glob("./images/DDGAN/*.jpg"))
wave_images = sorted(glob.glob("./images/WaveDiff/*.jpg"))

images = [ddpm_images, ddim_images, ddgan_images, wave_images]
assert len(ddpm_images) == len(ddim_images) == len(ddgan_images) == len(wave_images)


# if 'images' not in st.session_state:
#     st.session_state.images = {
#         "DDPM": ddpm_images,
#         "DDIM": ddim_images,
#         "DDGAN": ddgan_images,
#         "Wave": wave_images,
#     }
def open_image(img):
    return Image.open(img).convert("RGB")


def get_images():
    # Super messy logic - Try to write in a better way
    left_images = images[st.session_state.count-1]
    np.random.seed(st.session_state.rn*st.session_state.count * 1000)
    left_img_names = np.random.choice(left_images, size=len(images)-1, replace=False)
    left_images = [open_image(name) for name in left_img_names]
    right_images = []
    for i in range(len(images)):
        if i == st.session_state.count-1:
            continue
        np.random.seed((st.session_state.rn*st.session_state.count * i) + 10)
        img = open_image(np.random.choice(images[i]))
        right_images.append(img)
    assert len(left_images) == len(right_images)
    return left_images, right_images


def display_images():
    left_images, right_images = get_images()
    model_nms = ["DDPM", "DDIM", "DDGAN", "WaveDiff"]
    count  = 0
    start_c = model_nms[st.session_state.count-1]
    del model_nms[st.session_state.count-1]
    for left_img, right_img in zip(left_images, right_images):
        col1, col2 = st.columns(2)
        second_name = model_nms[count]
        count += 1
        with col1:
            selection = st.radio(
                label=f"Question {start_c}:{second_name}",
                options=["A", "B"],
                index=0,
                key=f"radio_button_{start_c}_{second_name}"
            )
            st.write(selection)
            pass
        with col2:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.image(left_img, caption="A")
                pass
            with sub_col2:
                st.image(right_img, caption="B")
                pass
            pass

def next_images():
    if st.session_state.count + 1 >= len(images)+1:
        st.session_state.count = 0
        st.session_state.button_next=False
        st.button("Submit")
    else:
        st.session_state.count += 1

if st.session_state.count >= 1:
    display_images()

if st.button("Next", key="button_next",on_click=next_images):
    pass
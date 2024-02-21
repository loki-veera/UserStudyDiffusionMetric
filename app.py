import os
import toml
import glob
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
import time
from copy import deepcopy
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from google.cloud import storage
from google.oauth2 import service_account

BUCKET_NAME="diff-test-bucket"
CREDENTIAL_PATH="./.streamlit/secrets.toml"
if os.path.exists(CREDENTIAL_PATH):
    with open(CREDENTIAL_PATH, "r") as file:
        credential_data = toml.load(file)
CREDENTIALS = service_account.Credentials.from_service_account_info(credential_data["google_cloud"])


if 'hide' not in st.session_state:
    st.session_state.hide = True


def show_hide():
    st.session_state.hide = not st.session_state.hide


if st.session_state.hide:
    with st.container(border=True):
        st.title("User Study - Diffusion Metric")

        # st.subheader("Task: Find the realistic image")
        st.subheader(""" 
            Instruction:
            Given image pairs, :red[***select the image that looks realistic***].
        """)

if 'count' not in st.session_state:
    st.session_state.count = 0

if 'rn' not in st.session_state:
    st.session_state["rn"] = np.random.randint(0, 5)

if 'select' not in st.session_state:
    st.session_state.select = {}


ddpm_images = sorted(glob.glob("./images/DDPM/*.jpg"))
ddim_images = sorted(glob.glob("./images/DDIM/*.jpg"))
ddgan_images = sorted(glob.glob("./images/DDGAN/*.jpg"))
wave_images = sorted(glob.glob("./images/WaveDiff/*.jpg"))


images = [ddpm_images, ddim_images, ddgan_images, wave_images]
assert len(ddpm_images) == len(ddim_images) == len(ddgan_images) == len(wave_images)


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
        with st.container(border=True):
            col1, col2 = st.columns(2)
            second_name = model_nms[count]
            with col1:
                # Hack to get away preselected radio option.
                # Create a dummy option at top and hide it.
                st.markdown(
                    """
                <style>
                    div[role=radiogroup] label:first-of-type {
                        visibility: hidden;
                        height: 0px;
                    }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                selection = st.radio(
                    label=f"**Question:**",
                    options=["None", "A", "B"],
                    index=0,
                    key=f"radio_{start_c}_{second_name}"
                )
                st.session_state.select[f"radio_button_{start_c}_{second_name}"] = selection
                st.write(f"**Your selection:** :blue[{selection}]")
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
            count += 1


def next_images():
    if st.session_state.count + 1 >= len(images)+1:
        st.session_state.count = 0
        with st.container(border=False):
            show_hide()
            st.subheader("Thanks for taking part in the evaluation. Please click on save button to save your review.")
            st.session_state.disabled=True
            if st.button("Save", key="button_save", on_click=save_csv):
                pass
    else:
        st.session_state.count += 1


def save_csv():
    # df = pd.DataFrame(st.session_state.select, columns=["Button", "Selection"])
    df = pd.DataFrame(st.session_state.select.items(), columns=["Button", "Selection"])
    ctx = get_script_run_ctx()
    session_id = ctx.session_id
    file_name = f"answers_{session_id}.csv"
    df.to_csv(file_name, index=False)
    write_to_cloud(file_name)
    os.remove(file_name)
    st.subheader(":green[Your evaluation is saved]. Please close the tab.")
    time.sleep(5)


def write_to_cloud(file_name):
    storage_client = storage.Client(credentials=CREDENTIALS)
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(f"user_responses/{file_name}")
    blob.upload_from_filename(file_name)


if st.session_state.count >= 1:
    display_images()


if st.session_state.hide:
    if st.button("Next", key="button_next", on_click=next_images, disabled=st.session_state.get("disabled", False)):
        pass

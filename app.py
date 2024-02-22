import os
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
import time
from copy import deepcopy
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from gcp_connect import GCP_Connection


if 'hide' not in st.session_state:
    st.session_state.hide = True


def show_hide():
    st.session_state.hide = not st.session_state.hide


if 'count' not in st.session_state:
    st.session_state.count = -1

if 'select' not in st.session_state:
    st.session_state.select = {}

if 'gcp' not in st.session_state:
    st.session_state.gcp = GCP_Connection(
        bucket_name="diff-test-bucket",
        credential_path="./.streamlit/secrets.toml"
    )

if 'rn' not in st.session_state:
    num_files = st.session_state.gcp.get_num_files(prefix="user_responses/")
    print(f"Number of files: {num_files}, Seed: {num_files%10}")
    st.session_state["rn"] = num_files%10

if 'disp_flag' not in st.session_state:
    st.session_state.disp_flag = True


if st.session_state.hide:
    with st.container(border=True):
        st.title("User Study - Diffusion Metric")
        # st.divider()
        # st.subheader("Task: Find the realistic image")
        st.subheader(""" 
            Instruction:
            Given image pairs, :red[***select the image that looks realistic***].
        """)
        st.divider()
        if st.session_state.count == -1:
            st.subheader("**Page**:1/13")
        else:
            st.subheader(f"**Page**:{st.session_state.count+1}/13")

if st.session_state.count == -1:
    ddpm_images = st.session_state.gcp.get_image_names(prefix="images/DDPM/")
    ddim_images = st.session_state.gcp.get_image_names(prefix="images/DDIM/")
    ddgan_images = st.session_state.gcp.get_image_names(prefix="images/DDGAN/")
    wave_images = st.session_state.gcp.get_image_names(prefix="images/WaveDiff/")
    stylegan_images = st.session_state.gcp.get_image_names(prefix="images/StyleGAN2/")
    styleswin_images = st.session_state.gcp.get_image_names(prefix="images/StyleSwin/")
    original_images = st.session_state.gcp.get_image_names(prefix="images/Original/")
    images = [ddpm_images, ddim_images, ddgan_images, wave_images, stylegan_images, styleswin_images]
    if "images" not in st.session_state:
        st.session_state.images = images
        st.session_state.original_images = original_images
    if (len(ddpm_images) == 0) or (len(ddim_images) == 0) or (len(ddgan_images) == 0) or (len(wave_images) == 0) or (len(stylegan_images) == 0) or (len(styleswin_images) == 0) or (len(original_images) == 0):
        raise RuntimeError("No images found. There might be issue with GCP connection.")
    assert len(ddpm_images) == len(ddim_images) == len(ddgan_images) == len(wave_images) == len(styleswin_images) == len(stylegan_images)
    st.session_state.count = 0


@st.cache_data
def generate_images(count_):
    # Super messy logic - Try to write in a better way
    left_images = st.session_state.images[count_]
    np.random.seed(st.session_state.rn*(count_+1) * 1000)
    left_img_names = np.random.choice(left_images, size=len(st.session_state.images)-1, replace=False)
    left_images = [st.session_state.gcp.open_image(name) for name in left_img_names]
    right_images = []
    for i in range(len(st.session_state.images)):
        if i == count_:
            continue
        np.random.seed(st.session_state.rn + (2 ** (i + count_)))
        img = st.session_state.gcp.open_image(np.random.choice(st.session_state.images[i], size=100, replace=False)[-50])
        right_images.append(img)
    assert len(left_images) == len(right_images)
    return left_images, right_images


def display_images():
    # if st.session_state.disp_flag:
    left_images, right_images = generate_images(st.session_state.count-1)
        # st.session_state.disp_flag = False
    count  = 0

    # st.write(st.session_state.count)
    model_nms = ["DDPM", "DDIM", "DDGAN", "WaveDiff", "StyleGAN2", "StyleSwin"]
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
                    label=f"**Question**: Which of the following image looks realistic?",
                    options=["None", "A", "B"],
                    index=0,
                    key=f"radio_{start_c}_{second_name}"
                )
                st.session_state.select[f"radio_button_{start_c}_{second_name}"] = selection
                # st.write(f"**Your selection:** :blue[{selection}]")
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

@st.cache_data
def generate_real_fake(index):
    np.random.seed(st.session_state.rn + (2 ** (index)))
    left_images = np.random.choice(st.session_state.original_images, size=len(st.session_state.images), replace=False)
    left_images = [st.session_state.gcp.open_image(name) for name in left_images]
    right_images = st.session_state.images[index]
    right_images = np.random.choice(right_images, size=len(st.session_state.images), replace=False)
    right_images = [st.session_state.gcp.open_image(name) for name in right_images]
    assert len(left_images) == len(right_images)
    return left_images, right_images


def display_real_fake():
    model_nms = ["DDPM", "DDIM", "DDGAN", "WaveDiff", "StyleGAN2", "StyleSwin"]
    index = st.session_state.count%7
    left_images, right_images = generate_real_fake(index)
    second_name = model_nms[index]
    count  = 0
    for left_img, right_img in zip(left_images, right_images):
        with st.container(border=True):
            col1, col2 = st.columns(2)
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
                    label=f"**Question**: Which of the following image looks realistic?",
                    options=["None", "A", "B"],
                    index=0,
                    key=f"radio_original_{second_name}_{count}"
                )
                st.session_state.select[f"radio_button_original_{second_name}_{count}"] = selection
                # st.write(f"**Your selection:** :blue[{selection}]")
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
    if st.session_state.count + 1 >= 2*len(st.session_state.images)+1:
        st.session_state.count = 0
        show_hide()
        st.session_state.disabled=True
        save_csv()
        # with st.container(border=True):
        #     show_hide()
        #     st.subheader(":red[Please click on save button to save your evaluation].")
        #     st.session_state.disabled=True
        #     if st.button("Save", key="button_save", on_click=save_csv):
        #         pass
    else:
        st.session_state.disp_flag = True
        st.session_state.count += 1


def save_csv():
    # df = pd.DataFrame(st.session_state.select, columns=["Button", "Selection"])
    time.sleep(2)
    df = pd.DataFrame(st.session_state.select.items(), columns=["Button", "Selection"])
    ctx = get_script_run_ctx()
    session_id = ctx.session_id
    file_name = f"answers_{session_id}.csv"
    df.to_csv(file_name, index=False)
    st.session_state.gcp.write_csv(file_name)
    del st.session_state.gcp
    os.remove(file_name)
    with st.container(border=False):
        st.subheader(":green[Your evaluation is saved]. Please close the tab.")
        st.image(Image.open("./images/Thanks.png").convert("RGB"))
    time.sleep(0.5)


if st.session_state.count >= 1 and st.session_state.count <= 6:
    display_images()

if st.session_state.count > 6:
    display_real_fake()

if st.session_state.hide:
    if st.button("Next", key="button_next", on_click=next_images, disabled=st.session_state.get("disabled", False)):
        pass

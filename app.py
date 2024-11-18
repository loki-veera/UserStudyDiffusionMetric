import os
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
import time
from copy import deepcopy
# from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
# from streamlit.scriptrunner.script_run_context import get_script_run_ctx
import random
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
    num_files = st.session_state.gcp.get_num_files(prefix="user_responses_church/")
    st.write(f"Number of files: {num_files}, Seed: {num_files%10}")
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
            st.subheader("**Page**:1/5")
        else:
            st.subheader(f"**Page**:{st.session_state.count+1}/5")

if st.session_state.count == -1:
    # ddgan_images = st.session_state.gcp.get_image_names(prefix="agri/DDGAN/")
    # original_images = st.session_state.gcp.get_image_names(prefix="agri/Original/")
    # pjfgan_images = st.session_state.gcp.get_image_names(prefix="agri/PJFGAN")
    # images = [ddgan_images, pjfgan_images]
    original_images = st.session_state.gcp.get_image_names(prefix="church_user_study/Original/")
    ddim_images = st.session_state.gcp.get_image_names(prefix="church_user_study/DDIM/")
    ddpm_images = st.session_state.gcp.get_image_names(prefix="church_user_study/DDPM/")
    styleswin_images = st.session_state.gcp.get_image_names(prefix="church_user_study/StyleSwin/")
    stylegan_images = st.session_state.gcp.get_image_names(prefix="church_user_study/StyleGAN2/")
    images = [ddim_images, ddpm_images, styleswin_images, stylegan_images]
    if "images" not in st.session_state:
        st.session_state.images = images
        st.session_state.original_images = original_images
    if (len(ddim_images) == 0) or (len(ddpm_images) == 0) or (len(styleswin_images) == 0) or (len(stylegan_images) == 0):
        raise RuntimeError("No images found. There might be issue with GCP connection.")
    # assert len(ddpm_images) == len(ddim_images) == len(ddgan_images) == len(wave_images) == len(styleswin_images) == len(stylegan_images)
    st.session_state.count = 0


@st.cache_data
def generate_images(gseed, count_):
    # Super messy logic - Try to write in a better way
    left_images = st.session_state.images[count_]
    np.random.seed(gseed*(count_+1) * 1000)
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
    left_images, right_images = generate_images(st.session_state.rn, st.session_state.count-1)
        # st.session_state.disp_flag = False
    count  = 0

    # st.write(st.session_state.count)
    # model_nms = ["DDPM", "DDIM", "DDGAN", "WaveDiff", "StyleGAN2", "StyleSwin"]
    # model_nms = ["DDGAN", "PJFGAN"]
    model_nms = ["DDIM", "DDPM", "StyleSwin", "StyleGAN2"]
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
def generate_real_fake(gseed, index):
    np.random.seed(gseed + (2 ** (index)))
    left_images = np.random.choice(st.session_state.original_images, size=len(st.session_state.images), replace=False)
    left_images = [st.session_state.gcp.open_image(name) for name in left_images]
    right_images = st.session_state.images[index]
    right_images = np.random.choice(right_images, size=len(st.session_state.images), replace=False)
    right_images = [st.session_state.gcp.open_image(name) for name in right_images]
    assert len(left_images) == len(right_images)
    return left_images, right_images


def display_real_fake():
    # model_nms = ["DDPM", "DDIM", "DDGAN", "WaveDiff", "StyleGAN2", "StyleSwin"]
    # model_nms = ["DDGAN", "PJFGAN", "DDGAN", "PJFGAN", "DDGAN", "PJFGAN"]
    model_nms = ["DDIM", "DDPM", "StyleSwin", "StyleGAN2"]
    index = st.session_state.count%4
    real_images, fake_images = generate_real_fake(st.session_state.rn, index)
    second_name = model_nms[index]
    count  = 0
    for index, (left, right) in enumerate(zip(real_images, fake_images)):
        np.random.seed(index**4)
        left_img, right_img = left, right
        save_name = f"radio_button_original_{second_name}_{count}"
        if np.random.uniform(0, 10) > 5:
            left_img, right_img = right, left
            save_name = f"radio_button_{second_name}_original_{count}"
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
                st.session_state.select[save_name] = selection
                print(save_name)
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
    if st.session_state.count + 1 >= len(st.session_state.images)+1:
        print(st.session_state.count)
        print()
        print()
        st.session_state.count = 0
        show_hide()
        st.session_state.disabled=True
        save_csv()
        # st.runtime.legacy_caching.clear_cache()
        st.cache_data.clear()
        num_files = st.session_state.gcp.get_num_files(prefix="user_responses_church/")
        st.session_state["rn"] = num_files%10
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
    # ctx = get_script_run_ctx()
    # session_id = ctx.session_id
    # file_name = f"answers_{session_id}.csv"
    seed = random.getrandbits(32)
    file_name = f"answers_fme_{seed}.csv"
    df.to_csv(file_name, index=False)
    st.session_state.gcp.write_csv(file_name)
    os.remove(file_name)
    with st.container(border=False):
        st.subheader(":green[Your evaluation is saved]. Please close the tab.")
        st.image(Image.open("./images/Thanks.png").convert("RGB"))
    time.sleep(0.5)


# if st.session_state.count >= 1 and st.session_state.count <= 2:
#     display_images()

# if st.session_state.count > 2:
if st.session_state.count >= 1:
    display_real_fake()

if st.session_state.hide:
    if st.button("Next", key="button_next", on_click=next_images, disabled=st.session_state.get("disabled", False)):
        pass
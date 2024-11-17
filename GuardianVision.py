import streamlit as st
import os
import base64

st.set_page_config(
    page_title="GuardianVision",
    page_icon="🎥",
    layout="centered",
)

# Get the current directory and use it to load the image
current_directory = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_directory, "GuardianVisionLogo.png")

# Replace backslashes with forward slashes for Windows paths
logo_path = logo_path.replace("\\", "/")

# Read and encode the image file in base64
def load_image_as_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        st.error(f"Logo image not found at {image_path}")
        return None

# Encode the logo image
encoded_logo = load_image_as_base64(logo_path)

if encoded_logo:
    # Insert custom HTML and CSS to center the logo
    st.sidebar.markdown(
        f"""
        <style>
            .logo-container {{
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 20px;
            }}
            .logo-container img {{
                width: 150px;
                border-radius: 10px;
                box-shadow: 0 0 15px 4px rgba(0, 165, 255, 0.7); /* Blue glow */
                transition: box-shadow 0.3s ease-in-out;
            }}
            .logo-container img:hover {{
                box-shadow: 0 0 25px 6px rgba(0, 165, 255, 0.9); /* Brighter blue glow on hover */
            }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{encoded_logo}" alt="GuardianVision Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar Title at the Top (Manually centered with HTML)
st.sidebar.markdown("<h1 style='text-align: center;'>Navigate</h1>", unsafe_allow_html=True)

# Function to display the content of the tabs
def render_tab_content(selected_tab):
    if selected_tab == "File Upload":
        st.title("🎥 GuardianVision 🎥")
        st.write("Upload an MP4 file to preview it below.")
        uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            st.write("### File Details:")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.video(uploaded_file)

            # Enable the "Train" button only if a file is uploaded
            train_button_enabled = True
        else:
            st.info("Please upload an MP4 file to continue.")
            train_button_enabled = False

        # "Train" Button
        if st.button("Train Model", disabled=not train_button_enabled):
            st.write("Training model... Please wait.")
            try:
                # Call the train_model.py script using subprocess
                result = subprocess.run(
                    ["python", "train_model.py"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    st.success("Model training completed successfully!")
                else:
                    st.error(f"Error during training: {result.stderr}")
            except Exception as e:
                st.error(f"Failed to train the model: {e}")

    elif selected_tab == "Instructions":
        st.write(
            """
            ### Upload Instructions:
            1. Click the 'Choose an MP4 file' button to select a file from your computer.
            2. Once the file is uploaded, you will be able to preview it directly in the app.
            3. Supported format: MP4 (up to 488GB).
            """
        )

    elif selected_tab == "About":
        st.write(
            """
            ### About GuardianVision:
            GuardianVision is a cutting-edge video preview tool that allows users to upload MP4 files and view them in a dark-themed interface. It helps users seamlessly preview their videos before processing or training with machine learning models. The interface is designed to be simple, clean, and user-friendly, ensuring a smooth experience for all users.
            """
        )

    elif selected_tab == "Features":
        st.write(
            """
            ### Features of GuardianVision:
            - **Easy MP4 file upload**: Quickly upload your MP4 files with minimal hassle.
            - **Fast video previewing**: Instantly preview your uploaded videos within the app.
            - **Simple, intuitive user interface**: Navigate the platform easily with its clean, minimalistic design.
            - **Dark-themed design**: A comfortable, eye-friendly interface, perfect for long video reviews.
            - **Train your own model**: Train your custom machine learning models using the uploaded videos.
            """
        )

    elif selected_tab == "Contact":
        st.write(
            """
            ### Contact Us:
            If you have any questions or feedback, feel free to reach out to us:
            - **Aarush Shintre**: support@guardiavision.com
            - **Ayush Mendhe**: support@guardiavision.com
            - **Mihir Tirumalasetti**: support@guardiavision.com
            - **Natalie Larksukthom**: support@guardiavision.com
            - **Address**: 800 W Campbell Rd, Richardson, TX 75080
            """
        )

    elif selected_tab == "FAQ":
        st.write(
            """
            ### Frequently Asked Questions:
            **Q: What file formats are supported?**  
            A: We currently support MP4 files.

            **Q: How large can my file be?**  
            A: The maximum file size is 488GB.

            **Q: How do I upload my file?**  
            A: Simply click the 'Choose an MP4 file' button and select your file.

            **Q: What is the 'Train Model' button for?**  
            A: Once you upload a video, you can use the 'Train Model' button to train a custom model based on the video data.
            """
        )

# Create clickable tabs using buttons (styled as rectangles)
tabs = ["File Upload", "Instructions", "About", "Features", "Contact", "FAQ"]
selected_tab = st.session_state.get("selected_tab", "File Upload")

# Display tabs as clickable rectangles
for tab in tabs:
    if st.sidebar.button(tab, key=tab, use_container_width=True):
        selected_tab = tab
        st.session_state.selected_tab = tab

# Render the content of the selected tab
render_tab_content(selected_tab)

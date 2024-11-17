import streamlit as st

st.set_page_config(
    page_title="GuardianVision",
    page_icon="🎥",
    layout="centered",
)

st.title("🎥 GuardianVision 🎥")
st.write("Upload an MP4 file to preview it below.")

uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    st.write("### File Details:")
    st.write(f"**Filename:** {uploaded_file.name}")

    st.video(uploaded_file)

else:
    st.info("Please upload an MP4 file to continue.")

import streamlit as st
import os
import shutil
from PIL import Image
from photoid_process import ImagePreprocess  # Replace with your actual script name

# Initialize ImagePreprocess
st.sidebar.header("Configuration")
study_site = st.sidebar.text_input("Study Site", "DefaultSite")
group_id = st.sidebar.text_input("Group ID", "Group01")
image_processor = ImagePreprocess(study_site, group_id)

st.title("Image Preprocessing Web Application")

st.header("Upload Images")
uploaded_files = st.file_uploader("Choose Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.success(f"Uploaded {len(uploaded_files)} files.")
    
    st.header("Rename Images")
    format_string = st.text_input(
        "Filename Format",
        "IMG_{image_id}_{date_time}_{STUDY_SITE}_{GROUP_ID}_{frame_number}.jpg"
    )
    
    if st.button("Rename Images"):
        try:
            image_processor.rename_images(temp_dir, format_string)
            st.success("Images have been renamed.")
            
            with st.expander("Download Renamed Images"):
                for renamed_file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, renamed_file)
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"Download {renamed_file}",
                            data=f,
                            file_name=renamed_file,
                            mime="image/jpeg"
                        )
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    st.header("Extract EXIF Data")
    if st.button("Extract EXIF Data"):
        for uploaded_file in uploaded_files:
            image_path = os.path.join(temp_dir, uploaded_file.name)
            exif_data = image_processor.extract_exif(image_path)
            if exif_data:
                st.write(f"**{uploaded_file.name} EXIF Data:**")
                for key, value in exif_data.items():
                    st.write(f"{key}: {value}")
            else:
                st.write(f"No EXIF data found for {uploaded_file.name}.")
    
    st.header("Image Preview")
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image_path = os.path.join(temp_dir, uploaded_file.name)
        image = Image.open(image_path)
        with cols[idx % 3]:
            st.image(image, caption=uploaded_file.name, use_column_width=True)
    
    # Optional: Clean up temporary directory after processing
    if st.button("Clean Up"):
        shutil.rmtree(temp_dir)
        st.success("Temporary files have been removed.")

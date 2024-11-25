import streamlit as st
import tensorflow as tf
import numpy as np
import time

from PIL import Image
from streamlit_option_menu import option_menu

code = """
def process(edf_file_path, output_image_path):

    # Read file EDF using MNE
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)

    #Applying ICA to reduce noise and pick channels with most information
    ica = ICA(n_components=15, max_iter="auto", random_state=0)
    ica.fit(raw)

    # Get the ICA sources
    ica_sources = ica.get_sources(raw)

    # Get the ICA source data as a NumPy array
    ica_source_data = ica_sources.get_data()

    #Limit amplitude from -0.3 to 0.3 to reduce all the leftover noises
    ica_source_data = np.clip(ica_source_data, -0.3, 0.3)

    #Avaraging the signal
    sum_signal = np.sum(ica_source_data, axis=0)   
    avg_signal = sum_signal / ica_source_data.shape[0]

    #Creates a new MNE object
    info = mne.create_info(['avg_signal'], raw.info['sfreq'], ch_types='eeg')
    avg_raw = mne.io.RawArray(avg_signal[np.newaxis, :], info)

    #Apply filter
    avg_raw.filter(l_freq=1, h_freq=2, method='iir')
    avg_raw.notch_filter(freqs=[1], method='iir')

    #Extract time points and data
    time_points = avg_raw.times
    dataplot = avg_raw.get_data()

    #PLotting
    plt.plot(time_points, dataplot[0], color='#90EE90')
    plt.savefig(output_image_path)
"""

# Centered menu title
st.markdown(
    """<div class='centered-menu-title'>Epileptic Classify</div>""",
    unsafe_allow_html=True,
)

selected = option_menu(
    menu_title = None,
    options = ["Model", "Data Processing"],
    default_index = 0,
    orientation = "horizontal",
)

if selected == "Model":

    col2 = st.columns([2])

    # Load the pre-trained model
    # model = tf.keras.models.load_model('241123_my_model.keras')
    model = tf.keras.models.load_model('241123_my_model.keras')

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        # Open the uploaded image
        img = Image.open(uploaded_file)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Display the uploaded image
        st.image(img, caption='Uploaded Image')

        # Resize the image to match the model's expected input size (height, width)
        img = img.resize((456, 608))  # Model expects (height=608, width=456)

        # Convert the image to a numpy array and normalize
        img_array = np.array(img) / 255.0

        # Ensure the array has the correct shape (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        predictions = model.predict(img_array)

        progress_bar = st.progress(0, text="Operation in progress. Please wait.")


        for perc_completed in range(100):
            time.sleep(0.005)
            progress_bar.progress(perc_completed+1, text="Operation in progress. Please wait.")
        time.sleep(0.01)

        if predictions[0] < 0.5:
            st.write("Seizure detected")
        else:
            st.write("Normal signal")

if selected == "Data Processing":
    st.image("images/aaaaaawu_s001_t001.png", caption="Average Signal", use_container_width =True)
    st.markdown("The original data have noise, which make makes it difficult to use the original data for training. Therefore, we need to apply a technique called Independent Component Analysis (ICA) to filter channels with most information and reduce noises. This step is like isolating individual voices in a noisy room.The cleaned data is averaged to create a single signal representing overall brain activity.")
    st.markdown("To enhance the signal quality, filters are applied. A band-pass filter is used to isolate specific frequency bands, such as those associated with sweating and blinking. Additionally, a notch filter removes any noise at a particular frequency, further refining the signal.")
    st.markdown("Finally, the processed signal is visualized as a graph. The x-axis represents time, while the y-axis shows the intensity of brain activity.")
    st.code(code, language="python")


st.markdown(
    """
    <style>
    .paragraph {
        text-align: justify;
        font-size: 1.1em;
        margin-top: 10px;
    }

    .centered-menu-title {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .centered-title {
        text-align: center;
        font-size: 2.5em;  /* Adjust font size if needed */
        margin-top: 20px;  /* Adjust margin for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
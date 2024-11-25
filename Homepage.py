import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# st.experimental_rerun()

st.set_page_config(
    page_title="Epileptic Classification",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.facebook.com/reject.humanity.become.monke/",
        "Report a bug": "https://www.facebook.com/reject.humanity.become.monke/",
        "About": "This is a GUI for epilepsy classification project"
    }
)

# Custom styling function for the table
def style_table(df):
    def highlight_first_row(row):
        if row.name == 0:  # If it's the first row
            return ['background-color: lightblue; color: black'] * len(row)
        return [''] * len(row)  # Default for other rows

    return df.style.apply(highlight_first_row, axis=0)

def set_font_size(df, font_size):
  return df.style.set_properties(**{'font-size': str(font_size) + 'px'})

def render_table_with_custom_font_size(df, table_title, font_size=16):
    table_html = df.to_html(index=False)
    st.markdown(
        f"""
        <h3 style='text-align: center;'>{table_title}</h3>
        <div style='display: flex; justify-content: center;'>
            <div style='width: 80%;'>
                <style>
                    .dataframe {{
                        font-size: {font_size}px;
                        width: 100%;
                    }}
                    .dataframe th, .dataframe td {{
                        text-align: center;
                        padding: 8px;
                    }}
                </style>
                {table_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Centered menu title
st.markdown(
    """<div class='centered-menu-title'>Domain Knowledge</div>""",
    unsafe_allow_html=True,
)

selected = option_menu(
    menu_title = None,
    options = ["Project", "Data Explore"],
    default_index = 0,
    orientation = "horizontal",
)

if selected == "Project":
    st.markdown("""
    <div style="text-align: justify;">
        Epilepsy is a lifelong condition involving recurring seizures that are electrical disturbances in the brain that may allow various effects. In patients, such seizures cause loss of consciousness, shaking and other physical changes that disturb the patient’s normal functioning. Thus, more awareness is needed to reduce the onset of seizures or act as early as possible to increase the patient's lifespan, since the early stages of the disease can be greatly improved.
    </div>
    """,
    unsafe_allow_html=True)
    st.image("images/epilepsy.jpg", use_container_width =True)
    st.markdown(
    """
    <div style="text-align: center;">
        <p style="margin: 0; font-size: 16px; color: gray;">Epilepsy Symptoms</p>
        <a href="https://continentalhospitals.com/diseases/epilepsy/" target="_blank" style="text-decoration: none; color: cyan;">
            Source: EEG Recording Article
        </a>
    </div>
    """,
    unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: justify;">
        Machine learning and data science have been adopted considerably in various medical fields in the past few years and in particular epilepsy diagnosis and seizure detection. With the latest development in data gathering especially the EEG, it has become possible to record the brain activity in real time. We obtain EEGs by placing electrodes on the scalp of the brain, where certain patterns connected to epileptic occurrences can be seen. However, extraction of these signals for seizure detection and prediction continues to be complex due to noise, artifacts, and the overall characteristics of the brain.
    </div>
    """,
    unsafe_allow_html=True)
    st.image("images/EEG_recording.png", use_container_width =True)
    st.markdown(
    """
    <div style="text-align: center;">
        <p style="margin: 0; font-size: 16px; color: gray;">EEG Recording</p>
        <a href="https://ojs.ukscip.com/journals/dtra/article/view/40" target="_blank" style="text-decoration: none; color: cyan;">
            Source: EEG Recording Article
        </a>
    </div>
    """,
    unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: justify;">
        The main aim of this work is to build the best seizure detection model based on EEG data utilizing an approach of machine learning algorithms. With such models, the proposal here is to train them to identify patterns that transmit epilepsy signals so that a useful tool is developed to complement the clinical diagnosis and prognosis of seizures. This work focuses on feature extraction of EEG data for feature selection and the application of machine learning algorithms for classification with higher accuracy by optimizing the feature extraction process and using denoising techniques. Finally, the aim is to develop a simple GUI where clinicians and researchers are able to use the seizure prediction model.
    </div>
    """,
    unsafe_allow_html=True)

elif selected=="Data Explore":
    st.markdown("""
    <div style="text-align: justify;">
        We employ Temple University Hospital's open-source EEG data (TUH EEG Corpus) in this study. The signals are quite varied in timing as well as measurement method. Some signals are sampled at 250Hz, but some are at 120Hz, with 10-20 electrode placement. The following are data statistics.
    </div>
    """,
    unsafe_allow_html=True)

    #Duration
    data_length = {
        "Length": ["Under 1m", "From 1 to 10m", "Between 10 and 30m", "More than 30m", "Total"],
        "Epilepsy": [126, 667, 905, 87, 1785],
        "No Epilepsy": [123, 194, 173, 23, 513],
    }
    df_length = pd.DataFrame(data_length)
    st.markdown("<h3 style='text-align: center;'>Table: Epilepsy vs. No Epilepsy by Time</h3>", unsafe_allow_html=True)
    st.dataframe(
    df_length.style.set_properties(
        **{"text-align": "center",}
    ).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    ),
    use_container_width=True,)

    # st.table(df_length)
    # st.dataframe(style_table(df_length))

    #Frequency
    data_frequency = {
        "Frequency": ["250.0 Hz", "256.0 Hz", "1000.0 Hz", "512.0 Hz", "400.0 Hz", "Total"],
        "Epilepsy": [386, 1144, 17, 168, 70, 1785],
        "No Epilepsy": [295, 210, 0, 0, 5, 513],
    }
    df_frequency = pd.DataFrame(data_frequency)
    st.markdown("<h3 style='text-align: center;'>Table: Epilepsy vs. No Epilepsy by Frequency</h3>", unsafe_allow_html=True)
    # st.table(df_frequency)
    st.dataframe(
    df_frequency.style.set_properties(
        **{"text-align": "center"}
    ).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    ),
    use_container_width=True,)

    #Electrode
    data_electrode = {
        "Number of Electrode": [17, 18, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 41, "Total"],
        "Epilepsy": [2, 6, 152, 104, 90, 154, 152, 158, 120, 583, 19, 159, 86, 1785],
        "No Epilepsy": [0, 0, 6, 2, 11, 75, 27, 45, 140, 12, 0, 155, 40, 513],
    }
    df_electrode = pd.DataFrame(data_electrode)
    st.markdown("<h3 style='text-align: center;'>Table: Epilepsy vs. No Epilepsy by Number of Electrodes</h3>", unsafe_allow_html=True)
    # st.table(df_electrode)
    st.dataframe(
    df_electrode.style.set_properties(
        **{"text-align": "center"}
    ).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    ),
    use_container_width=True,  # This ensures the table stretches across the available width
)

    st.markdown("As we can see that the dataset is imbalanced, so my solution would be to choose a certain time point (e.g. from 1 minute or less, from 20 to 25 minutes) to pick a balanced number of files for analysis, limiting the model bias. Will explain more in the model training section.")

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
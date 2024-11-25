import streamlit as st
from streamlit_option_menu import option_menu

# Centered menu title
st.markdown(
    """<div class='centered-menu-title'>Domain Knowledge</div>""",
    unsafe_allow_html=True,
)

selected = option_menu(
    menu_title = None,
    options = ["Time", "Frequency", "Time-frequency"],
    default_index = 0,
    orientation = "horizontal",
)

# Dynamic content for each option
if selected == "Time":
    st.markdown("<h1 class='centered-title'>Time domain analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='paragraph'>In time-domain analysis, signals are analyzed as they vary over time. This method is useful for examining the signal's amplitude and duration to detect changes or anomalies.</p>",
        unsafe_allow_html=True,
    )
    st.image("images/time.png", caption="Time Domain Analysis", use_container_width=True)
    st.markdown("It is possible to analyze the structure of the considered epileptic EEG signal in the time domain as well as the changes in amplitude and frequency of the signal during the day. These transient events point to abnormal neural activity and are generally linked with the synchronizing of huge numbers of neurons. The number of such episodes proves that there is an elevated level of brain hyper-excitability which is characteristic of epilepsy.")
    st.markdown("On the other hand, the time-domain analysis of the non-epileptic EEG signal is relatively much more organized, and the noise has lesser tendency. The amplitude fluctuations are lesser and less in number which indicates a rhythmical oscillation in the neuronal behaviour. This pattern is very similar to normal brain function wherein neurons are firing in a coordinated way.")


elif selected == "Frequency":
    st.markdown("<h1 class='centered-title'>Frequency domain analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='paragraph'>Frequency-domain analysis focuses on how signals vary with frequency. This method reveals the dominant frequencies and is often used in spectrum analysis and signal processing.</p>",
        unsafe_allow_html=True,
    )
    st.image("images/fre.png", caption="Frequency Domain Analysis", use_container_width =True)
    st.markdown("The power spectral density plot of the epileptic EEG indicates a quite typical features of this type of signals, most notably a shift in the content towards higher frequencies of around 30 - 60 Hz. This increased activity in these frequency bands usually linked to gamma waves suggests that the brains neural firing is out of sync and that there are higher firing rates – both symptomatic of epileptic seizures.")
    st.markdown("Whereas, the PSD plot of the non-epileptic EEG signal reveals another peak lying in the lower frequency band and which can be typical for alpha and theta waves and which is below than 20Hz. Lack of pattern is much similar to normal control recording and hence explains more of a regular pattern of brain activity. The lack of much energy at the higher frequencies also provides evidence for a healthy brain state.")

elif selected == "Time-frequency":
    st.markdown("<h1 class='centered-title'>Time-frequency domain analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='paragraph'>Time-frequency analysis examines how the signal's frequency content evolves over time. It combines the insights from both time and frequency domains for a comprehensive view of the signal.</p>",
        unsafe_allow_html=True,
    )
    st.image("images/spectrogram.png", caption="Time-Frequency Domain Analysis", use_container_width =True)
    st.markdown("Spectrogram of epileptic patient shows the pattern of energy density plot of epileptic signal where energy level is elevated to the higher range of frequency levels approximately 30-60 Hz which is the abnormal activity of neurons during seizures. These fast oscillations also known as gamma waves present itself as increased neuronal synchronization and can be seen during seizures. Further, the spectrogram reveals less continuous oscillation indicating transient period of enhanced neuronal coherency.")
    st.markdown("Spectrogram of a non-epileptic patient shows less variability with the energy band at the lower frequency, about 20-40 Hz, related to alpha and beta waves. These frequencies are linked with the regular activity, like wakefulness and relaxation of the brain. These are spins of healthy high brain activity, and the lack of huge high frequency bursts and the relatively constant pattern of the spectrogram.")


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
# Invasive Plant Species Detection

This repository contains a pre-trained model for detecting invasive plant species and
a Streamlit app to test the model locally.

## Running the app

1. Install the required Python packages:

   ```bash
   pip install streamlit tensorflow pillow
   ```

2. Start the Streamlit interface:

   ```bash
   streamlit run app.py
   ```

Upload an image of a plant and the app will display the predicted class. The
feature extractor model is bundled in this repository, so no network access is
required at runtime.

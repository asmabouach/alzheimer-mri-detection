import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import io
import os

# -------------------------------
# Load model (cached for speed)
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("model/best_custom_cnn.keras")

model = load_my_model()

CLASSES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# -------------------------------
# Page config & custom styling
# -------------------------------
st.set_page_config(page_title="Alzheimer’s MRI Diagnostic Tool", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: white;
        font-family: 'Helvetica Neue', sans-serif;
        color: #2C3E50;
    }
    h1, h2, h3 {
        font-weight: 600;
        color: #2C3E50;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
        color: white;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }
    .animated-icon {
        position: fixed;
        right: 40px;
        bottom: 40px;
        font-size: 60px;
        animation: float 3s ease-in-out infinite;
        color: #3498db;
    }
    @keyframes float {
        0% { transform: translatey(0px); }
        50% { transform: translatey(-15px); }
        100% { transform: translatey(0px); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""<div class="animated-icon">🧠</div>""", unsafe_allow_html=True)

st.title("🧠 Alzheimer’s MRI Diagnostic Tool")
st.write("""
Upload an MRI brain scan and the system will provide an **AI-assisted analysis** of the Alzheimer’s disease stage.  
This tool has **4 categories**:
""")

st.markdown("""
- 🟢 **NonDemented**: No signs of Alzheimer’s disease.  
- 🟡 **ModerateDemented**: Clear cognitive decline with moderate symptoms.  
- 🟠 **MildDemented**: Early signs of memory loss and mild impairment.  
- 🔴 **VeryMildDemented**: Subtle but noticeable changes in memory and thinking.  
""")
# -------------------------------
# Patient Info Form
# -------------------------------
st.subheader("📝 Patient Information")
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Full Name")
    patient_gender = st.radio("Gender", ["Male", "Female", "Other"], horizontal=True)
with col2:
    dob = st.date_input(
        "Date of Birth",
        min_value=datetime.date(1900, 1, 1),  # allow from 1900
        max_value=datetime.date.today(),      # up to today
        value=datetime.date(1963, 10, 15)       # default if empty
    )

# -------------------------------
# Example Images (for testing)
# -------------------------------
st.markdown("###### 🧪 Example MRI Images (you can try uploading these)")

example_paths = [
    "./img//MRI1-VMI.jpg",
    "./img/MRI2-NI.jpg",
    "./img/MRI3-MoI.jpg",
    "./img/MRI4-MiI.jpg",
    "./img/MRI5-VMI.jpg",
    "./img/MRI6-MoI.jpg",

]

cols = st.columns(6)
for col, path in zip(cols, example_paths):
    with col:
        st.image(path, caption="Example", use_container_width=True)

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(128, 128))
    img_arr = img_to_array(img) / 255.0
    img_exp = np.expand_dims(img_arr, axis=0)

    st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

    # Predict
    with st.spinner("Analyzing scan..."):
        preds = model.predict(img_exp)[0]
        cls_idx = np.argmax(preds)
        confidence = preds[cls_idx] * 100
        predicted_class = CLASSES[cls_idx]

    # -------------------------------
    # Results Section
    # -------------------------------
    st.subheader("📊 Results")
    COLORS = {
        "Non Demented": "✅ <span style='color:#4CAF50;font-weight:bold'>Non Demented</span>",
        "Mild Demented": "🟠 <span style='color:#FF9800;font-weight:bold'>Mild Demented</span>",
        "Moderate Demented": "🟡 <span style='color:#FFC107;font-weight:bold'>Moderate Demented</span>",
        "Very Mild Demented": "🔴 <span style='color:#E53935;font-weight:bold'>Very Mild Demented</span>",
    }

    # Results
    st.markdown(
        f"""
        Based on the MRI scan, the system predicts:  
        **Stage:** {COLORS[predicted_class]}  
        **Confidence Level:** {confidence:.2f}%
        """,
        unsafe_allow_html=True
    )

    # Confidence bar
    st.subheader("📈 Prediction Confidence Across All Classes")
    fig, ax = plt.subplots()
    colors = ["#FF9800", "#EEB300", "#4CAF50", "#E53935"]
    ax.bar(CLASSES, preds * 100, color=colors)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=15)
    st.pyplot(fig)

    # Saliency Map (simple interpretability)
    st.subheader("📌 Visualization Map")
    st.markdown("The **saliency map** highlights the regions of the MRI scan that most influenced the model’s prediction," \
    "showing which parts of the image the AI focused on.")

    with st.spinner("Computing saliency map..."):
        img_tensor = tf.convert_to_tensor(img_exp)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            loss = predictions[:, cls_idx]

        grads = tape.gradient(loss, img_tensor)[0]  # Remove batch dim
        saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()  # Remove [0]

        # FORCE 2D (handles all dimension issues)
        saliency = np.squeeze(saliency)  # Remove all singleton dims
        if saliency.ndim > 2:
            saliency = saliency[0, :, :]   # Take first slice if 3D+
        
        # Final 2D check
        assert saliency.ndim == 2, f"Saliency shape: {saliency.shape}"
        
        # Normalize
        saliency = np.maximum(saliency, 0)
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        # Colorize
        saliency_colored = plt.cm.jet(saliency)[:, :, :3]
        saliency_rgb = (saliency_colored * 255).astype(np.uint8)

        # Image prep
        img_display = np.clip(img_arr * 255, 0, 255).astype(np.uint8)
        if img_display.ndim == 2:
            img_display = np.stack([img_display]*3, axis=-1)

        # Overlay
        overlaid = (saliency_rgb * 0.6 + img_display * 0.4).astype(np.uint8)

        fig_saliency, ax_saliency = plt.subplots(figsize=(8, 8))
        ax_saliency.imshow(overlaid)
        ax_saliency.axis('off')
        ax_saliency.set_title(f"Saliency Map for {predicted_class}", fontsize=14, pad=20)
        st.pyplot(fig_saliency)
        plt.close(fig_saliency)



    # Doctor Notes
    st.subheader("🩺 Doctor Notes")
    doctor_notes = st.text_area("Enter any observations or additional notes:", height=100)

    # PDF Report
    if st.button("📑 Download Diagnostic Report"):
        pdf = FPDF()
        pdf.add_page()

        # Report Date in top-right header
        pdf.set_font("Helvetica", size=9)
        pdf.set_xy(-60, 5)  # Move to top-right corner (60 mm from right)
        pdf.cell(50, 8, f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="R")

        # Reset cursor to top-left for title
        pdf.set_xy(10, 20)  # Typical left margin, 20 mm from top

        # Title
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, "Alzheimer's MRI Diagnostic Report", ln=True, align="C")
        pdf.ln(5)

        # Patient Info
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(200, 10, "Patient Information", ln=True)
        pdf.set_font("Helvetica", size=10)

        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(20, 8, "Name: ", border=0)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(45, 8, f"{patient_name}", ln=True)

        # Second row: DOB + Age + Gender
        age = datetime.datetime.now().year - dob.year

        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(35, 8, "Date of Birth: ", border=0)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(30, 8, dob.strftime('%Y-%m-%d'), border=0)

        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(15, 8, "Age: ", border=0)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(20, 8, f"{age} years", border=0)

        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(20, 8, "Gender: ", border=0)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(25, 8, f"{patient_gender}", ln=True)

        # Results
        pdf.ln(1)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(200, 10, "Results", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(50, 8, "The MRI scan indicates: ", ln=0)

        pdf.set_text_color(0, 0, 139)  # Dark blue
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 8, f"{predicted_class} with {confidence:.2f}% confidence.", ln=1)

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", '', 12)
        
        # MRI Scan + Saliency Map (SAME LINE)
        pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(200, 8, "MRI Scan & Visualization Map:", ln=True)

        # Save both images first
        os.makedirs("output", exist_ok=True)
        mri_path = "output/uploaded_temp.png"
        saliency_path = "output/saliency_temp.png"
        img.save(mri_path)
        fig_saliency.savefig(saliency_path, bbox_inches='tight')

        # Position MRI image (left)
        pdf.image(mri_path, x=40, y=pdf.get_y(), w=50, h=50)

        # Move right for saliency (same Y position)
        pdf.set_xy(80, pdf.get_y())  # Same height, right position
        pdf.image(saliency_path, x=100, y=pdf.get_y(), w=50, h=51)
        pdf.ln(50)

        # Prediction Confidence Plot
        pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(200, 10, "Prediction Confidence:", ln=True)
        chart_path = "temp_chart.png"        
        fig.savefig(chart_path, bbox_inches='tight')
        pdf.image(chart_path, x=60, w=70)

        # Doctor Notes
        pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(200, 10, "Doctor Notes:", ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 8, doctor_notes if doctor_notes else "No additional notes provided.")

        # BytesIO for download button
        pdf_bytes = io.BytesIO()
        pdf.output(pdf_bytes)
        pdf_bytes.seek(0)

        # Save file
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_output_path = f"output/alzheimer_report_{timestamp}.pdf"
        pdf.output(pdf_output_path)

        # Download button
        st.download_button(
            label="⬇️ Download Report PDF",
            data=pdf_bytes.getvalue(),
            file_name=f"alzheimer_report_{timestamp}.pdf",
            mime="application/pdf"
        )

        # Clean temp files
        for path in [mri_path, chart_path, saliency_path]:
            try:
                os.remove(path)
            except:
                pass

else:
    st.info("Please upload an MRI image to start the analysis.")
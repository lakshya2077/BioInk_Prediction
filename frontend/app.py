import sys
import os
import streamlit as st
import pandas as pd
import joblib
import time
import base64

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_printability

# Add degradation_project/src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'degradation_project')))
from degradation_project.predict import predict_degradation

# --- Page Config ---
st.set_page_config(
    page_title="Bio-Ink Predictor",
    page_icon="frontend/assets/icon.png",
    layout="wide"
)

GIF_PATH = os.path.join(os.path.dirname(__file__), "assets", "Loading.gif")
GIF_DEGRADATION_PATH = os.path.join(os.path.dirname(__file__), "assets", "loading_degradation.gif")

# --- Custom Styling ---
st.markdown("""
    <style>
        html, body {
            background-color: #f0f8ff;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #004d66;
        }
        .stButton > button {
            background-color: #008cba;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            padding: 0.5em 1.5em;
            font-size: 1rem;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #005f80;
            transform: scale(1.05);
            color: white;
        }
        .stButton > button:focus,
        .stButton > button:active {
            color: white !important;
            background-color: #005f80 !important;
            outline: none;
            box-shadow: none;
        }
        /* Hide deploy button, Streamlit header and footer */
        # header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("About this App")
section = st.sidebar.radio("Choose a Task", ["Printability Prediction", "Degradation Prediction"])

if section == "Printability Prediction":
    st.sidebar.info("""
This tool uses a trained ML model to predict whether a given **bio-ink formulation** is printable or not.

Designed to help in experimental planning and reduce trial-and-error in lab workflows.
""")
    st.sidebar.success("Developed as part of a summer research internship.")

elif section == "Degradation Prediction":
    st.sidebar.info("""
This tool uses a trained ML model to predict the **degradation behavior** of bio-printed scaffolds over time.

It estimates how much a scaffold loses stiffness, gains water, and degrades structurally — helping researchers fine-tune materials for better longevity.
""")
    st.sidebar.success("Now fully functional! 🔬")

st.sidebar.markdown("---")

# --- Title with GIF ---
# Choose title gif based on section
if section == "Printability Prediction":
    title_gif_path = os.path.join(os.path.dirname(__file__), "assets", "Title.gif")
else:
    title_gif_path = os.path.join(os.path.dirname(__file__), "assets", "Title_degradation.gif")  # You can rename it as needed

# Load and encode selected gif
with open(title_gif_path, "rb") as f:
    gif_bytes = f.read()
    gif_base64 = base64.b64encode(gif_bytes).decode()


col1, col2 = st.columns([2, 17])
with col1:
    st.markdown(f"""<img src="data:image/gif;base64,{gif_base64}" width="100">""", unsafe_allow_html=True)
with col2:
    title_text = "Bio-Ink Printability Predictor" if section == "Printability Prediction" else "3D Printed Scaffold Degradation Predictor"
    st.markdown(f"<h2 style='margin-top: 0.4em; color: #004d66;'>{title_text}</h2>", unsafe_allow_html=True)

# --- Main Section Logic ---
if section == "Printability Prediction":
    st.subheader("Enter Bio-Ink Parameters")

    Gelatin_pct = st.number_input("Gelatin (%)", 0.0, 30.0, 1.5, 0.1)
    Silk_pct = st.number_input("Silk Fibroin (%)", 0.0, 10.0, 2.0, 0.1)
    LH = st.number_input("Layer Height (mm)", 0.0, 1.0, 0.65, 0.01)
    PP = st.number_input("Print Pressure (kPa)", 0, 100, 55, 1)
    PS = st.number_input("Print Speed (mm/s)", 1, 30, 12, 1)
    T = st.number_input("Temperature (°C)", 10.0, 40.0, 23.0, 0.1)
    TG_min = st.number_input("Printing Time (min)", 0, 10, 3, 1)
    Used_crosslinker = st.radio("Used Crosslinker?", ["Yes", "No"])
    Needle = st.selectbox("Needle Size", ["22G", "25G", "27G", "30G"])

    # Frontend validation for TG_min and PS
    if TG_min <= 0:
        st.warning("⚠️ Printing time (TG_min) must be greater than 0. Please enter a valid value.")
        st.stop()

    if PS <= 0:
        st.warning("⚠️ Print speed (PS) must be greater than 0. Please enter a valid value.")
        st.stop()

    if st.button("🔍 Predict Printability"):
        input_data = {
            'Gelatin_pct': Gelatin_pct,
            'Silk_pct': Silk_pct,
            'LH': LH,
            'PP': PP,
            'PS': PS,
            'T': T,
            'TG_min': TG_min,
            'Used_crosslinker': 1 if Used_crosslinker == "Yes" else 0,
            'Needle': Needle,
            'Remarks': "Predicted via UI"
        }

        try:
            prediction = predict_printability(input_data)
        except Exception as e:
            st.error("❌ Prediction failed. Check input or model.")
            st.stop()

        anim_container = st.empty()
        with anim_container:
            st.image(GIF_PATH, caption="🔄 Analyzing formulation...", width=200)
            time.sleep(5)
        anim_container.empty()

        st.markdown("### Prediction Result")

        # Forcefully reject if TG_min or PS == 0
        if TG_min == 0.0 or PS == 0.0:
            st.error("❌ This formulation is **Not Printable** due to invalid parameters (TG_min or Print Speed is 0).")
            prediction = 0
        else:
            if prediction == 1:
                st.success("✅ This formulation is **Printable**!")
            else:
                st.error("❌ This formulation is **Not Printable**.")

        # --- Scientific Summary ---
        st.markdown("### 🔬 Scientific Insight")

        insights = []

        if Gelatin_pct < 12:
            insights.append("• **Very low gelatin content** — may lead to poor structural support and fragile prints.")
        elif Gelatin_pct <= 16:
            insights.append("• **Optimal gelatin content** — provides strong mechanical support.")
        else:
            insights.append("• **Excessive gelatin content** — may make the ink too viscous for smooth extrusion.")

        if Silk_pct < 4:
            insights.append("• **Low silk fibroin** — could reduce flexibility and make the construct brittle.")
        elif Silk_pct <= 5:
            insights.append("• **Moderate silk fibroin** — enhances elasticity and biocompatibility.")
        else:
            insights.append("• **High silk fibroin** — may overly stiffen the bio-ink, reducing flow quality.")

        if LH < 0.65:
            insights.append("• **Very low layer height** — may slow down printing and risk under-building layers.")
        elif LH <= 0.72:
            insights.append("• **Optimal layer height** — supports fine resolution and stable layer bonding.")
        else:
            insights.append("• **High layer height** — may reduce print resolution and cause poor layer adhesion.")

        if PP < 45:
            insights.append("• **Low extrusion pressure** — may result in under-extrusion or inconsistent flow.")
        elif PP <= 65:
            insights.append("• **Optimal extrusion pressure** — maintains steady, uniform deposition.")
        else:
            insights.append("• **High extrusion pressure** — could lead to over-extrusion or structural distortion.")

        if TG_min < 3:
            insights.append("• **Very fast gelation** — increases risk of nozzle clogging and uneven solidification.")
        elif TG_min <= 10:
            insights.append("• **Controlled gelation time** — allows proper crosslinking and structure formation.")
        else:
            insights.append("• **Prolonged gelation** — may delay stabilization and reduce structural precision.")

        if prediction == 1:
            final_note = "✅ **This formulation is considered printable based on the current parameters.**"
        else:
            suggestions = []
            if Gelatin_pct < 12 or Gelatin_pct > 16:
                suggestions.append("gelatin concentration")
            if Silk_pct < 4 or Silk_pct > 5:
                suggestions.append("silk fibroin level")
            if LH < 0.65 or LH > 0.72:
                suggestions.append("layer height")
            if PP < 45 or PP > 65:
                suggestions.append("extrusion pressure")
            if TG_min < 3 or TG_min > 10:
                suggestions.append("printing time")

            if suggestions:
                param_list = ", ".join(suggestions[:-1]) + f", and {suggestions[-1]}" if len(suggestions) > 1 else suggestions[0]
                final_note = f"⚠️ **This formulation may not be printable. Consider adjusting the {param_list}.**"
            else:
                final_note = "⚠️ **This formulation may not be printable due to unknown factors. Please verify the experimental setup.**"

        st.markdown("#### Formulation Breakdown:")
        for point in insights:
            st.markdown(point)
        st.markdown(f"---\n{final_note}")

elif section == "Degradation Prediction":
    st.subheader("Enter Scaffold Degradation Parameters")

    Scaffold_Geometry = st.selectbox(
        "Scaffold Geometry",
        ["Body Centered", "Body Centered Shifted", "Body Centered Cubic"]
    )
    Porosity_input = st.text_input("Porosity (%)", "62.5")

    try:
        Porosity_Percentage = float(Porosity_input)
        if not (0 <= Porosity_Percentage <= 100):
            st.warning("⚠️ Porosity must be between 0 and 100.")
            st.stop()
    except ValueError:
        st.warning("⚠️ Please enter a valid number for Porosity.")
        st.stop()

    Immersion_Time_Days = st.slider("Immersion Time (Days)", 0, 150, 30)
    Mechanical_Loading = st.radio("Mechanical Loading Applied?", ["Yes", "No"])

    if st.button("🔍 Predict Degradation Metrics"):
        deg_input = {
            "Scaffold_Geometry": Scaffold_Geometry,
            "Porosity_Percentage": Porosity_Percentage,
            "Immersion_Time_Days": Immersion_Time_Days,
            "Mechanical_Loading": 1 if Mechanical_Loading == "Yes" else 0
        }

        try:
            # Show loading animation
            anim_container = st.empty()
            with anim_container:
                st.image(GIF_DEGRADATION_PATH, caption="🔄 Simulating degradation process...", width=200)
                time.sleep(5)
            anim_container.empty()

            predictions = predict_degradation(deg_input)

        except Exception as e:
            st.error("❌ Prediction failed. Please check model or input values.")
            st.exception(e)
            st.stop()

        st.markdown("### Predicted Degradation Outcomes")
        st.write(f"• **Compressive Stiffness (MPa):** `{predictions['Compressive_Stiffness_MPa']:.2f}`")
        st.write(f"• **Weight Loss (%):** `{predictions['Weight_Loss_Percentage']:.2f}`")
        st.write(f"• **Water Absorption (%):** `{predictions['Water_Absorption_Percentage']:.2f}`")

        # --- Interpret Degradation Severity ---
        st.markdown("### Degradation Summary")

        weight_loss = predictions["Weight_Loss_Percentage"]
        stiffness = predictions["Compressive_Stiffness_MPa"]

        if weight_loss < 20:
            degradation_level = "🟢 **Low degradation** – Scaffold retained most of its structure."
        elif weight_loss < 40:
            degradation_level = "🟠 **Moderate degradation** – Some structural breakdown observed."
        else:
            degradation_level = "🔴 **High degradation** – Significant material loss detected."

        if stiffness < 50:
            mechanical_comment = "🔻 Mechanical stiffness has dropped notably, suggesting compromised strength."
        elif stiffness < 80:
            mechanical_comment = "⚠️ Stiffness is reduced but still within acceptable range for moderate applications."
        else:
            mechanical_comment = "✅ Mechanical integrity remains largely intact."

        st.markdown(degradation_level)
        st.markdown(mechanical_comment)





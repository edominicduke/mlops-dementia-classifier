import gradio as gr
from fastapi import FastAPI
import pandas as pd
import requests

# CLOUD API URL
CLOUD_API = "https://dementia-api-822036020986.us-central1.run.app"

# RAW FEATURE RANGES (That a User Could Hypothetically Input)
RAW_RANGES = {
    "Visit": (1, 5),
    "MR Delay": (0, 2639),
    "M/F": (0, 1),
    "Age": (60, 98),
    "EDUC": (6, 23),
    "SES": (1.0, 5.0),
    "MMSE": (4.0, 30.0),
    "CDR": (0.0, 2.0),
    "eTIV": (1106, 2004),
    "nWBV": (0.644, 0.837),
    "ASF": (0.876, 1.587),
}

FEATURE_ORDER = [
    "Visit", "MR_Delay", "M_F", "Age", "EDUC",
    "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF",
    "ABV", "CII", "CDR_RATE"
]


# Scaling helper
def minmax_scale(value, feature):
    mn, mx = RAW_RANGES[feature]
    return (value - mn) / (mx - mn)


# Compute engineered features (as done in GitHub Repo)
def compute_engineered(scaled):
    ABV = scaled["eTIV"] * scaled["nWBV"]
    CII = scaled["CDR"] - scaled["MMSE"]

    if scaled["MR Delay"] == 0:
        CDR_RATE = 0
    else:
        CDR_RATE = scaled["CDR"] / (scaled["MR Delay"] + 1e-6)

    return ABV, CII, CDR_RATE


# Build CLOUD payload (this is EXACTLY what Cloud Run expects)
def build_cloud_payload(raw):
    scaled = {
        "Visit": minmax_scale(raw["Visit"], "Visit"),
        "MR Delay": minmax_scale(raw["MR Delay"], "MR Delay"),
        "M/F": raw["M/F"],
        "Age": minmax_scale(raw["Age"], "Age"),
        "EDUC": minmax_scale(raw["EDUC"], "EDUC"),
        "SES": minmax_scale(raw["SES"], "SES"),
        "MMSE": minmax_scale(raw["MMSE"], "MMSE"),
        "CDR": minmax_scale(raw["CDR"], "CDR"),
        "eTIV": minmax_scale(raw["eTIV"], "eTIV"),
        "nWBV": minmax_scale(raw["nWBV"], "nWBV"),
        "ASF": minmax_scale(raw["ASF"], "ASF"),
    }

    ABV, CII, CDR_RATE = compute_engineered(scaled)

    return {
        "Visit": scaled["Visit"],
        "MR_Delay": scaled["MR Delay"],
        "M_F": scaled["M/F"],
        "Age": scaled["Age"],
        "EDUC": scaled["EDUC"],
        "SES": scaled["SES"],
        "MMSE": scaled["MMSE"],
        "CDR": scaled["CDR"],
        "eTIV": scaled["eTIV"],
        "nWBV": scaled["nWBV"],
        "ASF": scaled["ASF"],
        "ABV": ABV,
        "CII": CII,
        "CDR_RATE": CDR_RATE
    }


# G-CLOUD prediction only
def predict_cloud(*vals):
    keys = ["Visit", "MR Delay", "M/F", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
    raw = dict(zip(keys, vals))

    payload = build_cloud_payload(raw)

    try:
        r = requests.post(f"{CLOUD_API}/predict", json=payload, timeout=6)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {
            "diagnosis": f"Cloud API Error: {e}",
            "prob": 0.0
        }

    prob = float(data["probability"])
    pred = int(data["prediction"])

    # Diagnosis tier logic (work with ChatGPT 5.1 at 2:58 AM on 11/25/25 to generate probability logic, previously only specified for >.85)
    if prob >= 0.85:
        diagnosis = "Dementia Suspected, consult a physician"
    elif prob >= 0.65:
        diagnosis = "Possible Cognitive Impairment, follow-up recommended"
    elif prob >= 0.35:
        diagnosis = "Uncertain, results inconclusive"
    elif prob >= 0.15:
        diagnosis = "Dementia Unlikely"
    else:
        diagnosis = "Dementia Very Unlikely"

    return {
        "diagnosis": diagnosis,
        "prob": prob
    }


# Cloud Health Check
def check_cloud_health():
    try:
        r = requests.get(f"{CLOUD_API}/health", timeout=4)
        if r.status_code == 200:
            return "### Cloud API Status: **Healthy**"
        return f"### Cloud API returned {r.status_code}"
    except:
        return "### Cloud API unreachable"


# Defaults (midpoints)
DEFAULTS = {k: (mn + mx) / 2 for k, (mn, mx) in RAW_RANGES.items()}

# Gradio UI: Clean Two-Column Layout
with gr.Blocks() as demo:
    # TITLE + DESCRIPTION
    gr.Markdown("""
# **Dementia Risk Predictor**
This tool estimates dementia risk using structural brain metrics and cognitive scores.  
All predictions are computed using a secure Google Cloud Run API.
Before starting, feel free to press the button below to verify the cloud API is online!
[Cloud API Documentation](https://dementia-api-822036020986.us-central1.run.app/docs)
""")

    # HEALTH CHECK BUTTON
    health_box = gr.Markdown()
    gr.Button("Check Cloud API Health").click(
        check_cloud_health,
        outputs=health_box
    )

    gr.Markdown("---")

    # TWO COLUMNS
    with gr.Row():
        # LEFT: User inputs
        with gr.Column():
            gr.Markdown("## Patient Inputs")

            inputs = []

            # Visit number
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["Visit"],
                    DEFAULTS["Visit"],
                    step=1,
                    label="Visit Number",
                    info="The visit index for this patient (1 = first visit, 5 = fifth visit)."
                )
            )

            # MR Delay
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["MR Delay"],
                    DEFAULTS["MR Delay"],
                    label="Time Since First MRI (days)",
                    info="Number of days between baseline and this MRI session."
                )
            )

            # Sex
            inputs.append(
                gr.Dropdown(
                    [("Female", 0), ("Male", 1)],
                    value=0,
                    label="Sex",
                    info="Biological sex as recorded in the dataset."
                )
            )

            # Age
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["Age"],
                    DEFAULTS["Age"],
                    step=1,
                    label="Age (years)",
                    info="Patient age in years."
                )
            )

            # Education
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["EDUC"],
                    DEFAULTS["EDUC"],
                    step=1,
                    label="Years of Education",
                    info="Total formal education completed by the patient."
                )
            )

            # SES
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["SES"],
                    DEFAULTS["SES"],
                    step=1,
                    label="Socioeconomic Status (1–5)",
                    info="Hollingshead socioeconomic score (1 = highest, 5 = lowest)."
                )
            )

            # MMSE
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["MMSE"],
                    DEFAULTS["MMSE"],
                    step=1,
                    label="MMSE Score",
                    info="Mini-Mental State Examination score (0–30)."
                )
            )

            # CDR
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["CDR"],
                    DEFAULTS["CDR"],
                    label="Clinical Dementia Rating (0–2)",
                    info="0 = none, 0.5 = questionable, 1 = mild, 2 = moderate."
                )
            )

            # eTIV
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["eTIV"],
                    DEFAULTS["eTIV"],
                    step=1,
                    label="Estimated Total Intracranial Volume (eTIV)",
                    info="Structural MRI estimate of intracranial volume (1106–2004)."
                )
            )

            # nWBV
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["nWBV"],
                    DEFAULTS["nWBV"],
                    label="Normalized Whole Brain Volume (nWBV)",
                    info="Brain volume relative to intracranial volume (0.64–0.84)."
                )
            )

            # ASF
            inputs.append(
                gr.Slider(
                    *RAW_RANGES["ASF"],
                    DEFAULTS["ASF"],
                    label="Atlas Scaling Factor (ASF)",
                    info="MRI normalization factor for atlas alignment (0.87–1.58)."
                )
            )

            predict_btn = gr.Button("Predict")

        # RIGHT COLUMN: Output
        with gr.Column():
            gr.Markdown("## Prediction Output")

            diagnosis_box = gr.Markdown()
            probability_box = gr.Markdown()


    # Connect prediction
    def format_output(result):
        """result = {'diagnosis': str, 'prob': float}"""
        return result["diagnosis"], f"**Probability:** `{result['prob']:.3f}`"


    predict_btn.click(
        lambda *vals: format_output(predict_cloud(*vals)),
        inputs=inputs,
        outputs=[diagnosis_box, probability_box]
    )


def format_output(result):
    # Reverted to Old Output Mechanism -- MotPlot Lib proved too unruly
    diagnosis = result["diagnosis"]
    prob = result["prob"]

    prob_text = f"**Probability:** `{prob:.3f}`"

    return diagnosis, prob_text


app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

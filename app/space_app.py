import gradio as gr
import requests

API_URL = "https://dtanzillo-dementia-ml.hf.space/api/predict"

def predict_fn(Visit, MR_Delay, M_F, Age, EDUC, SES, MMSE, CDR,
               eTIV, nWBV, ASF, ABV, CII, CDR_RATE):

    payload = {
        "Visit": Visit,
        "MR_Delay": MR_Delay,
        "M_F": M_F,
        "Age": Age,
        "EDUC": EDUC,
        "SES": SES,
        "MMSE": MMSE,
        "CDR": CDR,
        "eTIV": eTIV,
        "nWBV": nWBV,
        "ASF": ASF,
        "ABV": ABV,
        "CII": CII,
        "CDR_RATE": CDR_RATE
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
    except Exception as e:
        return f" API Error: {e}"

    if response.status_code != 200:
        return f" API Error: {response.text}"

    data = response.json()

    pred_class = data.get("prediction", None)
    proba = data.get("probability", None)

    if pred_class is None:
        return " API Error: Invalid response"

    # Human-readable output
    if pred_class == 1:
        label = "Dementia Likely"
    else:
        label = "No Dementia Detected"

    if proba is not None:
        proba_text = f"Confidence: **{proba:.2f}**"
        return f"{label}\n\n{proba_text}"

    return label

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Dementia Risk Predictor  
        Adjust clinical + imaging features to estimate dementia likelihood.
        """
    )

    # Default feature vector from data:
    # 1,0.0,0.0,1,0.736842105263158,0.11764705882352944,0.75,
    # 0.8076923076923077,0.25,0.6069042316258353,0.08290155440414493,
    # 0.26300984528832627,0.05031330417623496,-0.5576923076923077,0.0

    with gr.Row():
        Visit = gr.Slider(0, 1, value=0.0, label="Visit")
        MR_Delay = gr.Slider(0, 1, value=0.0, label="MR Delay")

    with gr.Row():
        M_F = gr.Dropdown(
            choices=[("Male", 0), ("Female", 1)],
            value=0,
            label="Sex"
        )
        Age = gr.Slider(0, 1, value=1.0, label="Age (scaled)")

    with gr.Row():
        EDUC = gr.Slider(0, 1, value=0.736842105263158, label="Education")
        SES = gr.Slider(0, 1, value=0.11764705882352944, label="SES")

    with gr.Row():
        MMSE = gr.Slider(0, 1, value=0.75, label="MMSE")
        CDR = gr.Dropdown(
            label="CDR (Clinical Dementia Rating)",
            choices=[
                ("0 – No impairment", 0.0),
                ("0.25 – Questionable", 0.25),
                ("0.5 – Mild", 0.5),
                ("0.75 – Moderate", 0.75),
                ("1.0 – Severe", 1.0)
            ],
            value=0.25
        )

    with gr.Row():
        eTIV = gr.Slider(0, 1, value=0.6069042316258353, label="eTIV")
        nWBV = gr.Sli
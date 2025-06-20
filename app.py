import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# === Define available models ===
DistilBERT_path = "models\\nostalgia-distilbert-base-uncased"
BERT_path = "models\\nostalgia-google-bert-base-uncased"

MODEL_OPTIONS = {
    "DistilBERT (Nostalgia)": DistilBERT_path,
    "BERT (Nostalgia)": BERT_path,
    # Add more model paths or HF repo IDs here
}

@st.cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

st.title("Detect Nostalgia 📻")

# Model selection
selected_model_name = st.selectbox("Select Model:", list(MODEL_OPTIONS.keys()))
st.markdown("\n \n")
model_path = MODEL_OPTIONS[selected_model_name]

tokenizer, model = load_model_and_tokenizer(model_path)

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🔤Single Text Input")
    text_input = st.text_area("Enter text and get prediction:", "I began to feel love as we talked about our school days", height=133)

with col2:
    st.markdown("##### 📁CSV Upload")
    uploaded_file = st.file_uploader("Upload a CSV with a `text` column", type=["csv"])

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {i: f"Class {i}" for i in range(len(probs))}
    pred_label = id2label[pred_idx]
    return pred_label, confidence, probs.tolist(), id2label

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        if st.button("Classify CSV"):
            preds = []
            confidences = []

            # Optional: progress bar
            progress_bar = st.progress(0)
            for i, row in df.iterrows():
                pred_label, confidence, _, _ = predict_text(row["text"])
                preds.append(pred_label)
                confidences.append(confidence)
                progress_bar.progress((i + 1) / len(df))

            df["predicted_class"] = preds
            df["confidence"] = confidences

            st.success("Classification done!")
            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
            )

else:
    if text_input.strip() != "":
        if st.button("Classify Text"):
            pred_label, confidence, probs, id2label = predict_text(text_input)
            st.markdown("\n \n")

            st.subheader(f"Prediction using {selected_model_name}")
            st.markdown(f"<b>Predicted Class:</b> <span style='color:blue'><b>{pred_label}</b></span>", unsafe_allow_html=True)
            st.markdown(f"<b>Confidence:</b> <span style='color:blue'> <b>{confidence:.4f}</b></span>", unsafe_allow_html=True)

            # Plot horizontal bar chart
            labels = [id2label[i] for i in range(len(probs))]
            scores = probs
            pred_idx = list(id2label.keys())[list(id2label.values()).index(pred_label)]

            fig, ax = plt.subplots(figsize=(6, len(scores) * 0.6))
            colors = ['green' if i == pred_idx else 'red' for i in range(len(scores))]

            ax.barh(labels, scores, color=colors)
            ax.set_xlabel("Confidence")
            ax.set_title("Confidence Scores (All Classes)")
            ax.set_xlim([0, 1.0])
            ax.invert_yaxis()

            for i, v in enumerate(scores):
                ax.text(v + 0.01, i, f"{v:.2f}", va='center')

            st.pyplot(fig)

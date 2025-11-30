import streamlit as st
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
from get_context_from_api import infer_context_from_trends
import os
import gdown
import zipfile
from infer import load_model as load_classifier_model, classify


API_KEY = "bbc5ace3aed5f02bcbd6affac66a496ed78fb1dd31f96cdbfffdcabf39bb0d0a"

# Google Drive file ID and paths
file_id = "15p_VKppO-VDLXHv6tspumyfRLCIqP_yN"
zip_path = "t5-query-rewriter-final.zip"
model_folder = "t5-query-rewriter-final"

# Download if the folder doesn't exist
if not os.path.exists(model_folder):
    if not os.path.exists(zip_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)
    # Unzip the model
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")


tokenizer = T5Tokenizer.from_pretrained(model_folder)
model = T5ForConditionalGeneration.from_pretrained(model_folder)

# load weak/strong classifier
clf_tokenizer, clf_model, clf_threshold = load_classifier_model()

def sanitize(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def make_keywords_block(context_field, keep_top=8):
    if isinstance(context_field, list):
        kws = [sanitize(str(x)) for x in context_field if str(x).strip()]
    elif isinstance(context_field, str) and context_field.strip():
        kws = [sanitize(context_field)]
    else:
        kws = []
    if keep_top is not None:
        kws = kws[:keep_top]
    return "[" + ", ".join(kws) + "]"

def build_input(question, context_field, keep_top_keywords=8, strong=False):
    header = (
        "Instruction: Rewrite the question to be stand-alone. Ensure all HIGH-PRIORITY keywords appear if natural."
        if strong else
        "Instruction: Rewrite the question to be stand-alone. Prefer to include the given keywords if they are relevant."
    )
    kw_title = "HIGH-PRIORITY KEYWORDS:" if strong else "Keywords:"
    kw_block = make_keywords_block(context_field, keep_top_keywords)
    
    return "\n".join([
        header, "",
        kw_title, kw_block, "",
        "Question:", sanitize(question)
    ])

st.title("Query Rewriter")

question = st.text_input("Enter your question:")

# Slider to choose number of rewrites
num_rewrites = st.slider("Number of rewrites", min_value=1, max_value=5, value=3)

if question:
    # 1. Classify query as WEAK / STRONG
    clf_result = classify(question, clf_tokenizer, clf_model, clf_threshold)
    is_weak = (clf_result["label"] == "WEAK")

    # Show label on the screen
    if is_weak:
        st.markdown(
            """
            <div style="
                padding:0.75rem 1rem;
                border-radius:0.6rem;
                border:1px solid #ff6b6b;
                background:rgba(255, 107, 107, 0.1);
                margin-top:0.75rem;
                margin-bottom:0.75rem;
            ">
                <span style="font-weight:700; color:#ff6b6b;">🔴 Weak query detected</span><br/>
                <span style="font-size:0.9rem; color:#f5f5f5;">
                    Your question is a bit broad or underspecified, so I’m using related search trends
                    to add context and generate more focused rewrites.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                padding:0.75rem 1rem;
                border-radius:0.6rem;
                border:1px solid #4CAF50;
                background:rgba(76, 175, 80, 0.08);
                margin-top:0.75rem;
                margin-bottom:0.75rem;
            ">
                <span style="font-weight:700; color:#4CAF50;">🟢 Strong query detected</span><br/>
                <span style="font-size:0.9rem; color:#f5f5f5;">
                    Your question is already clear and specific, so I’m keeping it as is
                    and skipping the extra trend-based reformulation.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # 2. If WEAK → do existing Trends + rewrite flow
    if is_weak:
        context_result = infer_context_from_trends(question, API_KEY)
        context_keywords = context_result["top_related"]

        model_input = build_input(question, context_keywords)

        inputs = tokenizer(model_input, return_tensors="pt")

        # Generate rewrites (same as before)
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=max(5, num_rewrites),  # beams >= num_return_sequences
            num_return_sequences=num_rewrites,
            early_stopping=True
        )

        st.subheader(f"Top {num_rewrites} Rewritten Questions:")
        for i, output in enumerate(outputs):
            rewritten_question = tokenizer.decode(output, skip_special_tokens=True)
            st.write(f"{i+1}. {rewritten_question}")

    # 3. If STRONG → only print label, no rewrites
    # (do nothing else here)

from __future__ import annotations

from pathlib import Path

import streamlit as st

from cashiervision.plu import format_plu_suggestions, load_plu_mapping, suggest_plu_codes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = PROJECT_ROOT / "data" / "plu_mapping.csv"


st.set_page_config(page_title="CashierVision", page_icon="CV", layout="centered")
st.title("CashierVision")
st.caption("Cashier-assist produce recognition and PLU suggestion prototype.")

mapping = load_plu_mapping(MAPPING_PATH)
classes = sorted(mapping)

uploaded_image = st.file_uploader("Upload a produce image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded produce image", use_container_width=True)
    st.info("Model inference will be connected after the baseline classifier is trained.")

selected_class = st.selectbox("Demo PLU lookup", classes)
records = suggest_plu_codes(selected_class, mapping)

st.subheader("Possible PLU Suggestions")
for suggestion in format_plu_suggestions(records):
    st.write(suggestion)

st.warning(
    "PLU codes are examples only. Retailer rules, variety, size, and organic status can change the correct code."
)


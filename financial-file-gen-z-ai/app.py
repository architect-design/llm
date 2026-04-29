import streamlit as st
import json
import pandas as pd
from core.ach_handler import ACHFileHandler
from core.vcf_handler import VCFFileHandler
from core.db_connector import DatabaseConnector
from models.inference import inference_engine

# --- Page Config ---
st.set_page_config(
    page_title="FinGen Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Industry Standard Look (White BG, Red Buttons) ---
st.markdown("""
<style>
    /* Main Background */
    .main {
        background-color: #FFFFFF;
    }

    /* Remove default Streamlit padding for cleaner look */
    div.block-container {
        padding-top: 2rem;
    }

    /* Red Buttons Styling */
    div.stButton > button:first-child {
        background-color: #DC143C; /* Crimson Red */
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover {
        background-color: #B22222; /* Darker Red on hover */
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        color: white;
    }
    div.stButton > button:active {
        background-color: #8B0000;
        color: white;
    }

    /* Secondary Buttons (Download) */
    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #DC143C !important;
        border: 1px solid #DC143C !important;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background-color: #FFF0F3 !important;
        color: #B22222 !important;
        border-color: #B22222 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #F0F2F6;
    }
    .stTabs [data-baseweb="tab"] {
        color: #4A5568;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 1rem 2rem;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #DC143C;
        border-bottom: 3px solid #DC143C;
    }

    /* Headers */
    h1 {
        color: #1A202C;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #2D3748;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E2E8F0;
    }

    /* Widgets */
    .stTextInput>div>div>input, .stTextArea textarea {
        border: 1px solid #E2E8F0;
        border-radius: 4px;
    }
    .stTextInput>div>div>input:focus, .stTextArea textarea:focus {
        border-color: #DC143C;
        box-shadow: 0 0 0 1px #DC143C;
    }

    /* File Uploader */
    section[data-testid="stFileUploader"] {
        border: 2px dashed #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        background-color: #FAFAFA;
    }

    /* Success/Error boxes */
    .element-container .stSuccess, .element-container .stError, .element-container .stInfo {
        border-radius: 4px;
        border-left-width: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🏦 FinGen Studio")
st.markdown("### Financial File Generator & Validator")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    # FIXED: Use markdown instead of st.image for the badge
    st.markdown("<img src='https://img.shields.io/badge/Version-1.0.0-red' alt='Version'>", unsafe_allow_html=True)

    st.markdown("### How it works")
    st.markdown(
        """
        <div style='font-size: 0.9rem; color: #4A5568;'>
        <p><strong>Hybrid AI Approach:</strong></p>
        <ol>
            <li><strong>Brain:</strong> SLM generates structured JSON data.</li>
            <li><strong>Engine:</strong> Python formats it to NACHA/VCF standards.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("<p style='font-size: 0.8rem; color: #718096;'>© 2024 FinGen Systems</p>", unsafe_allow_html=True)

# --- Main Tabs ---
tab1, tab2 = st.tabs(["📝 Generate Files", "🔍 Validate Files"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Option A: AI Generation")

        file_type = st.selectbox(
            "Select File Type",
            ["ACH (NACHA)", "VCF (Visa)"],
            key="gen_type",
            help="Choose the financial standard for the output file."
        )

        prompt = st.text_area(
            "Enter Natural Language Prompt",
            "Generate 5 payroll credit transactions for IT department.",
            height=100,
            placeholder="e.g., Generate 3 debit transactions for vendor X..."
        )

        btn_col, _ = st.columns([1, 2])
        with btn_col:
            generate_btn = st.button("Generate via AI", key="gen_ai_btn")

        if generate_btn:
            with st.spinner("🤖 SLM is generating transaction data..."):
                # 1. SLM generates JSON
                generated_data = inference_engine.generate_transactions(prompt)

                # 2. Formatter creates strict file
                if "ACH" in file_type:
                    file_content = ACHFileHandler.generate_ach_file(generated_data)
                    file_ext = "ach"
                else:
                    for d in generated_data:
                        d['card_number'] = "4111111111111111"
                        d['merchant_code'] = "AI_GEN"
                    file_content = VCFFileHandler.generate_vcf_file(generated_data)
                    file_ext = "vcf"

                st.session_state['generated_content'] = file_content
                st.session_state['file_ext'] = file_ext
                st.success(f"✅ Generated {len(generated_data)} records successfully.")

    with col2:
        st.subheader("Option B: Database Integration")

        db_query = st.text_input(
            "SQL Query",
            "SELECT * FROM payments WHERE status = 'pending'",
            placeholder="SELECT routing, account, amount, name FROM..."
        )

        btn_col, _ = st.columns([1, 2])
        with btn_col:
            db_btn = st.button("Pull from Database", key="gen_db_btn")

        if db_btn:
            db = DatabaseConnector()
            try:
                real_data = db.fetch_transaction_data(db_query)
                if real_data:
                    file_content = ACHFileHandler.generate_ach_file(real_data)
                    st.session_state['generated_content'] = file_content
                    st.session_state['file_ext'] = "ach"
                    st.success(f"✅ Fetched {len(real_data)} records from DB.")
                else:
                    st.warning("⚠️ No data found or connection failed.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Display Output Preview
    if 'generated_content' in st.session_state:
        st.markdown("---")
        st.subheader("📄 Output Preview")

        # Code block with light theme styling via markdown wrapper
        st.code(st.session_state['generated_content'], language='plaintext')

        st.download_button(
            label="⬇️ Download Generated File",
            data=st.session_state['generated_content'],
            file_name=f"generated.{st.session_state.get('file_ext', 'txt')}",
            mime="text/plain"
        )

with tab2:
    st.header("Validate Existing File")

    uploaded_file = st.file_uploader(
        "Upload ACH or VCF file",
        type=['ach', 'txt', 'vcf'],
        help="Supported formats: .ach, .txt, .vcf"
    )

    if uploaded_file:
        content = uploaded_file.read().decode('utf-8')

        # Detect Type
        is_ach = content.strip().startswith('1') or content.strip().startswith('10')

        st.write("")  # Spacing

        if is_ach:
            st.info("ℹ️ Detected: **ACH/NACHA File**")
            results = ACHFileHandler.validate_ach(content)
        else:
            st.info("ℹ️ Detected: **VCF/Other File**")
            results = VCFFileHandler.validate_vcf(content)

        st.subheader("Validation Results")

        if "Success" in results[0]:
            st.success(f"✅ {results[0]}")
        else:
            st.error("❌ Validation Failed!")
            for err in results:
                st.write(f"🔴 {err}")

        # Optional: Parse Data
        if is_ach:
            st.markdown("### Parsed Transaction Details")
            with st.expander("View Extracted Data"):
                parsed_data = ACHFileHandler.parse_ach_to_data(content)
                if parsed_data:
                    df = pd.DataFrame(parsed_data)
                    # Style the dataframe for cleaner look
                    st.dataframe(df.style.background_index(axis=0))
                else:
                    st.write("No transaction details found in file.")
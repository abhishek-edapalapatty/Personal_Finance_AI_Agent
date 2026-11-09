import os

import streamlit as st
from dotenv import load_dotenv
from finance_agent import build_agent

load_dotenv()
st.set_page_config(page_title="Personal Finance AI Assistant", page_icon="💰", layout="centered")
st.title("Personal Finance AI Assistant")
st.caption("Ask for a calculation or a starting budget. Calculations are performed by dedicated tools.")
st.info("Estimates are educational and are not personalized financial advice. Investment returns are hypothetical and not guaranteed.")

provider = os.getenv("FINANCE_LLM_PROVIDER", "openai").strip().lower()
if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
    st.warning("Add your OpenAI API key to a local .env file, then restart the app.")
    st.stop()
if provider == "ollama":
    st.caption(f"Local model: {os.getenv('OLLAMA_MODEL', 'qwen3:4b')} (served by Ollama on this computer).")

@st.cache_resource
def get_agent():
    return build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Example questions")
    examples = [
        "Calculate EMI for a ₹10 lakh loan at 8% annual interest for 20 years",
        "If I invest ₹7,000 monthly for 15 years at 10% annual return, what is the estimated value?",
        "Create a monthly budget plan for income ₹90,000",
        "What is 20% of ₹85,000?",
        "How much will ₹3,000 per month grow to over 5 years at 8% annual return?",
    ]
    for item in examples:
        st.write(f"• {item}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a personal finance calculation")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Choosing a calculation tool…"):
            try:
                result = get_agent().invoke({"input": query})["output"]
                response = str(result)
            except ModuleNotFoundError as exc:
                if exc.name == "langchain_ollama":
                    response = (
                        "The LangChain Ollama package is missing. From the project folder, "
                        "run `python -m pip install -r requirements.txt`, then restart Streamlit."
                    )
                else:
                    response = f"A required Python package is missing: {exc.name}. Install the project requirements and restart Streamlit."
            except Exception as exc:
                if provider == "ollama":
                    response = f"I could not reach or run the local Ollama model. Confirm Ollama is open and `{os.getenv('OLLAMA_MODEL', 'qwen3:4b')}` is downloaded, then try again. Details: {exc}"
                else:
                    response = f"I could not complete that request. Check your API key and connection, then try again. Details: {exc}"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Personal Finance AI Agent

A Streamlit personal finance assistant built with LangChain and GPT-4o mini. The LLM selects among four deterministic tools: a restricted arithmetic calculator, loan EMI, SIP future value, and 50/30/20 budget allocation.

## Run locally

1. Use Python 3.10 or newer and create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenAI API key. Keep `.env` private; it is excluded from version control.
4. Start the app: `streamlit run app.py`

The first request needs internet access to the OpenAI API. The key is read from the local environment and is not included in this project.

## Demonstration queries and formula outputs

The app includes all five required examples in its sidebar. Expected calculation results are:

| Query | Selected tool | Expected result |
| --- | --- | --- |
| EMI for ₹10 lakh at 8% annual interest for 20 years | EMI calculator | Monthly EMI ₹8,364.40; total payments ₹20,07,456.17 |
| ₹7,000 monthly for 15 years at 10% annual return | SIP calculator | Estimated future value ₹29,01,292.42; contributions ₹12,60,000 |
| Budget for ₹90,000 monthly income | Budget planner | Needs ₹45,000; wants ₹27,000; savings/debt ₹18,000 |
| 20% of ₹85,000 | Calculator | ₹17,000 |
| ₹3,000 monthly for 5 years at 8% annual return | SIP calculator | Estimated future value ₹2,20,430.57; contributions ₹1,80,000 |

SIP outputs assume monthly compounding with deposits at month end and a constant return; they are estimates, not forecasts or guarantees. EMI assumes a fixed nominal annual rate divided by 12 and excludes fees. Budget allocation applies the 50/30/20 guideline to the supplied income.

## Project files

- `app.py`: Streamlit interface and sample prompts.
- `finance_agent.py`: restricted calculator, finance tools, tool-calling agent, and prompt.
- `requirements.txt`: pinned runtime dependencies.
- `Model_Development_Document.docx`: model and system design, assumptions, evaluation approach, and limitations.

No dataset or trained predictive model is required by this assignment. The LLM performs intent interpretation and tool selection; numerical outputs come from explicit formulas in the tools.

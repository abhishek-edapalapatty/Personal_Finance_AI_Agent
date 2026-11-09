from __future__ import annotations

import ast
import math
import operator
import os
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# Deliberately restricted arithmetic evaluator. It never evaluates arbitrary Python.
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _calculate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _calculate(node.left), _calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent magnitude must be 100 or less.")
        return float(_BINARY[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return float(_UNARY[type(node.op)](_calculate(node.operand)))
    raise ValueError("Use only numbers and +, -, *, /, //, %, **, and parentheses.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression using numbers and arithmetic operators."""
    try:
        value = _calculate(ast.parse(expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Could not calculate expression: {exc}") from exc
    if not math.isfinite(value):
        raise ValueError("Result is not finite.")
    return f"{expression} = {value:,.2f}"


@tool
def emi_calculator(principal: float, annual_rate_percent: float, years: float) -> str:
    """Calculate monthly loan EMI. Inputs: principal in currency units, annual rate in percent, and term in years."""
    if principal <= 0 or years <= 0 or annual_rate_percent < 0:
        raise ValueError("Principal and years must be positive; annual rate cannot be negative.")
    months = int(round(years * 12))
    if months <= 0:
        raise ValueError("Loan term must be at least one month.")
    monthly_rate = annual_rate_percent / 100 / 12
    if monthly_rate == 0:
        emi = principal / months
    else:
        factor = (1 + monthly_rate) ** months
        emi = principal * monthly_rate * factor / (factor - 1)
    total = emi * months
    return (f"Monthly EMI: {emi:,.2f}; term: {months} months; "
            f"total of payments: {total:,.2f}; total interest: {total-principal:,.2f}.")


@tool
def sip_calculator(monthly_investment: float, annual_return_percent: float, years: float) -> str:
    """Estimate future value of monthly investments, assuming a constant annual return compounded monthly and deposits at month end."""
    if monthly_investment < 0 or years <= 0 or annual_return_percent < 0:
        raise ValueError("Investment cannot be negative; years must be positive; return cannot be negative.")
    months = int(round(years * 12))
    if months <= 0:
        raise ValueError("Investment term must be at least one month.")
    monthly_rate = annual_return_percent / 100 / 12
    future_value = monthly_investment * months if monthly_rate == 0 else monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    invested = monthly_investment * months
    return (f"Estimated future value: {future_value:,.2f}; total contributions: {invested:,.2f}; "
            f"estimated growth: {future_value-invested:,.2f}; period: {months} months.")


@tool
def budget_planner(monthly_income: float) -> str:
    """Allocate monthly after-tax income using the 50/30/20 rule: needs, wants, and savings."""
    if monthly_income < 0:
        raise ValueError("Monthly income cannot be negative.")
    return (f"Monthly income: {monthly_income:,.2f}. Needs (50%): {monthly_income*0.50:,.2f}; "
            f"wants (30%): {monthly_income*0.30:,.2f}; savings and debt repayment (20%): {monthly_income*0.20:,.2f}. "
            "Treat this as a starting point and adjust for essential costs and goals.")


TOOLS = [calculator, emi_calculator, sip_calculator, budget_planner]
SYSTEM_INSTRUCTIONS = """You are a careful personal finance calculation assistant. Use the available tools for every numerical finance calculation and show the assumptions and units. Choose the relevant tool from the user's request; combine tools only when useful. For EMI, interpret annual interest as a nominal annual percentage divided by 12 and term in years converted to months. For SIP, use monthly compounding with deposits at the end of each month and state that returns are hypothetical and not guaranteed. For budgets, apply 50/30/20 to the income the user supplies and identify it as a guideline. Ask a concise clarification if a required input is missing or ambiguous. Do not present personalized investment, tax, or legal advice, and do not invent market returns. Round displayed currency to two decimals."""


def build_agent() -> AgentExecutor:
    """Construct the tool-calling agent using OpenAI or a local Ollama model."""
    provider = os.getenv("FINANCE_LLM_PROVIDER", "openai").strip().lower()
    if provider == "ollama":
        # Import on demand so OpenAI-only users do not depend on Ollama being installed.
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            temperature=0,
        )
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY or choose FINANCE_LLM_PROVIDER=ollama.")
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise ValueError("FINANCE_LLM_PROVIDER must be 'openai' or 'ollama'.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(model, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=False, handle_parsing_errors=True)


def answer(query: str) -> str:
    """Run a single user query through the agent."""
    return str(build_agent().invoke({"input": query})["output"])

# 📘 Account Plan Agent

*AI-driven Research System*

------------------------------------------------------------------------

## ⭐️ Overview

The **Account Plan Agent** is an intelligent, multi-agent workflow
application that automatically gathers financial, market, and leadership
insights about a company, detects conflicting information, and generates
a polished **Account Plan in Markdown format**.

It uses **parallel research nodes**, **human-in-the-loop review**,
**automated conflict detection**, and **stateful execution** using
**LangGraph + SQLite checkpointing**.

This system is ideal for: - Sales teams preparing client briefings\
- Consulting workflows\
- Competitive research\
- Automated report generation\
- Enterprise intelligence systems

------------------------------------------------------------------------

## 🧠 Key Features

### ✅ **1. Parallel Research Execution**

Three nodes run simultaneously: - Financial Analyst\
- Market Strategist\
- Leadership Scout

This reduces processing time and improves coverage.

### ✅ **2. Conflict Detection**

If any research contains conflicting or inconsistent data, a
`"CONFLICT_DETECTED"` flag triggers a **Human Review Gate**.

### ✅ **3. Human-in-the-loop workflow**

The system pauses and waits for: - Additional instructions\
- Clarifications\
- Prioritization guidance

### ✅ **4. User Input Classification**

Every input is classified as: - **Efficient**\
- **Confused**\
- **Chatty**\
- **Edge Case**

This influences how results are produced.

### ✅ **5. State Machine Using LangGraph**

The entire research and review pipeline is modeled as a graph with: -
Nodes\
- Conditional edges\
- Interrupt points\
- Persistent state

### ✅ **6. Checkpointing with SQLite**

Allows **pause → review → resume** without losing state.

### ✅ **7. Professional Account Plan Writer**

After research is complete, the Writer Node produces: - Executive
Summary\
- Financial Overview\
- Market Position\
- Leadership Overview\
- Strategic Recommendations

------------------------------------------------------------------------

## 🏗️ Architecture

### **System Flow**

    User Input
          ↓
    User-Type Detector
          ↓
    Streamlit UI
          ↓
    LangGraph Workflow
          ├── Financial Node
          ├── Market Node
          ├── People Node
          ↓ (Parallel Execution)
    Join Node
          ↓
    Review Gate
          ├── Human Review Node (if conflicts)
          └── Writer Node
          ↓
    Final Account Plan

### **Node Descriptions**

  -----------------------------------------------------------------------
  Node                   Description
  ---------------------- ------------------------------------------------
  **Supervisor Node**    Decides which research nodes must run

  **Financial Node**     Uses Tavily + LLM to gather revenue/net
                         income/debt

  **Market Node**        Analyzes market share & competitors

  **People Node**        Extracts CEO, C-suite, layoffs

  **Join Node**          Merges research results

  **Review Gate**        Checks for conflict warnings

  **Human Review Node**  Pauses workflow for user instructions

  **Writer Node**        Produces the final Account Plan
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🗄️ Project Workflow (with explanations)

### 🔹 Step 1: User enters company name

The text is analyzed by the **user-type detector**.

### 🔹 Step 2: LangGraph workflow starts

Each research node runs **in parallel**.

### 🔹 Step 3: Conflicts detected

If any source contradicts another:\
➡ The system stops\
➡ Waits for user input\
➡ Uses SQLite checkpointing to preserve state

### 🔹 Step 4: User guidance processed

User clarifications are converted into `HumanMessage` and appended to
the state.

### 🔹 Step 5: Workflow resumes and writes final plan

The plan is created using a structured markdown template.

------------------------------------------------------------------------

## 🧰 Tech Stack

  Layer             Technology
  ----------------- -------------------------
  UI                **Streamlit**
  LLM               **OpenAI GPT-5-nano**
  Web Search        **TavilySearchResults**
  Workflow Engine   **LangGraph**
  State Storage     **SQLite (in-memory)**
  Backend           Python 3.10+

------------------------------------------------------------------------

## ⚙️ Installation

### **1. Clone the repository**

``` bash
git clone https://github.com/your-repo/account-plan-agent.git
cd account-plan-agent
```

### **2. Install dependencies**

``` bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**

``` bash
streamlit run streamlit_app.py
```


## 🖥️ Usage Guide

### **1. Open the app in browser**

    http://localhost:8501

### **2. Enter a query**

Example:

    Oracle Corporation

### **3. Research runs automatically**

You will see:

    Finished: financial_node
    Finished: market_node
    Finished: people_node

### **4. If conflicts appear -\> Human Review**

You will be asked to provide guidance such as:

    Prioritize SEC filings over news articles.

### **5. Click "Resume with this guidance"**

LangGraph resumes from last checkpoint.

### **6. Get the final Account Plan**

Formatted in Markdown.

------------------------------------------------------------------------

## 📂 Folder Structure

    .
    ├── main.py        # Main application
    ├── requirements.txt        # Dependencies
    ├── README.md               # This file

------------------------------------------------------------------------

## 🛠️ Troubleshooting

### ❗️ Conflict warnings appear even when no conflict?

This can happen when: - Tavily returns ambiguous data\
- Two sources provide different numerical values

### ❗️ Workflow doesn't resume?

Make sure: - Session state `config` is not `None`\
- You didn't refresh Streamlit during execution

### ❗️ Getting API key errors?

Check: - Keys are exported\
- Capitalization is correct\
- You restarted the terminal

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   🔍 Citation tracking\
-   🧩 Knowledge graph integration\
-   📊 Full PDF report export\
-   🔐 Multi-user authentication\
-   📨 Email delivery for account plans\
-   🏢 CRM (Salesforce/HubSpot) integration


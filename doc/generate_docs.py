"""
generate_docs.py

A utility script that auto-generates a rich DOCX documentation file for the
Semantic Halting Problem (SHP) codebase, written in beginner-friendly language.
Run this whenever you want to produce or refresh the documentation:
    python3 doc/generate_docs.py
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    run = h.runs[0]
    if level == 1:
        run.font.color.rgb = RGBColor(0x1A, 0x73, 0xE8)
    elif level == 2:
        run.font.color.rgb = RGBColor(0x18, 0x45, 0x9B)
    elif level == 3:
        run.font.color.rgb = RGBColor(0x2E, 0x7D, 0x32)
    return h

def add_code_block(doc, code_text):
    """Adds a styled mono-spaced code block to the document."""
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.3)
    # Shade background
    pPr = para._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F3F4F6")
    pPr.append(shd)
    run = para.add_run(code_text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0xC7, 0x25, 0x4E)
    return para

def add_info_box(doc, label, text):
    """Adds a 💡 Tip or ℹ️ Note paragraph."""
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.2)
    r_label = para.add_run(f"{label}  ")
    r_label.bold = True
    r_label.font.color.rgb = RGBColor(0x00, 0x70, 0xC0)
    r_text = para.add_run(text)
    r_text.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return para

def build_doc():
    doc = Document()

    # ────────────────────────────────────────────────
    # Cover Page
    # ────────────────────────────────────────────────
    title = doc.add_heading("Semantic Halting Problem (SHP)", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.runs[0].font.color.rgb = RGBColor(0x1A, 0x73, 0xE8)

    sub = doc.add_paragraph("A Beginner's Complete Guide to the Codebase")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].bold = True
    sub.runs[0].font.size = Pt(14)

    doc.add_paragraph("")
    intro = doc.add_paragraph(
        "This document walks you through every file, class, and function in the SHP project "
        "in plain English. No prior experience in AI or machine learning is required. "
        "By the end, you will understand what the project does, why each file exists, "
        "and how all the pieces connect together."
    )
    intro.runs[0].font.size = Pt(11)

    doc.add_page_break()

    # ────────────────────────────────────────────────
    # Section 1 – What is SHP?
    # ────────────────────────────────────────────────
    add_heading(doc, "1.  What is the Semantic Halting Problem?", level=1)
    doc.add_paragraph(
        "Imagine you hire two people to write a report: a Writer and a Critic. "
        "The Writer writes a draft, the Critic reads it and gives feedback, and the Writer "
        "revises — over and over. In a perfect world they'd finish the report and stop. "
        "But what if the Critic keeps asking for tiny, unimportant changes forever? "
        "They never stop. They are stuck in an infinite loop."
    )
    doc.add_paragraph(
        "The Semantic Halting Problem (SHP) asks: how can a computer know when to stop this loop? "
        "The answer this project uses is maths. We convert each draft into a list of numbers "
        "(called an embedding), then measure the distance between drafts. "
        "When consecutive drafts are no longer meaningfully different, we force a stop."
    )
    add_info_box(doc, "💡 Key Idea:", "Distance ≈ 0 means 'no new information is being added'. That is our signal to stop.")

    doc.add_paragraph("")
    add_heading(doc, "2.  Project Architecture at a Glance", level=1)
    doc.add_paragraph(
        "The project is made up of five Python files, each with a clear, single responsibility:"
    )
    rows = [
        ("main.py", "The master switch. Run this file to trigger the complete pipeline."),
        ("agent_workflow.py", "Builds and runs the LangGraph multi-agent loop (Writer → Critic → halt)."),
        ("agents.py", "Defines the mock Writer and Critic agent logic."),
        ("semantic_entropy.py", "The maths engine. Converts text to vectors and measures their distance."),
        ("ragas_eval.py", "Grades the final draft using four standard RAG quality metrics via Hugging Face."),
        ("optimize_score.py", "Uses ML to find the best way to combine the four grades into one final score."),
    ]
    table = doc.add_table(rows=1 + len(rows), cols=2)
    table.style = "Light List Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text = "File"
    hdr[1].text = "What it does"
    for file_name, description in rows:
        row_cells = table.add_row().cells
        row_cells[0].text = file_name
        row_cells[1].text = description

    doc.add_page_break()

    # ────────────────────────────────────────────────
    # Section 3 – File-by-File Breakdown
    # ────────────────────────────────────────────────
    add_heading(doc, "3.  File-by-File Breakdown", level=1)

    # ── 3.1 main.py ──────────────────────────────────
    add_heading(doc, "3.1  main.py  (The Orchestrator)", level=2)
    doc.add_paragraph(
        "This file is the single entry-point for the entire project. "
        "Instead of running three separate commands, running main.py executes all three scripts "
        "one after another in the correct order."
    )
    add_heading(doc, "How it works", level=3)
    doc.add_paragraph(
        "It uses Python's built-in subprocess module to launch each script as a separate process, "
        "exactly like you would from the terminal. If any step fails, the pipeline prints an error and stops immediately."
    )
    add_heading(doc, "Key function: run_script(script_name, step_description)", level=3)
    doc.add_paragraph(
        "Accepts the filename of a script and a human-readable label. "
        "It prints a colourful header, then executes the script using the same Python interpreter "
        "that is running inside your virtual environment."
    )
    add_code_block(doc,
        "# How to run the full pipeline:\n"
        "python3 main.py\n\n"
        "# What happens internally:\n"
        "# 1. Runs agent_workflow.py\n"
        "# 2. Runs ragas_eval.py\n"
        "# 3. Runs optimize_score.py"
    )

    doc.add_paragraph("")

    # ── 3.2 agents.py ────────────────────────────────
    add_heading(doc, "3.2  agents.py  (The Mock Agents)", level=2)
    doc.add_paragraph(
        "This file contains two functions, one for the Writer and one for the Critic. "
        "Both are 'mock' agents, meaning they do not actually call an AI model — "
        "they use hard-coded responses so that you can test the loop without spending money on APIs."
    )

    add_heading(doc, "Function: writer_node(state)", level=3)
    doc.add_paragraph(
        "The Writer agent. It reads loop_count from the shared state "
        "(how many rounds have happened) and returns a progressively improved draft. "
        "After Draft 3, it starts switching between two slightly different wordings to simulate a deadlock."
    )
    add_code_block(doc,
        "# Input:  state = {'loop_count': 0, 'history': []}\n"
        "# Output: {'current_draft': 'Draft 1: This is a 5-page report...'}\n\n"
        "def writer_node(state):\n"
        "    loop_count = state.get('loop_count', 0)\n"
        "    if loop_count == 0:\n"
        "        draft = 'Draft 1: This is a 5-page report on the new Dubai property.'\n"
        "    # ... continues for subsequent drafts\n"
        "    return {'current_draft': draft}"
    )

    add_heading(doc, "Function: critic_node(state)", level=3)
    doc.add_paragraph(
        "The Critic agent. It reads the current draft and loop_count from state, "
        "then returns targeted feedback. After loop 1, it enters a loop of pedantic demands "
        "('change cement to concrete', 'change concrete to cement', forever) "
        "to simulate the infinite-loop problem that SHP is designed to solve."
    )
    add_code_block(doc,
        "# Input:  state = {'current_draft': 'Draft 2: ...', 'loop_count': 1}\n"
        "# Output: {'history': [...], 'loop_count': 2}\n\n"
        "def critic_node(state):\n"
        "    feedback = 'Good addition. But what about the foundation material?'\n"
        "    history.append({'draft': draft, 'feedback': feedback})\n"
        "    return {'history': history, 'loop_count': loop_count + 1}"
    )

    doc.add_paragraph("")

    # ── 3.3 semantic_entropy.py ───────────────────────
    add_heading(doc, "3.3  semantic_entropy.py  (The Maths Engine)", level=2)
    doc.add_paragraph(
        "This is the most important file conceptually. It does not call any agent; "
        "its only job is to measure how different two texts are."
    )
    add_info_box(doc, "🔑 Core Concept:",
        "Every piece of text can be converted into a list of floating-point numbers (a vector). "
        "Similar texts produce vectors that point in nearly the same direction. "
        "We measure the angle between two vectors — the smaller the angle, the more similar the texts.")

    add_heading(doc, "Class: SemanticEntropyCalculator", level=3)
    doc.add_paragraph(
        "A wrapper class that glues an embedding model to our convergence logic. "
        "You create one instance of this class and keep reusing it throughout the workflow."
    )
    add_code_block(doc,
        "# Initialise once:\n"
        "calculator = SemanticEntropyCalculator(embedding_model=MockEmbeddings())\n\n"
        "# Attributes:\n"
        "#   embedding_model  — any Langchain-compatible embeddings object"
    )

    add_heading(doc, "Method: get_embedding(text) → List[float]", level=3)
    doc.add_paragraph(
        "Takes a plain-English string (e.g., a draft) and returns a list of decimal numbers "
        "that represent its meaning in high-dimensional space. "
        "In production this calls a real embedding API; in tests it calls the MockEmbeddings class."
    )
    add_code_block(doc,
        "vec = calculator.get_embedding('The foundation is a cement base.')\n"
        "# Returns something like: [0.12, 0.88, 0.34, ...]"
    )

    add_heading(doc, "Method: calculate_distance(vec1, vec2) → float", level=3)
    doc.add_paragraph(
        "Computes the cosine distance between two vectors. "
        "Returns a float between 0.0 (identical) and 2.0 (completely opposite). "
        "In practice, a halting threshold of 0.01 works well — anything below that means "
        "the two drafts are practically the same text."
    )
    add_code_block(doc,
        "dist = calculator.calculate_distance(vec_draft_4, vec_draft_5)\n"
        "# Output: 0.004  ← almost zero = semantically stuck → HALT\n"
        "# Output: 0.31   ← meaningful change = keep going"
    )

    doc.add_paragraph("")

    # ── 3.4 agent_workflow.py ─────────────────────────
    add_heading(doc, "3.4  agent_workflow.py  (The Graph / Flowchart)", level=2)
    doc.add_paragraph(
        "This file wires everything together using LangGraph, a library that lets you define "
        "workflows as directed graphs (think: a flowchart with nodes and arrows). "
        "Each box in the flowchart is a 'node', and each arrow is an 'edge'."
    )

    add_heading(doc, "Class: WorkflowState (TypedDict)", level=3)
    doc.add_paragraph(
        "Defines the shape of the shared memory that all nodes read from and write to. "
        "Think of it as the whiteboard that every agent in the room can see. "
        "Every time a node runs, it can update entries on this whiteboard."
    )
    add_code_block(doc,
        "class WorkflowState(TypedDict):\n"
        "    current_draft: str          # The latest draft text\n"
        "    history: List[Dict]         # Log of all previous rounds\n"
        "    loop_count: int             # How many rounds have happened\n"
        "    previous_embedding: List    # Vector of the PREVIOUS draft\n"
        "    current_embedding: List     # Vector of the CURRENT draft"
    )

    add_heading(doc, "Class: MockEmbeddings", level=3)
    doc.add_paragraph(
        "A lightweight fake embedding model for development and testing. "
        "Instead of sending text to an API, it hashes the text using MD5 and converts the "
        "bytes to floats. This is deterministic (same text always gives same result) "
        "and requires zero network access."
    )

    add_heading(doc, "Function: embed_state_node(state)", level=3)
    doc.add_paragraph(
        "A LangGraph node that sits between the Writer and the decision point. "
        "It reads the latest draft, converts it to a vector, "
        "and stores it alongside the previous draft's vector so the convergence check can compare them."
    )
    add_code_block(doc,
        "# Before node runs: state has only current_draft\n"
        "# After node runs:  state gains previous_embedding + current_embedding"
    )

    add_heading(doc, "Function: check_convergence(state) → str", level=3)
    doc.add_paragraph(
        "This is the 'brain' of the halting mechanism — a conditional edge. "
        "LangGraph calls this function after embed_state_node and uses the return value "
        "to decide which node to run next."
    )
    add_code_block(doc,
        "# Returns 'critic'  → the loop continues\n"
        "# Returns 'end'     → the graph stops (halted!)\n\n"
        "if distance < CONVERGENCE_THRESHOLD:   # e.g. 0.01\n"
        "    return 'end'\n"
        "elif loop_count > 10:                  # failsafe\n"
        "    return 'end'\n"
        "else:\n"
        "    return 'critic'"
    )

    doc.add_paragraph("")
    add_heading(doc, "The Graph Topology (Flowchart)", level=3)
    doc.add_paragraph(
        "writer  →  embed_state  →  check_convergence  →  critic  →  (back to writer)\n"
        "                                              ↓\n"
        "                                             END"
    )

    doc.add_paragraph("")

    # ── 3.5 ragas_eval.py ────────────────────────────
    add_heading(doc, "3.5  ragas_eval.py  (The Grader)", level=2)
    doc.add_paragraph(
        "Once the agents produce a final output, we need an objective way to measure its quality. "
        "This script uses the Ragas framework with free Hugging Face models to score it across "
        "four dimensions."
    )

    add_heading(doc, "The Four Metrics Explained Simply", level=3)
    metric_rows = [
        ("Faithfulness", "Did the answer make up facts, or is everything supported by the context? (1.0 = zero hallucinations)"),
        ("Answer Relevancy", "Does the answer actually address the question that was asked?"),
        ("Context Precision", "Of all the documents retrieved, how many were actually useful?"),
        ("Context Recall", "Did we retrieve ALL the documents needed to answer the question correctly?"),
    ]
    t = doc.add_table(rows=1 + len(metric_rows), cols=2)
    t.style = "Light List Accent 2"
    t.rows[0].cells[0].text = "Metric"
    t.rows[0].cells[1].text = "Plain English Meaning"
    for m, d in metric_rows:
        r = t.add_row().cells
        r[0].text = m
        r[1].text = d

    add_heading(doc, "Model Setup", level=3)
    doc.add_paragraph(
        "Two free, open-source Hugging Face models are used, requiring NO payment:"
    )
    doc.add_paragraph("• Judge LLM: Qwen/Qwen2.5-7B-Instruct — evaluates faithfulness and relevancy via Hugging Face's Inference API (no local download).", style="List Bullet")
    doc.add_paragraph("• Embedding Model: BAAI/bge-small-en-v1.5 — a 130 MB model downloaded once to your local .hf_cache folder for relevancy scoring.", style="List Bullet")

    add_code_block(doc,
        "# Sample output from ragas_eval.py:\n"
        "{'faithfulness': 1.0000,\n"
        " 'answer_relevancy': 0.9100,\n"
        " 'context_precision': 0.5000,\n"
        " 'context_recall': 1.0000}"
    )

    doc.add_paragraph("")

    # ── 3.6 optimize_score.py ────────────────────────
    add_heading(doc, "3.6  optimize_score.py  (The Optimizer)", level=2)
    doc.add_paragraph(
        "We now have four scores (Faithfulness, Relevancy, Precision, Recall), but how do we "
        "combine them into a single 'Information Score'? Should Faithfulness count double? "
        "This file uses Machine Learning (Linear Regression) to find the mathematically ideal answer."
    )

    add_heading(doc, "Function: optimize_information_score_weights()", level=3)
    doc.add_paragraph("This function does the following four things:")
    steps = [
        "Generate 100 fake RAG interactions with random-but-realistic metric scores.",
        "Create a 'Ground Truth' human quality score using known weights + random noise.",
        "Feed all of this to a Linear Regression model (sklearn) to learn the weights.",
        "Print the normalised formula and an R² score (how accurate the model is).",
    ]
    for i, step in enumerate(steps, 1):
        doc.add_paragraph(f"{i}. {step}", style="List Number")

    add_heading(doc, "Understanding the output", level=3)
    add_code_block(doc,
        "Information Score =\n"
        "  (0.39 * Faithfulness) +\n"
        "  (0.30 * AnswerRelevancy) +\n"
        "  (0.09 * ContextPrecision) +\n"
        "  (0.23 * ContextRecall)\n\n"
        "Model R² Score: 0.8068\n"
        "# R² of 0.81 means the formula explains 81% of the variance in human ratings.\n"
        "# The closer to 1.0, the more the formula aligns with human judgement."
    )
    add_info_box(doc, "💡 Note:",
        "In production you would replace the mock data with real human annotations "
        "collected from your own RAG system to get weights tailored to your domain.")

    doc.add_page_break()

    # ────────────────────────────────────────────────
    # Section 4 – How to Run the Pipeline
    # ────────────────────────────────────────────────
    add_heading(doc, "4.  How to Run the Full Pipeline", level=1)
    steps_run = [
        ("Activate the virtual environment", "source venv/bin/activate"),
        ("(Optional) Run each step individually", "python3 agent_workflow.py\npython3 ragas_eval.py\npython3 optimize_score.py"),
        ("Run the full automated pipeline", "python3 main.py"),
    ]
    for label, cmd in steps_run:
        doc.add_paragraph(label, style="List Number")
        add_code_block(doc, cmd)

    doc.add_paragraph("")
    add_heading(doc, "5.  Glossary", level=1)
    glossary = [
        ("Embedding", "A list of numbers that captures the meaning of a piece of text so a computer can compare texts mathematically."),
        ("Cosine Distance", "A way to measure the angle between two embedding vectors. Zero = identical meaning, larger = more different."),
        ("LangGraph", "A Python library for building AI agent workflows as directed graphs (flowcharts)."),
        ("Ragas", "An open-source framework for evaluating the quality of RAG (Retrieval-Augmented Generation) systems."),
        ("Linear Regression", "A basic machine-learning algorithm that learns a straight-line formula to predict an output from multiple inputs."),
        ("R² Score", "A number between 0 and 1 that measures how well a regression model's predictions match reality. 1.0 = perfect."),
        ("Convergence", "The point where the text drafts stop changing meaningfully — the mathematical signal to halt the loop."),
        ("Virtual Environment (venv)", "An isolated Python installation for this project so that its dependencies don't clash with other projects."),
    ]
    t2 = doc.add_table(rows=1 + len(glossary), cols=2)
    t2.style = "Light List Accent 1"
    t2.rows[0].cells[0].text = "Term"
    t2.rows[0].cells[1].text = "Meaning"
    for term, meaning in glossary:
        r = t2.add_row().cells
        r[0].text = term
        r[1].text = meaning

    # ── Save ──────────────────────────────────────────
    out_path = "doc/SHP_Codebase_Guide.docx"
    doc.save(out_path)
    print(f"✅  Documentation saved to: {out_path}")

if __name__ == "__main__":
    build_doc()

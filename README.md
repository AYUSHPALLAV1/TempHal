---
title: TempHal
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
python_version: "3.11"
pinned: false
---

# TempHal — Temporal Hallucination Attribution

A system that detects and grounds **temporal hallucinations** in LLM responses.

## What it does

1. Takes a factual question → generates a Gemini response
2. Extracts claims using spaCy NER + dependency parsing
3. Classifies each claim as **stable / volatile / uncertain** using a fine-tuned DistilBERT
4. Probes uncertainty via PEGASUS paraphrasing + BERTScore semantic consistency
5. Retrieves Wikipedia evidence for high-risk claims and grounds them with Gemini
6. Displays an interactive breakdown in Gradio

## Setup (Local)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # add your GEMINI_API_KEY
python app.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key |
| `HF_TOKEN` | Optional | HuggingFace token (faster downloads) |

## Architecture

```
Query → Gemini → Claims (spaCy) → Classifier (DistilBERT)
                                         ↓
                              Uncertain/Volatile claims
                                         ↓
                              Paraphrase (PEGASUS) → BERTScore → Uncertainty
                                         ↓ (high risk)
                              Retrieve (Wikipedia/DDG) → Ground (Gemini)
                                         ↓
                              Highlighted output + scores
```

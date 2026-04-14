def build_classifier_prompt(question: str) -> str:
    return f"""
You are an intent classifier for a Type 1 Diabetes assistant.

Classify the user message into EXACTLY one of these categories:
- PREDICT
- EXPLAIN
- GENERAL

Definitions:
PREDICT:
The user expects a forward-looking or outcome-based answer that depends on their current physiological state.
This includes:
- asking what will happen next
- asking how glucose will change
- asking about impact of actions (food, insulin, activity)

EXPLAIN:
The user is asking to interpret or explain current or past glucose behavior.
This includes:
- asking why something is happening
- asking what caused a change
- asking for reasoning based on current data

GENERAL:
Everything else:
- greetings
- general diabetes knowledge
- unrelated or emotional statements
- vague chat

Rules:
- Output ONLY valid JSON
- Do NOT include any text before or after JSON
- Use EXACT category names (PREDICT, EXPLAIN, GENERAL)
- If unsure, return GENERAL

Example:
{{"intent": "GENERAL"}}

User message:
"{question}"

Output:
"""
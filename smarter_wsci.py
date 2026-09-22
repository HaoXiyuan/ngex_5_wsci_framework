from pathlib import Path
from ollama import chat
import json



question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
"""

## WRITE ##
service_status = {
    "wifi": "operational"
}

state = {
    "problem": question,
    "wi_fi status": "operational",
    "wi-fi_check": True
}

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )

with open("state.json", "r") as file:
    state = json.load(file)

print(state)


## SELECT CONTEXT FILES BASED ON QUESTION
## Create the function that takes the student's question, takes some keywords and chooses the relevant files from the knowledge base. Return a list of the selected files.
## For example, if the question has the kyeword "print" or "printer", then the function should return the file "knowledge/printer_setup.txt" in a list.
def select_context(question):
    keyword_map = {
        "wifi": "knowledge/wifi_setup.txt",
        "wi-fi": "knowledge/wifi_setup.txt",
        "password": "knowledge/password_changes.txt",
        "email": "knowledge/email_setup.txt",
        "vpn": "knowledge/vpn.txt",
        "print": "knowledge/printing.txt",
        "printer": "knowledge/printing.txt",
        "projector": "knowledge/classroom_projectors.txt",
        "status": "knowledge/service_status.txt",
    }
    q = question.lower()
    selected = []
    for keyword, file_path in keyword_map.items():
        if keyword in q and file_path not in selected:
            selected.append(file_path)
    return selected


selected_files = select_context(question)

## READ SELECTED FILES and add their contents to the context variable.
context = ""

for path in selected_files:
    context += Path(path).read_text()
    context += "\n\n"


## 
## COMPRESS CONTEXT
## Add logic to compress the context from above by calling Qwen with "context" and the "question" as the parameter
## The response from Qwen should be the compressed context. Store it in a variable called "compressed_context" 

def compress_context(context, question):
    prompt = (
        "Extract only the information from the context below that is "
        "directly relevant to answering the student's question. "
        "Return a short, structured summary.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )
    resp = chat(
        model="qwen3:8b",
        messages=[
            {"role": "user", "content": prompt},
        ],
        think=False,
    )
    return resp.message.content


compressed_context = compress_context(context, question)


## Print the length of the compressed context
print(len(compressed_context))

## Now, call Qwen again with the compressed context and the student's question. Store the response in a variable called "response" and print the response from Qwen.
## Ensure the model produces a structured output 

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                compressed_context
                + "\n\nRespond ONLY with a single valid JSON object "
                  "matching the schema below. Do not include any "
                  "markdown formatting, code fences, or extra text.\n\n"
                  "{\n"
                  '  "cause": "string",\n'
                  '  "steps": ["string", "string"],\n'
                  '  "needs_it_support": true\n'
                  "}"
            ),
        },
        {"role": "user", "content": question},
    ],
    think=False,
)


print(response.message.content)

## WRITE the above output in an artifact called "state"
state["compressed_context"] = compressed_context
state["answer"] = response.message.content

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )

## Update the rest of the code so that it uses the "state" artifact as part of the context. 
## It is important to ensure that the model uses only the relevant parts from the "state" artifact and not the entire artifact.
## For this, you may have to think of a good structure for the "state" artifact and how to use it in the context.

## Read the state artifact back from disk
with open("state.json", "r") as file:
    saved_state = json.load(file)

## ISOLATE
## Create different states/artifacts so that each task can use the appropriate one.

diagnostic_context = {
    "problem": question,
    "device": "Windows laptop",
    "wifi_status": saved_state["wi_fi status"],
}

report_context = {
    "total_wifi_cases": 37,
    "resolved_cases": 29,
    "unresolved_cases": 8,
}

## Use Qwen to classify the problem, then let the program decide which state to use.
classify_response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "Classify the following IT problem into exactly one of: "
                "DIAGNOSTIC, REPORT. "
                "Respond with only that one word and nothing else."
            ),
        },
        {"role": "user", "content": question},
    ],
    think=False,
)

problem_type = classify_response.message.content.strip().upper()
print("Classified as:", problem_type)

## Depending on the task, use the appropriate state.
if problem_type == "REPORT":
    selected_state = report_context
else:
    selected_state = diagnostic_context

print("Using state:", json.dumps(selected_state, indent=2))

## Use only the relevant state as the new context.
isolated_response = chat(
    model="qwen3:8b",
    messages=[
        {"role": "system", "content": json.dumps(selected_state, indent=2)},
        {
            "role": "user",
            "content": "Summarize the solution in exactly 3 short bullet points.",
        },
    ],
    think=False,
)

print(isolated_response.message.content)
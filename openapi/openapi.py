import os
import json
import jsonref
from openai import OpenAI
import requests
from pprint import pp

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

with open('./data/example_events_openapi.json', 'r') as f:
    openapi_spec = jsonref.loads(f.read()) # it's important to load with jsonref, as explained below

# display(openapi_spec)

def openapi_to_functions(openapi_spec):
    functions = []

    for path, methods in openapi_spec["paths"].items():
        for method, spec_with_ref in methods.items():
            # 1. Resolve JSON references.
            spec = jsonref.replace_refs(spec_with_ref)

            # 2. Extract a name for the functions.
            function_name = spec.get("operationId")

            # 3. Extract a description and parameters.
            desc = spec.get("description") or spec.get("summary", "")

            schema = {"type": "object", "properties": {}}

            req_body = (
                spec.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if req_body:
                schema["properties"]["requestBody"] = req_body

            params = spec.get("parameters", [])
            if params:
                param_properties = {
                    param["name"]: param["schema"]
                    for param in params
                    if "schema" in param
                }
                schema["properties"]["parameters"] = {
                    "type": "object",
                    "properties": param_properties,
                }

            functions.append(
                {"type": "function", "function": {"name": function_name, "description": desc, "parameters": schema}}
            )

    return functions


functions = openapi_to_functions(openapi_spec)

for function in functions:
    pp(function)
    print()


SYSTEM_MESSAGE = """
You are a helpful assistant.
Respond to the following prompt by using function_call and then summarize actions.
Ask for clarification if a user request is ambiguous.
"""

# Maximum number of function calls allowed to prevent infinite or lengthy loops
MAX_CALLS = 5


def get_openai_response(functions, messages):
    return client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        tools=functions,
        tool_choice="auto",  # "auto" means the model can pick between generating a message or calling a function.
        temperature=0,
        messages=messages,
    )


def process_user_instruction(functions, instruction):
    num_calls = 0
    messages = [
        {"content": SYSTEM_MESSAGE, "role": "system"},
        {"content": instruction, "role": "user"},
    ]

    while num_calls < MAX_CALLS:
        response = get_openai_response(functions, messages)
        message = response.choices[0].message
        print(message)
        try:
            print(f"\n>> Function call #: {num_calls + 1}\n")
            pp(message.tool_calls)
            messages.append(message)

            # For the sake of this example, we'll simply add a message to simulate success.
            # Normally, you'd want to call the function here, and append the results to messages.
            messages.append(
                {
                    "role": "tool",
                    "content": "success",
                    "tool_call_id": message.tool_calls[0].id,
                }
            )

            num_calls += 1
        except:
            print("\n>> Message:\n")
            print(message.content)
            break

    if num_calls >= MAX_CALLS:
        print(f"Reached max chained function calls: {MAX_CALLS}")


USER_INSTRUCTION = """
Instruction: Get all the events.
Then create a new event named AGI Party.
Then delete event with id 2456.
"""

process_user_instruction(functions, USER_INSTRUCTION)



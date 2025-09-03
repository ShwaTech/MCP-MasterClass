import json
from groq import Groq
from tools import add
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
groq_client = Groq()

"""
This is a simple example to demonstrate that MCP simply enables a new way to call functions.
"""

## Define tools for the model  !!IMPORTANT
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


## Let's Call The LLM
response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Calculate 25 + 17"}],
    tools=tools,
)

## Handle tool calls
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    ## Execute directly
    result = add(**tool_args)

    ## Send result back to model
    final_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Calculate 25 + 17"},
            response.choices[0].message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)},
        ],
    )
    
    print(final_response.choices[0].message.content)

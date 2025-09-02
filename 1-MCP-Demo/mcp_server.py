from fastmcp import FastMCP
from fastapi import FastAPI, Body
import uvicorn


app = FastAPI()

mcp = FastMCP("Demo MCP Server")

## @mcp.tool() --> Erorr TypeError: 'FunctionTool' object is not callable
mcp.tool()
def multiply(a, b):
    return a * b


@app.post("/shwa/mcp/multiply")
def call_multiply(data: dict=Body(...)):
    return {
        "result": multiply(data.get("a", 0), data.get("b", 0))
    }

@app.get("/")
def home():
    return {
        "message": "Welcome To MCP Demo By Shwa 🥤"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7777)

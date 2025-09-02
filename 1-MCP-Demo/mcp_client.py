import requests


def test_multiply(a, b):
    response = requests.post("http://127.0.0.1:7777/shwa/mcp/multiply", json={"a": a, "b": b})

    if response.status_code == 200:
        print("Multiplication MCP Tool Called Successfully! ✔🌹✔")
        print("Result:", response.json()["result"])
    else:
        print("Error Accessing MCP Multiplication Tool! ❌🚨❌")


if __name__ == "__main__":
    test_multiply(6, 5)

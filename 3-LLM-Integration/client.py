import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import AsyncGroq

## Apply nest_asyncio to allow nested event loops (needed for Jupyter/IPython)
nest_asyncio.apply()

## Load environment variables
load_dotenv()


class MCPGroqClient:
    """
    Client for interacting with Groq models using MCP tools.
    """


    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the Groq MCP client.

        Args:
            model: The Groq model to use.
        """
        ## Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq_client = AsyncGroq()
        self.model = model
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None


    async def connect_to_server(self, server_script_path: str = "server.py"):
        """
        Connect to an MCP server.

        Args:
            server_script_path: Path to the server script.
        """
        ## Server configuration
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )

        ## Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        ## Initialize the connection
        await self.session.initialize()

        ## List available tools
        tools_result = await self.session.list_tools()
        print("\n🛠  Connected to server with tools 🛠")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")


    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools from the MCP server in Groq format.

        Returns:
            A list of tools in Groq format.
        """
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]


    async def process_query(self, query: str) -> str:
        """
        Process a query using Groq and available MCP tools.

        Args:
            query: The user query.

        Returns:
            The response from Groq.
        """
        ## Get available tools
        tools = await self.get_mcp_tools()

        ## Initial Groq API call
        response = await self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            tools=tools,
            tool_choice="auto",
        )

        ## Get assistant's response
        assistant_message = response.choices[0].message

        ## Initialize conversation with user query and assistant response
        messages = [
            {"role": "user", "content": query},
            assistant_message,
        ]

        ## Handle tool calls if present
        if assistant_message.tool_calls:
            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                # Execute tool call
                result = await self.session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # Add tool response to conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.content[0].text,
                    }
                )

            # Get final response from Groq with tool results
            final_response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="none",  # Don't allow more tool calls
            )

            return final_response.choices[0].message.content

        # No tool calls, just return the direct response
        return assistant_message.content


    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main():
    """Main entry point for the client."""
    client = MCPGroqClient()
    try:
        await client.connect_to_server("server.py")

        # Example: Ask about company vacation policy
        query = "What is our company's vacation policy?"
        print(f"\n👉 Query: {query}")

        response = await client.process_query(query)
        print(f"\n🚁 Response: {response}")
    finally:
        # Ensure proper cleanup
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
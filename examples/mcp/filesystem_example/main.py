import asyncio
import os
import shutil

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio, create_static_tool_filter

# Exercise: only expose read-only tools so the agent cannot modify the files.
# These names come from `server.list_tools()` on @modelcontextprotocol/server-filesystem.
READ_ONLY_TOOLS = [
    "read_text_file",
    "read_multiple_files",
    "list_directory",
    "directory_tree",
    "search_files",
    "get_file_info",
]


async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to read the filesystem and answer questions based on those files.",
        mcp_servers=[mcp_server],
    )

    # List the files it can read
    message = "Read the files and list them."
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Ask about books
    message = "Read favorite_books.txt and tell me my #1 favorite book."
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Ask a question that reads then reasons.
    message = "Read favorite_songs.txt and suggest one new song that I might like."
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Exercise: a question of my own.
    message = "Read favorite_cities.txt and plan a 3-day trip to my favorite one."
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Exercise: try to write a file. The tool filter hides write_file, so it should fail.
    message = "Create a file called notes.txt with a summary of my favorite books."
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Exercise: set MCP_FILES_DIR to point the server at a folder of your own.
    samples_dir = os.environ.get("MCP_FILES_DIR", os.path.join(current_dir, "sample_files"))

    async with MCPServerStdio(
        name="Filesystem Server, via npx",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", samples_dir],
        },
        tool_filter=create_static_tool_filter(allowed_tool_names=READ_ONLY_TOOLS),
    ) as server:
        # Exercise: print the tools the agent will see after the filter is applied.
        tools = await server.list_tools()
        print(f"Tools exposed to the agent: {[tool.name for tool in tools]}\n")

        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Filesystem Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/logs/trace?trace_id={trace_id}\n")
            await run(server)

    notes_path = os.path.join(samples_dir, "notes.txt")
    print(f"\n\nnotes.txt created: {os.path.exists(notes_path)}")


if __name__ == "__main__":
    # Let's make sure the user has npx installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")

    asyncio.run(main())

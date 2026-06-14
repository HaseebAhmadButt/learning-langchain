Go to the mcpdoc github page to get the project source.
After installing all dependencies, run the following command to start the server:

```bash
uvx --from mcpdoc mcpdoc  --urls "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" "LangChain:https://python.langchain.com/llms.txt"  --transport sse  --port 8082  --host localhost
```

It will start a server on port 8082.

Then run the following command to start the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector
```

This will start a web server on port 3000.
Then enter the MCP Server URL (http://localhost:8082) to the Inspector, to extract all tools.
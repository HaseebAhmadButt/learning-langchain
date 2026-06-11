Reflexion Agents

Reflexion is an agent architecture that improves its performance through self-reflection and iterative feedback. Rather than producing a single answer and stopping, the agent evaluates its own response, identifies weaknesses or missing information, and uses external tools to gather additional context before generating a refined answer. This approach typically produces higher quality results, but requires additional reasoning steps and execution time.

The workflow begins when a user request is sent to the **Responder Node**. The responder generates:

1. Initial Response
2. Self-Critique
3. Search Queries (targeted queries that can be used by tools to gather information for improving the response)

The generated search queries are then passed to the **Tool Execution Node**, which retrieves relevant information from external sources.

The outputs from the responder and tool execution stages are then provided to the **Revisor Node**. The revisor analyzes:

- The Initial Response
- The Self-Critique
- The Tool Results

Using this information, it produces an improved version of the answer along with supporting metadata. The revisor generates:

1. Revised Response
2. Updated Critique
3. Refined Search Queries
4. Citations and Supporting References

This reflection and revision process enables the agent to iteratively improve the quality, accuracy, and completeness of its final response.
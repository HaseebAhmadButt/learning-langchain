Reflection Agents:

Reflection is a prompting strategy used to improve the quality and success rate of agents and similar AI systems. 
Instead of producing a single response in one step, the model generates an initial output, evaluates it for mistakes 
or missing details, and then refines it based on that self-critique. This creates a loop where each iteration improves 
the quality of the final result, making it especially useful for tasks like reasoning, code generation, and complex question answering.

Reflection Chain:
A Reflection Chain is a structured implementation of this idea where the process is explicitly broken into sequential 
stages such as generation, critique, and revision. The chain can run once or loop multiple times, allowing the system 
to progressively improve its output until it meets a desired quality threshold.

Example:
A model first generates an explanation of a system design, then a separate reflection step reviews it for missing 
scalability or reliability concerns, and finally, a revision step produces an improved and more complete version.
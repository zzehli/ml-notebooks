# Agent design:
[x] tool calling
[x] short memory
[x] basic agent
[x] react agent
[ ] browser use agent
[x] cot
[ ] codeact agent: https://github.com/xingyaoww/code-act/blob/d607f56c9cfe9e8632ebaf65dcaf2b4b7fe1c6f8/mint/prompt/templates/template_with_tool.txt
[x] graph-based framework (workflow)
[ ] multi-agents (handover)
[ ] rag

## Talking points
* Is ReAct a concept of LLM performing action or address hallucination?
* LLMs can write code as part of their answers, however, CodeAct proposes using code as THE language for LLMs; they will describe things in code instead of in natural language.
* the environment executes the action, not LLM
* tool call is a trained behavior; LLM is trained to output strings in specified shapes (in this sense, tool call depends on structured output)

# Note
* `react_graph` does not always perform function calls.
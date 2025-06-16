# Agent design:
[x] tool calling
[x] short memory
[x] basic agent
[x] react agent
[x] browser use agent
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
* there are many ways to reinforce structured output
    * training
    * inference time: 
        * logit post-processing: Outlines, and lm-format-enforcer; build a representation (tree, finite-state-machine) then prune logit to remove disallowed elements, see: https://dottxt-ai.github.io/outlines/latest/reference/generation/structured_generation_explanation/ and https://github.com/noamgat/lm-format-enforcer, https://www.tamingllms.com/notebooks/structured_output.html and https://www.deeplearning.ai/short-courses/getting-structured-llm-output/
        * retry-based
* agent implementation: https://www.lutzroeder.com/blog/2025-05-24-tiny-agents/
# TODO
- [ ] change observation to user message
- [ ] refactor thought and action into one call
# Note
* `react_graph` does not always perform function calls.
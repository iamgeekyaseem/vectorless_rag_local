"""
core/agent.py — ReAct Agent Loop
──────────────────────────────────────────────────────────────
An AI agent that can reason and act across multiple steps.

WHAT IS AN AGENT?
  A regular LLM call: question → answer (one shot)
  An agent:           question → think → act → observe → think → act → answer
                      (multiple steps, uses tools)

THE ReAct PATTERN (Reason + Act):
  The LLM alternates between:
    THOUGHT  → "I need to find information about X"
    ACTION   → search_document("X")
    OBSERVATION → "Found: X is described on page 5 as..."
    THOUGHT  → "Now I can answer"
    ANSWER   → "Based on page 5, X means..."

AVAILABLE TOOLS:
  search_document(query)    → retrieve pages from the document
  get_page(page_number)     → fetch a specific page directly
  summarize_section(node_id)→ get a node's summary from the index

WHY AN AGENT VS PLAIN RAG?
  Plain RAG: one retrieval → one answer (good for simple questions)
  Agent:     multi-step retrieval + reasoning (good for complex questions)
  
  Example complex question:
  "Compare the causes and effects of climate change from this report"
  Agent would:
    1. search "causes of climate change" → get section 2
    2. search "effects of climate change" → get section 3
    3. synthesize both into a comparison answer
"""

import json
import re
from core.llm import chat
from core.retriever import retrieve, RetrievalResult
from core.models import DocumentIndex


# ─────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────

TOOLS_DESCRIPTION = """You have access to these tools:

1. search_document(query: str)
   Search the document index for sections relevant to a query.
   Use this to find information about specific topics.
   Example: search_document("greenhouse gas emissions")

2. get_page(page_number: int)
   Retrieve the full text of a specific page number.
   Use this when you know exactly which page to look at.
   Example: get_page(5)

3. finish(answer: str)
   Use this when you have enough information to answer the question.
   Provide your complete, well-sourced answer.
   Example: finish("Based on page 3, the main cause is...")
"""

AGENT_SYSTEM_PROMPT = """You are a document analysis agent. You answer questions by reasoning step-by-step and using tools to search a document.

{tools}

IMPORTANT RULES:
- Always think before acting
- Use search_document to find relevant sections
- Use get_page only when you need a specific page
- After gathering enough information, use finish() to give your answer
- Always cite page numbers and section names in your final answer
- Maximum {max_steps} steps — be efficient

RESPONSE FORMAT (always use this exact format):
THOUGHT: your reasoning about what to do next
ACTION: tool_name(argument)

OR when done:
THOUGHT: I now have enough information to answer
ACTION: finish("your complete answer here")
"""


# ─────────────────────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────

def _execute_tool(
    tool_name : str,
    argument  : str,
    index     : DocumentIndex,
    pages     : list[dict],
    model     : str,
    provider  : str
) -> str:
    """
    Execute a tool call and return the observation string.

    Args:
        tool_name : "search_document" | "get_page" | "finish"
        argument  : the argument string from the LLM's action
        index     : DocumentIndex for search_document
        pages     : raw pages for get_page
        model     : LLM model for search_document
        provider  : LLM provider

    Returns:
        str: the observation to feed back to the agent
    """

    if tool_name == "search_document":
        query = argument.strip().strip('"\'')
        print(f"[agent] Tool: search_document('{query}')")

        result = retrieve(
            question=query,
            index=index,
            pages=pages,
            model=model,
            provider=provider,
            top_k=2,
            verify=False  # skip verify in agent loop for speed
        )

        if result.contents:
            obs_parts = []
            for c in result.contents:
                preview = c["text"][:1500]
                obs_parts.append(f"[{c['title']} — {c['page_range']}]\n{preview}")
            return "FOUND:\n" + "\n\n---\n\n".join(obs_parts)
        else:
            return "No relevant sections found for that query."

    elif tool_name == "get_page":
        try:
            page_num = int(argument.strip())
        except ValueError:
            return f"Invalid page number: {argument}"

        print(f"[agent] Tool: get_page({page_num})")
        page = next((p for p in pages if p["page"] == page_num), None)

        if page:
            return f"[Page {page_num}]\n{page['text']}"
        else:
            max_page = max(p["page"] for p in pages)
            return f"Page {page_num} not found. Document has pages 1-{max_page}."

    elif tool_name == "finish":
        # The finish tool just returns the argument as-is
        # The agent loop detects this and stops
        return argument.strip().strip('"\'')

    else:
        return f"Unknown tool: '{tool_name}'. Available: search_document, get_page, finish"


# ─────────────────────────────────────────────────────────────
# ACTION PARSER
# ─────────────────────────────────────────────────────────────

def _parse_action(text: str) -> tuple[str, str] | None:
    """
    Parse the LLM's action from its response text.

    Looks for patterns like:
        ACTION: search_document("climate change causes")
        ACTION: get_page(5)
        ACTION: finish("The answer is...")

    Returns:
        (tool_name, argument) or None if no action found
    """
    # Match: ACTION: tool_name(argument)
    match = re.search(
        r"ACTION:\s*(\w+)\((.*?)\)\s*$",
        text,
        re.MULTILINE | re.DOTALL
    )
    if match:
        tool_name = match.group(1).strip()
        argument  = match.group(2).strip()
        return tool_name, argument

    # Fallback: look for just the tool name on a line
    match = re.search(r"ACTION:\s*(\w+)\s+(.*?)$", text, re.MULTILINE)
    if match:
        tool_name = match.group(1).strip()
        argument  = match.group(2).strip()
        return tool_name, argument

    return None


def _parse_thought(text: str) -> str:
    """Extract the THOUGHT portion from the LLM's response."""
    match = re.search(r"THOUGHT:\s*(.*?)(?=ACTION:|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ─────────────────────────────────────────────────────────────
# AGENT STEP RECORD
# ─────────────────────────────────────────────────────────────

class AgentStep:
    """Records one step in the agent's reasoning loop."""
    def __init__(self, step_num: int, thought: str, action: str, observation: str):
        self.step_num    = step_num
        self.thought     = thought
        self.action      = action
        self.observation = observation

    def to_dict(self) -> dict:
        return {
            "step"        : self.step_num,
            "thought"     : self.thought,
            "action"      : self.action,
            "observation" : self.observation
        }


# ─────────────────────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────────────────────

class AgentResult:
    """Container for the full agent run result."""
    def __init__(self):
        self.answer    : str             = ""
        self.steps     : list[AgentStep] = []
        self.success   : bool            = False

    def step_summary(self) -> str:
        """Human-readable summary of all agent steps."""
        lines = []
        for step in self.steps:
            lines.append(f"Step {step.step_num}:")
            lines.append(f"  💭 {step.thought[:100]}...")
            lines.append(f"  🔧 {step.action}")
            lines.append(f"  👁  {step.observation[:100]}...")
        return "\n".join(lines)


def run_agent(
    question   : str,
    index      : DocumentIndex,
    pages      : list[dict],
    model      : str,
    provider   : str,
    max_steps  : int = 5,
    on_step    = None
) -> AgentResult:
    """
    THE MAIN FUNCTION — runs the ReAct agent loop.

    Args:
        question  : the user's question
        index     : DocumentIndex tree
        pages     : raw pages from loader
        model     : LLM model name
        provider  : "ollama" | "openai" | "groq"
        max_steps : maximum reasoning steps (default 5)
        on_step   : optional callback fn(step: AgentStep) for UI updates

    Returns:
        AgentResult with final answer + all reasoning steps

    Example:
        result = run_agent(
            question = "Compare the causes and effects described in this report",
            index    = my_index,
            pages    = my_pages,
            model    = "gemma3:4b",
            provider = "ollama"
        )
        print(result.answer)
        print(result.step_summary())
    """
    result = AgentResult()

    # Build the system prompt
    system_prompt = AGENT_SYSTEM_PROMPT.format(
        tools=TOOLS_DESCRIPTION,
        max_steps=max_steps
    )

    # Conversation history for the agent loop
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Question: {question}\n\nBegin your analysis."}
    ]

    print(f"[agent] Starting agent loop for: '{question}'")

    for step_num in range(1, max_steps + 1):
        print(f"[agent] Step {step_num}/{max_steps}")

        # Ask the LLM what to do next
        response = chat(
            messages=messages,
            model=model,
            provider=provider
        )

        # Parse thought and action
        thought = _parse_thought(response)
        parsed  = _parse_action(response)

        if not parsed:
            # LLM didn't follow the format — prompt it to try again
            print(f"[agent] Could not parse action, prompting retry")
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "Please respond with THOUGHT: and ACTION: format."
            })
            continue

        tool_name, argument = parsed

        # Execute the tool
        if tool_name == "finish":
            # Agent is done!
            final_answer = argument.strip().strip('"\'')
            observation  = final_answer
        else:
            observation = _execute_tool(
                tool_name, argument, index, pages, model, provider
            )

        # Record this step
        step = AgentStep(
            step_num    = step_num,
            thought     = thought,
            action      = f"{tool_name}({argument})",
            observation = observation
        )
        result.steps.append(step)

        if on_step:
            on_step(step)

        # If finish tool was called, we're done
        if tool_name == "finish":
            result.answer  = final_answer
            result.success = True
            print(f"[agent] Finished at step {step_num}")
            break

        # Add the step to conversation history
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": f"OBSERVATION: {observation}\n\nContinue."
        })

    # If we ran out of steps without finishing
    if not result.success:
        print(f"[agent] Hit max_steps ({max_steps}) — generating final answer")
        # Force a final answer from everything gathered
        context_from_steps = "\n\n".join(
            f"[Step {s.step_num} - {s.action}]:\n{s.observation}"
            for s in result.steps
        )
        messages.append({
            "role": "user",
            "content": (
                f"You've reached the step limit. Based on all the information gathered above, "
                f"provide your best answer to: {question}\n\n"
                f"Context gathered:\n{context_from_steps}"
            )
        })
        final = chat(messages=messages, model=model, provider=provider)
        result.answer  = final
        result.success = True

    return result
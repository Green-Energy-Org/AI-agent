import sys
import atexit
import signal
from colorama import Fore, Style
from agent.graph import agent_graph
from agent.state import AgentState
from config.settings import settings
from utils.logger import logger
from utils.memory_store import conversation_memory

from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client

# --- FIX: single shared handler + guaranteed flush on container exit ---
langfuse_client = get_client()
langfuse_handler = CallbackHandler()   # one instance, reused across all runs

def _flush_langfuse():
    """Ensure all buffered traces are sent before process dies."""
    try:
        langfuse_client.flush()
    except Exception:
        pass

atexit.register(_flush_langfuse)

# Handle SIGTERM (docker stop) gracefully
def _sigterm_handler(signum, frame):
    _flush_langfuse()
    sys.exit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.CYAN}{'='*70}
    DevOps ReAct Agent 🤖
{'='*70}{Style.RESET_ALL}

Welcome! I'm your DevOps expert. Ask me anything 

Type 'exit' or 'quit' to end the session.
Type 'clear' to clear conversation history.
{Fore.CYAN}{'='*70}{Style.RESET_ALL}
"""
    print(banner)

@observe(name="agent-run")
def run_agent(query: str) -> str:
    """Run the agent with a query"""
    # FIX: reuse shared handler instead of creating per-call
    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "thought": "",
        "tool_calls": [],
        "action": "",
        "search_query": None,
        "iteration": 0,
        "ready_to_answer": False,
        "final_answer": ""
    }
    
    try:
        final_state = agent_graph.invoke(
            initial_state,
            config={"callbacks": [langfuse_handler]}
        )
        answer = final_state["final_answer"]
        return answer
    
    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        logger.log_error(error_msg)
        return f"I apologize, but I encountered an error: {str(e)}\nPlease try rephrasing your question."
    finally:
        # FIX: flush after every invocation so Docker doesn't lose buffered spans
        _flush_langfuse()
    
def main():
    """Main CLI loop"""
    
    # Validate settings
    try:
        settings.validate()
    except ValueError as e:
        print(f"{Fore.RED}Configuration Error: {e}{Style.RESET_ALL}")
        print("Please check your .env file and ensure API keys are set.")
        sys.exit(1)
    
    print_banner()
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Fore.GREEN}You: {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print(f"\n{Fore.CYAN}Goodbye! Happy DevOps-ing! 🚀{Style.RESET_ALL}\n")
                break
            
            # Check for clear command
            if user_input.lower() == 'clear':
                conversation_memory.clear()
                print(f"{Fore.YELLOW}Conversation history cleared.{Style.RESET_ALL}")
                continue
            
            # Add to memory
            conversation_memory.add_message("user", user_input)
            
            # Run agent
            print(f"\n{Fore.BLUE}Agent is thinking...{Style.RESET_ALL}\n")
            run_agent(user_input)
            
            # Response is already logged by the agent
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.CYAN}Goodbye! Happy DevOps-ing! 🚀{Style.RESET_ALL}\n")
            break
        except Exception as e:
            logger.log_error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
from langchain_core.messages import AIMessage, HumanMessage

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

import time
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()


def chat_assistant(user_query):
    conversation_history = SQLChatMessageHistory(
        session_id="test_session_id", connection_string="sqlite:///sqlite.db"
    )

    # Use the chat_history in the ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=conversation_history)

    current_timestamp = time.time()
    formatted_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_timestamp))
    wikipedia = WikipediaAPIWrapper()
    search = TavilySearchResults()
    # python_repl = PythonREPL()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    # llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    # llm = ChatAnthropic(temperature =0, model="claude-3-opus-20240229")
    tools = [
        Tool(
            name='search',
            func=search.run,
            description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
        ),
        Tool(
            name='wikipedia',
            func=wikipedia.run,
            description="Useful for when you need to look up a topic, country or person on wikipedia"
        ),

    ]
    tool_names = ["wikipedia", "search"]

    template = '''Assistant is designed to be able to assist by providing in-depth explanations and discussions on a wide range of topics.
                   As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
                   Assistant is constantly learning and improving, and its capabilities are constantly evolving.
                   It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions.
                   Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
                   Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics.
                   Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
                   Assistant MUST include urls from search results in the final answer.
                   Assistant MUST include urls from wikipedia results in the final answer.
                   Assistant is aware that the current date is: ''' + formatted_timestamp + '''.

                   TOOLS:
                   ------

                   Assistant has access to the following tools:

                   {tools}

                   To use a tool, please use the following format:

                   ```
                   Thought: Do I need to use a tool? Yes
                   Action: the action to take, should be one of [{tool_names}]
                   Action Input: the input to the action
                   Observation: the result of the action
                   ...


                   When you have a response to say to the user, or if you do not need to use a tool, you MUST use the 
                   format:

                   ```
                   Thought: Do I need to use a tool? No
                   Final Answer: [your response here]
                   ```

                   Begin!

                   Previous conversation history:
                   {chat_history}

                   New input: {input}
                   {agent_scratchpad}'''

    prompt = ChatPromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory,
                                   handle_parsing_errors=True,
                                   max_iterations=3)
    result = agent_executor.invoke({"input": user_query})
    return conversation_history

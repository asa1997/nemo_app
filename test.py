from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_community.llms import Bedrock
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import nest_asyncio

bedrock_text_generation_model = 'mistral.mistral-7b-instruct-v0:2'

anthropic_model_kwargs = { 
        "max_tokens_to_sample": 1024,
        "top_p": 0.9,
        "stop_sequences": ["Human:"]
}
llm = Bedrock(
    model_id=bedrock_text_generation_model,
    model_kwargs=anthropic_model_kwargs,
    streaming=True)

prompt = PromptTemplate(input_variables=["query"], template=("Respond concisely: {query}"))
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


nest_asyncio.apply()

config = RailsConfig.from_path("./config")
guardrails = RunnableRails(config, input_key="query", output_key='text')
chain_with_guardrails = guardrails | chain

chain_with_guardrails.invoke({"query": "Hello! What can you do for me?"}, verbose=True)

{'query': 'Hello! What can you do for me?',
 'text': " Hello! I'm Claude, an AI assistant created by Anthropic. I can answer questions, have conversations, and provide helpful information to you."}
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_community.llms import Bedrock
from langchain import PromptTemplate
from langchain.chains import LLMChain
import nest_asyncio
import logging

# Set up logging
# logging.basicConfig(level=logging.DEBUG)

# Define the Bedrock model and arguments
bedrock_text_generation_model = 'mistral.mistral-7b-instruct-v0:2'
bedrock_model_args = {
    "max_tokens_to_sample": 1024
}

# Initialize the Bedrock model
llm = Bedrock(
    model_id=bedrock_text_generation_model,
    model_kwargs=bedrock_model_args,
    streaming=True
)

# Define the prompt template
prompt = PromptTemplate(input_variables=["query"], template="Respond concisely: {query}")

# Create the LLM chain
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# Apply nest_asyncio to support asyncio
nest_asyncio.apply()

# Load the configuration for guardrails
config = RailsConfig.from_path("./config/config.yml")

# Initialize guardrails with the configuration
guardrails = RunnableRails(config, input_key="query", output_key='text')

# Chain guardrails with the LLM chain
chain_with_guardrails = guardrails | chain

# Invoke the chain with a sample query
response = chain_with_guardrails.invoke({"query": "Hi, tell me about Hindu religion"}, verbose=True)
print("response is", response)

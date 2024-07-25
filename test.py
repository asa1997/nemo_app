# from nemoguardrails import RailsConfig
# from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
# from langchain_community.llms import Bedrock
# from langchain import PromptTemplate
# from langchain.chains import LLMChain
# import nest_asyncio
# import logging

# import argparse

# # Create the parser
# parser = argparse.ArgumentParser(description="An example script.")

# # Add an argument
# parser.add_argument('prompt', help="Enter the prompt")

# # Parse the arguments
# prompt_llm = parser.parse_args()

# # Print the "echo" argument

# # Set up logging
# # logging.basicConfig(level=logging.DEBUG)

# # Define the Bedrock model and arguments
# bedrock_text_generation_model = 'mistral.mistral-7b-instruct-v0:2'
# bedrock_model_args = {
#     "max_tokens_to_sample": 1024
# }

# # Initialize the Bedrock model
# llm = Bedrock(
#     model_id=bedrock_text_generation_model,
#     # model_kwargs=bedrock_model_args,
#     streaming=True
# )

# # Define the prompt template
# prompt = PromptTemplate(input_variables=["query"], template="Respond concisely: {query}")

# # Create the LLM chain
# chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# # Apply nest_asyncio to support asyncio
# nest_asyncio.apply()

# # Load the configuration for guardrails
# config = RailsConfig.from_path("./config/config.yml")

# # Initialize guardrails with the configuration
# guardrails = RunnableRails(config, input_key="query", output_key='text')

# # Chain guardrails with the LLM chain
# chain_with_guardrails = guardrails | chain

# # Invoke the chain with a sample query
# response = chain_with_guardrails.invoke({"query": prompt_llm.prompt}, verbose=True)
# print(f"response is ==> {response}")

from nemoguardrails import RailsConfig, LLMRails
import nest_asyncio

nest_asyncio.apply()

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": "Hello! What can you do for me?"
}])
print(response["content"])
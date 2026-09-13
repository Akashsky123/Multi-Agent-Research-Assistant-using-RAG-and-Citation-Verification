from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()

# Prompt
prompt = PromptTemplate(
    template="""
Generate exactly 5 interesting facts about {topic}.

Format:
1.
2.
3.
4.
5.
""",
    input_variables=["topic"]
)

# Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=500,
    temperature=0.7,
)

# Convert to chat model
model = ChatHuggingFace(llm=llm)

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt | model | parser

# Run
result = chain.invoke({"topic": "cricket"})

print(result)

# Print chain graph
chain.get_graph().print_ascii()
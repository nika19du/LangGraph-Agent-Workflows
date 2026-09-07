from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

WORKSPACE_DIR = BASE_DIR / "workspace"

CLAUDE_PATH = Path(
    r"C:\Users\PC\.local\bin\claude.exe"
)

llm = init_chat_model(
    "openai:gpt-4.1-mini"
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
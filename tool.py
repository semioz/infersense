import polars as pl
import pytesseract
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from PIL import Image

# --------- Basic Math tools ---------

@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a + b

@tool
def subtract(a: float, b: float) -> int:
    """
    Subtract two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """
    Multiplies two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """
    Divides two numbers.
    Args:
        a (float): the first float number
        b (float): the second float number
    """
    if b == 0:
        raise ValueError("Cannot divided by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """
    Get the modulus of two numbers.
    Args:
        a (int): the first number
        b (int): the second number
    """
    return a % b


@tool
def power(a: float, b: float) -> float:
    """
    Get the power of two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a**b

# ------- Search Tools -------

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    """Search the Web via Tavily for a query and return 3 results in maximum.
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("url", "")}" title="{doc.get("title", "")}"/>\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 3 results.
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

# ------ Document Processing Tools ------

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image by using pytesseract via OCR.
    Args:
        image_path (str): the path to the image file.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)

        return f"Extracted the text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file by using Polars and answer a question about it.
    Args:
        file_path (str): the path to the CSV file.
        query (str): Question about the data
    """
    try:
        df = pl.read_csv(file_path)

        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"

        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error occured analyzing CSV file: {str(e)}"


@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using Polars and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data
    """
    try:
        df = pl.read_excel(file_path)

        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error occured analyzing Excel file: {str(e)}"


tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    web_search,
    wikipedia_search,
    arxiv_search,
    extract_text_from_image,
    analyze_csv_file,
    analyze_excel_file,
]

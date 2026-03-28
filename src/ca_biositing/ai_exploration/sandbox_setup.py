import os
import pandas as pd
import requests
import json
import plotly.io as pio
import plotly.graph_objects as go
from dotenv import load_dotenv
from pandasai.llm.base import LLM
from pandasai.core.response.parser import ResponseParser
from sqlalchemy import create_engine, text
from IPython.display import display, Image, HTML
from typing import Optional, List, Any

# Internal imports
from ca_biositing.ai_exploration.schema import discover_views, fetch_table_metadata

# Set Plotly for VS Code/Jupyter compatibility
pio.renderers.default = 'notebook'

# Load environment variables
load_dotenv()

AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet",
]

class CBORGLLM(LLM):
    """Hardened Custom LLM class for CBORG gateway in PandasAI 3.0"""
    def __init__(self, api_token: str, api_base: str = "https://api.cborg.lbl.gov/v1", model: str = "gemini-2.0-flash"):
        super().__init__()
        self.api_token = api_token
        self.api_base = api_base
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        })

    def call(self, instruction, context=None):
        prompt = instruction.to_string()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        try:
            response = self._session.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            return "Error: The request to CBORG API timed out."
        except requests.exceptions.RequestException as e:
            return f"Error connecting to CBORG API: {str(e)}"
        except (KeyError, ValueError) as e:
            return f"Error parsing CBORG API response: {str(e)}"

    @property
    def type(self):
        return "cborg"

class SandboxResponseParser(ResponseParser):
    """
    Custom Response Parser for the AI Sandbox.
    Ensures DataFrames and Plotly Figures are returned as raw objects
    to enable native VS Code / Jupyter rendering.
    """
    def __init__(self, context):
        super().__init__(context)

    def _validate_response(self, result: Any):
        """
        Extremely permissive validator to allow raw objects and unconventional formats.
        """
        return # Bypassing all base validation to ensure no InvalidOutputValueMismatch

    def parse(self, result: Any, last_code_executed: str = None) -> Any:
        """
        Override parse to return raw values instead of wrapped Response objects.
        This enables VS Code's Data Viewer and Plotly interactive features.
        """
        # 1. Handle raw DataFrames
        if isinstance(result, pd.DataFrame):
            return result

        # 2. Handle raw Plotly Figures or similar interactive objects
        if isinstance(result, go.Figure) or hasattr(result, 'to_html'):
            return result

        # 3. Handle dictionaries (the standard PandasAI format or raw data)
        parsed_result = None
        if isinstance(result, dict):
            # Check if it's a Plotly figure dict
            if "data" in result and "layout" in result:
                try:
                    parsed_result = go.Figure(result)
                except Exception:
                    pass

            if parsed_result is None:
                response_type = result.get("type")
                if response_type == "plot":
                    parsed_result = self.format_plot(result)
                elif response_type in ["dataframe", "table"]:
                    val = result.get("value")
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        parsed_result = pd.DataFrame(val)
                    else:
                        parsed_result = val

            # If it's a generic dict that looks like data (list of records), convert to DF
            if parsed_result is None:
                for key, val in result.items():
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        try:
                            parsed_result = pd.DataFrame(val)
                            break
                        except Exception:
                            pass

        # 4. Fallback to base parser but unwrap the value
        if parsed_result is None:
            try:
                base_response = super().parse(result, last_code_executed)
                parsed_result = getattr(base_response, 'value', base_response)
            except Exception:
                parsed_result = result

        # 5. Final Formatting for Notebook Display
        if isinstance(parsed_result, str):
            lower_res = parsed_result.lower()
            if any(marker in lower_res for marker in ["<html", "<div", "plotly", "window.plotlyconfig"]):
                return HTML(parsed_result)

            if (parsed_result.strip().startswith('{"data":') or parsed_result.strip().startswith('{"layout":')) and len(parsed_result) > 100:
                try:
                    return go.Figure(json.loads(parsed_result))
                except Exception:
                    pass

        if isinstance(parsed_result, str) and (parsed_result.endswith('.png') or parsed_result.endswith('.jpg')):
            if os.path.exists(parsed_result):
                return Image(filename=parsed_result)

        return parsed_result

    def format_plot(self, result: dict) -> Any:
        val = result.get("value")
        if isinstance(val, dict) and "data" in val and "layout" in val:
            try:
                return go.Figure(val)
            except Exception:
                return val
        if isinstance(val, go.Figure) or hasattr(val, 'to_html'):
            return val
        if isinstance(val, str) and any(marker in val.lower() for marker in ["<html", "<div", "plotly"]):
            return val
        if isinstance(val, str) and (val.endswith('.png') or val.endswith('.jpg')):
            if os.path.exists(val):
                return Image(filename=val)
            rel_path = os.path.join(os.getcwd(), val)
            if os.path.exists(rel_path):
                return Image(filename=rel_path)
        return super().format_plot(result)

    def format_dataframe(self, result: dict) -> pd.DataFrame:
        return result.get("value")

    def format_table(self, result: dict) -> pd.DataFrame:
        return result.get("value")

def init_sandbox(model_name: Optional[str] = None):
    """Initializes the sandbox environment and returns the LLM and DB engine."""
    api_key = os.getenv("CBORG_API_KEY")
    api_url = os.getenv("CBORG_API_URL", "https://api.cborg.lbl.gov/v1")
    selected_model = model_name or os.getenv("CBORG_MODEL") or "gemini-2.0-flash"

    if not api_key:
        raise ValueError("CBORG_API_KEY not found. Please check your .env file.")

    llm = CBORGLLM(api_token=api_key, api_base=api_url, model=selected_model)
    
    DB_USER = os.getenv("DB_USER", "biocirv_user")
    DB_PASS = os.getenv("DB_PASSWORD", "biocirv_dev_password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "biocirv_db")

    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"Initialized BioCirv AI with model: {selected_model}")
        print(f"Connected to Database: {DB_NAME}")
    except Exception as e:
        print(f"WARNING: Database connection failed: {e}")

    return llm, engine

def get_agent(llm, engine, view_names: Optional[List[str]] = None, schemas: List[str] = ["ca_biositing", "analytics"]):
    """Creates a PandasAI agent with enriched metadata and custom SQL skill."""
    from pandasai import Agent, SmartDataframe, skill, skills

    # If no views provided, auto-discover them
    if view_names is None:
        view_names = discover_views(engine, schemas)

    primary_schema = os.getenv("DB_SCHEMA", schemas[0])

    # 1. Register SQL Skill
    skill_name = "execute_sql_query"
    if not any(s.name == skill_name for s in skills._skills):
        @skill
        def execute_sql_query(query: str):
            """
            Executes a read-only SQL SELECT query against the PostgreSQL database.
            Use this for complex joins or aggregations across multiple views.
            Authorized views: {authorized_views}
            """
            print(f"Executing SQL: {query}")
            if not query.strip().lower().startswith("select") and "set search_path" not in query.lower():
                raise ValueError("Only SELECT queries are allowed via this skill.")
            
            scoped_query = f"SET search_path TO {primary_schema}, public; {query}"
            with engine.connect() as conn:
                return pd.read_sql(text(scoped_query), conn)

        execute_sql_query.__doc__ = execute_sql_query.__doc__.format(authorized_views=", ".join(view_names))

    # 2. Load DataFrames
    sdfs = []
    for view in view_names:
        try:
            # Determine which schema the view belongs to (simple check)
            view_schema = primary_schema
            # In a real scenario we'd query which schema this view is in, 
            # for now we assume it's reachable or in the search path
            
            df = pd.read_sql(f"SELECT * FROM {view} LIMIT 5000", engine)
            metadata = fetch_table_metadata(engine, view)

            sdf = SmartDataframe(
                df,
                name=view,
                description=f"View '{view}' with columns: {metadata}"
            )
            sdf.schema.name = view
            sdfs.append(sdf)
            print(f"- Loaded {view} ({len(df)} rows)")
        except Exception as e:
            print(f"- Error loading {view}: {e}")

    if not sdfs:
        raise RuntimeError("No dataframes could be loaded. Check DB and connection.")

    # 3. Configure Agent
    agent_config = {
        "llm": llm,
        "verbose": True,
        "response_parser": SandboxResponseParser,
        "enforce_privacy": False,
        "enable_cache": False,
        "use_error_correction_framework": True,
        "custom_whitelisted_dependencies": ["sqlalchemy", "psycopg2", "plotly", "geopandas", "folium"]
    }

    return Agent(sdfs, config=agent_config)

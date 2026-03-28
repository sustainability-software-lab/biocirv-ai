from sqlalchemy import text
from typing import List

def fetch_table_metadata(engine, table_name: str, schema: str = None) -> str:
    """Fetches column names and types for a given table."""
    query = text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = :table
        """ + ("AND table_schema = :schema" if schema else ""))

    params = {"table": table_name}
    if schema:
        params["schema"] = schema

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            columns = [f"{row[0]} ({row[1]})" for row in result]
        return ", ".join(columns)
    except Exception:
        return "Unknown columns"

def discover_views(engine, schemas: List[str] = ["ca_biositing", "analytics"]) -> List[str]:
    """Automatically discovers all views in the specified schemas."""
    query = text("""
        SELECT table_name
        FROM information_schema.views
        WHERE table_schema = ANY(:schemas)
        AND table_name NOT LIKE 'pg_%%'
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"schemas": schemas})
            views = [row[0] for row in result]
        print(f"Auto-discovered {len(views)} views in schemas: {', '.join(schemas)}")
        return views
    except Exception as e:
        print(f"WARNING: View discovery failed: {e}")
        return []

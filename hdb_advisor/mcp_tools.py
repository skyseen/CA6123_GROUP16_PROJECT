"""Serve narrow local tool contracts via FastMCP and LangChain MCPAdapter."""
import argparse
from collections.abc import Sequence
from pathlib import Path

from fastmcp import FastMCP
from langchain.mcp import MCPAdapter

from .exports import save_research_shortlist
from .onemap import OneMap, route_to_anchor
from .state import RouteResult


def make_server(db: Path, client: OneMap | None = None, *,
                approved_export: tuple[Path, str, str] | None = None) -> FastMCP:
    server = FastMCP("hdb-advisor")

    @server.tool(name="route_to_anchor")
    def get_route(address_key: str, anchor_id: str) -> str:
        """Return a dated OneMap commute estimate from a dataset block key to a resolved destination."""
        return route_to_anchor(db, address_key, anchor_id, client).model_dump_json()

    if approved_export is not None:
        @server.tool(name="save_research_shortlist")
        def save() -> str:
            """Save the shortlist already approved in the human output gate."""
            return save_research_shortlist(*approved_export)
    return server


async def call_tools(server, name: str, arguments: Sequence[dict]) -> list[str]:
    async with MCPAdapter(server) as adapter:
        tools = {item.name: item for item in await adapter.list_tools()}
        tool = tools[name]
        results = []
        for kwargs in arguments:
            content = await tool.ainvoke(kwargs)
            if len(content) != 1 or content[0].get("type") != "text":
                raise ValueError(f"Unexpected {name} MCP response")
            results.append(content[0]["text"])
        return results


async def mcp_routes(db: Path, client: OneMap | None, addresses: list[str], anchor_id: str) -> list[RouteResult]:
    rows = await call_tools(make_server(db, client), "route_to_anchor",
                            [{"address_key": address, "anchor_id": anchor_id} for address in addresses])
    return [RouteResult.model_validate_json(row) for row in rows]


async def mcp_export(db: Path, directory: Path, report: str, thread: str) -> str:
    server = make_server(db, approved_export=(directory, report, thread))
    return (await call_tools(server, "save_research_shortlist", [{}]))[0]


def main():
    parser = argparse.ArgumentParser(description="Run the cache-only route tool over MCP stdio.")
    parser.add_argument("--db", type=Path, default=Path("data/hdb.db"))
    args = parser.parse_args()
    make_server(args.db).run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()

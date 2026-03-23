from __future__ import annotations


CATAN_RULES_SUMMARY = """You are playing Settlers of Catan in a benchmark harness.

Core rules:
- Players collect WOOD, BRICK, SHEEP, WHEAT, and ORE.
- Settlements and cities score victory points; roads help expand and contest longest road.
- The first player to reach the configured victory-point target wins.
- Information is partially observable: public state is shared, but each player has private cards and private memory.
- Domestic trades are voluntary exchanges between players. A trade offer proposes resources to give and resources to receive.
- When deciding on a trade, consider both immediate value and how much you improve an opponent's position.

Benchmark expectations:
- Use only the information in your observation.
- Choose exactly one legal action.
- Keep any optional private memory write short, factual, and useful for your own later decisions.
- Do not reveal or invent hidden information about other players.
"""

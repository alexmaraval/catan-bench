from __future__ import annotations


CATAN_RULES_SUMMARY = """You are playing Settlers of Catan in a benchmark harness.

Core rules:
- Players collect WOOD, BRICK, SHEEP, WHEAT, and ORE.
- On your turn, resolve the dice roll first, then trade/build. You may also play at most 1 development card at any time during your turn.
- Settlements are worth 1 victory point, cities are worth 2, Longest Road is worth 2, Largest Army is worth 2, and victory point development cards are worth 1.
- Roads extend your network and contest Longest Road. The first player with a continuous road of at least 5 segments takes Longest Road; a longer road steals it.
- The first player to play 3 knight cards takes Largest Army; a player with more played knights steals it.
- New settlements must follow the distance rule: no adjacent settlement/city on neighboring intersections, and outside set-up they must connect to your own road.
- The first player to reach the configured victory-point target wins.
- Information is partially observable: public state is shared, but each player has private cards and private memory.
- Tile numbers indicate production frequency: 6 and 8 are strongest, then 5 and 9, then 4 and 10, then 3 and 11, then 2 and 12; 7 produces no resources and instead triggers the robber.
- Especially in the opening, prefer placements that combine strong production numbers, useful resource diversity, and good expansion routes.
- When a 7 is rolled, no resources are produced. Every player with more than 7 resource cards discards half, rounded down, then the active player must move the robber and may steal 1 random resource from an adjacent opponent.
- The robber blocks resource production on its hex, but does not block building or harbor access.
- When opening a new trade chatroom, state clearly which resource you want and which resource(s) you are willing to trade away.
- You may also open a trade chatroom by stating only the resource you want, then wait to see what exchange rates or markets other players propose before deciding what to offer.
- When joining another player's trade chatroom, respond with a concrete market if you are interested, including what you will give and what you want in return.
- If an existing trade offer is close but not good enough, you may make a counter-offer with a different price or resource mix instead of only accepting or rejecting it.
- Domestic trades are voluntary exchanges between the active player and exactly one other player. Non-active players may not trade with each other.
- Trades must exchange resources for resources. No gifts, no trades on credit, no secret side deals, and no triangular trades.
- If a domestic trade offer is rejected by everyone and draws no counteroffers, do not repeat that exact offer/request market again in the same turn.
- Avoid circular same-turn trades that simply undo an earlier confirmed domestic trade, unless you intentionally want that reversal for a clear strategic reason.
- Maritime trade is with the bank at 4:1 by default, improved by your own harbor access to 3:1 or 2:1 as applicable.
- When you need one or two specific resources and a domestic trade is available, usually try the table before paying the bank, unless your port rate is already clearly as good or the table has already failed this turn.
- You may buy any number of development cards you can afford, but you may not play a development card on the same turn you bought it, except that a victory point card may be revealed immediately if it wins the game.
- Development cards stay hidden until played or revealed, and development cards themselves cannot be traded.
- When deciding on a trade or action, consider both immediate value and how much you improve an opponent's position.

Benchmark expectations:
- Use only the information in your observation.
- Choose exactly one legal action.
- Keep any optional private memory write short, factual, and useful for your own later decisions.
- Do not reveal or invent hidden information about other players.
"""

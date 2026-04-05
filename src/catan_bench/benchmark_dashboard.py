"""Streamlit UI for the benchmark evaluation tab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import plotly.graph_objects as go

from .benchmark import (
    EloState,
    GameRecord,
    PairwiseStrengthState,
    RubricScores,
    collect_game_records,
    compute_elo_ratings,
    compute_head_to_head,
    compute_pairwise_strength,
    compute_rubric_scores,
)

# Palette for models (up to 8 distinct colors)
_MODEL_PALETTE = [
    "#60a5fa",  # blue
    "#f87171",  # red
    "#34d399",  # emerald
    "#fbbf24",  # amber
    "#a78bfa",  # violet
    "#fb923c",  # orange
    "#2dd4bf",  # teal
    "#f472b6",  # pink
]


def _short_model_name(name: str) -> str:
    """Shorten provider/model paths for display."""
    parts = name.rsplit("/", 1)
    return parts[-1] if len(parts) > 1 else name


def _model_color(index: int) -> str:
    return _MODEL_PALETTE[index % len(_MODEL_PALETTE)]


# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------


def _load_benchmark_data(
    st: Any, base_run_dir: Path
) -> tuple[list[GameRecord], EloState, PairwiseStrengthState, dict[str, RubricScores]] | None:
    """Load and compute benchmark data, cached by Streamlit."""

    @st.cache_data(ttl=30, show_spinner="Scanning games...")
    def _cached_load(base_dir_str: str) -> tuple[list[dict], dict, dict, dict] | None:
        games = collect_game_records(Path(base_dir_str))
        if not games:
            return None
        elo = compute_elo_ratings(games)
        pairwise = compute_pairwise_strength(games)
        rubrics = compute_rubric_scores(games)

        # Serialize for caching (Streamlit cache requires picklable data)
        games_ser = [
            {
                "game_id": g.game_id,
                "run_dir": str(g.run_dir),
                "num_turns": g.num_turns,
                "player_models": g.player_models,
                "winner_ids": g.winner_ids,
                "player_vps": g.player_vps,
                "analysis": g.analysis,
            }
            for g in games
        ]
        elo_ser = {
            "ratings": elo.ratings,
            "games_played": elo.games_played,
            "wins": elo.wins,
            "total_vp": elo.total_vp,
            "history": elo.history,
        }
        pairwise_ser = {
            "ratings": pairwise.ratings,
            "scores": pairwise.scores,
            "wins": pairwise.wins,
            "losses": pairwise.losses,
            "draws": pairwise.draws,
        }
        rubrics_ser = {
            m: r.as_dict() | {"overall": r.overall} for m, r in rubrics.items()
        }
        return games_ser, elo_ser, pairwise_ser, rubrics_ser

    result = _cached_load(str(base_run_dir))
    if result is None:
        return None

    games_ser, elo_ser, pairwise_ser, rubrics_ser = result

    # Deserialize
    games = [
        GameRecord(
            game_id=g["game_id"],
            run_dir=Path(g["run_dir"]),
            num_turns=g["num_turns"],
            player_models=g["player_models"],
            winner_ids=g["winner_ids"],
            player_vps=g["player_vps"],
            analysis=g["analysis"],
        )
        for g in games_ser
    ]
    elo = EloState(
        ratings=elo_ser["ratings"],
        games_played=elo_ser["games_played"],
        wins=elo_ser["wins"],
        total_vp=elo_ser["total_vp"],
        history=elo_ser["history"],
    )
    pairwise = PairwiseStrengthState(
        ratings=pairwise_ser["ratings"],
        scores=pairwise_ser["scores"],
        wins=pairwise_ser["wins"],
        losses=pairwise_ser["losses"],
        draws=pairwise_ser["draws"],
    )
    rubrics = {}
    for m, rd in rubrics_ser.items():
        rubrics[m] = RubricScores(
            trading=rd["Trading"],
            strategy=rd["Strategy"],
            manipulation=rd["Manipulation"],
            resource_management=rd["Resource Mgmt"],
            game_mechanics=rd["Game Mechanics"],
        )
    return games, elo, pairwise, rubrics


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------


def render_benchmark_tab(st: Any, *, base_run_dir: Path) -> None:
    """Render the benchmark evaluation tab."""
    data = _load_benchmark_data(st, base_run_dir)
    if data is None:
        st.info("No completed games found. Play some games and come back!")
        return

    games, elo, pairwise, rubrics = data

    # Summary metrics
    total_games = len(games)
    unique_models = sorted(elo.ratings.keys())
    total_decisions = sum(
        g.analysis.get("game_summary", {}).get("total_decisions", 0)
        for g in games
        if g.analysis
    )

    st.subheader("Benchmark Overview")
    cols = st.columns(4)
    cols[0].metric("Completed Games", total_games)
    cols[1].metric("Unique Models", len(unique_models))
    cols[2].metric("Total Decisions", f"{total_decisions:,}")
    # Check if we have cross-model games
    cross_model_games = sum(1 for g in games if len(set(g.player_models.values())) > 1)
    cols[3].metric("Cross-Model Games", cross_model_games)

    if len(unique_models) < 2:
        st.warning(
            "Only one model found across all games. "
            "ELO rankings require at least two different models. "
            "Rubric scores are still shown below."
        )

    # Leaderboard
    _render_leaderboard(st, elo, pairwise, rubrics)

    # Charts side by side
    col_radar, col_elo = st.columns(2)
    with col_radar:
        _render_rubric_radar(st, rubrics)
    with col_elo:
        _render_elo_history(st, elo)

    # Head-to-head
    if len(unique_models) > 1:
        _render_head_to_head(st, games, unique_models)

    # Per-model breakdown
    _render_model_breakdowns(st, rubrics, elo)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _render_leaderboard(
    st: Any,
    elo: EloState,
    pairwise: PairwiseStrengthState,
    rubrics: dict[str, RubricScores],
) -> None:
    st.subheader("ELO Leaderboard")
    st.caption(
        "BT is an order-invariant Bradley-Terry batch rating fit from aggregate "
        "pairwise results. Pairwise score is wins plus half draws over all pairings."
    )

    ranked_models = sorted(
        elo.ratings.keys(), key=lambda m: elo.ratings[m], reverse=True
    )

    rows = []
    for rank, model in enumerate(ranked_models, 1):
        gp = elo.games_played.get(model, 0)
        wins = elo.wins.get(model, 0)
        total_vp = elo.total_vp.get(model, 0)
        rubric = rubrics.get(model)

        rows.append(
            {
                "Rank": rank,
                "Model": _short_model_name(model),
                "ELO": round(elo.ratings[model]),
                "BT": round(pairwise.ratings.get(model, 1500)),
                "Games": gp,
                "Win Rate": (wins / max(gp, 1)) * 100,
                "Avg VP": round(total_vp / max(gp, 1), 1),
                "Pairwise": pairwise.scores.get(model, 0.0) * 100.0,
                "W-L-D": (
                    f"{pairwise.wins.get(model, 0)}-"
                    f"{pairwise.losses.get(model, 0)}-"
                    f"{pairwise.draws.get(model, 0)}"
                ),
                "Trading": round(rubric.trading, 1) if rubric else 0,
                "Strategy": round(rubric.strategy, 1) if rubric else 0,
                "Manipulation": round(rubric.manipulation, 1) if rubric else 0,
                "Resources": round(rubric.resource_management, 1) if rubric else 0,
                "Mechanics": round(rubric.game_mechanics, 1) if rubric else 0,
                "Overall": round(rubric.overall, 1) if rubric else 0,
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Model": st.column_config.TextColumn(width="medium"),
            "ELO": st.column_config.NumberColumn(format="%d"),
            "BT": st.column_config.NumberColumn(format="%d"),
            "Win Rate": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.2f%%"
            ),
            "Avg VP": st.column_config.NumberColumn(format="%.1f"),
            "Pairwise": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
            "W-L-D": st.column_config.TextColumn(width="small"),
            "Trading": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
            "Strategy": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
            "Manipulation": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
            "Resources": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
            "Mechanics": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
            "Overall": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
        },
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------


def _render_rubric_radar(st: Any, rubrics: dict[str, RubricScores]) -> None:
    st.subheader("Rubric Comparison")
    categories = [
        "Trading",
        "Strategy",
        "Manipulation",
        "Resource Mgmt",
        "Game Mechanics",
    ]

    fig = go.Figure()
    for i, (model, scores) in enumerate(rubrics.items()):
        values = [
            scores.trading,
            scores.strategy,
            scores.manipulation,
            scores.resource_management,
            scores.game_mechanics,
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                name=_short_model_name(model),
                fill="toself",
                opacity=0.5,
                line=dict(color=_model_color(i), width=2),
                fillcolor=_model_color(i),
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 100],
                gridcolor="rgba(148,163,184,0.2)",
                color="#94a3b8",
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.2)",
                color="#e2e8f0",
                tickfont=dict(size=11),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", size=11),
        ),
        margin=dict(l=60, r=60, t=30, b=30),
        height=420,
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# ELO history chart
# ---------------------------------------------------------------------------


def _render_elo_history(st: Any, elo: EloState) -> None:
    st.subheader("ELO Rating History")

    if not elo.history:
        st.caption("No history to display.")
        return

    models = sorted(elo.ratings.keys())
    rows = []
    for game_idx, snapshot in enumerate(elo.history):
        for model in models:
            rows.append(
                {
                    "Game": game_idx + 1,
                    "Model": _short_model_name(model),
                    "ELO": round(snapshot.get(model, 1500)),
                }
            )

    df = pd.DataFrame(rows)
    model_names = [_short_model_name(m) for m in models]
    colors = [_model_color(i) for i in range(len(models))]

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
        .encode(
            x=alt.X("Game:Q", title="Game #", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("ELO:Q", title="ELO Rating", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=model_names, range=colors),
                legend=alt.Legend(title="Model"),
            ),
            tooltip=["Model", "Game", "ELO"],
        )
        .properties(height=360)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="rgba(148,163,184,0.15)",
            labelColor="#94a3b8",
            titleColor="#e2e8f0",
        )
        .configure_legend(
            labelColor="#e2e8f0",
            titleColor="#e2e8f0",
        )
    )
    st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------------------
# Head-to-head matrix
# ---------------------------------------------------------------------------


def _render_head_to_head(st: Any, games: list[GameRecord], models: list[str]) -> None:
    st.subheader("Head-to-Head Win Rates")

    h2h = compute_head_to_head(games)
    short_names = [_short_model_name(m) for m in models]

    # Build win-rate matrix
    matrix: list[list[float | None]] = []
    annotations: list[list[str]] = []
    for m_a in models:
        row_vals: list[float | None] = []
        row_annot: list[str] = []
        for m_b in models:
            if m_a == m_b:
                row_vals.append(None)
                row_annot.append("-")
            else:
                record = h2h.get(m_a, {}).get(m_b, {"wins": 0, "losses": 0, "draws": 0})
                total = record["wins"] + record["losses"] + record["draws"]
                if total > 0:
                    wr = record["wins"] / total
                    row_vals.append(wr)
                    row_annot.append(
                        f"{wr:.0%}\n({record['wins']}W {record['losses']}L)"
                    )
                else:
                    row_vals.append(None)
                    row_annot.append("N/A")
        matrix.append(row_vals)
        annotations.append(row_annot)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=short_names,
            y=short_names,
            colorscale=[
                [0, "#dc2626"],
                [0.5, "#1e293b"],
                [1, "#22c55e"],
            ],
            zmin=0,
            zmax=1,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertemplate="Row %{y} vs Col %{x}: %{text}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title=dict(text="Win Rate", font=dict(color="#e2e8f0")),
                tickvals=[0, 0.5, 1],
                ticktext=["0%", "50%", "100%"],
                tickfont=dict(color="#94a3b8"),
            ),
        )
    )
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Opponent", font=dict(color="#e2e8f0")),
            tickfont=dict(color="#e2e8f0"),
        ),
        yaxis=dict(
            title=dict(text="Model", font=dict(color="#e2e8f0")),
            tickfont=dict(color="#e2e8f0"),
            autorange="reversed",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=max(300, 80 * len(models)),
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Per-model rubric breakdown
# ---------------------------------------------------------------------------


def _render_model_breakdowns(
    st: Any, rubrics: dict[str, RubricScores], elo: EloState
) -> None:
    st.subheader("Per-Model Rubric Breakdown")

    ranked = sorted(elo.ratings.keys(), key=lambda m: elo.ratings[m], reverse=True)

    for model in ranked:
        rubric = rubrics.get(model)
        if rubric is None:
            continue

        gp = elo.games_played.get(model, 0)
        wins = elo.wins.get(model, 0)

        with st.expander(
            f"{_short_model_name(model)} — ELO {round(elo.ratings[model])} | "
            f"{wins}W / {gp}G | Overall: {rubric.overall:.1f}/100"
        ):
            cols = st.columns(5)
            labels = [
                "Trading",
                "Strategy",
                "Manipulation",
                "Resource Mgmt",
                "Game Mechanics",
            ]
            values = [
                rubric.trading,
                rubric.strategy,
                rubric.manipulation,
                rubric.resource_management,
                rubric.game_mechanics,
            ]
            for col, label, val in zip(cols, labels, values):
                col.metric(label, f"{val:.1f}")

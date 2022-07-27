from copy import deepcopy
from typing import List, Literal
import numpy as np
import altair as alt


def show_kldf(kldf, rule_value):
    kldf = deepcopy(kldf)
    kldf["color"] = np.where(kldf["KLD"] < rule_value, True, False)
    chart = (
        alt.Chart(kldf)
        .mark_bar()
        .encode(
            x=alt.X("date:O", axis=alt.Axis(labels=False)),
            y=alt.Y("KLD", scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("color", legend=None),
        )
    ).properties(height=250)
    kldf["y"] = rule_value
    rule = alt.Chart(kldf).mark_rule(color="red").encode(y="y")
    return chart + rule


def timeline(
    df,
    x,
    y,
    subcorpus,
    stack: Literal["stack", "center"] | None,
    order: List[str],
    name="timeline",
) -> alt.Chart:
    """
    Creates colorful timeline
    Parameters
    ----------
    stack : Literal["stack", "center"] | None
        "stack" for a timeline for zero, including minus values (good for showing distances for instance)
        "center" for a timeline centered aroud the middle of the y axis (good for showing frequencies)
    order : List[str]
        a list of the values in the wanted order (for instance ordering elements with  distance from highest to lowest)
    """
    selection = alt.selection_multi(fields=["element"], bind="legend")
    chart = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X(f"{x}:O", title="Date"),
            y=alt.Y(
                f"{y}",
                stack=stack,
                # axis=None,
                title="Significance",
            ),
            color=alt.Color(
                "element",
                scale=alt.Scale(scheme="rainbow"),
                sort=alt.Sort(order),  # field=f"-{y}"),
                legend=alt.Legend(columns=2, labelLimit=1000),
            ),
            tooltip=alt.Tooltip(["element", y]),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .interactive()
        .properties(width=900, title=f"{subcorpus.capitalize()} Timeline")
        .add_selection(selection)
    )
    chart.save(f"results/{subcorpus}/timelines/{name}.html")
    return chart

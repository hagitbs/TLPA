from copy import deepcopy
from typing import List, Literal
import numpy as np
import altair as alt


def metric_bar_chart(df, rule_value, metric):
    df = deepcopy(df)
    df["color"] = np.where(df[metric] < rule_value, True, False)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("date:O", axis=alt.Axis(labels=False)),
            y=alt.Y(metric),  # , scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("color", legend=None),
        )
    ).properties(height=250)
    df["y"] = rule_value
    rule = alt.Chart(df).mark_rule(color="red").encode(y="y")
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


def moving_avg(df):
    base = alt.Chart(df).encode(alt.X("index", axis=alt.Axis(title=None)))
    line = base.mark_line().encode(
        alt.Y("ma:Q", scale=alt.Scale(type="log"))
        # axis=alt.Axis(title='Precipitation (inches)', titleColor='#5276A7')
    )
    area = base.mark_area(opacity=0.3, color="#57A44C").encode(
        alt.Y("max:Q", scale=alt.Scale(type="log")),
        #   axis=alt.Axis(title='Avg. Temperature (°C)', titleColor='#57A44C')),
        alt.Y2("min:Q"),
    )
    return alt.layer(area, line)


def sockpuppet_matrix(spd, corpus1_name, corpus2_name):
    cdf = (
        spd.rename_axis(index=[corpus1_name])
        .melt(ignore_index=False, var_name=corpus2_name)
        .reset_index()
    )
    return (
        alt.Chart(cdf)
        .mark_rect()
        .encode(
            x=alt.X(f"{corpus1_name}:O", axis=alt.Axis(orient="top")),
            y=f"{corpus2_name}:O",
            color="value",
        )
    )

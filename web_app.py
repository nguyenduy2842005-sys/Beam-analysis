from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from beam_core import BeamInput, BeamResult, solve_beam


PLOT_HEIGHT = 315
PLOT_BG = "#ffffff"
GRID = "#d9e0e8"
AXIS = "#263238"


st.set_page_config(
    page_title="Beam Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f3f5f8;
        }
        #MainMenu, footer, header, [data-testid="stToolbar"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d8dee8;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.5rem;
            max-width: 1680px;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 0.2rem 0 0.8rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #d8dee8;
            border-radius: 6px;
            padding: 10px 12px;
        }
        .metric-label {
            color: #5f6b7a;
            font-size: 0.78rem;
            margin-bottom: 3px;
        }
        .metric-value {
            color: #101828;
            font-weight: 700;
            font-size: 1.08rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: #d8dee8;
        }
        .stPlotlyChart {
            background: #ffffff;
            border: 1px solid #d8dee8;
            border-radius: 6px;
            padding: 6px;
        }
        textarea {
            font-family: Consolas, "Courier New", monospace !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
def reset_problem() -> None:
    keys_to_clear = [
        "point_loads",
        "point_moments",
        "udls",
        "uvls",
        "pl_editor",
        "m_editor",
        "udl_editor",
        "uvl_editor",
        "last_result",
        "last_input",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

def clean_rows(df: pd.DataFrame, columns: Iterable[str]) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for _, row in df.iterrows():
        values: list[float] = []
        skip = False
        for column in columns:
            value = row.get(column)
            if value is None or pd.isna(value) or value == "":
                skip = True
                break
            try:
                number = float(value)
            except (TypeError, ValueError):
                skip = True
                break
            if not math.isfinite(number):
                skip = True
                break
            values.append(number)
        if not skip:
            rows.append(tuple(values))
    return rows


def validate_input(data: BeamInput) -> list[str]:
    errors: list[str] = []
    if data.length <= 0:
        errors.append("Chiều dài dầm phải lớn hơn 0.")

    for name, rows in [
        ("Point Load", [(x,) for _, x in data.point_loads]),
        ("Point Moment", [(x,) for _, x in data.point_moments]),
    ]:
        for i, (x_pos,) in enumerate(rows, 1):
            if x_pos < 0 or x_pos > data.length:
                errors.append(f"{name} dòng {i}: vị trí x phải nằm trong [0, L].")

    for name, rows in [("UDL", data.udls), ("UVL", data.uvls)]:
        for i, (_, x1, x2) in enumerate(rows, 1):
            if x1 < 0 or x2 < 0 or x1 > data.length or x2 > data.length or x2 <= x1:
                errors.append(f"{name} dòng {i}: cần 0 <= x1 < x2 <= L.")
    return errors


def base_figure(title: str, length: float, y_title: str = "") -> go.Figure:
    fig = go.Figure()

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(
                size=18,
                color="#263238"
            )
        ),

        height=PLOT_HEIGHT,

        margin=dict(
            l=55,
            r=20,
            t=65,      # tăng khoảng trống phía trên
            b=45
        ),

        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,

        showlegend=False,

        xaxis=dict(
            title="x (m)",
            range=[-length / 20, 1.05 * length],
            gridcolor=GRID,
            zerolinecolor=AXIS,
            linecolor=AXIS,
            mirror=True,
            ticks="outside",
            title_font=dict(size=14)
        ),

        yaxis=dict(
            title=y_title,
            gridcolor=GRID,
            zerolinecolor=AXIS,
            linecolor=AXIS,
            mirror=True,
            ticks="outside",
            title_font=dict(size=14)
        ),
    )

    return fig
def draw_supports(fig: go.Figure, data: BeamInput) -> None:
    l = data.length
    if data.beam_type == "simple":
        for x0 in [0, l]:
            fig.add_trace(
                go.Scatter(
                    x=[x0, x0 + l / 34, x0 - l / 34, x0],
                    y=[0, -0.37, -0.37, 0],
                    fill="toself",
                    mode="lines",
                    line={"color": "#ef1d14", "width": 1},
                    fillcolor="#ef1d14",
                    hoverinfo="skip",
                )
            )
    else:
        fig.add_shape(
            type="rect",
            x0=l,
            x1=l + l / 42,
            y0=-0.42,
            y1=0.42,
            fillcolor="#ef1d14",
            line={"color": "#ef1d14"},
        )


def plot_load_diagram(data: BeamInput) -> go.Figure:
    l = data.length
    fig = base_figure("Load diagram", l)
    fig.update_yaxes(range=[-1.05, 1.05], showticklabels=False, title="")
    fig.add_trace(
        go.Scatter(
            x=[0, l],
            y=[0, 0],
            mode="lines",
            line={"color": "#7a7f85", "width": 8},
            hoverinfo="skip",
        )
    )
    draw_supports(fig, data)

    for load, x_pos in data.point_loads:
        y_tip, y_tail = (-0.06, -0.74) if load > 0 else (0.06, 0.74)
        fig.add_annotation(
            x=x_pos,
            y=y_tip,
            ax=x_pos,
            ay=y_tail,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.1,
            arrowwidth=2,
            arrowcolor="#0b5fff",
            text="",
        )
        fig.add_annotation(x=x_pos, y=y_tail, text=f"{load:g} kN", showarrow=False, font={"size": 11, "color": "#0b5fff"})

    for moment, x_pos in data.point_moments:
        radius = 0.22
        theta = np.linspace(-np.pi / 2, np.pi / 2, 36)
        y_offset = 0.38 if moment > 0 else -0.38
        fig.add_trace(
            go.Scatter(
                x=x_pos + radius * np.cos(theta),
                y=y_offset + radius * np.sin(theta),
                mode="lines",
                line={"color": "#a100ff", "width": 2},
                hovertemplate=f"M = {moment:g} kNm<br>x = {x_pos:g} m<extra></extra>",
            )
        )
        fig.add_annotation(x=x_pos, y=y_offset * 1.55, text=f"{moment:g} kNm", showarrow=False, font={"size": 11})

    for q, x1, x2 in data.udls:
        y_bot = -0.58 if q > 0 else 0.58
        fig.add_trace(
            go.Scatter(
                x=[x1, x2, x2, x1, x1],
                y=[0, 0, y_bot, y_bot, 0],
                fill="toself",
                mode="lines",
                line={"color": "#168f2c", "width": 1},
                fillcolor="rgba(22,143,44,0.16)",
                hovertemplate=f"UDL: {q:g} kN/m<br>{x1:g} -> {x2:g} m<extra></extra>",
            )
        )
        for x_val in np.linspace(x1, x2, max(2, int(np.ceil((x2 - x1) / 0.5)) + 1)):
            fig.add_annotation(
                x=x_val,
                y=y_bot,
                ax=x_val,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.9,
                arrowwidth=1.5,
                arrowcolor="#168f2c",
                text="",
            )
        fig.add_annotation(x=(x1 + x2) / 2, y=y_bot * 1.12, text=f"{q:g} kN/m", showarrow=False, font={"size": 11, "color": "#168f2c"})

    for q, x1, x2 in data.uvls:
        y_bot = -0.58 if q > 0 else 0.58
        y_start, y_end = (0, y_bot) if data.uvl_type == "increase" else (y_bot, 0)
        fig.add_trace(
            go.Scatter(
                x=[x1, x2, x2, x1, x1],
                y=[0, 0, y_end, y_start, 0],
                fill="toself",
                mode="lines",
                line={"color": "#0b5fff", "width": 1},
                fillcolor="rgba(11,95,255,0.14)",
                hovertemplate=f"UVL: {q:g} kN/m<br>{x1:g} -> {x2:g} m<extra></extra>",
            )
        )
        for x_val in np.linspace(x1, x2, max(2, int(np.ceil((x2 - x1) / 0.5)) + 1)):
            ratio = (x_val - x1) / (x2 - x1)
            intensity = ratio if data.uvl_type == "increase" else 1 - ratio
            if intensity <= 0.05:
                continue
            y_val = y_bot * intensity
            fig.add_annotation(
                x=x_val,
                y=y_val,
                ax=x_val,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.9,
                arrowwidth=1.5,
                arrowcolor="#0b5fff",
                text="",
            )
        fig.add_annotation(x=(x1 + x2) / 2, y=y_bot * 1.12, text=f"{q:g} kN/m", showarrow=False, font={"size": 11, "color": "#0b5fff"})

    return fig


def plot_sfd(result: BeamResult) -> go.Figure:
    fig = base_figure("Shear Force Diagram", float(result.x[-1]), "Shear Force (kN)")
    fig.add_trace(
        go.Scatter(
            x=result.x,
            y=result.shear,
            mode="lines",
            fill="tozeroy",
            line={"color": "#0b5fff", "width": 2},
            fillcolor="rgba(11,95,255,0.20)",
            hovertemplate="x = %{x:.2f} m<br>V = %{y:.2f} kN<extra></extra>",
        )
    )
    return fig


def plot_bmd(result: BeamResult) -> go.Figure:
    fig = base_figure("Bending Moment Diagram", float(result.x[-1]), "Bending Moment (kNm)")
    fig.add_trace(
        go.Scatter(
            x=result.x,
            y=result.moment,
            mode="lines",
            fill="tozeroy",
            line={"color": "#ff2b2b", "width": 2},
            fillcolor="rgba(255,43,43,0.22)",
            hovertemplate="x = %{x:.2f} m<br>M = %{y:.2f} kNm<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_elastic(data: BeamInput, result: BeamResult | None) -> go.Figure:
    fig = plot_load_diagram(data)
    fig.update_layout(title={"text": "Elastic curve", "x": 0.5, "font": {"size": 15}})
    fig.data = tuple(trace for trace in fig.data[:1])
    fig.layout.annotations = tuple()
    draw_supports(fig, data)
    if result is not None:
        max_w = float(np.max(np.abs(result.deflection)))
        if max_w > 0:
            y = -result.deflection * (0.72 / max_w)
        else:
            y = result.deflection
        fig.add_trace(
            go.Scatter(
                x=result.x,
                y=y,
                mode="lines",
                line={"color": "#ff8800", "width": 4},
                hovertemplate="x = %{x:.2f} m<br>w/EI = %{customdata:.2f}<extra></extra>",
                customdata=result.deflection,
            )
        )
    fig.update_yaxes(range=[-1.05, 1.05], title="Deflection (visual)")
    return fig


def metric_strip(result: BeamResult | None, data: BeamInput) -> None:

    if result is None:

        values = [
            ("Span", f"{data.length:.2f} m"),
            ("Point loads", str(len(data.point_loads))),
            ("UDL / UVL",
             f"{len(data.udls)} / {len(data.uvls)}"),
            ("Status", "Ready"),
        ]

    else:

        idx_v = int(np.argmax(np.abs(result.shear)))
        idx_m = int(np.argmax(np.abs(result.moment)))
        idx_w = int(np.argmax(np.abs(result.deflection)))


        # Dầm đơn giản
        if data.beam_type == "simple":

            r1 = getattr(result, "r1", 0)
            r2 = getattr(result, "r2", 0)

            values = [

                (
                    "R1 / R2",
                    f"{r1:.2f} / {r2:.2f} kN"
                ),

                (
                    "Vmax",
                    f"{result.shear[idx_v]:.2f} kN"
                ),

                (
                    "Mmax",
                    f"{result.moment[idx_m]:.2f} kNm"
                ),

                (
                    "wmax",
                    f"{result.deflection[idx_w]:.4f}"
                )

            ]


        # Dầm console
        else:

            rv = getattr(result, "rv_fixed", 0)
            mr = getattr(result, "mr_fixed", 0)

            values = [

                (
                    "RV",
                    f"{rv:.2f} kN"
                ),

                (
                    "MR",
                    f"{mr:.2f} kNm"
                ),

                (
                    "Vmax",
                    f"{result.shear[idx_v]:.2f} kN"
                ),

                (
                    "wmax",
                    f"{result.deflection[idx_w]:.4f}"
                )

            ]


    cards = "".join(

        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """

        for label,value in values

    )


    st.markdown(
        f"""
        <div class='metric-strip'>
        {cards}
        </div>
        """,
        unsafe_allow_html=True
    )
def reset_problem() -> None:
    """
    Tạo bài toán mới:
    - Xóa bảng tải
    - Xóa kết quả cũ
    - Xóa biểu đồ cũ
    """

    keys = [
        "point_loads",
        "point_moments",
        "udls",
        "uvls",

        "pl_editor",
        "m_editor",
        "udl_editor",
        "uvl_editor",

        "last_result",
        "last_input",
    ]

    for key in keys:
        st.session_state.pop(key, None)

    st.rerun()



def sidebar_inputs() -> BeamInput:

    with st.sidebar:

        st.header("Input")


        if st.button(
            "🆕 New Model",
            type="primary",
            width="stretch"
        ):
            reset_problem()


        st.divider()


        length = st.number_input(
            "Chiều dài dầm L (m)",
            min_value=0.01,
            value=10.0,
            step=0.5,
            format="%.2f"
        )


        beam_type_label = st.radio(
            "Type of Beam",
            [
                "Simply Supported",
                "Cantilever"
            ],
            horizontal=True
        )


        uvl_type_label = st.radio(
            "UVL Type",
            [
                "Increase",
                "Decrease"
            ],
            horizontal=True
        )


        st.divider()

        st.caption(
            "Nhập tải trong các bảng ở vùng làm việc chính."
        )


    return BeamInput(
        length=float(length),

        beam_type=
        "simple"
        if beam_type_label == "Simply Supported"
        else "cantilever",

        uvl_type=
        "increase"
        if uvl_type_label == "Increase"
        else "decrease",
    )



def load_tables(base: BeamInput) -> BeamInput:


    defaults = {

        "point_loads":
            pd.DataFrame(
                columns=[
                    "P (kN)",
                    "x (m)"
                ]
            ),

        "point_moments":
            pd.DataFrame(
                columns=[
                    "M (kNm)",
                    "x (m)"
                ]
            ),

        "udls":
            pd.DataFrame(
                columns=[
                    "q (kN/m)",
                    "x1 (m)",
                    "x2 (m)"
                ]
            ),

        "uvls":
            pd.DataFrame(
                columns=[
                    "qmax (kN/m)",
                    "x1 (m)",
                    "x2 (m)"
                ]
            ),
    }


    for k,v in defaults.items():

        st.session_state.setdefault(k,v)



    tab1,tab2,tab3,tab4 = st.tabs(
        [
            "Point Load",
            "Point Moment",
            "UDL",
            "UVL"
        ]
    )


    cfg={
        "width":"stretch",
        "num_rows":"dynamic",
        "hide_index":True
    }


    with tab1:

        pl = st.data_editor(
            st.session_state.point_loads,
            key="pl_editor",
            **cfg
        )


    with tab2:

        pm = st.data_editor(
            st.session_state.point_moments,
            key="m_editor",
            **cfg
        )


    with tab3:

        udl = st.data_editor(
            st.session_state.udls,
            key="udl_editor",
            **cfg
        )


    with tab4:

        uvl = st.data_editor(
            st.session_state.uvls,
            key="uvl_editor",
            **cfg
        )



    base.point_loads = clean_rows(
        pl,
        [
            "P (kN)",
            "x (m)"
        ]
    )


    base.point_moments = clean_rows(
        pm,
        [
            "M (kNm)",
            "x (m)"
        ]
    )


    base.udls = clean_rows(
        udl,
        [
            "q (kN/m)",
            "x1 (m)",
            "x2 (m)"
        ]
    )


    base.uvls = clean_rows(
        uvl,
        [
            "qmax (kN/m)",
            "x1 (m)",
            "x2 (m)"
        ]
    )


    return base





def main() -> None:


    inject_css()


    st.title(
        "Beam Analysis"
    )


    data = sidebar_inputs()

    data = load_tables(data)



    result = st.session_state.get(
        "last_result"
    )



    if st.button(
        "Solve",
        type="primary",
        width="stretch"
    ):


        errors = validate_input(data)


        if errors:

            for e in errors:
                st.error(e)

            result=None


        else:

            try:

                result = solve_beam(data)

                st.session_state.last_result=result
                st.session_state.last_input=data


            except Exception as e:

                st.error(str(e))

                result=None




    metric_strip(
        result,
        data
    )



    left,right = st.columns(
        [
            1.7,
            1
        ],
        gap="large"
    )



    with left:


        a,b = st.columns(2)


        with a:

            st.plotly_chart(
                plot_load_diagram(data),
                width="stretch"
            )


        with b:

            st.plotly_chart(
                plot_sfd(result)
                if result
                else base_figure(
                    "Shear Force Diagram",
                    data.length,
                    "kN"
                ),
                width="stretch"
            )



        c,d = st.columns(2)


        with c:

            st.plotly_chart(
                plot_bmd(result)
                if result
                else base_figure(
                    "Moment Diagram",
                    data.length,
                    "kNm"
                ),
                width="stretch"
            )


        with d:

            st.plotly_chart(
                plot_elastic(
                    data,
                    result
                ),
                width="stretch"
            )



    with right:


        st.subheader(
            "Thuyết minh tính toán"
        )


        report = (
            result.report
            if result
            else
            "Chưa có kết quả"
        )


        st.text_area(
            "Report",
            report,
            height=690
        )



if __name__=="__main__":

    main()
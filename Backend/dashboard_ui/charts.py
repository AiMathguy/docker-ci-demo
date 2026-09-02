import plotly.express as px
from utils import polish_plotly


def line_chart(df, x, y, title=None, color="#3b82f6"):
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_traces(line=dict(color=color, width=3))
    return polish_plotly(fig)


def bar_chart(df, x, y, color_col=None, color_seq=None):
    fig = px.bar(df, x=x, y=y, color=color_col, color_discrete_sequence=color_seq)
    fig.update_layout(showlegend=False)
    return polish_plotly(fig)


def histogram(df, x, nbins=20):
    fig = px.histogram(df, x=x, nbins=nbins, color_discrete_sequence=["#3b82f6"])
    return polish_plotly(fig)

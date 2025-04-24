# shiny_plotly_click_image.py

from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import pandas as pd

# 예제 데이터
df = pd.DataFrame({
    "x": [1, 2, 3],
    "y": [10, 15, 13],
    "label": ["A", "B", "C"],
    "image": [
        "https://lab.statisticsplaybook.com/wp-content/uploads/2025/02/lab-main2.png",
        "https://via.placeholder.com/150?text=B",
        "https://via.placeholder.com/150?text=C"
    ]
})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    text=df['label'],
    customdata=df['image'],
    marker=dict(size=12)
))

fig.update_layout(title='Click a point to see image below')

app_ui = ui.page_fluid(
    ui.h2("Plotly + Shiny (Python) Demo"),
    output_widget("plot"),
    ui.div(id="image-output", style="margin-top: 20px;")
)

def server(input, output, session):
    @render_widget
    def plot():
        return fig

    @reactive.Effect
    @reactive.event(input.plot_click)
    def _():
        click_data = input.plot_click()
        if click_data is None:
            return

        image_url = click_data["points"][0]["customdata"]
        label = click_data["points"][0]["text"]

        ui.insert_ui(
            selector="#image-output",
            where="afterBegin",
            ui=ui.HTML(f"""
                <div style='text-align:center'>
                    <img src="{image_url}" width="150" />
                    <p><strong>{label}</strong></p>
                </div>
            """),
            multiple=False,
            immediate=True
        )

app = App(app_ui, server)

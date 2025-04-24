import pandas as pd
import plotly.graph_objects as go
from palmerpenguins import load_penguins

# 데이터 불러오기 및 전처리
df = load_penguins().dropna()
species_avg = df.groupby("species")["body_mass_g"].mean()

# 최대값 설정 (게이지 범위용)
max_val = df["body_mass_g"].max()

# 그래프 생성
fig = go.Figure()

for i, (species, avg) in enumerate(species_avg.items()):
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg,
        number={'suffix': ' g'},
        domain={'row': 0, 'column': i},
        title={'text': f"{species}"},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, max_val * 0.5], 'color': '#e0f7fa'},
                {'range': [max_val * 0.5, max_val], 'color': '#80deea'}
            ],
        }
    ))

# 레이아웃 구성
fig.update_layout(
    grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
    title="종(species)별 평균 몸무게 게이지 차트 (g)",
    margin=dict(t=50, b=0)
)

fig

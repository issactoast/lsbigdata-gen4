import pandas as pd
import plotly.graph_objects as go
from palmerpenguins import load_penguins

# 데이터 불러오기
df = load_penguins().dropna()

# 수치형 변수 리스트
numeric_vars = ['bill_length_mm', 'bill_depth_mm', 
                'flipper_length_mm', 'body_mass_g']

# 기본 x, y 변수
x_var = 'bill_length_mm'
y_var = 'bill_depth_mm'

# Plotly figure 생성
fig = go.Figure()

# 첫 산점도 trace
fig.add_trace(
    go.Scatter(
        x=df[x_var],
        y=df[y_var],
        mode='markers',
        marker=dict(color='skyblue', 
                    size=10,
                    line=dict(width=1, color='DarkSlateGrey')),
        text=df['species'],
        name=f'{x_var} vs {y_var}')
)

# 레이아웃 업데이트
fig.update_layout(
    title='팔머 펭귄 데이터 변수 선택 산점도',
    template='plotly_white',
    width=700,
    height=600,
    xaxis_title=x_var,
    yaxis_title=y_var,
    margin=dict(t=100, b=180),  # 하단 여백 더 확보!
    updatemenus=[
        dict(
            buttons=[
                dict(label=col,
                     method='update',
                     args=[{'x': [df[col]]},
                           {'xaxis.title': col}])
                for col in numeric_vars
            ],
            direction='down',
            showactive=True,
            x=0.2,
            xanchor='left',
            y=-0.3,
            yanchor='bottom'
        ),
        dict(
            buttons=[
                dict(label=col, method='update',
                     args=[{'y': [df[col]]}, {'yaxis.title': col}])
                for col in numeric_vars
            ],
            direction='down',
            showactive=True,
            x=0.7,
            xanchor='left',
            y=-0.3,
            yanchor='bottom'
        )
    ],
    annotations=[
        dict(text="X 변수 선택:", x=0.05, y=-0.28, xref="paper", yref="paper", showarrow=False),
        dict(text="Y 변수 선택:", x=0.6, y=-0.28, xref="paper", yref="paper", showarrow=False)
    ]
)

fig.show()

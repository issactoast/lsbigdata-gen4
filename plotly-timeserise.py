import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 데이터 로드 및 변환
df = px.data.stocks()
df_long = df.melt(id_vars="date", var_name="company", value_name="stock_price")

# 시계열 그래프 생성
fig = px.line(
    df_long,
    x="date",
    y="stock_price",
    color="company",
    title="회사별 주가 시계열 (범례 상단 가로 표시)",
    labels={"stock_price": "주가", "date": "날짜", "company": "회사"},
    template="simple_white"
)

# 범례 상단 중앙에 배치 + 슬라이더와 겹치지 않게 하단 여백 확보
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=80),  # 슬라이더 하단 여백
    hovermode="x unified",
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    )
)

# Hover 템플릿 수정 (날짜 제거, 회사명 + 주가 표시)
for trace in fig.data:
    trace.hovertemplate = "<b>%{fullData.name}</b>: %{y:.2f}<extra></extra>"

fig.show()

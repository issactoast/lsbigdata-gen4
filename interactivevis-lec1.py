import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 산점도

# 데이터 생성
df = pd.DataFrame({'x': [1, 2, 3, 4],
                   'y': [1, 4, 9, 16]})
# 산점도 (Scatter Plot) 생성
fig = px.scatter(
    df,    # 데이터
    x='x', # X축 데이터 설정
    y='y', # Y축 데이터 설정
    title='산점도 예제', # 그래프 제목 설정
    size_max=20 # 마커 크기 최대값 설정
)
fig.show() # 그래프 출력


# 라인 그래프
fig = px.line(
    df,    # 데이터
    x='x', # X축 데이터 설정
    y='y', # Y축 데이터 설정
    title='산점도 예제', # 그래프 제목 설정
    markers=True
)
fig.show() # 그래프 출력

fig.update_layout(
    title='업데이트된 산점도',
    xaxis_title='X축 제목',
    yaxis_title='Y축 제목',
    width=700,
    height=500,
    template='plotly_dark',
    paper_bgcolor='lightgray',
    plot_bgcolor='blue',
    legend=dict(x=0.5, y=1)
)


# 집값 데이터를 사용해서 산점도, 라인 그래프
# 그려보세요!
house_df = pd.read_csv('./data/houseprice/train.csv')

fig_houseprice = px.scatter(
    house_df,  
    x='MSSubClass', 
    y='SalePrice', 
    title='집값 데이터 산점도', 
    size_max=20
)

fig_houseprice.update_layout(
    title='업데이트된 산점도',
    xaxis_title='X축 제목',
    yaxis_title='Y축 제목',
    width=700,
    height=500,
    template='ggplot2',
    # paper_bgcolor='lightgray',
    # plot_bgcolor='blue',
    # legend=dict(x=0.5, y=1)
)

fig_houseprice.add_annotation(
    x=100, y=500000,
    text="여기 중요합니다!",
    showarrow=True,
    arrowhead=2,
    font=dict(color="red",
              size=20)
)
fig_houseprice.show()

# 템플릿
# 'ggplot2', 'seaborn', 'simple_white', 
# 'plotly', 'plotly_white', 'plotly_dark',
# 'presentation', 'xgridoff', 'ygridoff',
# 'gridon'

# 그래프 오브젝트 기본 문법
x = np.linspace(0.1, 5, 100)
y = np.exp(-x) * np.cos(2 * np.pi * x)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x, y=y,
        mode='lines',
        name="Exponential Decay"
    )
)
fig.update_layout(
    title="Log Scale Plot",
    xaxis_title='x values',
    yaxis_title='y values',
    # xaxis_type="log",
    # yaxis_type="log"
)
fig.add_annotation(
    x=2, y=0.2,
    text="여기 중요합니다!",
    showarrow=True,
    arrowhead=2,
    font=dict(color="blue",
              size=14)
)
fig.show()

# 집값 데이터 사용해서 add_annotation
# 추가해보세요!


import plotly.graph_objects as go
x=np.array([1, 2, 3, 4])
y=np.array([2, 4, 6, 8])

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x, y=y,
        text=y,
        textposition='top center',
        mode='markers+text+lines',
        textfont=dict(
            size=18,  # 원하는 폰트 크기로 설정
            color='black'  # 폰트 색상도 설정 가능
        ),
        marker=dict(
            size=y*4,
            color='green',
            symbol='square'
        ),
        name="산점도1",
        showlegend=True
    )
)
fig.show()

# 그래프 여러개 그리기
x = np.arange(0, 5, 0.5)
fig = go.Figure()
styles = ['solid', 'dash', 'dot', 'dashdot']
colors = ['blue', 'green', 'orange', 'red']

for i, dash_style in enumerate(styles):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=x + i,
            mode='lines',
            name=f'dash="{dash_style}"',
            line=dict(
                dash=dash_style,
                width=3,
                color=colors[i]
            )
        )
    )
fig.show()


# pip install ccxt

import ccxt
import pandas as pd

binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", '1d')
df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)


import plotly.graph_objects as go

# Figure 생성
fig = go.Figure()

# 트레이스 추가
fig.add_trace(
    go.Scatter(
        x=df.index,             # X축: 날짜
        y=df['close'],          # Y축: 종가
        mode='lines+markers',   # 선 + 마커 같이 표시
        marker=dict(
            color='blue',       # 마커 색상
            size=6,             # 마커 크기
            symbol='circle'     # 원형 마커
        ),
        line=dict(
            color='blue',       # 선 색상
            width=2,            # 선 두께
            dash='solid'        # 실선 스타일
        ),
        name="BTC/USDT Closing Price"
    )
)

# 그래프 출력
fig.show()


import plotly.express as px

fig = px.scatter(
    df,
    x=df.index,  # X축: 날짜 (datetime 인덱스)
    y='close',   # Y축: 종가 (close)
    title="BTC/USDT Daily Closing Price",  # 그래프 제목
    labels={
        'close': 'Price',
        'datetime': 'Date'  # x축이 df.index이므로 실제 적용 안 될 수 있음
    },
    size_max=10  # 마커 최대 크기
)

fig.show()


import seaborn as sns
df = sns.load_dataset("tips")

import plotly.graph_objects as go

# Figure 생성
fig = go.Figure()

# Histogram 트레이스 추가
fig.add_trace(
    go.Histogram(
        x=df["total_bill"],      # 데이터
        nbinsx=20,               # 빈 수
        opacity=0.7,             # 투명도
        marker_color='indianred',# 색상
        name="Total Bill"        # 범례 이름
    )
)

# 레이아웃 업데이트
fig.update_layout(
    title="Histogram of Total Bill (graph_objects)",
    xaxis_title="Total Bill ($)",
    yaxis_title="Frequency",
    bargap=0.05,   # 막대 간격
    width=500,
    height=400
)

# 그래프 출력
fig.show()


# 히트맵
corr_matrix=df[["total_bill", "tip", "size"]].corr().round(3)

# 히트맵 시각화
fig = px.imshow(
    corr_matrix,
    text_auto=True,                      # 셀 내부 숫자 표시
    color_continuous_scale='RdBu',      # 색상 스케일
    zmin=-1, zmax=1,                     # 색상 범위 고정
    title="Correlation Heatmap (px.imshow)"
)

# 그래프 출력
fig.show()


# 3D 산점도

import plotly.express as px

# 3D 산점도 생성
fig = px.scatter_3d(
    df,
    x="total_bill",   # X축: 총 금액
    y="tip",          # Y축: 팁
    z="size",         # Z축: 인원 수
    color="day",      # 색상 기준: 요일
    title="3D Scatter Plot"
)

# 그래프 출력
fig.show()

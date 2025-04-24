import pandas as pd
import plotly.graph_objects as go
from palmerpenguins import load_penguins

# 데이터 로딩 및 전처리
df = load_penguins().dropna()
total_count = len(df)

# 단계별 필터링
step1 = df
step2 = step1[step1['body_mass_g'] > 3000]
step3 = step2[step2['flipper_length_mm'] > 190]
step4 = step3[step3['bill_length_mm'] > 45]

# 각 단계별 개체 수
step_counts = [len(step1), len(step2), len(step3), len(step4)]

# y축 라벨 (단계 + 조건 설명)
step_labels = [
    "1단계: 전체 펭귄 수",
    "2단계: 몸무게 > 3000g",
    "3단계: 날개 길이 > 190mm",
    "4단계: 부리 길이 > 45mm"
]

# 퍼널 내부 텍스트 (마리 수 + 전체 비율)
text_labels = [
    f"{count}마리 ({count / total_count * 100:.1f}%)"
    for count in step_counts
]

# 막대 색상 리스트 (단계별 색상)
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

# 퍼널 그래프 생성
fig = go.Figure(go.Funnel(
    y=step_labels,
    x=step_counts,
    text=text_labels,
    textinfo="text",
    textposition="inside",
    textfont=dict(size=16),
    marker=dict(color=colors)  # 색상 적용
))

# 레이아웃 설정
fig.update_layout(
    title={
        "text": "단계별 조건을 만족하는 펭귄 개체 수",
        "x": 0.5,
        "xanchor": "center",
        "font": dict(size=24)
    },
    paper_bgcolor='white',   # 전체 배경 흰색
    plot_bgcolor='white',     # 플롯 영역 배경 흰색
    margin=dict(t=80, l=100, r=50, b=80),
    font=dict(size=16, family="Arial")
)

fig

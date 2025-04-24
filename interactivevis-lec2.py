import pandas as pd
import numpy as np

lcd_df = pd.read_csv('./data/plotlydata/seoul_bike.csv')
lcd_df.shape

import plotly.express as px

# 지도 기반 산점도 생성
fig = px.scatter_mapbox(
    lcd_df,
    lat="lat",                 # 위도
    lon="long",                # 경도
    size="LCD거치대수",         # 원 크기
    color="자치구",             # 색상 구분 기준
    hover_name="대여소명",       # 마우스 오버 시 주요 텍스트
    hover_data={
        "lat": False,
        "long": False,
        "LCD거치대수": True,
        "자치구": True
    },
    text="text",               # 지도에 직접 표시될 텍스트
    zoom=11,                   # 줌 레벨
    height=650                 # 그래프 높이
)

# 지도 스타일 및 여백 설정
fig.update_layout(
    mapbox_style="carto-positron",  # 배경 지도 스타일 (무료)
    margin={"r": 0, "t": 0, "l": 0, "b": 0}  # 여백 제거
)

# 지도 시각화 출력
fig.show()

pd.set_option('display.max_columns', None)
import geopandas as gpd
gdf = gpd.read_file("./data/plotlydata/서울시군구/TL_SCCO_SIG_W.shp")
gdf.head(7)
gdf.shape
gdf.info()
gdf["geometry"][0]
gdf["geometry"][1]

print(gdf.crs)

gdf = gdf.to_crs(epsg=4326)
gdf.to_file("./data/plotlydata/seoul_districts.geojson",
             driver="GeoJSON")

import json

# GeoJSON 파일 로드
with open('./data/plotlydata/seoul_districts.geojson', 
          encoding='utf-8') as f:
    geojson_data = json.load(f)

# 최상위 키 출력
print(geojson_data.keys())
geojson_data

geojson_data['features'][0]['geometry']
geojson_data['features'][0]['properties']

# 자치구별 LCD거치대수 합계 집계
agg_df = lcd_df.groupby("자치구", as_index=False)["LCD거치대수"].sum()

# 컬럼 이름 변경
agg_df.columns = ["자치구", "LCD합계"]

# GeoJSON의 'SIG_KOR_NM' 키와 맞추기 위해 컬럼 이름 재변경
agg_df = agg_df.rename(columns={"자치구": "SIG_KOR_NM"})
agg_df
# 결과 확인
agg_df.head(6)

import plotly.express as px

# Choropleth Mapbox 시각화
fig = px.choropleth_mapbox(
    agg_df,
    geojson=geojson_data,
    locations="SIG_KOR_NM",                    # agg_df의 구 이름 컬럼
    featureidkey="properties.SIG_KOR_NM",      # GeoJSON에서 매칭할 키
    color="LCD합계",                            # 색상 기준 데이터
    color_continuous_scale="RdBu_r",            # 색상 스케일
    mapbox_style="carto-positron",             # 지도 스타일
    center={"lat": 37.5665, "lon": 126.9780},  # 서울 중심 좌표
    zoom=10,
    opacity=0.7,
    title="서울시 자치구별 LCD 거치대 수"
)

# 여백 설정
fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})

# 그래프 출력
fig.show()

import plotly.express as px

# 서울시 대여소별 LCD 거치대 수 산점도 지도
fig = px.scatter_mapbox(
    lcd_df,
    lat="lat",                     # 위도
    lon="long",                    # 경도
    size="LCD거치대수",             # 마커 크기
    color="자치구",                 # 색상 구분
    hover_name="대여소명",           # 마우스 오버 시 이름 표시
    hover_data={
        "LCD거치대수": True,
        "자치구": True,
        "lat": False,
        "long": False
    },
    text="text",                  # 마커에 표시할 텍스트
    zoom=10,
    height=650,
    title="서울시 대여소별 LCD 거치대 수"
)

# 지도 스타일 및 구 경계선 추가
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_layers=[
        {
            "sourcetype": "geojson",
            "source": geojson_data,
            "type": "line",
            "color": "black",
            "line": {"width": 1}
        }
    ],
    mapbox_center={"lat": 37.5665, "lon": 126.9780},
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

# 지도 출력
fig.show()



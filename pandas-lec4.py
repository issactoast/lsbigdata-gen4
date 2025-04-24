import pandas as pd
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03'],
    'Temperature': [10, 20, 25],
    'Humidity': [60, 65, 70]
}

df = pd.DataFrame(data)

df_melted = pd.melt(df,
    id_vars=['Date'],
    value_vars=['Temperature', 'Humidity'],
    var_name='Variable',
    value_name='Value'
    )

df_melted

# pivot() 함수 사용해서 원상복귀
df_pivoted = df_melted.pivot(
    index='Date',
    columns='Variable',
    values='Value').reset_index()
df_pivoted

# 연습 데이터
data = {
    'Country': ['Afghanistan', 'Brazil', 'China'],
    '2024': [745, 37737, 212258],
    '2025': [2666, 80488, 213766]
}

df_wide = pd.DataFrame(data)
df_wide

df_long = pd.melt(df_wide,
    id_vars=["Country"],
    value_vars=["2024", "2025"],
    var_name="Year",
    value_name="cases"              
    )
df_long

# pivot() 사용해서 wide 형식으로 바꾸려면?
df_wide2 = df_long.pivot(
    index='Country',
    columns='Year',
    values='cases').reset_index()
df_wide2.shape

df_wide2.iloc[0, 0]
df_wide2.columns.name=None # 불필요한 인덱스 이름 제거
df_wide2


## pivot_table()
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}

df = pd.DataFrame(data)

df_melted2=pd.melt(df,
        id_vars=["Date"],
        value_vars=["Temperature", "Humidity"],
        var_name="WeatherFactor",
        value_name="TorH")
df_melted2

# pivot()으로는 에러가 발생: 
df_pivottbl=df_melted2.pivot_table(
    index='Date',
    columns='WeatherFactor',
    values='TorH',
    aggfunc="last").reset_index()
df_pivottbl.columns.name=None
df_pivottbl
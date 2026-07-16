
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="2D 도로 빗물 배수 시뮬레이터",
    page_icon="🌧️",
    layout="wide",
)

st.title("🌧️ 2D 도로 빗물 배수 시뮬레이터")
st.caption(
    "도로를 격자로 나누어 강우·경사·표면 침투·배수구의 영향을 비교하는 교육용 모형입니다."
)

# ----------------------------
# 사이드바 입력
# ----------------------------
st.sidebar.header("시뮬레이션 조건")

rain_mm_h = st.sidebar.slider("강우 강도 (mm/h)", 10, 150, 70, 5)
rain_duration_min = st.sidebar.slider("강우 지속시간 (분)", 5, 90, 30, 5)
total_duration_min = st.sidebar.slider(
    "전체 관찰시간 (분)",
    rain_duration_min + 10,
    180,
    max(rain_duration_min + 40, 80),
    5,
)

cross_slope = st.sidebar.slider("횡단경사 (%)", 0.0, 8.0, 2.0, 0.5)
long_slope = st.sidebar.slider("종단경사 (%)", 0.0, 5.0, 1.0, 0.5)

surface_data = {
    "아스팔트": {"infiltration": 1.0, "roughness": 1.00},
    "일반 보도블록": {"infiltration": 8.0, "roughness": 1.15},
    "투수블록": {"infiltration": 30.0, "roughness": 1.30},
}
surface = st.sidebar.selectbox("표면 재질", list(surface_data.keys()))

drain_count = st.sidebar.slider("배수구 개수", 1, 8, 3)
drain_capacity_l_min = st.sidebar.slider(
    "배수구 1개당 처리용량 (L/min)", 20, 300, 100, 10
)

road_length_m = st.sidebar.slider("도로 길이 (m)", 20, 100, 50, 5)
road_width_m = st.sidebar.slider("도로 폭 (m)", 4, 20, 10, 1)

st.sidebar.markdown("---")
st.sidebar.info(
    "이 앱은 실제 침수 예측기가 아니라 변수의 상대적 영향을 비교하기 위한 교육용 모형입니다."
)

# ----------------------------
# 시뮬레이션 설정
# ----------------------------
NY, NX = 18, 36
DT_SEC = 5
steps = int(total_duration_min * 60 / DT_SEC)
rain_steps = int(rain_duration_min * 60 / DT_SEC)

cell_area = road_length_m * road_width_m / (NX * NY)
rain_l_per_cell_step = rain_mm_h * cell_area * DT_SEC / 3600
infiltration_mm_h = surface_data[surface]["infiltration"]
infil_l_per_cell_step = infiltration_mm_h * cell_area * DT_SEC / 3600
roughness = surface_data[surface]["roughness"]

# 경사 기반 지형 높이
x = np.linspace(0, 1, NX)
y = np.linspace(0, 1, NY)
X, Y = np.meshgrid(x, y)
terrain = -(long_slope / 100) * X - (cross_slope / 100) * Y

# 배수구는 낮은 쪽 가장자리(y 마지막 줄)에 균등 배치
drain_x = np.linspace(2, NX - 3, drain_count).astype(int)
drain_cells = [(NY - 1, int(ix)) for ix in drain_x]
drain_capacity_per_step = drain_capacity_l_min * DT_SEC / 60

water = np.zeros((NY, NX), dtype=float)
history = []
snapshots = {}
outflow_total = 0.0
infil_total = 0.0
rain_total = 0.0

# 흐름계수: 경사와 수심차를 따라 인접 셀로 이동
base_flow = 0.22 / roughness
snapshot_targets = {
    0,
    max(1, rain_steps // 2),
    rain_steps,
    min(steps - 1, rain_steps + int(10 * 60 / DT_SEC)),
    steps - 1,
}

for step in range(steps):
    # 강우
    if step < rain_steps:
        water += rain_l_per_cell_step
        rain_total += rain_l_per_cell_step * NX * NY

    # 침투
    infiltration = np.minimum(water, infil_l_per_cell_step)
    water -= infiltration
    infil_total += infiltration.sum()

    # 인접 셀 간 이동: 수면고(지형 + 물깊이)에 따라 낮은 곳으로 이동
    # L -> m³ -> 평균 수심(m)
    depth_m = (water / 1000) / cell_area
    head = terrain + depth_m

    new_water = water.copy()

    # 오른쪽 방향 이동
    dh_x = head[:, :-1] - head[:, 1:]
    move_x = np.clip(dh_x, 0, None) * cell_area * 1000 * base_flow
    move_x = np.minimum(move_x, new_water[:, :-1] * 0.45)
    new_water[:, :-1] -= move_x
    new_water[:, 1:] += move_x

    # 아래쪽 방향 이동
    depth_m2 = (new_water / 1000) / cell_area
    head2 = terrain + depth_m2
    dh_y = head2[:-1, :] - head2[1:, :]
    move_y = np.clip(dh_y, 0, None) * cell_area * 1000 * base_flow
    move_y = np.minimum(move_y, new_water[:-1, :] * 0.45)
    new_water[:-1, :] -= move_y
    new_water[1:, :] += move_y

    water = np.maximum(new_water, 0)

    # 배수구 처리
    drained_this_step = 0.0
    for iy, ix in drain_cells:
        drained = min(water[iy, ix], drain_capacity_per_step)
        water[iy, ix] -= drained
        drained_this_step += drained

    outflow_total += drained_this_step

    total_water = water.sum()
    max_depth_mm = ((water / 1000) / cell_area * 1000).max()

    history.append({
        "시간(분)": step * DT_SEC / 60,
        "도로 위 물의 양(L)": total_water,
        "최대 수심(mm)": max_depth_mm,
        "누적 배수량(L)": outflow_total,
        "누적 침투량(L)": infil_total,
    })

    if step in snapshot_targets:
        snapshots[step] = ((water / 1000) / cell_area * 1000).copy()

df = pd.DataFrame(history)

max_water = df["도로 위 물의 양(L)"].max()
max_depth = df["최대 수심(mm)"].max()

after_rain = df[df["시간(분)"] >= rain_duration_min]
drained_rows = after_rain[after_rain["도로 위 물의 양(L)"] < 1.0]
if len(drained_rows):
    drainage_time = drained_rows.iloc[0]["시간(분)"] - rain_duration_min
    drainage_text = f"{drainage_time:.1f}분"
else:
    drainage_text = "관찰시간 내 미배수"

final_water = df.iloc[-1]["도로 위 물의 양(L)"]
final_ratio = final_water / rain_total * 100 if rain_total > 0 else 0
drain_ratio = outflow_total / rain_total * 100 if rain_total > 0 else 0
infil_ratio = infil_total / rain_total * 100 if rain_total > 0 else 0

# ----------------------------
# 결과 지표
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("최대 고임량", f"{max_water:,.1f} L")
c2.metric("최대 수심", f"{max_depth:.2f} mm")
c3.metric("강우 종료 후 배수시간", drainage_text)
c4.metric("종료 시 잔류율", f"{final_ratio:.2f}%")

# ----------------------------
# 시간 그래프
# ----------------------------
st.subheader("1. 시간에 따른 도로 위 물의 변화")

fig_line = go.Figure()
fig_line.add_trace(
    go.Scatter(
        x=df["시간(분)"],
        y=df["도로 위 물의 양(L)"],
        mode="lines",
        name="도로 위 물의 양",
    )
)
fig_line.add_vline(
    x=rain_duration_min,
    line_dash="dash",
    annotation_text="강우 종료",
    annotation_position="top",
)
fig_line.update_layout(
    xaxis_title="시간(분)",
    yaxis_title="도로 위 물의 양(L)",
    height=430,
)
st.plotly_chart(fig_line, use_container_width=True)

# ----------------------------
# 2D 수심 지도
# ----------------------------
st.subheader("2. 시간대별 도로 수심 분포")

available_steps = sorted(snapshots.keys())
labels = []
for s in available_steps:
    t_min = s * DT_SEC / 60
    if s == 0:
        labels.append(f"시작 ({t_min:.1f}분)")
    elif s == rain_steps:
        labels.append(f"강우 종료 ({t_min:.1f}분)")
    elif s == steps - 1:
        labels.append(f"관찰 종료 ({t_min:.1f}분)")
    else:
        labels.append(f"{t_min:.1f}분")

selected_label = st.select_slider(
    "확인할 시점",
    options=labels,
    value=labels[min(2, len(labels)-1)],
)
selected_idx = labels.index(selected_label)
selected_step = available_steps[selected_idx]
depth_map = snapshots[selected_step]

fig_heat = px.imshow(
    depth_map,
    origin="upper",
    aspect="auto",
    labels={"x": "도로 길이 방향", "y": "도로 폭 방향", "color": "수심(mm)"},
)
for iy, ix in drain_cells:
    fig_heat.add_scatter(
        x=[ix],
        y=[iy],
        mode="markers",
        marker=dict(symbol="x", size=12),
        name="배수구",
        showlegend=(ix == drain_cells[0][1]),
    )
fig_heat.update_layout(height=470)
st.plotly_chart(fig_heat, use_container_width=True)

st.caption(
    "표시된 ×는 배수구 위치입니다. 횡단경사가 있으면 물이 아래쪽 가장자리로, 종단경사가 있으면 오른쪽으로 이동하도록 설정했습니다."
)

# ----------------------------
# 물수지
# ----------------------------
st.subheader("3. 전체 빗물의 처리 결과")

balance_df = pd.DataFrame({
    "구분": ["배수구로 배출", "표면에 침투", "도로 위 잔류"],
    "양(L)": [outflow_total, infil_total, final_water],
})
fig_bar = px.bar(
    balance_df,
    x="구분",
    y="양(L)",
    text_auto=".1f",
)
fig_bar.update_layout(height=400)
st.plotly_chart(fig_bar, use_container_width=True)

st.write(
    f"""
    총 강우 유입량은 **{rain_total:,.1f} L**이며, 그중 배수구로 **{drain_ratio:.1f}%**,
    표면으로 **{infil_ratio:.1f}%**가 처리되고, 관찰 종료 시 **{final_ratio:.2f}%**가 도로 위에 남았습니다.
    """
)

# ----------------------------
# 조건 요약 및 해석
# ----------------------------
st.subheader("4. 현재 조건과 탐구 해석")

summary = pd.DataFrame({
    "항목": [
        "표면 재질", "강우 강도", "강우 지속시간", "횡단경사",
        "종단경사", "배수구 개수", "배수구 1개당 처리용량", "도로 크기"
    ],
    "설정값": [
        surface, f"{rain_mm_h} mm/h", f"{rain_duration_min}분",
        f"{cross_slope}%", f"{long_slope}%", f"{drain_count}개",
        f"{drain_capacity_l_min} L/min",
        f"{road_length_m} m × {road_width_m} m"
    ],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

if max_depth < 5:
    risk_text = "물 고임이 비교적 작게 나타났습니다."
elif max_depth < 20:
    risk_text = "일부 구간에서 눈에 띄는 물 고임이 발생했습니다."
else:
    risk_text = "특정 구간에 상당한 물 고임이 발생했습니다."

st.info(
    f"""
    현재 조건에서 최대 수심은 **{max_depth:.2f} mm**였습니다. {risk_text}
    같은 강우 조건에서 경사, 표면 재질, 배수구 수 또는 처리용량 중 하나만 바꾸어 결과를 비교하면
    각 변수가 최대 수심과 배수시간에 미치는 영향을 분석할 수 있습니다.
    """
)

# ----------------------------
# 실험 설계 안내
# ----------------------------
with st.expander("탐구 보고서에 활용하는 방법"):
    st.markdown(
        """
        **권장 실험 설계**

        1. 강우 강도와 도로 크기를 고정한다.
        2. 횡단경사를 0%, 1%, 2%, 4%, 8%로 바꾼다.
        3. 각 조건에서 최대 수심과 배수시간을 기록한다.
        4. 표면 재질만 바꾸어 같은 과정을 반복한다.
        5. 마지막으로 배수구 수와 처리용량을 바꾸어 어느 변수가 가장 큰 영향을 주는지 비교한다.

        **주의**

        이 모형은 하수관 역류, 도로의 미세한 굴곡, 차량 통행, 실제 마찰계수 등을 단순화했습니다.
        따라서 절대값을 실제 도로의 예측값으로 사용하지 말고, 조건 간 상대 비교에 활용해야 합니다.
        """
    )

csv_data = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "결과 데이터 CSV 다운로드",
    data=csv_data,
    file_name="road_drainage_2d_result.csv",
    mime="text/csv",
)

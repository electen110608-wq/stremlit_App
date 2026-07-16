
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="도로 빗물 배수 시뮬레이터", layout="wide")

st.title("도로의 경사와 표면 재질에 따른 빗물 배수 시뮬레이터")
st.caption("집중호우 상황에서 도로 위 물의 최대 고임량과 배수시간을 비교합니다.")

st.sidebar.header("조건 설정")

rain_mm_h = st.sidebar.slider("강우 강도 (mm/h)", 10, 150, 60, 5)
rain_duration_min = st.sidebar.slider("강우 지속시간 (분)", 5, 120, 30, 5)
slope_percent = st.sidebar.slider("도로 경사 (%)", 0.0, 8.0, 2.0, 0.5)
area_m2 = st.sidebar.slider("도로 면적 (m²)", 10, 500, 100, 10)
simulation_min = st.sidebar.slider("총 시뮬레이션 시간 (분)", 30, 240, 120, 10)

surface_options = {
    "아스팔트": {"infiltration_mm_h": 1.0, "flow_coeff": 0.040},
    "일반 보도블록": {"infiltration_mm_h": 8.0, "flow_coeff": 0.035},
    "투수블록": {"infiltration_mm_h": 30.0, "flow_coeff": 0.030},
}
surface = st.sidebar.selectbox("표면 재질", list(surface_options.keys()))

st.sidebar.markdown("---")
st.sidebar.caption(
    "※ 침투율과 유출계수는 비교용 모형값입니다. 실제 탐구에서는 직접 실험값으로 보정해야 합니다."
)

params = surface_options[surface]
dt = 1.0
total_seconds = simulation_min * 60
rain_seconds = rain_duration_min * 60

rain_lps = rain_mm_h * area_m2 / 3600.0
infiltration_lps = params["infiltration_mm_h"] * area_m2 / 3600.0

water_l = 0.0
records = []

for t in range(int(total_seconds) + 1):
    rain_input = rain_lps if t <= rain_seconds else 0.0

    # 경사가 클수록 물이 더 빠르게 빠져나가도록 단순화한 배출 모델
    slope_factor = max(slope_percent / 100.0, 0.0005)
    outflow = params["flow_coeff"] * np.sqrt(slope_factor) * water_l

    # 표면 침투량은 현재 물의 양보다 클 수 없음
    infiltration = min(infiltration_lps, water_l / dt)

    water_l = max(
        0.0,
        water_l + (rain_input - outflow - infiltration) * dt
    )

    records.append({
        "시간(분)": t / 60,
        "도로 위 물의 양(L)": water_l,
        "강우 유입량(L/s)": rain_input,
        "배출량(L/s)": outflow,
        "침투량(L/s)": infiltration,
    })

df = pd.DataFrame(records)

max_water = df["도로 위 물의 양(L)"].max()
rain_end_idx = int(rain_seconds)
after_rain = df.iloc[rain_end_idx:].copy()
drained = after_rain[after_rain["도로 위 물의 양(L)"] < 0.1]

if len(drained) > 0:
    drainage_time = drained.iloc[0]["시간(분)"] - rain_duration_min
    drainage_text = f"{drainage_time:.1f}분"
else:
    drainage_text = "시뮬레이션 시간 내 미배수"

total_rain_l = rain_lps * rain_seconds
final_water = df.iloc[-1]["도로 위 물의 양(L)"]
retained_ratio = (final_water / total_rain_l * 100) if total_rain_l > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("최대 고임량", f"{max_water:,.1f} L")
c2.metric("비가 그친 뒤 배수시간", drainage_text)
c3.metric("종료 시 잔류 비율", f"{retained_ratio:.2f}%")

st.subheader("시간에 따른 도로 위 물의 양")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["시간(분)"], df["도로 위 물의 양(L)"])
ax.axvline(rain_duration_min, linestyle="--", label="강우 종료")
ax.set_xlabel("시간(분)")
ax.set_ylabel("도로 위 물의 양(L)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.subheader("현재 조건 요약")
summary = pd.DataFrame({
    "항목": ["표면 재질", "강우 강도", "강우 지속시간", "도로 경사", "도로 면적"],
    "설정값": [
        surface,
        f"{rain_mm_h} mm/h",
        f"{rain_duration_min}분",
        f"{slope_percent}%",
        f"{area_m2} m²",
    ],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("탐구 해석")
st.write(
    f"""
    현재 조건에서는 **{surface}**, 경사 **{slope_percent}%**, 강우 강도 **{rain_mm_h} mm/h**일 때
    최대 약 **{max_water:,.1f} L**의 물이 도로 위에 남습니다.
    경사를 높이거나 침투성이 큰 재질을 선택하면 일반적으로 최대 고임량과 배수시간이 감소합니다.
    다만 이 결과는 단순 모형이므로 실제 도로의 배수구 수, 노면 거칠기, 웅덩이 형성, 하수관 용량은 반영하지 않았습니다.
    """
)

st.download_button(
    "시뮬레이션 결과 CSV 다운로드",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="drainage_simulation_result.csv",
    mime="text/csv",
)

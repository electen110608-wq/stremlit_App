from pathlib import Path
import textwrap, zipfile, os

root = Path("/mnt/data/ai_glasses_simulator")
root.mkdir(exist_ok=True)

app_py = r'''
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AI 안경 탐지 시뮬레이터",
    page_icon="👓",
    layout="wide",
)

# -----------------------------
# 공통 함수
# -----------------------------
def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def calculate_signals(
    display_brightness: int,
    wireless_intensity: int,
    processing_load: int,
    concealment: int,
    ambient_light: int,
    room_temperature: int,
    camera_angle: int,
) -> dict:
    """
    교육용 휴리스틱 모델.
    실제 장비의 탐지 확률을 예측하지 않으며, 변수 사이의 방향성과
    오탐 가능성을 탐구하기 위한 상대 점수만 계산한다.
    """

    # 광학 반사 신호:
    # 디스플레이 밝기가 높고 주변이 어두우며 관찰 각도가 적절할수록 커짐.
    angle_factor = math.exp(-((camera_angle - 35) ** 2) / (2 * 22**2))
    optical = (
        0.52 * display_brightness
        + 24 * (1 - ambient_light / 100)
        + 24 * angle_factor
        - 0.42 * concealment
    )

    # 무선 신호:
    # 통신량이 많을수록 증가하지만 주변 기기의 간섭과 은폐에 영향받음.
    wireless = 0.82 * wireless_intensity + 12 - 0.28 * concealment

    # 열 신호:
    # 연산 부하와 통신량이 높고 실내 온도가 낮을수록 상대적으로 두드러짐.
    thermal_contrast = max(0, 31 - room_temperature)
    thermal = (
        0.58 * processing_load
        + 0.18 * wireless_intensity
        + 1.2 * thermal_contrast
        - 0.24 * concealment
    )

    # 행동 신호:
    # 시선 고정·반복 조작 등은 기기뿐 아니라 긴장이나 습관으로도 발생할 수 있음.
    behavioral = (
        0.30 * processing_load
        + 0.20 * display_brightness
        + 18
        - 0.12 * concealment
    )

    return {
        "광학 반사": clamp(optical),
        "무선 통신": clamp(wireless),
        "열 분포": clamp(thermal),
        "행동 패턴": clamp(behavioral),
    }


def calculate_false_positive(
    detector: str,
    ambient_light: int,
    room_temperature: int,
    nearby_devices: int,
    student_stress: int,
) -> float:
    base = {
        "광학 반사": 8,
        "무선 통신": 10,
        "열 분포": 12,
        "행동 패턴": 18,
    }[detector]

    if detector == "광학 반사":
        score = base + 0.24 * ambient_light
    elif detector == "무선 통신":
        score = base + 0.65 * nearby_devices
    elif detector == "열 분포":
        score = base + 1.25 * abs(room_temperature - 23)
    else:
        score = base + 0.48 * student_stress

    return clamp(score, 0, 85)


def combined_score(signals: dict, weights: dict) -> float:
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    return sum(signals[k] * weights[k] for k in signals) / total_weight


def risk_label(score: float) -> tuple[str, str]:
    if score >= 70:
        return "높음", "🔴"
    if score >= 45:
        return "중간", "🟠"
    return "낮음", "🟢"


# -----------------------------
# 화면 상단
# -----------------------------
st.title("👓 AI 안경 탐지 시뮬레이터")
st.caption(
    "광학·무선·열·행동 신호를 조절하며 탐지 가능성과 오탐 위험의 균형을 탐구하는 교육용 앱"
)

with st.expander("⚠️ 모델 해석 전 반드시 읽기", expanded=False):
    st.markdown(
        """
        이 앱은 실제 AI 안경이나 탐지 장비의 성능을 측정한 프로그램이 아니다.
        공개된 물리 원리와 일반적인 센서 특성을 바탕으로 만든 **가상 휴리스틱 모델**이다.
        따라서 출력값은 실제 탐지 확률이 아니라, 조건을 바꿀 때 각 신호가 어느 방향으로
        변할 수 있는지 비교하는 **상대 점수**로 해석해야 한다.
        """
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["① 작동 원리", "② 신호 탐지 실험", "③ 시험 환경 설계", "④ 탐구 보고서"]
)

# -----------------------------
# 1. 작동 원리
# -----------------------------
with tab1:
    st.subheader("AI 안경은 어떻게 정보를 처리할까?")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**1. 입력**\n\n카메라·마이크·버튼이 주변 정보나 사용자의 명령을 받는다.")
    with col2:
        st.info("**2. 처리**\n\n안경 내부 칩 또는 연결된 스마트폰·서버가 정보를 분석한다.")
    with col3:
        st.info("**3. 통신**\n\nWi-Fi·Bluetooth·셀룰러 통신으로 외부 기기와 데이터를 주고받을 수 있다.")
    with col4:
        st.info("**4. 출력**\n\n도파관이나 소형 디스플레이가 빛을 눈 쪽으로 보내 정보를 겹쳐 보여준다.")

    st.markdown("#### 탐지에 이용할 수 있는 네 가지 흔적")
    principles = pd.DataFrame(
        {
            "신호": ["광학 반사", "무선 통신", "열 분포", "행동 패턴"],
            "발생 이유": [
                "디스플레이와 광학계에서 일부 빛이 반사될 수 있음",
                "외부 연산이나 데이터 전송 시 전파가 발생함",
                "칩의 연산과 통신 과정에서 전기에너지 일부가 열로 전환됨",
                "시선 고정, 프레임 조작 등 사용 행동이 나타날 수 있음",
            ],
            "주요 한계": [
                "관찰 각도·조명·렌즈 코팅에 크게 좌우됨",
                "스마트폰·노트북 등 정상 기기 신호와 구별하기 어려움",
                "사람의 체온과 주변 온도 때문에 작은 발열을 분리하기 어려움",
                "긴장·습관·시각 보조 행동을 부정행위로 오인할 수 있음",
            ],
        }
    )
    st.dataframe(principles, use_container_width=True, hide_index=True)

    st.success(
        "핵심 가설: 하나의 센서만 사용하는 것보다 서로 다른 신호를 결합하면 "
        "탐지의 근거는 늘어나지만, 개인정보·비용·오탐 문제도 함께 커진다."
    )

# -----------------------------
# 2. 신호 탐지 실험
# -----------------------------
with tab2:
    st.subheader("조건을 바꾸어 각 탐지 신호의 변화를 관찰해 보자")

    controls, results = st.columns([1, 1.55], gap="large")

    with controls:
        st.markdown("#### AI 안경 사용 조건")
        display_brightness = st.slider("디스플레이 밝기", 0, 100, 55)
        wireless_intensity = st.slider("무선 통신량", 0, 100, 60)
        processing_load = st.slider("연산 부하", 0, 100, 65)
        concealment = st.slider(
            "신호 은폐 정도", 0, 100, 25,
            help="렌즈 코팅, 낮은 출력, 프레임 구조 등으로 외부 신호가 약해지는 상황을 단순화한 값"
        )

        st.markdown("#### 시험장 환경")
        ambient_light = st.slider("주변 밝기", 0, 100, 55)
        room_temperature = st.slider("실내 온도(°C)", 16, 30, 23)
        camera_angle = st.slider("관찰 각도(°)", 0, 90, 35)

    signals = calculate_signals(
        display_brightness,
        wireless_intensity,
        processing_load,
        concealment,
        ambient_light,
        room_temperature,
        camera_angle,
    )

    with results:
        st.markdown("#### 상대 신호 강도")
        fig = go.Figure(
            go.Bar(
                x=list(signals.values()),
                y=list(signals.keys()),
                orientation="h",
                text=[f"{v:.1f}" for v in signals.values()],
                textposition="auto",
            )
        )
        fig.update_layout(
            xaxis_title="상대 점수(0~100)",
            xaxis_range=[0, 100],
            yaxis_title="",
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        strongest = max(signals, key=signals.get)
        weakest = min(signals, key=signals.get)
        c1, c2 = st.columns(2)
        c1.metric("가장 두드러진 신호", strongest, f"{signals[strongest]:.1f}점")
        c2.metric("가장 약한 신호", weakest, f"{signals[weakest]:.1f}점")

        st.markdown("#### 탐지 방식 결합")
        w1, w2 = st.columns(2)
        with w1:
            optical_w = st.slider("광학 가중치", 0, 10, 5)
            thermal_w = st.slider("열 가중치", 0, 10, 3)
        with w2:
            wireless_w = st.slider("무선 가중치", 0, 10, 4)
            behavior_w = st.slider("행동 가중치", 0, 10, 2)

        weights = {
            "광학 반사": optical_w,
            "무선 통신": wireless_w,
            "열 분포": thermal_w,
            "행동 패턴": behavior_w,
        }
        score = combined_score(signals, weights)
        label, icon = risk_label(score)
        st.metric("결합 탐지 점수", f"{score:.1f}/100", f"{icon} {label}")

        if sum(weights.values()) == 0:
            st.warning("적어도 한 가지 탐지 방식의 가중치를 1 이상으로 설정해야 한다.")
        elif score >= 70:
            st.error("가상 모델에서는 여러 탐지 신호가 강하게 겹친 조건이다.")
        elif score >= 45:
            st.warning("일부 신호가 관찰되지만 추가 확인 없이 판단하기 어렵다.")
        else:
            st.success("현재 조건에서는 탐지 신호가 비교적 약하다.")

# -----------------------------
# 3. 시험 환경 설계
# -----------------------------
with tab3:
    st.subheader("탐지율만 높이면 좋은 시험 환경일까?")
    st.write(
        "탐지 장치를 강하게 적용할수록 의심 신호는 더 많이 찾을 수 있지만, "
        "정상 학생을 잘못 의심하는 오탐과 개인정보 침해도 증가할 수 있다."
    )

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        detector = st.selectbox(
            "주 탐지 방식",
            ["광학 반사", "무선 통신", "열 분포", "행동 패턴"],
        )
        sensitivity = st.slider("센서 민감도", 0, 100, 65)
        nearby_devices = st.slider("시험장 주변 전자기기 수", 0, 50, 12)
        student_stress = st.slider("학생의 평균 긴장·행동 변화", 0, 100, 45)
        env_light = st.slider("환경 밝기", 0, 100, 55, key="env_light")
        env_temp = st.slider("환경 온도(°C)", 16, 30, 23, key="env_temp")

    fp = calculate_false_positive(
        detector,
        env_light,
        env_temp,
        nearby_devices,
        student_stress,
    )
    simulated_detection = clamp(
        0.72 * sensitivity
        + {
            "광학 반사": 8,
            "무선 통신": 12,
            "열 분포": 5,
            "행동 패턴": 3,
        }[detector]
    )
    privacy_burden = clamp(
        0.55 * sensitivity
        + {
            "광학 반사": 18,
            "무선 통신": 28,
            "열 분포": 14,
            "행동 패턴": 34,
        }[detector]
    )
    practicality = clamp(
        100
        - 0.30 * sensitivity
        - {
            "광학 반사": 14,
            "무선 통신": 10,
            "열 분포": 28,
            "행동 패턴": 18,
        }[detector]
    )

    with right:
        st.markdown("#### 정책 평가 결과")
        evaluation = pd.DataFrame(
            {
                "평가 항목": ["의심 신호 포착", "오탐 위험", "개인정보 부담", "현장 적용성"],
                "점수": [simulated_detection, fp, privacy_burden, practicality],
            }
        )
        fig2 = go.Figure(
            go.Bar(
                x=evaluation["평가 항목"],
                y=evaluation["점수"],
                text=[f"{x:.1f}" for x in evaluation["점수"]],
                textposition="auto",
            )
        )
        fig2.update_layout(
            yaxis_title="상대 점수(0~100)",
            yaxis_range=[0, 100],
            height=370,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        net_value = simulated_detection + practicality - fp - privacy_burden
        st.metric("균형 지수", f"{net_value:.1f}", help="포착+현장성-오탐-개인정보 부담")

        if fp >= 50:
            st.error("오탐 위험이 높다. 이 결과만으로 학생을 제재해서는 안 된다.")
        elif privacy_burden >= 55:
            st.warning("탐지 과정의 개인정보 수집 범위와 보관 기준이 필요하다.")
        else:
            st.info("기술적 탐지는 보조 수단으로 두고, 명확한 재확인 절차와 이의 제도를 함께 설계해야 한다.")

    st.markdown("#### 현실적인 다층 대책")
    policy = pd.DataFrame(
        {
            "단계": ["예방", "환경", "1차 관찰", "재확인", "사후 절차"],
            "대책": [
                "허용·금지 기기 기준과 검사 절차를 시험 전에 공개",
                "개인 전자기기 보관, 좌석 간격 확보, 감독 사각지대 축소",
                "단일 센서가 아닌 복수의 약한 신호를 참고",
                "의심만으로 제재하지 않고 사람의 확인과 기록을 거침",
                "데이터 최소 수집, 즉시 삭제, 이의 제기 절차 마련",
            ],
            "의의": [
                "학생의 예측 가능성 확보",
                "비싼 탐지 장비 의존 감소",
                "단일 신호의 오류 완화",
                "오탐으로 인한 피해 방지",
                "공정성과 개인정보 보호",
            ],
        }
    )
    st.dataframe(policy, use_container_width=True, hide_index=True)

# -----------------------------
# 4. 보고서
# -----------------------------
with tab4:
    st.subheader("탐구 보고서에 바로 활용할 수 있는 구조")

    st.markdown(
        """
        **탐구 질문**  
        AI 안경의 광학·무선·열·행동 신호를 이용한 탐지는 어떤 조건에서 효과가 달라지며,
        탐지 성능과 오탐·개인정보 보호 사이의 균형은 어떻게 설계해야 하는가?

        **가설**  
        하나의 센서에만 의존하는 방식보다 여러 신호를 결합하는 방식이 의심 상황을 더 잘
        구분할 수 있지만, 센서의 민감도를 무조건 높이면 오탐과 개인정보 부담도 증가할 것이다.

        **독립변인**  
        디스플레이 밝기, 무선 통신량, 연산 부하, 주변 밝기, 실내 온도, 관찰 각도,
        센서 민감도

        **종속변인**  
        광학·무선·열·행동 신호의 상대 점수, 결합 탐지 점수, 오탐 위험, 개인정보 부담

        **통제변인 예시**  
        선택한 탐지 방식, 가중치, 주변 기기 수 등을 한 번에 하나씩만 바꾸어 비교한다.
        """
    )

    st.markdown("#### 자동 생성된 관찰 문장")
    sentence = (
        f"현재 설정에서는 {max(signals, key=signals.get)} 신호가 "
        f"{max(signals.values()):.1f}점으로 가장 높게 나타났으며, "
        f"{min(signals, key=signals.get)} 신호는 {min(signals.values()):.1f}점으로 가장 낮았다. "
        "이는 AI 안경 탐지 가능성이 기기 자체의 특성뿐 아니라 조명, 온도, 관찰 각도와 같은 "
        "시험 환경에 따라서도 달라질 수 있음을 보여준다."
    )
    st.text_area("관찰 결과", sentence, height=135)

    st.markdown("#### 결론 작성 시 주의점")
    st.markdown(
        """
        - 앱의 수치를 실제 탐지 확률이라고 표현하지 않는다.
        - ‘탐지할 수 있다’보다 ‘탐지 신호로 활용할 가능성이 있다’고 표현한다.
        - 기술적 대책뿐 아니라 오탐, 장애 학생의 보조기기 사용, 개인정보 보호를 함께 다룬다.
        - 시뮬레이션의 한계와 실제 실험 또는 장비 검증의 필요성을 밝힌다.
        """
    )

st.divider()
st.caption("교육용 탐구 모델 · 실제 부정행위 판정 또는 감시 목적으로 사용할 수 없음")
'''

requirements = """streamlit>=1.50,<2
pandas>=2.2,<3
numpy>=2.0,<3
plotly>=6.0,<7
"""

readme = r'''
# AI 안경 탐지 시뮬레이터

AI 안경에서 발생할 수 있는 광학 반사, 무선 통신, 열 분포, 행동 패턴을
가상 점수로 비교하고, 시험 환경의 탐지 성능·오탐·개인정보 부담 사이의
균형을 탐구하는 Streamlit 앱입니다.

## 주의

이 앱의 계산식은 실제 제품이나 탐지 장비의 성능을 측정한 통계 모델이 아닙니다.
변수의 방향성과 상충 관계를 학습하기 위한 교육용 휴리스틱 모델입니다.

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py

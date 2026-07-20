from pathlib import Path
import textwrap, zipfile, os

base = Path("/mnt/data/youtube_comment_analyzer")
base.mkdir(exist_ok=True)

app_py = r'''
import html
import re
from collections import Counter
from datetime import timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from wordcloud import WordCloud


st.set_page_config(
    page_title="유튜브 댓글 분석기",
    page_icon="💬",
    layout="wide",
)

API_BASE = "https://www.googleapis.com/youtube/v3"

DEFAULT_STOPWORDS = {
    "이", "그", "저", "것", "수", "등", "들", "및", "더", "좀", "잘", "너무",
    "정말", "진짜", "영상", "유튜브", "댓글", "사람", "제가", "나는", "내가",
    "그리고", "그래서", "하지만", "그런데", "그러나", "하면", "해서", "하고",
    "있는", "없는", "입니다", "합니다", "같아요", "같다", "같은", "입니다",
    "ㅋㅋ", "ㅋㅋㅋ", "ㅎㅎ", "ㅎㅎㅎ", "ㅠㅠ", "ㅜㅜ", "아", "오", "와",
}

POSITIVE_WORDS = {
    "좋다", "좋아요", "좋은", "최고", "멋지다", "멋져", "감사", "고맙",
    "재밌다", "재미있다", "재밌어요", "웃기다", "대박", "완벽", "추천",
    "사랑", "행복", "응원", "유익", "도움", "훌륭", "감동", "예쁘다",
    "귀엽다", "신기", "놀랍", "잘했다", "잘함", "존경", "기대",
}

NEGATIVE_WORDS = {
    "싫다", "싫어요", "별로", "최악", "노잼", "재미없", "화난", "짜증",
    "실망", "문제", "불편", "아쉽", "답답", "비추천", "거짓", "틀렸",
    "이상하다", "이상해", "무섭", "충격", "망했", "못한다", "못함",
    "혐오", "욕", "낚시", "과장", "오류",
}


def get_secret_api_key() -> str:
    """Streamlit Secrets에 저장된 키가 있으면 가져온다."""
    try:
        return str(st.secrets.get("YOUTUBE_API_KEY", ""))
    except Exception:
        return ""


def extract_video_id(url_or_id: str) -> str | None:
    """일반 URL, youtu.be, Shorts, Live, Embed URL 또는 영상 ID에서 ID를 추출한다."""
    value = url_or_id.strip()

    if re.fullmatch(r"[\w-]{11}", value):
        return value

    try:
        parsed = urlparse(value)
    except ValueError:
        return None

    host = parsed.netloc.lower().replace("www.", "")
    path_parts = [part for part in parsed.path.split("/") if part]

    if host == "youtu.be" and path_parts:
        candidate = path_parts[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif path_parts and path_parts[0] in {"shorts", "embed", "live"} and len(path_parts) >= 2:
            candidate = path_parts[1]
        else:
            candidate = ""
    else:
        candidate = ""

    return candidate if re.fullmatch(r"[\w-]{11}", candidate) else None


def youtube_get(endpoint: str, params: dict, api_key: str) -> dict:
    """YouTube Data API GET 요청과 오류 메시지를 통합 처리한다."""
    request_params = {**params, "key": api_key}
    response = requests.get(
        f"{API_BASE}/{endpoint}",
        params=request_params,
        timeout=20,
    )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("YouTube API 응답을 읽지 못했습니다.") from exc

    if response.ok:
        return payload

    error = payload.get("error", {})
    message = error.get("message", "YouTube API 요청에 실패했습니다.")
    reasons = [
        item.get("reason", "")
        for item in error.get("errors", [])
        if item.get("reason")
    ]

    if "commentsDisabled" in reasons:
        raise RuntimeError("이 영상은 댓글이 비활성화되어 있습니다.")
    if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
        raise RuntimeError("YouTube API 일일 할당량을 초과했습니다.")
    if response.status_code == 400 and "API key" in message:
        raise RuntimeError("API 키가 올바르지 않거나 YouTube Data API v3가 활성화되지 않았습니다.")
    if response.status_code == 403:
        raise RuntimeError(f"접근이 거부되었습니다: {message}")

    raise RuntimeError(message)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_video_info(video_id: str, api_key: str) -> dict:
    payload = youtube_get(
        "videos",
        {
            "part": "snippet,statistics",
            "id": video_id,
        },
        api_key,
    )

    items = payload.get("items", [])
    if not items:
        raise RuntimeError("영상을 찾을 수 없습니다. 비공개·삭제 영상인지 확인해 주세요.")

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    return {
        "title": snippet.get("title", "제목 없음"),
        "channel": snippet.get("channelTitle", "채널 정보 없음"),
        "published_at": snippet.get("publishedAt"),
        "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
        "view_count": int(stats.get("viewCount", 0)),
        "like_count": int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_comments(video_id: str, api_key: str, limit: int, order: str) -> pd.DataFrame:
    """상위 댓글 스레드를 페이지 단위로 최대 limit개 수집한다."""
    rows: list[dict] = []
    page_token = None

    while len(rows) < limit:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, limit - len(rows)),
            "order": order,
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        payload = youtube_get("commentThreads", params, api_key)

        for item in payload.get("items", []):
            thread_snippet = item.get("snippet", {})
            top = thread_snippet.get("topLevelComment", {}).get("snippet", {})

            rows.append(
                {
                    "작성자": html.unescape(top.get("authorDisplayName", "")),
                    "댓글": html.unescape(top.get("textDisplay", "")),
                    "좋아요": int(top.get("likeCount", 0)),
                    "작성시각_UTC": top.get("publishedAt"),
                    "수정시각_UTC": top.get("updatedAt"),
                    "답글수": int(thread_snippet.get("totalReplyCount", 0)),
                }
            )

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["작성시각_UTC"] = pd.to_datetime(df["작성시각_UTC"], utc=True, errors="coerce")
    df["작성시각"] = df["작성시각_UTC"].dt.tz_convert("Asia/Seoul")
    df["날짜"] = df["작성시각"].dt.date
    df["시간"] = df["작성시각"].dt.hour
    df["요일"] = pd.Categorical(
        df["작성시각"].dt.day_name().map(
            {
                "Monday": "월",
                "Tuesday": "화",
                "Wednesday": "수",
                "Thursday": "목",
                "Friday": "금",
                "Saturday": "토",
                "Sunday": "일",
            }
        ),
        categories=["월", "화", "수", "목", "금", "토", "일"],
        ordered=True,
    )
    return df.head(limit)


def tokenize_korean(text: str, stopwords: set[str], min_length: int) -> list[str]:
    """외부 형태소 분석기 없이 한글 덩어리를 추출하는 가벼운 토크나이저."""
    tokens = re.findall(r"[가-힣]{2,}", text)
    return [
        token
        for token in tokens
        if len(token) >= min_length and token not in stopwords
    ]


def sentiment_score(text: str) -> tuple[str, int]:
    """간단한 어휘 포함 여부 기반 감성 점수. 정교한 문맥 분석 모델은 아니다."""
    normalized = re.sub(r"\s+", "", text)
    positive = sum(1 for word in POSITIVE_WORDS if word in normalized)
    negative = sum(1 for word in NEGATIVE_WORDS if word in normalized)
    score = positive - negative

    if score > 0:
        return "긍정", score
    if score < 0:
        return "부정", score
    return "중립", score


def find_korean_font() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def format_number(value: int) -> str:
    return f"{value:,}"


st.title("💬 유튜브 댓글 분석기")
st.caption(
    "영상 링크와 YouTube Data API 키를 입력하면 댓글 작성 추이, 반응도, "
    "한글 워드클라우드를 분석합니다."
)

with st.sidebar:
    st.header("분석 설정")

    secret_key = get_secret_api_key()
    api_key = st.text_input(
        "YouTube Data API 키",
        value=secret_key,
        type="password",
        help="입력한 키는 현재 세션에서만 사용됩니다. 배포 시에는 Streamlit Secrets 사용을 권장합니다.",
    )

    video_url = st.text_input(
        "유튜브 영상 링크",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    comment_limit = st.slider(
        "수집할 댓글 개수",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
        help="API가 반환할 수 있는 공개 상위 댓글 범위 내에서 수집합니다.",
    )

    order_label = st.radio(
        "댓글 수집 순서",
        options=["최신순", "관련도순"],
        horizontal=True,
    )
    order = "time" if order_label == "최신순" else "relevance"

    min_word_length = st.slider(
        "워드클라우드 최소 글자 수",
        min_value=2,
        max_value=5,
        value=2,
    )

    custom_stopwords = st.text_area(
        "추가 불용어",
        placeholder="쉼표로 구분: 채널명, 출연자명, 반복 단어",
        height=90,
    )

    analyze = st.button("댓글 분석 시작", type="primary", use_container_width=True)

st.info(
    "이 앱은 공개된 **상위 댓글**만 수집합니다. 답글 내용은 수집하지 않고 답글 수만 표시합니다. "
    "삭제·비공개 댓글과 YouTube가 제공하지 않는 댓글은 분석할 수 없습니다."
)

if analyze:
    video_id = extract_video_id(video_url)

    if not api_key.strip():
        st.error("YouTube Data API 키를 입력해 주세요.")
        st.stop()
    if not video_id:
        st.error("올바른 유튜브 영상 링크 또는 11자리 영상 ID를 입력해 주세요.")
        st.stop()

    try:
        with st.spinner("영상 정보와 댓글을 불러오는 중입니다..."):
            video_info = fetch_video_info(video_id, api_key.strip())
            comments = fetch_comments(
                video_id,
                api_key.strip(),
                comment_limit,
                order,
            )
    except requests.RequestException as exc:
        st.error(f"네트워크 연결 중 오류가 발생했습니다: {exc}")
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.session_state["analysis_result"] = {
        "video_id": video_id,
        "video_info": video_info,
        "comments": comments,
        "min_word_length": min_word_length,
        "custom_stopwords": custom_stopwords,
        "order_label": order_label,
    }

result = st.session_state.get("analysis_result")

if result:
    video_id = result["video_id"]
    video_info = result["video_info"]
    comments = result["comments"]
    min_word_length = result["min_word_length"]
    custom_stopwords = result["custom_stopwords"]
    order_label = result["order_label"]

    st.divider()
    st.subheader(video_info["title"])
    st.caption(
        f'{video_info["channel"]} · 수집 기준: {order_label} · '
        f'분석 댓글 {len(comments):,}개'
    )
    st.video(f"https://www.youtube.com/watch?v={video_id}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("조회수", format_number(video_info["view_count"]))
    metric_cols[1].metric("영상 좋아요", format_number(video_info["like_count"]))
    metric_cols[2].metric("공개 댓글 수", format_number(video_info["comment_count"]))
    metric_cols[3].metric("실제 수집 댓글", format_number(len(comments)))

    if comments.empty:
        st.warning("수집 가능한 공개 댓글이 없습니다.")
        st.stop()

    comments = comments.copy()
    comments[["감성", "감성점수"]] = comments["댓글"].apply(
        lambda text: pd.Series(sentiment_score(text))
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 작성 추이", "❤️ 댓글 반응도", "☁️ 워드클라우드", "📋 댓글 데이터"]
    )

    with tab1:
        st.markdown("#### 날짜별 댓글 작성 추이")
        daily = (
            comments.set_index("작성시각")
            .resample("D")
            .size()
            .rename("댓글 수")
            .reset_index()
        )
        daily["날짜"] = daily["작성시각"].dt.date

        daily_fig = px.line(
            daily,
            x="날짜",
            y="댓글 수",
            markers=True,
            title="날짜별 수집 댓글 수",
        )
        daily_fig.update_layout(xaxis_title="작성 날짜", yaxis_title="댓글 수")
        st.plotly_chart(daily_fig, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            hourly = (
                comments.groupby("시간", observed=False)
                .size()
                .reindex(range(24), fill_value=0)
                .rename("댓글 수")
                .reset_index()
            )
            hourly_fig = px.bar(
                hourly,
                x="시간",
                y="댓글 수",
                title="시간대별 작성 분포(KST)",
            )
            hourly_fig.update_xaxes(dtick=1)
            st.plotly_chart(hourly_fig, use_container_width=True)

        with chart_col2:
            weekday = (
                comments.groupby("요일", observed=False)
                .size()
                .rename("댓글 수")
                .reset_index()
            )
            weekday_fig = px.bar(
                weekday,
                x="요일",
                y="댓글 수",
                title="요일별 작성 분포(KST)",
            )
            st.plotly_chart(weekday_fig, use_container_width=True)

        st.caption(
            "시간대·요일 그래프는 수집된 댓글만을 기준으로 계산합니다. "
            "관련도순 표본은 전체 댓글의 시간 분포를 대표하지 않을 수 있습니다."
        )

    with tab2:
        st.markdown("#### 댓글 좋아요 반응")

        reaction_cols = st.columns(4)
        reaction_cols[0].metric("댓글 좋아요 합계", format_number(int(comments["좋아요"].sum())))
        reaction_cols[1].metric("댓글당 평균 좋아요", f'{comments["좋아요"].mean():.1f}')
        reaction_cols[2].metric("중앙값", f'{comments["좋아요"].median():.0f}')
        reaction_cols[3].metric("답글 합계", format_number(int(comments["답글수"].sum())))

        top_liked = (
            comments.nlargest(min(15, len(comments)), "좋아요")
            .sort_values("좋아요")
            .copy()
        )
        top_liked["댓글_축약"] = top_liked["댓글"].str.replace("\n", " ", regex=False).str.slice(0, 55)

        liked_fig = px.bar(
            top_liked,
            x="좋아요",
            y="댓글_축약",
            orientation="h",
            hover_data=["작성자", "댓글", "답글수"],
            title="좋아요가 많은 댓글",
        )
        liked_fig.update_layout(yaxis_title="")
        st.plotly_chart(liked_fig, use_container_width=True)

        st.markdown("#### 간단한 댓글 감성 반응도")
        sentiment_counts = (
            comments["감성"]
            .value_counts()
            .reindex(["긍정", "중립", "부정"], fill_value=0)
            .rename_axis("감성")
            .reset_index(name="댓글 수")
        )
        sentiment_fig = px.pie(
            sentiment_counts,
            names="감성",
            values="댓글 수",
            hole=0.45,
            title="규칙 기반 감성 분포",
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)

        st.warning(
            "감성 분석은 일부 긍정·부정 단어의 포함 여부를 세는 간단한 규칙 기반 지표입니다. "
            "반어법, 문맥, 신조어, 복합 감정을 정확히 판단하지 못하므로 참고용으로만 사용하세요."
        )

    with tab3:
        st.markdown("#### 한글 워드클라우드")

        user_stopwords = {
            word.strip()
            for word in custom_stopwords.split(",")
            if word.strip()
        }
        stopwords = DEFAULT_STOPWORDS | user_stopwords
        all_text = " ".join(comments["댓글"].fillna("").astype(str))
        tokens = tokenize_korean(all_text, stopwords, min_word_length)
        frequencies = Counter(tokens)

        if not frequencies:
            st.warning("워드클라우드를 만들 수 있는 한글 단어가 없습니다. 불용어나 최소 글자 수 설정을 조정해 보세요.")
        else:
            font_path = find_korean_font()

            if not font_path:
                st.error(
                    "한글 폰트를 찾지 못했습니다. Streamlit Cloud 저장소에 packages.txt를 함께 올렸는지 확인하세요."
                )
            else:
                max_words = st.slider(
                    "표시할 최대 단어 수",
                    min_value=20,
                    max_value=200,
                    value=100,
                    step=10,
                    key="max_words",
                )

                wordcloud = WordCloud(
                    font_path=font_path,
                    width=1400,
                    height=800,
                    background_color="white",
                    max_words=max_words,
                    collocations=False,
                ).generate_from_frequencies(frequencies)

                fig, ax = plt.subplots(figsize=(14, 8))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                freq_df = pd.DataFrame(
                    frequencies.most_common(30),
                    columns=["단어", "빈도"],
                )
                freq_fig = px.bar(
                    freq_df.sort_values("빈도"),
                    x="빈도",
                    y="단어",
                    orientation="h",
                    title="상위 30개 한글 단어",
                )
                st.plotly_chart(freq_fig, use_container_width=True)

                st.download_button(
                    "단어 빈도 CSV 다운로드",
                    data=freq_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{video_id}_word_frequency.csv",
                    mime="text/csv",
                )

        st.caption(
            "형태소 분석기 대신 연속된 한글 문자열을 단어로 추출하는 가벼운 방식입니다. "
            "조사·어미가 붙은 단어가 서로 다른 단어로 집계될 수 있습니다."
        )

    with tab4:
        display_df = comments[
            ["작성시각", "작성자", "댓글", "좋아요", "답글수", "감성"]
        ].copy()
        display_df["작성시각"] = display_df["작성시각"].dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "댓글": st.column_config.TextColumn(width="large"),
                "좋아요": st.column_config.NumberColumn(format="%d"),
                "답글수": st.column_config.NumberColumn(format="%d"),
            },
        )

        st.download_button(
            "전체 댓글 CSV 다운로드",
            data=display_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{video_id}_comments.csv",
            mime="text/csv",
        )
'''

requirements = '''
streamlit>=1.40,<2
pandas>=2.2,<3
requests>=2.32,<3
plotly>=5.24,<7
matplotlib>=3.9,<4
wordcloud>=1.9.4,<2
'''

packages = '''
fonts-nanum
'''

gitignore = '''
.streamlit/secrets.toml
__pycache__/
*.pyc
'''

secrets_example = '''
# 이 파일은 예시입니다.
# 실제 키가 들어간 secrets.toml은 GitHub에 올리지 마세요.
YOUTUBE_API_KEY = "여기에_본인의_API_키"
'''

readme = r'''
# 유튜브 댓글 분석기

YouTube 영상 링크를 입력하면 다음 기능을 제공합니다.

- 영상 임베드 및 기본 통계
- 수집 댓글 개수 설정
- 날짜별·시간대별·요일별 댓글 작성 추이
- 댓글 좋아요 및 답글 수 분석
- 간단한 규칙 기반 한국어 감성 분석
- 한글 워드클라우드와 단어 빈도
- 댓글·단어 빈도 CSV 다운로드

## 1. Google API 키 준비

1. Google Cloud Console에서 프로젝트를 만듭니다.
2. **YouTube Data API v3**를 활성화합니다.
3. 사용자 인증 정보에서 API 키를 생성합니다.
4. 가능하면 API 키에 API 제한과 사용처 제한을 설정합니다.

## 2. Streamlit Cloud 배포

GitHub 저장소 루트에 다음 파일을 올립니다.

- `app.py`
- `requirements.txt`
- `packages.txt`
- `.gitignore`

Streamlit Community Cloud에서 새 앱을 만들고 `app.py`를 실행 파일로 선택합니다.

앱 화면에서 키를 직접 입력할 수도 있습니다. 배포자가 키를 기본 제공하려면
Streamlit Cloud 앱 설정의 **Secrets**에 다음 내용을 입력합니다.

```toml
YOUTUBE_API_KEY = "실제_API_키"

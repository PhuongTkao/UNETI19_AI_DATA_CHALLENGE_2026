
from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_noi_bo"))

from predictor_v3 import InferenceV3
from model_utils import check_schema

st.set_page_config(
    page_title="NYC Taxi Predictor | Gia Cát Lượng",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_TRIP = {
    "vendor_id": 1,
    "pickup_month": 3,
    "pickup_dayofweek": 2,
    "pickup_hour": 10,
    "pickup_minute_bucket": 15,
    "passenger_count": 1,
    "pickup_lon_approx": -73.99,
    "pickup_lat_approx": 40.75,
    "dropoff_lon_approx": -73.98,
    "dropoff_lat_approx": 40.76,
    "store_and_fwd_flag": "N",
}

NEAR_TRIP = DEFAULT_TRIP.copy()

FAR_TRIP = {
    **DEFAULT_TRIP,
    "vendor_id": 2,
    "pickup_month": 6,
    "pickup_dayofweek": 4,
    "pickup_hour": 18,
    "pickup_minute_bucket": 30,
    "passenger_count": 2,
    "dropoff_lon_approx": -73.78,
    "dropoff_lat_approx": 40.65,
}


st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #F6F8FB;
}

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
    width: 270px !important;
    min-width: 270px !important;
    max-width: 270px !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 12px 13px 16px;
}

section[data-testid="stSidebar"] h2 {
    color: #152C49 !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    line-height: 1.3 !important;
}

section[data-testid="stSidebar"] h3 {
    position: relative;
    padding-left: 13px;
    color: #152C49 !important;
    font-size: 16px !important;
    font-weight: 750 !important;
    line-height: 1.35 !important;
}

section[data-testid="stSidebar"] h3::before {
    content: "";
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    height: 18px;
    width: 3px;
    border-radius: 3px;
    background: #FFC83D;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #5C6C84 !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
}

/* Tổ hợp mô hình: rõ hơn và vẫn trên một hàng */
section[data-testid="stSidebar"] .st-key-ensemble_line [data-testid="stCaptionContainer"] p {
    color: #43566E !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: -0.25px;
    white-space: nowrap !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: #243B68;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #E6F6EF;
    border: 1px solid #C8EBDD;
    border-radius: 10px;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] p {
    color: #087760 !important;
    font-weight: 700;
}

section[data-testid="stSidebar"] [data-testid="stMetric"] {
    min-width: 0;
    padding: 8px 6px;
    background: #F6F8FB;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

section[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
    font-size: 13px !important;
}

section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #152C49 !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    white-space: nowrap;
}

/* Thu gọn khoảng cách trước Thông tin đội */
.st-key-sidebar_team {
    border-top: 1px solid #E2E8F0;
    padding-top: 12px;
    margin-top: -6px !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1350px;
    padding-top: 2.5rem !important;
    padding-bottom: 1.5rem;
}

.st-key-hero_banner {
    background: #152C49;
    border-left: 4px solid #FFC83D;
    border-radius: 14px;
    padding: 25px 24px 21px;
    box-shadow: 0 5px 20px rgba(21,44,73,.09);
}

.st-key-hero_banner [data-testid="stVerticalBlock"] {
    gap: .35rem;
}

.st-key-hero_banner [data-testid="stMarkdownContainer"] p {
    color: #DCE8F7 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    margin: 0 !important;
}

.st-key-hero_banner h2 {
    color: #FFFFFF !important;
    font-size: clamp(25px, 2.2vw, 30px) !important;
    font-weight: 800 !important;
    line-height: 1.3 !important;
    margin: 0 !important;
}

.st-key-template_card,
.st-key-input_card,
.st-key-result_card {
    background: #FFFFFF;
    border-color: #E2E8F0 !important;
    border-radius: 14px;
    box-shadow: 0 4px 18px rgba(21,44,73,.04);
}

.st-key-input_card h3 {
    font-weight: 900 !important;
}

/* Tiêu đề nhóm: thanh vàng tự căn giữa */
.st-key-input_card h4 {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    color: #152C49 !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
    padding-left: 0 !important;
}

.st-key-input_card h4::before {
    content: "";
    position: static !important;
    display: block;
    flex: 0 0 3px;
    width: 3px;
    height: 20px;
    border-radius: 3px;
    background: #FFC83D;
    transform: none !important;
}

[data-testid="stWidgetLabel"] p {
    color: #253B59 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    line-height: 1.4;
}

[data-testid="stNumberInput"] input {
    color: #172B4D !important;
    font-size: 16px !important;
    font-weight: 600;
}

[data-baseweb="select"] span {
    font-size: 15px;
}

.st-key-input_card [data-testid="stCaptionContainer"] p,
.st-key-result_card [data-testid="stCaptionContainer"] p {
    color: #5B6B82 !important;
    font-size: 14px !important;
    line-height: 1.5;
}

[data-testid="stButton"] button {
    min-height: 43px;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 700;
}

[data-testid="stButton"] button[kind="primary"] {
    background: #FFC83D;
    color: #152C49;
    border-color: #FFC83D;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background: #EFBA2F;
    color: #152C49;
    border-color: #EFBA2F;
}

[data-testid="stButton"] button[kind="secondary"] {
    background: #FFFFFF;
    color: #152C49;
    border-color: #CBD5E1;
}

.st-key-predict_trip button[kind="primary"] {
    background: #152C49 !important;
    color: #FFFFFF !important;
    border-color: #152C49 !important;
}

.st-key-predict_trip button[kind="primary"]:hover {
    background: #27486D !important;
    border-color: #27486D !important;
}

.st-key-result_card {
    border: 1px solid #DBE5F0 !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 20px rgba(21,44,73,.06);
}

.st-key-result_card h3 {
    color: #152C49 !important;
    font-size: 23px !important;
    font-weight: 800 !important;
}

.st-key-result_value {
    background: #F7F9FC;
    border-left: 4px solid #FFC83D;
    border-radius: 12px;
    padding: 12px 18px;
}

.st-key-result_value h2 {
    color: #152C49 !important;
    font-size: clamp(36px, 4vw, 48px) !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    letter-spacing: -.03em;
}

.st-key-result_value [data-testid="stCaptionContainer"] p {
    color: #52657E !important;
    font-size: 15px !important;
}

/* SỬA LỖI: thông báo xanh không bị ép chiều cao */
.st-key-result_status [data-testid="stAlert"] {
    background: #E9F8F0 !important;
    border: 1px solid #D0EDE0 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

.st-key-result_status [data-testid="stAlert"] p {
    color: #08765F !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    line-height: 1.5 !important;
}

.st-key-result_card h4 {
    color: #152C49 !important;
    font-size: 18px !important;
    font-weight: 750 !important;
}

/* Khung khoảng cách: căn giữa toàn bộ nội dung */
.st-key-distance_info {
    background: #F8FAFD;
    border: 1px solid #DDE7F1 !important;
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center !important;
}

.st-key-distance_info [data-testid="stMetric"] {
    width: 100%;
    text-align: center !important;
}

.st-key-distance_info [data-testid="stMetricLabel"] {
    width: 100%;
    display: flex;
    justify-content: center !important;
    text-align: center !important;
}

.st-key-distance_info [data-testid="stMetricLabel"] p {
    color: #253B59 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    text-align: center !important;
}

.st-key-distance_info [data-testid="stMetricValue"] {
    width: 100%;
    display: flex;
    justify-content: center !important;
    text-align: center !important;
}

.st-key-distance_info [data-testid="stCaptionContainer"] p {
    color: #5B6B82 !important;
    font-size: 13px !important;
    text-align: center !important;
}

/* Ba ô thông tin hành trình */
.st-key-journey_distance,
.st-key-journey_pickup,
.st-key-journey_dropoff {
    background: #F8FAFD;
    border: 1px solid #DDE7F1 !important;
    border-radius: 12px;
    min-height: 106px;
    text-align: center;
}

.st-key-journey_distance [data-testid="stCaptionContainer"] p,
.st-key-journey_pickup [data-testid="stCaptionContainer"] p,
.st-key-journey_dropoff [data-testid="stCaptionContainer"] p {
    color: #52657E !important;
    font-size: 14px !important;
    text-align: center !important;
}

.st-key-journey_distance [data-testid="stMarkdownContainer"] p,
.st-key-journey_pickup [data-testid="stMarkdownContainer"] p,
.st-key-journey_dropoff [data-testid="stMarkdownContainer"] p {
    color: #152C49 !important;
    font-size: 19px !important;
    font-weight: 800 !important;
    text-align: center !important;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
        padding-left: 12px;
        padding-right: 12px;
    }

    .st-key-hero_banner {
        padding: 22px 17px 18px;
    }
}
</style>
""", unsafe_allow_html=True)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        math.radians,
        (lon1, lat1, lon2, lat2)
    )

    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin((lon2 - lon1) / 2) ** 2
    )

    return 6371.0 * 2 * math.asin(
        min(1.0, math.sqrt(a))
    )


def distance_km():
    s = st.session_state

    return haversine_km(
        s["pickup_lon_approx"],
        s["pickup_lat_approx"],
        s["dropoff_lon_approx"],
        s["dropoff_lat_approx"]
    )


def clear_prediction(mark_stale=False):
    had_result = (
        st.session_state.get("prediction_seconds") is not None
        or st.session_state.get("stale_prediction", False)
    )

    st.session_state.prediction_seconds = None
    st.session_state.prediction_signature = None
    st.session_state.prediction_error = None
    st.session_state.stale_prediction = bool(
        mark_stale and had_result
    )


def on_manual_edit():
    st.session_state.active_template = None
    clear_prediction(mark_stale=True)


def select_sample(name):
    values = NEAR_TRIP if name == "near" else FAR_TRIP

    st.session_state.update(values)
    st.session_state.active_template = name

    clear_prediction(mark_stale=True)


def reset_inputs():
    st.session_state.update(DEFAULT_TRIP)
    st.session_state.active_template = None
    clear_prediction()


def initialize_state():
    values = {
        **DEFAULT_TRIP,
        "active_template": None,
        "prediction_seconds": None,
        "prediction_signature": None,
        "prediction_error": None,
        "stale_prediction": False,
    }

    for key, value in values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_trip():
    s = st.session_state

    return pd.DataFrame([{
        "vendor_id": int(s["vendor_id"]),
        "pickup_month": int(s["pickup_month"]),
        "pickup_dayofweek": int(s["pickup_dayofweek"]),
        "pickup_hour": int(s["pickup_hour"]),
        "pickup_minute_bucket": int(s["pickup_minute_bucket"]),
        "passenger_count": int(s["passenger_count"]),
        "pickup_lon_approx": float(s["pickup_lon_approx"]),
        "pickup_lat_approx": float(s["pickup_lat_approx"]),
        "dropoff_lon_approx": float(s["dropoff_lon_approx"]),
        "dropoff_lat_approx": float(s["dropoff_lat_approx"]),
        "straight_line_km_approx": float(distance_km()),
        "store_and_fwd_flag": s["store_and_fwd_flag"],
    }])


def input_signature(trip):
    return tuple(trip.iloc[0].tolist())


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if (
        y_true.shape != y_pred.shape
        or y_true.size == 0
        or not np.isfinite(y_true).all()
        or not np.isfinite(y_pred).all()
        or (y_true < 0).any()
        or (y_pred < 0).any()
    ):
        raise ValueError("Dữ liệu kiểm định không hợp lệ")

    return float(
        np.sqrt(
            np.mean(
                (np.log1p(y_pred) - np.log1p(y_true)) ** 2
            )
        )
    )


@st.cache_data(show_spinner=False)
def load_validation_metrics():
    scores = {}

    files = [
        ("Selection", "selection_predictions.csv"),
        ("Audit", "audit_predictions.csv"),
    ]

    for name, filename in files:
        path = ROOT / "ket_qua" / filename

        if not path.is_file():
            continue

        try:
            df = pd.read_csv(path)

            if {
                "y_true_seconds",
                "v3_locked"
            }.issubset(df.columns):
                scores[name] = rmsle(
                    df["y_true_seconds"],
                    df["v3_locked"]
                )

        except (OSError, ValueError, KeyError):
            pass

    return scores


@st.cache_resource(show_spinner=False)
def load_model():
    return InferenceV3(ROOT / "_noi_bo")


initialize_state()

try:
    model = load_model()

except Exception as exc:
    st.error(
        "Không thể nạp mô hình V3. "
        "Kiểm tra thư mục _noi_bo/models."
    )
    st.exception(exc)
    st.stop()


with st.sidebar:
    st.markdown("## Nhóm Gia Cát Lượng")
    st.caption("Olympic Trí tuệ nhân tạo UNETI 2026")
    st.success("Mô hình đã sẵn sàng")

    st.markdown("### Thông tin mô hình")
    st.markdown("**Kiến trúc:** V3 Hybrid")
    st.markdown("**Tổ hợp mô hình**")

    with st.container(key="ensemble_line"):
        st.caption("LightGBM V2 (60%) + LightGBM Geo (40%)")

    st.markdown("### Kết quả kiểm định")
    scores = load_validation_metrics()

    col_selection, col_audit = st.columns(
        2,
        gap="small"
    )

    with col_selection:
        st.metric(
            "Selection",
            f"{scores['Selection']:.6f}"
            if "Selection" in scores else "—"
        )

    with col_audit:
        st.metric(
            "Audit",
            f"{scores['Audit']:.6f}"
            if "Audit" in scores else "—"
        )

    st.caption(
        "Kết quả trên là RMSLE kiểm định nội bộ. "
    )

    with st.container(key="sidebar_team"):
        st.markdown("### Thông tin đội")
        st.markdown("**Mã đội:** UNETI19")
        st.markdown("**Bộ Test:** UNETI04")


with st.container(key="hero_banner"):
    st.markdown(
        "**OLYMPIC TRÍ TUỆ NHÂN TẠO UNETI 2026 · VÒNG 2**"
    )

    st.markdown(
        "## Dự đoán thời gian di chuyển taxi"
    )

    st.markdown(
        "Nhập thông tin chuyến xe tại New York "
        "để ước tính thời gian di chuyển."
    )


with st.container(border=True, key="template_card"):
    st.markdown("**Thử một chuyến đi mẫu**")

    near, far, reset = st.columns(
        [1, 1, 0.75],
        gap="small"
    )

    with near:
        st.button(
            "Chuyến đi gần",
            key="sample_near",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.active_template == "near"
                else "secondary"
            ),
            on_click=select_sample,
            args=("near",),
        )

    with far:
        st.button(
            "Chuyến đi xa",
            key="sample_far",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.active_template == "far"
                else "secondary"
            ),
            on_click=select_sample,
            args=("far",),
        )

    with reset:
        st.button(
            "Đặt lại",
            key="sample_reset",
            use_container_width=True,
            on_click=reset_inputs,
        )


with st.container(border=True, key="input_card"):
    st.markdown("### Thông tin chuyến xe")

    left, right = st.columns(
        2,
        gap="large"
    )

    with left:
        st.markdown("#### Thời gian đón khách")

        t1, t2 = st.columns(
            2,
            gap="small"
        )

        with t1:
            st.selectbox(
                "Tháng",
                range(1, 7),
                key="pickup_month",
                on_change=on_manual_edit
            )

            st.selectbox(
                "Giờ đón khách",
                range(24),
                key="pickup_hour",
                on_change=on_manual_edit
            )

        with t2:
            st.selectbox(
                "Thứ trong tuần",
                options=range(7),
                format_func=lambda x: [
                    "Thứ Hai",
                    "Thứ Ba",
                    "Thứ Tư",
                    "Thứ Năm",
                    "Thứ Sáu",
                    "Thứ Bảy",
                    "Chủ Nhật"
                ][x],
                key="pickup_dayofweek",
                on_change=on_manual_edit
            )

            st.selectbox(
                "Phút đón khách",
                [0, 15, 30, 45],
                key="pickup_minute_bucket",
                on_change=on_manual_edit
            )

        st.markdown("#### Thông tin bổ sung")

        extra1, extra2 = st.columns(
            2,
            gap="small"
        )

        with extra1:
            st.number_input(
                "Số hành khách",
                min_value=1,
                max_value=6,
                step=1,
                key="passenger_count",
                on_change=on_manual_edit
            )

        with extra2:
            st.selectbox(
                "Mã đơn vị ghi nhận chuyến đi",
                options=[1, 2],
                format_func=lambda x: f"Đơn vị {x}",
                key="vendor_id",
                on_change=on_manual_edit
            )

        st.selectbox(
            "Trạng thái lưu dữ liệu",
            options=["N", "Y"],
            format_func=lambda x: {
                "N": "Truyền trực tiếp (N)",
                "Y": "Lưu để gửi sau (Y)"
            }[x],
            key="store_and_fwd_flag",
            on_change=on_manual_edit
        )

    with right:
        st.markdown("#### Điểm đón và điểm trả")

        pickup, dropoff = st.columns(
            2,
            gap="small"
        )

        with pickup:
            st.number_input(
                "Kinh độ đón",
                min_value=-75.0,
                max_value=-72.0,
                step=0.01,
                format="%.2f",
                key="pickup_lon_approx",
                on_change=on_manual_edit
            )

            st.number_input(
                "Vĩ độ đón",
                min_value=40.0,
                max_value=42.0,
                step=0.01,
                format="%.2f",
                key="pickup_lat_approx",
                on_change=on_manual_edit
            )

        with dropoff:
            st.number_input(
                "Kinh độ trả",
                min_value=-75.0,
                max_value=-72.0,
                step=0.01,
                format="%.2f",
                key="dropoff_lon_approx",
                on_change=on_manual_edit
            )

            st.number_input(
                "Vĩ độ trả",
                min_value=40.0,
                max_value=42.0,
                step=0.01,
                format="%.2f",
                key="dropoff_lat_approx",
                on_change=on_manual_edit
            )

        with st.container(border=True, key="distance_info"):
            st.metric(
                "Khoảng cách đường chim bay",
                f"{distance_km():.2f} km"
            )

            st.caption(
                "Tự tính từ tọa độ, "
                "không phải quãng đường giao thông thực tế."
            )

    if st.button(
        "Ước tính thời gian",
        key="predict_trip",
        type="primary",
        icon=":material/schedule:",
        use_container_width=True,
    ):
        try:
            trip = build_trip()
            check_schema(trip)

            with st.spinner(
                "Đang tính toán thời gian di chuyển..."
            ):
                predicted = float(
                    model.predict(trip)[0]
                )

            if (
                not math.isfinite(predicted)
                or predicted <= 0
            ):
                raise ValueError(
                    "Mô hình trả về thời gian không hợp lệ"
                )

            st.session_state.prediction_seconds = predicted
            st.session_state.prediction_signature = (
                input_signature(trip)
            )
            st.session_state.prediction_error = None
            st.session_state.stale_prediction = False

        except ValueError as exc:
            clear_prediction()
            st.session_state.prediction_error = str(exc)

        except Exception:
            clear_prediction()
            st.session_state.prediction_error = (
                "Không thể dự đoán. "
                "Kiểm tra dữ liệu và mô hình."
            )

    if st.session_state.prediction_error:
        st.error(st.session_state.prediction_error)


with st.container(border=True, key="result_card"):
    st.markdown("### Thời gian di chuyển dự kiến")

    current = build_trip()

    matching = (
        st.session_state.prediction_signature
        == input_signature(current)
    )

    result = (
        st.session_state.prediction_seconds
        if matching
        else None
    )

    if result is None:
        if st.session_state.stale_prediction or (
            st.session_state.prediction_seconds is not None
            and not matching
        ):
            st.warning(
                "Thông tin đã thay đổi. "
                "Nhấn Ước tính thời gian để xem kết quả mới."
            )
        else:
            st.info(
                "Chọn chuyến mẫu hoặc nhập thông tin, "
                "sau đó nhấn Ước tính thời gian."
            )

    else:
        minutes, seconds = divmod(
            int(round(result)),
            60
        )

        with st.container(key="result_value"):
            st.markdown(
                f"## {minutes} phút {seconds} giây"
            )

            st.caption(
                f"Tổng thời gian dự đoán: {result:,.1f} giây"
            )

        # Dùng st.success để chữ không bị cắt ở mép khung
        with st.container(key="result_status"):
            st.success(
                "Đã tính trực tiếp bằng mô hình "
                "V3 Hybrid đã huấn luyện"
            )

    st.markdown("#### Thông tin hành trình")

    c1, c2, c3 = st.columns(
        3,
        gap="small"
    )

    with c1:
        with st.container(
            border=True,
            key="journey_distance"
        ):
            st.caption("Khoảng cách chim bay")
            st.markdown(
                f"**{distance_km():.2f} km**"
            )

    with c2:
        with st.container(
            border=True,
            key="journey_pickup"
        ):
            st.caption("Điểm đón (vĩ độ, kinh độ)")
            st.markdown(
                f"**{st.session_state['pickup_lat_approx']:.2f}, "
                f"{st.session_state['pickup_lon_approx']:.2f}**"
            )

    with c3:
        with st.container(
            border=True,
            key="journey_dropoff"
        ):
            st.caption("Điểm trả (vĩ độ, kinh độ)")
            st.markdown(
                f"**{st.session_state['dropoff_lat_approx']:.2f}, "
                f"{st.session_state['dropoff_lon_approx']:.2f}**"
            )

    st.caption(
        "Lưu ý: Thời gian di chuyển được ước tính từ mô hình AI "
        "đã huấn luyện và có thể khác với thực tế."
    )

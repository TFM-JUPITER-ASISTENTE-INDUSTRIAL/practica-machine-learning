from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / 'dataset_practica_final.csv'
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "is_canceled"
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]

CATEGORICAL_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type"
]
NUMERICAL_COLS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests"
]

SPECIAL_COLS = [
    "agent",
    "company"
]

TOP_COUNTRIES = [
    "PRT",
    "GBR",
    "FRA",
    "ESP",
    "DEU"
]

TOP_AGENTS = [
    9.0, 
    240.0, 
    1.0, 
    14.0, 
    7.0,
    6.0, 
    250.0, 
    241.0, 
    28.0, 
    8.0,
    3.0, 
    37.0, 
    19.0, 
    40.0, 
    314.0,
    21.0, 
    229.0, 
    242.0, 
    83.0, 
    29.0
]
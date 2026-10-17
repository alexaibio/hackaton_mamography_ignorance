import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mammography.config import get_config
from mammography.io import read_table

HORIZONS = ["screening_3m_flag", "screening_6m_flag", "screening_12m_flag"]
CONTACT_BINS = [-1, 0, 2, 5, 10, 20, 1_000_000]
CONTACT_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "20+"]
PERCENT_COLUMNS = ["non_response_%", "share_of_non_responders_%"]
TOP_AREAS = 5

st.set_page_config(page_title="Screening outreach", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


@st.cache_data
def person_outcomes(invitations: pd.DataFrame) -> pd.DataFrame:
    return invitations.groupby("person_id")[HORIZONS].any()


def breakdown(invitations: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    grouped = invitations.groupby(column, dropna=False, observed=True).agg(
        invitations=(target, "size"),
        non_responders=(target, lambda flags: int((~flags).sum())),
    )
    grouped["non_response_%"] = (100 * grouped["non_responders"] / grouped["invitations"]).round(1)
    grouped["share_of_non_responders_%"] = (
        100 * grouped["non_responders"] / grouped["non_responders"].sum()
    ).round(1)
    return grouped


def draw_pie(values: pd.Series, title: str) -> None:
    figure, axes = plt.subplots(figsize=(5, 5))
    axes.pie(values.values, labels=values.index, autopct="%1.1f%%", startangle=90)
    axes.set_title(title)
    axes.axis("equal")
    st.pyplot(figure)


config = get_config()
target = config["target"]
invitations = load_invitations()
persons = person_outcomes(invitations)

start = invitations["invitation_date"].min()
end = invitations["invitation_date"].max()

st.title("Mammography screening outreach")

first, second, third, fourth = st.columns([1, 1, 2, 1.2])
first.metric("Invitations", f"{len(invitations):,}")
second.metric("Patients", f"{len(persons):,}")
third.metric("Study period", f"{start:%b %Y} – {end:%b %Y}")
fourth.metric(
    "Never responded",
    f"{(~persons['screening_12m_flag']).mean():.1%}",
    help="patients with no screening within 12 months of any of their invitations",
)

st.header("Responders")
st.caption("Share of patients screened within the horizon of at least one invitation.")

twelve, six, three = st.columns(3)
twelve.metric("within 12 months", f"{persons['screening_12m_flag'].mean():.1%}")
six.metric("within 6 months", f"{persons['screening_6m_flag'].mean():.1%}")
three.metric("within 3 months", f"{persons['screening_3m_flag'].mean():.1%}")

st.header("Non-responders")


st.subheader("By age band")

st.markdown(
    "- `non_response_%` - pecentage og non responders among all invitation\n"
    "- `share_of_non_responders_%` — share of this age among non-responders\n"
)

age = breakdown(invitations, "age_band", target)
age_left, age_right = st.columns([3, 2])
age_left.table(age)
age_right.bar_chart(age["non_response_%"])

st.subheader("By general medical activity")

st.markdown(
    "- **T0 - the invitation date**\n\n"
    "- `gp_registered` — had a valid registration with a general practitioner\n"
    "- `gyn_registered_flag` — the same for a gynaecologist\n"
    "- Healthcare contacts — number of contacts with any provider in the 12 months "
    "*before* T0, so it describes her engagement before the invitation was sent, "
    "never after."
)

gp, gyn = st.columns(2)
with gp:
    st.caption("Registered with a GP")
    st.table(breakdown(invitations, "gp_registered", target)[PERCENT_COLUMNS])
with gyn:
    st.caption("Registered with a gynaecologist")
    st.table(breakdown(invitations, "gyn_registered_flag", target)[PERCENT_COLUMNS])

activity = invitations.copy()
activity["healthcare_contact_12m"] = pd.cut(
    activity["healthcare_contact_12m_cnt"], CONTACT_BINS, labels=CONTACT_LABELS
)

st.caption("Healthcare contacts in the 12 months before the invitation")
contacts = breakdown(activity, "healthcare_contact_12m", target)
contact_left, contact_right = st.columns([3, 2])
contact_left.table(contacts[PERCENT_COLUMNS])
contact_right.bar_chart(contacts["non_response_%"])

st.subheader("By territory")
st.caption(
    f"{invitations['vzp_territorial_area'].nunique()} VZP branch areas — "
    "an approximation of region, not an administrative okres."
)
territory = breakdown(invitations, "vzp_territorial_area", target)

table_column, pie_column = st.columns([3, 2])

with table_column:
    st.dataframe(
        territory.sort_values("non_response_%", ascending=False)[PERCENT_COLUMNS],
        width="stretch",
        height=420,
    )

with pie_column:
    national_rate = territory["non_responders"].sum() / territory["invitations"].sum()
    excess = (
        territory["non_responders"] - territory["invitations"] * national_rate
    ).sort_values(ascending=False)
    above_average = excess[excess > 0]

    slices = above_average.head(TOP_AREAS).round(0)
    remainder = above_average.iloc[TOP_AREAS:].sum().round(0)
    if remainder > 0:
        slices = pd.concat([slices, pd.Series({"All other areas above average": remainder})])

    draw_pie(slices, "Excess non-responders, size-adjusted")
    st.caption(
        f"Non-responders above what this area's invitation volume would predict at the "
        f"national rate of {national_rate:.1%}. "
        f"{len(above_average)} of {len(territory)} areas are above average; the rest are "
        "at or below it and do not appear. A large area with an average rate contributes "
        "nothing here, however many non-responders it has."
    )

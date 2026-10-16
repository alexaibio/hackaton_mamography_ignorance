import json

from mammography.config import get_config
from mammography.llm.client import get_client

# never sent: direct identifiers, the exact invitation date, and the labels themselves
EXCLUDED_FIELDS = (
    "person_id",
    "invitation_id",
    "invitation_date",
    "screening_3m_flag",
    "screening_6m_flag",
    "screening_12m_flag",
)

# plain-language names, so the model can use the values in a sentence
FEATURE_LABELS = {
    "age_band": "age band",
    "vzp_territorial_area": "region (VZP branch area)",
    "insurance_duration_years": "years insured with VZP",
    "insurance_duration_band": "years insured with VZP (band)",
    "prior_invitation_count": "screening invitations she has already received",
    "months_last_invitation": "months since the previous invitation",
    "invitation_sequence": "which invitation in the sequence this is",
    "first_invitation": "this is her first invitation",
    "healthcare_contact_12m_cnt": "contacts with any doctor in the last 12 months",
    "gp_registered": "registered with a general practitioner",
    "gp_reg_duration_band": "years registered with that general practitioner",
    "gyn_registered_flag": "registered with a gynaecologist",
    "gyn_reg_duration_band": "years registered with that gynaecologist",
    "prior_screen_count_5y": "mammography screenings in the last 5 years",
    "months_since_last_screen": "months since her last mammography",
    "never_screened_before_flag": "has never had a mammography screening",
    "gp_prevention_24m_cnt": "preventive check-ups at the GP in the last 24 months",
    "gyn_prevention_24m_cnt": "preventive check-ups at the gynaecologist in the last 24 months",
    "other_screening_36m_cnt": "participations in other screening programmes in the last 36 months",
    "dental_prevention_24m_cnt": "dental preventive check-ups in the last 24 months",
    "invitation_count": "invitations she has received in total",
    "notification_channel": "channel this message will be sent through",
    "suggested_actions": "obstacles worth addressing, identified from her record",
}

SHARED_RULES = """Rules:
- use the attributes to make the message specific and persuasive: refer to concrete \
facts about her situation, such as having seen her gynaecologist recently, never having \
been screened, or having been invited several times already
- when an attribute shows an obstacle, address that obstacle directly
- plain language a non-medical reader understands immediately
- state one clear next action
- it is fine to say that the chance of breast cancer rises with age and that she is now \
in a given age band, but NEVER state a numeric statistic, percentage or risk figure — \
write [STATISTIKA] where one belongs and a human will fill it in
- never mention a model, a score, a prediction or a risk ranking, and never imply she \
has been singled out by a computer
- no diagnosis, no medical advice, no guilt, no fear, no pressure
- do not invent names, dates, phone numbers, addresses or appointment times; use \
placeholders in square brackets
- write in {language}
- channel is {channel}: for sms stay under 160 characters, for letter use at most 120 \
words, for email use at most 150 words with a short subject line"""

INVITATION_PROMPT = (
    """You write invitation and reminder messages for a national breast cancer screening \
programme. You are given every attribute on record for one woman whom the programme \
expects not to respond.

Write the message that should be sent to her next.

"""
    + SHARED_RULES
    + """

Return the message text only."""
)

REMINDER_SEQUENCE_PROMPT = (
    """You write reminder messages for a national breast cancer screening programme. You \
are given every attribute on record for one woman who has not responded to her \
screening invitations.

Write a sequence of {count} reminders, to be sent several weeks apart, as JSON:
{{"reminders": ["first", "second", ...]}}

The sequence escalates in concreteness, never in pressure: a gentle nudge first, then \
removing the practical obstacle her record points to, then offering direct help such as \
booking on her behalf. Each message must use different facts from her record so the \
three do not read as copies.

"""
    + SHARED_RULES
    + """

Return only the JSON object."""
)


CLINICIAN_EMAIL_PROMPT = """You write to physicians on behalf of a national breast cancer \
screening programme. You are given every attribute on record for one woman in the \
eligible cohort who has not attended screening.

Write one short email to her {clinician}, asking them to encourage her to attend.

Rules:
- address the physician, never the patient
- say briefly why she is being raised: her age band, how many invitations she has \
received, and that no screening is recorded for her
- use her record to make the ask concrete: if she attends preventive check-ups with \
this physician, point out that the next visit is an opportunity; if she has never been \
screened, say so
- the ask must suit the recipient: a gynaecologist can issue the referral and book her \
directly; a general practitioner should raise it opportunistically and direct her to a \
gynaecologist or a screening centre
- it is fine to say this came from the programme's outreach analysis, but never mention \
a model, a score, a probability or a risk ranking
- NEVER state a numeric statistic, percentage or risk figure — write [STATISTIKA] where \
one belongs
- do not invent the patient's name or insurance number, or any date, phone number or \
address; use placeholders in square brackets
- professional, factual, at most 150 words, with a short subject line
- write in {language}

Return the email text only."""


def _plain(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def describe_patient(attributes: dict) -> dict:
    """Everything known about the patient, relabelled, minus identifiers and labels."""
    described = {}
    for name, value in attributes.items():
        if name in EXCLUDED_FIELDS:
            continue
        plain = _plain(value)
        if plain is None:
            continue
        described[FEATURE_LABELS.get(name, name)] = plain
    return described


def _complete(system_prompt: str, attributes: dict, channel: str, **prompt_kwargs) -> str:
    config = get_config().get("llm", {})
    response = get_client().chat.completions.create(
        model=config["model"],
        temperature=config.get("temperature", 0.4),
        max_completion_tokens=prompt_kwargs.pop(
            "max_tokens", config.get("max_output_tokens", 400)
        ),
        messages=[
            {
                "role": "system",
                "content": system_prompt.format(
                    language=config.get("language", "Czech"), channel=channel, **prompt_kwargs
                ),
            },
            {
                "role": "user",
                "content": json.dumps(describe_patient(attributes), ensure_ascii=False),
            },
        ],
        **({"response_format": {"type": "json_object"}} if "count" in prompt_kwargs else {}),
    )
    return response.choices[0].message.content.strip()


def draft_invitation_message(invitation: dict, channel: str) -> str:
    return _complete(INVITATION_PROMPT, invitation, channel)


def draft_clinician_email(patient: dict, clinician: str) -> str:
    return _complete(CLINICIAN_EMAIL_PROMPT, patient, "email", clinician=clinician)


def draft_reminder_sequence(patient: dict, channel: str, count: int = 3) -> list[str]:
    config = get_config().get("llm", {})
    content = _complete(
        REMINDER_SEQUENCE_PROMPT,
        patient,
        channel,
        count=count,
        max_tokens=config.get("max_output_tokens", 400) * count,
    )
    return json.loads(content)["reminders"]

import json

from mammography.config import get_config
from mammography.llm.client import get_client

# only these reach the API — no names, ids, addresses or dates of birth
ALLOWED_FIELDS = (
    "age_band",
    "region",
    "notification_type",
    "notification_channel",
    "n_prior_invitations",
    "n_prior_attended",
    "preferred_language",
    "has_mobile",
    "has_email",
    "has_registered_gp",
    "gp_registered",
    "gyn_registered_flag",
    "never_screened_before_flag",
    "prior_screen_count_5y",
    "healthcare_contact_12m_cnt",
    "invitation_count",
    "risk_of_ignoring",
    "top_driver",
)

SYSTEM_PROMPT = """You write reminder messages for a national breast cancer screening \
programme. You are given an anonymous description of one invitation that the model \
predicts will be ignored, including the likely reason.

Write the reminder that should be sent next. Rules:
- address the specific barrier in the description, do not write a generic reminder
- plain language a non-medical reader understands immediately
- state one clear next action
- no diagnosis, no medical advice, no statistics about cancer risk
- no guilt, fear or pressure
- do not invent names, dates, phone numbers, addresses or appointment times; use \
placeholders in square brackets if a detail is needed
- write in {language}
- channel is {channel}: for sms stay under 160 characters, for letter use at most \
120 words, for email use at most 150 words with a short subject line

Return the message text only."""


REMINDER_SEQUENCE_PROMPT = """You write reminder messages for a national breast cancer \
screening programme. You are given an anonymous description of one woman who has not \
responded to her screening invitations.

Write a sequence of {count} reminders, to be sent several weeks apart, as JSON:
{{"reminders": ["first", "second", ...]}}

The sequence should escalate in concreteness, never in pressure:
- the first is a gentle nudge
- the middle one removes a practical obstacle visible in the description
- the last offers direct help, such as booking on her behalf

Rules for every message:
- use what the description actually says: her age band, whether she has a registered \
general practitioner or gynaecologist, how many times she has been invited, whether she \
has ever been screened
- it is fine to say that the chance of breast cancer rises with age and that she is now \
in a given age band, but NEVER state a numeric statistic, percentage or risk figure — \
write [STATISTIKA] where one belongs and a human will fill it in
- never mention a model, a score, a prediction or a risk ranking
- no diagnosis, no medical advice, no guilt, no fear
- do not invent names, dates, phone numbers, addresses or appointment times; use \
placeholders in square brackets
- write in {language}
- channel is {channel}: for sms stay under 160 characters, for letter use at most 120 \
words, for email use at most 150 words with a short subject line

Return only the JSON object."""


def draft_reminder_sequence(patient: dict, channel: str, count: int = 3) -> list[str]:
    config = get_config().get("llm", {})
    payload = {field: patient[field] for field in ALLOWED_FIELDS if field in patient}

    response = get_client().chat.completions.create(
        model=config["model"],
        temperature=config.get("temperature", 0.4),
        max_completion_tokens=config.get("max_output_tokens", 400) * count,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": REMINDER_SEQUENCE_PROMPT.format(
                    count=count,
                    language=config.get("language", "Czech"),
                    channel=channel,
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return json.loads(response.choices[0].message.content)["reminders"]


def draft_invitation_message(invitation: dict, channel: str) -> str:
    config = get_config().get("llm", {})
    payload = {field: invitation[field] for field in ALLOWED_FIELDS if field in invitation}

    response = get_client().chat.completions.create(
        model=config["model"],
        temperature=config.get("temperature", 0.4),
        max_completion_tokens=config.get("max_output_tokens", 400),
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    language=config.get("language", "Czech"), channel=channel
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return response.choices[0].message.content.strip()

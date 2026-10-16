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

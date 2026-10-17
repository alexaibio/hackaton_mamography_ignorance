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

NEVER = """Never do any of the following:
- state a numeric statistic, percentage, survival figure or risk number. If one belongs \
somewhere, write [STATISTIKA] and a human will fill it in from the programme's own \
materials
- mention a model, a score, a probability, a prediction or a ranking, or imply that a \
computer selected her
- use guilt, fear, pressure, or any suggestion that she has done something wrong
- invent a name, a date, a time, a phone number, an address or a price. Use a \
placeholder in square brackets instead, such as [TELEFON] or [TERMÍN]
- promise a result, a diagnosis, or medical advice of any kind"""

REMINDER_SYSTEM = """# Who you are

You are a communication specialist for the Czech national breast cancer screening \
programme, run with the VZP health insurance fund. You have spent years writing to women \
who were invited for a screening mammography and did not come. You know that the reason \
is almost never refusal of the idea — it is postponement, uncertainty about what happens \
at the examination, not knowing whom to ask, or a letter that read like an administrative \
notice and was put aside.

Your job is not to inform her. She has already been informed, several times. Your job is \
to get her to pick up the phone and book.

# Who you are writing to

One woman, in the eligible age group, who has already received at least one invitation \
and has not attended. Her anonymous record is given below: her age band, her region, how \
many invitations she has had, whether she has a registered general practitioner and \
gynaecologist, how often she sees a doctor at all, and whether she has ever been screened.

Read that record before writing. It tells you which obstacle is hers. A woman who sees \
her gynaecologist twice a year and has never been screened has a different problem from a \
woman with no registered doctor at all, and the two should not receive the same letters.

# The sequence

Write {count} messages, sent several weeks apart over roughly three months. She reads \
them in order. Each must stand on its own, and each must add something the previous one \
did not — if the second could be swapped for the first without anyone noticing, you have \
written the sequence wrongly.

**Message 1 — make it feel easy.**
Short. Warm. One idea only: the examination is quick, it is already paid for, and here is \
the single step to arrange it. Do not list everything you know. Do not explain the whole \
programme. If she reads one sentence, it should be the one telling her what to do.

**Message 2 — remove her specific obstacle.**
This is where her record matters most:
- no registered gynaecologist → tell her she does not need one to attend, and how to find \
a screening centre directly
- a gynaecologist she visits → tell her to raise it at the next appointment, and that they \
can issue the referral and book it there
- never screened before → describe the examination plainly: how long it takes, that it is \
done by a woman radiographer in most centres, that the compression lasts seconds, that \
she can ask them to stop at any moment
- invited many times already → acknowledge it without reproach. "We have written before, \
and we understand that life gets in the way" opens a door that another reminder closes
- sees a doctor rarely → keep it even simpler and make the one action unmissable

**Message 3 — offer to do the work for her.**
By now the gentle approach has not worked, so stop asking and start offering: a slot \
already held in her name, help arranging transport, a named contact person rather than a \
general line, or an appointment outside working hours. The message should feel like \
someone has taken the task off her hands.

# What belongs in these messages

- that the examination is fully covered by her insurance and costs her nothing
- that it takes a few minutes
- that the overwhelming majority of results are normal
- one concrete next action, in its own sentence, impossible to miss
- that she can bring someone with her, and can ask questions before anything happens
- it is fine and useful to say that the chance of breast cancer rises with age and that \
she is now in the age group where regular screening matters most

# Fears to address, gently and without naming them as fears

Many women postpone because they expect pain, because undressing in front of strangers \
feels humiliating, because they are afraid of what might be found, or because they feel \
foolish for having left it so long. Do not announce these. Defuse them in passing — \
"most women find it easier than they expected", "the staff are used to explaining \
everything first", "it is never too late to start".

# How to write

- short sentences, one idea each. If a sentence needs reading twice, rewrite it
- the register of a good nurse speaking to a patient she respects, not of an institution \
addressing a file number
- no bureaucratic phrasing: no "we hereby inform you", no "in accordance with", no \
"dear insured person"
- no medical jargon, no abbreviations she would have to decode
- vary the opening of each message, so the three do not look mass-produced
- speak to her as "you", never about "patients" in the third person
- use her region or age band naturally if it helps, never as a label

{never}

# Channel

Channel is {channel}:
- **sms** — under 160 characters, no salutation, one action, a phone number placeholder
- **letter** — at most 120 words, open with a proper salutation, close with the name of \
the screening centre as a placeholder
- **email** — at most 150 words, begin with a short subject line of at most eight words, \
then the body

# Language

Write in {language}. Use the natural, current form of the language, not a translation of \
English phrasing.

# Output

Return only JSON, in this shape, with no commentary:
{{"reminders": ["first message", "second message", "third message"]}}"""

CLINICIAN_SYSTEM = """You are writing on behalf of the Czech national breast cancer \
screening programme to a {clinician} about one of their patients.

Your task now: write one email asking this physician to encourage the patient to attend a \
screening mammography. She is in the eligible age group, has been invited, and has not \
come. You are given her anonymous record below.

Why this email exists: a recommendation from the doctor a woman already knows and trusts \
moves her far more than another letter from an insurer. You are asking for thirty seconds \
of their next consultation, not for a new workflow.

The email must contain:
- one line on why she is being raised: her age group, that she has been invited, and that \
no screening is recorded for her
- one or two concrete facts from her record that make the ask easy to act on. If she \
attends preventive check-ups with this physician, point out that the next visit is the \
opportunity. If she has never been screened, say so, because the first time needs more \
explanation than a repeat
- a specific request that suits the recipient. A gynaecologist can issue the referral and \
book her during the visit. A general practitioner should raise it opportunistically and \
direct her to a gynaecologist or a screening centre
- an offer of support from the programme: materials for her, or help with booking

It is fine to say the programme identified her through its outreach analysis of insurance \
records.

{never}

Format: a short subject line, then at most 150 words. Professional and factual — this is \
a colleague, not a patient. No marketing tone.

Write in {language}."""


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


def patient_brief(attributes: dict) -> str:
    """The record as readable lines rather than raw json."""
    lines = []
    for label, value in describe_patient(attributes).items():
        if isinstance(value, bool):
            value = "yes" if value else "no"
        elif isinstance(value, (list, tuple)):
            value = "; ".join(str(item) for item in value)
        lines.append(f"- {label}: {value}")
    return "Her record, anonymous:\n" + "\n".join(lines)


def reminder_messages(patient: dict, channel: str, count: int = 3) -> list[dict]:
    language = get_config().get("llm", {}).get("language", "Czech")
    return [
        {
            "role": "system",
            "content": REMINDER_SYSTEM.format(
                count=count, channel=channel, language=language, never=NEVER
            ),
        },
        {"role": "user", "content": patient_brief(patient)},
    ]


def clinician_messages(patient: dict, clinician: str) -> list[dict]:
    language = get_config().get("llm", {}).get("language", "Czech")
    return [
        {
            "role": "system",
            "content": CLINICIAN_SYSTEM.format(
                clinician=clinician, language=language, never=NEVER
            ),
        },
        {"role": "user", "content": patient_brief(patient)},
    ]


def render_messages(messages: list[dict]) -> str:
    return "\n\n".join(f"=== {message['role']} ===\n{message['content']}" for message in messages)


def _complete(messages: list[dict], max_tokens: int, as_json: bool = False) -> str:
    config = get_config().get("llm", {})
    response = get_client().chat.completions.create(
        model=config["model"],
        temperature=config.get("temperature", 0.4),
        max_completion_tokens=max_tokens,
        messages=messages,
        **({"response_format": {"type": "json_object"}} if as_json else {}),
    )
    return response.choices[0].message.content.strip()


def draft_reminder_sequence(patient: dict, channel: str, count: int = 3) -> list[str]:
    budget = get_config().get("llm", {}).get("max_output_tokens", 400)
    content = _complete(
        reminder_messages(patient, channel, count), max_tokens=budget * count, as_json=True
    )
    return json.loads(content)["reminders"]


def draft_clinician_email(patient: dict, clinician: str) -> str:
    budget = get_config().get("llm", {}).get("max_output_tokens", 400)
    return _complete(clinician_messages(patient, clinician), max_tokens=budget)

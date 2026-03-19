import json
import os

DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "questions.jsonl")

ANSWERABLE = [
    {
        "question": "How many PTO days are accrued per month for full-time employees?",
        "ground_truth": "1.5 days of PTO per month",
        "evidence_doc_id": "policy_pto",
        "evidence_quote": "Full-time employees accrue 1.5 days of PTO per month.",
    },
    {
        "question": "What is the PTO carryover limit into the next year?",
        "ground_truth": "10 days",
        "evidence_doc_id": "policy_pto",
        "evidence_quote": "PTO can be carried over up to 10 days into the next calendar year.",
    },
    {
        "question": "How far in advance must PTO requests be submitted?",
        "ground_truth": "at least 5 business days in advance",
        "evidence_doc_id": "policy_pto",
        "evidence_quote": "PTO requests must be submitted at least 5 business days in advance.",
    },
    {
        "question": "What class must domestic flights be booked in?",
        "ground_truth": "economy class",
        "evidence_doc_id": "policy_travel",
        "evidence_quote": "Domestic flights must be economy class.",
    },
    {
        "question": "When are international flights allowed to be premium economy?",
        "ground_truth": "over 8 hours",
        "evidence_doc_id": "policy_travel",
        "evidence_quote": "International flights over 8 hours may be premium economy.",
    },
    {
        "question": "What is the nightly hotel cap for lodging?",
        "ground_truth": "$210",
        "evidence_doc_id": "policy_travel",
        "evidence_quote": "nightly hotel cap is $210",
    },
    {
        "question": "What authentication requirement applies to corporate accounts?",
        "ground_truth": "Multi-factor authentication (MFA) is required",
        "evidence_doc_id": "policy_security",
        "evidence_quote": "Multi-factor authentication (MFA) is required for all corporate accounts.",
    },
    {
        "question": "How long must passwords be and how often rotated?",
        "ground_truth": "at least 12 characters and rotated every 180 days",
        "evidence_doc_id": "policy_security",
        "evidence_quote": "Passwords must be at least 12 characters and rotated every 180 days.",
    },
    {
        "question": "Within how many hours must security incidents be reported?",
        "ground_truth": "within 4 hours",
        "evidence_doc_id": "policy_security",
        "evidence_quote": "Incidents must be reported to security within 4 hours.",
    },
    {
        "question": "How many remote work days per week are allowed?",
        "ground_truth": "up to 3 days per week",
        "evidence_doc_id": "policy_remote",
        "evidence_quote": "Employees may work remotely up to 3 days per week.",
    },
    {
        "question": "What are the core collaboration hours?",
        "ground_truth": "10:00-15:00 local time",
        "evidence_doc_id": "policy_remote",
        "evidence_quote": "Core collaboration hours are 10:00-15:00 local time.",
    },
    {
        "question": "Is remote work allowed outside the employee's country of employment?",
        "ground_truth": "Remote work is not permitted outside the employee's country",
        "evidence_doc_id": "policy_remote",
        "evidence_quote": "Remote work is not permitted outside the employee's country of employment.",
    },
    {
        "question": "What is the daily meal reimbursement limit?",
        "ground_truth": "$55 per day",
        "evidence_doc_id": "policy_expenses",
        "evidence_quote": "Meals are reimbursed up to $55 per day with receipts.",
    },
    {
        "question": "What is the mileage reimbursement rate?",
        "ground_truth": "$0.52 per mile",
        "evidence_doc_id": "policy_expenses",
        "evidence_quote": "Mileage reimbursement is $0.52 per mile.",
    },
    {
        "question": "When are expense reports due after travel completion?",
        "ground_truth": "within 10 days",
        "evidence_doc_id": "policy_expenses",
        "evidence_quote": "Expense reports are due within 10 days of travel completion.",
    },
    {
        "question": "What is the default admin IP for the AetherX AX-200 router?",
        "ground_truth": "192.168.1.1",
        "evidence_doc_id": "manual_router",
        "evidence_quote": "Default admin IP is 192.168.1.1.",
    },
    {
        "question": "How long should the reset pin be held for a factory reset on AX-200?",
        "ground_truth": "12 seconds",
        "evidence_doc_id": "manual_router",
        "evidence_quote": "Factory reset requires holding the reset pin for 12 seconds.",
    },
    {
        "question": "What does a solid blue status LED indicate on the AX-200?",
        "ground_truth": "connected",
        "evidence_doc_id": "manual_router",
        "evidence_quote": "The status LED is solid blue when connected",
    },
    {
        "question": "Which Wi-Fi bands does the AX-200 support?",
        "ground_truth": "2.4 GHz and 5 GHz bands",
        "evidence_doc_id": "manual_router",
        "evidence_quote": "Supports Wi-Fi 6 on 2.4 GHz and 5 GHz bands.",
    },
    {
        "question": "Which app is used for initial AetherX router setup?",
        "ground_truth": "AetherX mobile app",
        "evidence_doc_id": "manual_setup",
        "evidence_quote": "Initial setup uses the AetherX mobile app (iOS/Android).",
    },
    {
        "question": "What are the default admin username and password for the AX-200?",
        "ground_truth": "username admin and password aetherx",
        "evidence_doc_id": "manual_setup",
        "evidence_quote": "Default admin username is \"admin\" and password is \"aetherx\".",
    },
    {
        "question": "Where can the guest network be enabled in the AX-200 settings?",
        "ground_truth": "Settings > Wi-Fi",
        "evidence_doc_id": "manual_setup",
        "evidence_quote": "Guest network can be enabled under Settings > Wi-Fi.",
    },
    {
        "question": "How often are firmware updates released for the AX-200?",
        "ground_truth": "quarterly",
        "evidence_doc_id": "manual_maintenance",
        "evidence_quote": "Firmware updates are released quarterly.",
    },
    {
        "question": "How long should the router remain plugged in during an update?",
        "ground_truth": "6 minutes",
        "evidence_doc_id": "manual_maintenance",
        "evidence_quote": "do not unplug the router for 6 minutes.",
    },
    {
        "question": "Where do you go to back up router settings?",
        "ground_truth": "Admin > Backup",
        "evidence_doc_id": "manual_maintenance",
        "evidence_quote": "To back up settings, go to Admin > Backup",
    },
    {
        "question": "What is the operating temperature range for the AX-200?",
        "ground_truth": "0-40 C",
        "evidence_doc_id": "manual_maintenance",
        "evidence_quote": "Operating temperature range is 0-40 C.",
    },
    {
        "question": "What does the AS-10 sensor measure?",
        "ground_truth": "temperature and humidity",
        "evidence_doc_id": "manual_sensor",
        "evidence_quote": "The AS-10 sensor measures temperature and humidity.",
    },
    {
        "question": "What is the battery life of the AS-10 sensor?",
        "ground_truth": "18 months",
        "evidence_doc_id": "manual_sensor",
        "evidence_quote": "Battery life is 18 months under normal use.",
    },
    {
        "question": "How long should the pairing button be pressed on the AS-10?",
        "ground_truth": "3 seconds",
        "evidence_doc_id": "manual_sensor",
        "evidence_quote": "The pairing button must be pressed for 3 seconds.",
    },
    {
        "question": "How many devices can the AH-5 hub connect?",
        "ground_truth": "up to 32 devices",
        "evidence_doc_id": "manual_hub",
        "evidence_quote": "The hub connects up to 32 devices.",
    },
    {
        "question": "By how much did a 20% canopy cover increase reduce peak surface temperature?",
        "ground_truth": "3.5 C",
        "evidence_doc_id": "article_heat_islands",
        "evidence_quote": "Increasing canopy cover by 20% reduced peak surface temperature by 3.5 C.",
    },
    {
        "question": "At what time of day were thermal images taken in the study?",
        "ground_truth": "2 pm",
        "evidence_doc_id": "article_heat_islands",
        "evidence_quote": "The study used thermal imaging at 2 pm during summer months.",
    },
    {
        "question": "At what height were sensors placed in the measurement protocol?",
        "ground_truth": "1.5 meters above ground",
        "evidence_doc_id": "article_methods",
        "evidence_quote": "Sensors were placed at 1.5 meters above ground.",
    },
    {
        "question": "How frequently was data logged in the study?",
        "ground_truth": "every 10 minutes",
        "evidence_doc_id": "article_methods",
        "evidence_quote": "Data were logged every 10 minutes.",
    },
    {
        "question": "What statistical test and alpha were used?",
        "ground_truth": "paired t-test with alpha 0.05",
        "evidence_doc_id": "article_methods",
        "evidence_quote": "Statistical analysis used a paired t-test with alpha 0.05.",
    },
    {
        "question": "How much did high-albedo pavement reduce surface temperature?",
        "ground_truth": "2.1 C",
        "evidence_doc_id": "article_findings",
        "evidence_quote": "High-albedo pavement reduced surface temperature by 2.1 C.",
    },
    {
        "question": "How did dark asphalt affect surface temperature relative to control?",
        "ground_truth": "increased by 1.8 C",
        "evidence_doc_id": "article_findings",
        "evidence_quote": "Dark asphalt increased surface temperature by 1.8 C relative to control.",
    },
    {
        "question": "What instrument measured reflectivity in the study?",
        "ground_truth": "portable spectrometer",
        "evidence_doc_id": "article_findings",
        "evidence_quote": "Reflectivity was measured with a portable spectrometer.",
    },
    {
        "question": "What key limitation was noted about nighttime temperatures?",
        "ground_truth": "The study did not measure nighttime temperatures",
        "evidence_doc_id": "article_limitations",
        "evidence_quote": "The study did not measure nighttime temperatures.",
    },
    {
        "question": "What is one policy implication mentioned in the abstract?",
        "ground_truth": "zoning incentives for canopy cover",
        "evidence_doc_id": "article_abstract",
        "evidence_quote": "Policy implications include zoning incentives for canopy cover.",
    },
]

UNANSWERABLE = [
    "What is the parental leave duration in the company policies?",
    "Does the travel policy allow business class for domestic flights?",
    "What is the company's data retention period?",
    "How many days of bereavement leave are provided?",
    "Is remote work allowed from any country during holidays?",
    "What is the AX-200 router's Wi-Fi 7 capability?",
    "Does the AX-200 support WPA3 Enterprise?",
    "What is the price of the AetherX AX-200?",
    "How long does the AX-200 warranty last?",
    "What is the AS-10 sensor's IP rating?",
    "How many antennas does the AH-5 hub have?",
    "What is the maximum range of the AS-10 sensor?",
    "Did the study measure nighttime temperatures and report results?",
    "What is the humidity effect size in the heat island study?",
    "What were the authors' names in the study?",
    "Which coastal cities were included in the sample?",
    "What was the study's funding source?",
    "What is the DOI of the heat island paper?",
    "What is the standard deviation of temperature reduction?",
    "How did rainfall affect the measurements?",
]


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    records = []
    qid = 1
    for item in ANSWERABLE:
        records.append(
            {
                "id": f"q{qid:03d}",
                "question": item["question"],
                "answerable": True,
                "ground_truth": item["ground_truth"],
                "evidence_doc_id": item["evidence_doc_id"],
                "evidence_quote": item["evidence_quote"],
            }
        )
        qid += 1
    for question in UNANSWERABLE:
        records.append(
            {
                "id": f"q{qid:03d}",
                "question": question,
                "answerable": False,
                "ground_truth": "",
                "evidence_doc_id": "",
                "evidence_quote": "",
            }
        )
        qid += 1

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()

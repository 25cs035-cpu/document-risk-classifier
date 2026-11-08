"""Generate a synthetic labelled dataset of business documents (Low / Medium / High risk).

Run:  python generate_data.py
Creates: data/labels.csv (300 docs) and samples/*.txt (3 demo files for the app).
Some sentences are deliberately borrowed from other classes so the data is
noisy and the confidence threshold actually matters.
"""
import os
import random

import pandas as pd

random.seed(7)

DOC_TYPES = [
    "Service Agreement", "Vendor Contract", "Internal Policy", "Incident Report",
    "Audit Summary", "Invoice Review", "Data Processing Addendum", "Compliance Memo",
]

NEUTRAL = [
    "This document is issued by the Operations department and applies to all business units.",
    "The parties have agreed to the terms set out in the schedules attached to this document.",
    "Records related to this matter are retained for seven years in line with company practice.",
    "The responsible manager will review this document at the next quarterly meeting.",
    "Payment terms are thirty days from the date of invoice unless otherwise agreed.",
    "All notices must be sent in writing to the registered address of each party.",
    "This document should be read together with the master services agreement.",
    "The project team met on Tuesday to discuss deliverables and timelines.",
]

LOW = [
    "The vendor has passed all routine compliance checks and holds valid certifications.",
    "Liability is capped at the total fees paid under this agreement in the previous twelve months.",
    "No incidents or complaints were recorded during the review period.",
    "The contract follows the standard company template with no deviations.",
    "All required approvals were obtained and signed before the effective date.",
    "Personal data is processed only within approved systems and encrypted at rest.",
    "The supplier is on the approved vendor list and was audited successfully this year.",
    "Both parties confirmed that all obligations were fulfilled on schedule.",
    "The policy has been reviewed by legal and requires no changes.",
    "Insurance coverage meets the minimum requirements set by the procurement policy.",
    "The invoice matches the purchase order and was approved by the budget owner.",
    "Training records show that all staff completed the mandatory compliance course.",
]

MEDIUM = [
    "Two required signatures are still missing and approval is pending from the legal team.",
    "The renewal clause is ambiguous and may allow automatic extension without notice.",
    "A minor deviation from the standard template was identified in the payment section.",
    "Several invoices were paid late and the vendor has requested a revised schedule.",
    "Data is shared with a subcontractor whose security review is not yet complete.",
    "The liability cap exists but excludes certain categories of damages that need clarification.",
    "An internal audit found documentation gaps that must be fixed within sixty days.",
    "The service levels are not clearly defined and could lead to disputes.",
    "A policy exception was granted temporarily and needs to be reassessed next quarter.",
    "Staff training completion is below the required target and a reminder was issued.",
    "The vendor certificate expires next month and renewal has not been confirmed.",
    "A low-value complaint was raised by a customer and is currently under investigation.",
]

HIGH = [
    "The agreement contains unlimited liability and uncapped indemnification obligations.",
    "A data breach exposed customer personal information and regulators must be notified.",
    "The transaction shows indicators of possible money laundering and was escalated to compliance.",
    "The company faces a regulatory fine for non-compliance with data protection law.",
    "Litigation has been filed against the company for breach of contract and damages.",
    "The counterparty appears on a sanctions list and the payment was blocked.",
    "An employee reported suspected bribery involving a government official through a whistleblower channel.",
    "The vendor may terminate without notice and impose severe financial penalties.",
    "Critical security controls were found missing and sensitive data was transferred without consent.",
    "The audit identified fraud and falsified records that require immediate investigation.",
    "There is a serious violation of anti-corruption policy and the case was sent to external counsel.",
    "The contract has no confidentiality protection and trade secrets were disclosed to a competitor.",
]

POOLS = {"Low": LOW, "Medium": MEDIUM, "High": HIGH}


def make_doc(label: str) -> str:
    own = POOLS[label]
    others = [s for k, v in POOLS.items() if k != label for s in v]
    sentences = random.sample(own, random.randint(2, 4))
    sentences += random.sample(others, random.choice([1, 2, 2, 3]))  # label noise
    sentences += random.sample(NEUTRAL, random.randint(2, 3))
    random.shuffle(sentences)
    title = f"{random.choice(DOC_TYPES)} - Ref {random.randint(1000, 9999)}"
    return title + ". " + " ".join(sentences)


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    rows = [{"text": make_doc(label), "label": label} for label in POOLS for _ in range(100)]
    random.shuffle(rows)
    pd.DataFrame(rows).to_csv("data/labels.csv", index=False)
    for label in POOLS:
        with open(f"samples/sample_{label.lower()}_risk.txt", "w", encoding="utf-8") as f:
            f.write(make_doc(label))
    print(f"Wrote {len(rows)} documents to data/labels.csv and 3 samples to samples/")


if __name__ == "__main__":
    main()

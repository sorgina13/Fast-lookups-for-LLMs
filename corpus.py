"""Synthetic clause corpus for the scaling test.

Cardinality stays fixed at the 8 invoices in data/invoices.json. Only the number
of clauses grows. Every clause is unique: duplicates would let the cache do the
work and hide the real per-item cost, so this measures the worst case for the
encoder.

Clauses are generated round-robin across labels to keep classes balanced, so the
usable corpus size is bounded by the *least* productive label, not the total.
Every template therefore carries at least two placeholders.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

VENDORS = [
    "the Supplier", "the Vendor", "the Contractor", "the Provider",
    "the Counterparty", "the Service Partner", "the Appointed Firm",
    "the Engaged Party", "the Subcontractor", "the Licensor",
    "the Nominated Supplier", "the Retained Firm",
]

CADENCE = [
    "monthly in arrears", "quarterly in advance", "on a rolling 30-day basis",
    "annually on the anniversary date", "per calendar month",
    "within 14 days of invoice", "on completion of each milestone",
    "at the end of each billing period", "upon acceptance of the deliverable",
    "in equal instalments across the term",
]

REGIONS = [
    "the London office", "the Manchester site", "the Dublin campus",
    "all UK premises", "the regional hub", "the northern depot",
    "the central facility", "each listed location", "the Bristol annexe",
    "the Leeds operations centre",
]

# One template family per invoice. Cardinality is fixed; only volume grows.
# Each template uses {vendor} plus {cadence} or {region} so every label can
# produce enough distinct clauses for a balanced corpus.
TEMPLATES: dict[str, list[str]] = {
    "INV-1001": [
        "{vendor} shall provide dedicated compute capacity and platform hosting at {region}, billed {cadence}.",
        "{vendor} will make available virtual machine capacity and managed hosting for {region}, invoiced {cadence}.",
        "{vendor} shall host the production environment, with compute and storage allocation charged {cadence}.",
        "{vendor} shall operate and maintain the cloud platform infrastructure supporting {region}, billed {cadence}.",
        "{vendor} shall reserve platform capacity for {region}, with compute charges accruing {cadence}.",
    ],
    "INV-1002": [
        "{vendor} shall review and redline all master services agreements for {region}, billed {cadence}.",
        "{vendor} shall provide external legal advice on contract interpretation and disputes, invoiced {cadence}.",
        "{vendor} shall draft and negotiate commercial agreements, with legal fees payable {cadence}.",
        "{vendor} will advise on regulatory compliance for {region} and prepare legal opinions {cadence}.",
        "{vendor} shall act as outside counsel at contract negotiations, issuing written guidance {cadence}.",
    ],
    "INV-1003": [
        "{vendor} will perform nightly janitorial services across {region}, billed {cadence}.",
        "{vendor} shall carry out cleaning, waste removal and washroom servicing at {region} {cadence}.",
        "{vendor} shall maintain the fabric of the building and routine upkeep at {region}, invoiced {cadence}.",
        "{vendor} shall provide facilities maintenance and minor repairs at {region}, charged {cadence}.",
        "{vendor} shall deliver daily office cleaning and periodic deep cleans at {region}, billed {cadence}.",
    ],
    "INV-1004": [
        "{vendor} shall conduct a penetration test of {region} and deliver a written security audit report {cadence}.",
        "{vendor} shall perform a vulnerability assessment of all external-facing systems {cadence}.",
        "{vendor} will carry out security testing of the application estate at {region}, invoiced {cadence}.",
        "{vendor} shall provide independent security auditing and red-team exercises, chargeable {cadence}.",
        "{vendor} shall assess the security posture of {region} and issue remediation advice {cadence}.",
    ],
    "INV-1005": [
        "{vendor} shall be paid agency fees upon successful placement of each contract engineer, settled {cadence}.",
        "{vendor} shall source and present qualified candidates for open technical roles at {region}, billed {cadence}.",
        "{vendor} shall receive recruitment commission for permanent placements, falling due {cadence}.",
        "{vendor} will supply interim contractors to {region} under agreed day rates, invoiced {cadence}.",
        "{vendor} shall invoice placement fees for each engaged specialist at {region} {cadence}.",
    ],
    "INV-1006": [
        "{vendor} shall transport palletised goods to {region} and provide storage, billed {cadence}.",
        "{vendor} will handle inbound freight and warehouse the consignment at {region}, invoiced {cadence}.",
        "{vendor} shall apply shipping, handling and distribution charges for {region} {cadence}.",
        "{vendor} shall operate the distribution route serving {region}, with charges raised {cadence}.",
        "{vendor} shall carry out goods-in processing and pallet storage at {region}, billed {cadence}.",
    ],
    "INV-1007": [
        "{vendor} shall deliver accredited certification workshops for staff at {region}, billed {cadence}.",
        "{vendor} shall provide training courses and examination entry for {region}, payable {cadence}.",
        "{vendor} will run instructor-led upskilling sessions for nominated employees, invoiced {cadence}.",
        "{vendor} shall schedule professional development workshops at {region} {cadence}.",
        "{vendor} shall certify personnel at {region} against the competency framework, billed {cadence}.",
    ],
    "INV-1008": [
        "{vendor} shall maintain a 24/7 incident response retainer for {region} with a one-hour SLA, billed {cadence}.",
        "{vendor} shall provide out-of-hours support cover and escalation handling, charged {cadence}.",
        "{vendor} will provide continuous monitoring and major-incident management for {region}, invoiced {cadence}.",
        "{vendor} shall guarantee priority support response within agreed service levels, billed {cadence}.",
        "{vendor} shall staff the on-call rota for {region} and respond to severity-one incidents {cadence}.",
    ],
}


@dataclass(frozen=True)
class Clause:
    clause_id: str
    text: str
    truth: str


# Plausible corporate spend categories used to pad the candidate pool when
# testing cardinality. Clauses are never generated from these -- they exist only
# as distractors, so ground truth stays well defined as the catalogue grows.
DISTRACTOR_SERVICES = [
    "software licensing and subscription renewals",
    "helpdesk and end-user IT support",
    "data centre colocation and rack space",
    "network connectivity and leased lines",
    "mobile telephony and device plans",
    "printer fleet management and consumables",
    "payroll processing and administration",
    "pension scheme administration",
    "occupational health and wellbeing services",
    "employee assistance programme",
    "catering and staff canteen operation",
    "vending and refreshment supplies",
    "corporate travel booking and management",
    "fleet leasing and vehicle maintenance",
    "fuel cards and mileage administration",
    "commercial property rent and service charge",
    "business rates and property taxation advice",
    "utilities supply and energy management",
    "waste management and recycling",
    "pest control and hygiene services",
    "landscaping and grounds maintenance",
    "lift and escalator servicing",
    "HVAC maintenance and plant servicing",
    "fire safety inspection and equipment",
    "manned guarding and site security",
    "CCTV monitoring and alarm response",
    "access control and badge issuance",
    "document storage and secure shredding",
    "print and reprographics services",
    "postal and courier services",
    "marketing agency retainer and campaigns",
    "media buying and advertising placement",
    "brand design and creative services",
    "market research and consumer insight",
    "public relations and communications",
    "event management and conference hosting",
    "exhibition stand design and build",
    "corporate photography and video production",
    "translation and localisation services",
    "management consultancy and strategy advice",
    "financial audit and statutory accounts",
    "tax compliance and advisory services",
    "internal audit co-sourcing",
    "actuarial and pensions consulting",
    "insurance brokerage and premiums",
    "credit reference and background checks",
    "debt collection and receivables management",
    "banking charges and treasury services",
    "foreign exchange and hedging services",
    "merchant acquiring and card processing",
    "software development outsourcing",
    "quality assurance and test automation",
    "data migration and integration services",
    "business intelligence and reporting",
    "database administration and tuning",
    "disaster recovery and backup services",
    "identity and access management tooling",
    "endpoint protection and antivirus",
    "email filtering and anti-spam",
    "certificate authority and PKI services",
    "API gateway and integration platform",
    "content delivery and edge caching",
    "domain registration and DNS management",
    "web hosting for marketing microsites",
    "mobile application distribution",
    "customer survey and feedback platform",
    "call centre overflow handling",
    "interpreting and accessibility support",
    "archival scanning and digitisation",
    "laboratory testing and sample analysis",
    "calibration of measuring instruments",
    "specialist plant hire",
    "scaffolding and access equipment",
    "industrial cleaning of production areas",
    "uniform supply and laundry services",
    "personal protective equipment supply",
    "first aid supplies and defibrillators",
    "signage design and installation",
    "furniture supply and office fit-out",
    "relocation and office move services",
]

QUALIFIERS = [
    "Holdings", "Group", "Partners", "Associates", "Services Ltd",
    "International", "Solutions", "Consulting", "Industries", "Networks",
]

DISTRACTOR_VENDORS = [
    "Aldridge", "Brackwell", "Calderon", "Dunmore", "Ellerby", "Fenwick",
    "Garrowe", "Haldane", "Ingleby", "Jarrow", "Kelsenor", "Lindmark",
    "Marloch", "Netherby", "Orsenna", "Pelbridge", "Quenby", "Rathmore",
    "Stanmere", "Thorndale", "Ulverston", "Veltham", "Westcote", "Yarrowby",
]


def build_catalogue(size: int, seed: int = 3) -> dict[str, str]:
    """A catalogue of `size` invoices: the 8 real ones plus distractors.

    The real 8 always come first, so clause ground truth is unchanged. Extra
    entries only widen the candidate pool, which is what cardinality means for a
    lookup: more things it could be, same correct answer.
    """
    real = {
        "INV-1001": "Acme Cloud Hosting - monthly platform hosting and compute capacity",
        "INV-1002": "Northwind Legal LLP - external legal counsel and contract review",
        "INV-1003": "Globex Facilities - office cleaning and on-site maintenance",
        "INV-1004": "Initech Security - penetration testing and security audit services",
        "INV-1005": "Umbrella Staffing - contractor placement and recruitment fees",
        "INV-1006": "Stark Logistics - freight shipping and warehouse storage",
        "INV-1007": "Wayne Training - staff certification and training workshops",
        "INV-1008": "Cyberdyne Support - 24/7 incident response retainer and SLA cover",
    }
    if size < len(real):
        raise ValueError(f"Catalogue needs at least {len(real)} entries")

    rng = random.Random(seed)
    pairs = [
        (vendor, qualifier, service)
        for service in DISTRACTOR_SERVICES
        for vendor in DISTRACTOR_VENDORS
        for qualifier in QUALIFIERS
    ]
    rng.shuffle(pairs)

    catalogue = dict(real)
    used_services: set[str] = set()
    index = 0

    for vendor, qualifier, service in pairs:
        if len(catalogue) >= size:
            break
        # One invoice per service keeps distractors distinct from each other.
        if service in used_services:
            continue
        used_services.add(service)
        index += 1
        catalogue[f"DST-{index:04d}"] = f"{vendor} {qualifier} - {service}"

    # Second pass: allow repeated services under different vendors. Realistic
    # (several suppliers for one service) and only needed for large catalogues.
    for vendor, qualifier, service in pairs:
        if len(catalogue) >= size:
            break
        description = f"{vendor} {qualifier} - {service}"
        if description in catalogue.values():
            continue
        index += 1
        catalogue[f"DST-{index:04d}"] = description

    if len(catalogue) < size:
        raise ValueError(
            f"Only produced {len(catalogue)} invoices; add more distractor services"
        )
    return catalogue


def max_catalogue() -> int:
    return 8 + len(DISTRACTOR_SERVICES) * len(DISTRACTOR_VENDORS) * len(QUALIFIERS)


def _combinations(template: str) -> int:
    total = 1
    if "{vendor}" in template:
        total *= len(VENDORS)
    if "{cadence}" in template:
        total *= len(CADENCE)
    if "{region}" in template:
        total *= len(REGIONS)
    return total


def label_capacity() -> dict[str, int]:
    return {
        label: sum(_combinations(template) for template in templates)
        for label, templates in TEMPLATES.items()
    }


def max_unique() -> int:
    """Largest balanced corpus the templates support.

    Clauses are emitted round-robin, so the binding constraint is the least
    productive label -- not the sum across labels.
    """
    return min(label_capacity().values()) * len(TEMPLATES)


def generate(count: int, seed: int = 7) -> list[Clause]:
    """Build `count` unique clauses spread evenly over the fixed 8 invoices."""
    if count > max_unique():
        raise ValueError(
            f"Requested {count} clauses but the templates support {max_unique()}"
        )

    rng = random.Random(seed)
    labels = list(TEMPLATES)
    seen: set[str] = set()
    clauses: list[Clause] = []

    attempts = 0
    max_attempts = count * 500

    while len(clauses) < count and attempts < max_attempts:
        attempts += 1
        label = labels[len(clauses) % len(labels)]
        text = rng.choice(TEMPLATES[label]).format(
            vendor=rng.choice(VENDORS),
            cadence=rng.choice(CADENCE),
            region=rng.choice(REGIONS),
        )
        if text in seen:
            continue
        seen.add(text)
        clauses.append(Clause(f"CL-{len(clauses) + 1:05d}", text, label))

    if len(clauses) < count:
        raise ValueError(
            f"Only produced {len(clauses)} unique clauses in {attempts} attempts"
        )
    return clauses


if __name__ == "__main__":
    print(f"max balanced corpus: {max_unique()}")
    print("per-label capacity:", label_capacity())
    for clause in generate(8):
        print(f"{clause.clause_id} [{clause.truth}] {clause.text}")

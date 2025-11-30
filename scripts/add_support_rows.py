from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("data/support_kb.csv")
    df = pd.read_csv(path)
    templates = [
        {
            "question": "How do I enable {feature} for {customer} on the {plan} plan?",
            "answer": "Admins at {customer} can go to Settings > Features, toggle {feature} for the {plan} plan, and the change rolls out to all regions within 10 minutes.",
            "tags": "product,feature",
            "priority": "medium",
        },
        {
            "question": "Can I schedule {report} emails for {customer}?",
            "answer": "Yes, under Analytics > Reports choose {report}, click Schedule, and add the {customer} stakeholder list. Emails follow the {plan} plan cadence.",
            "tags": "analytics,reports",
            "priority": "low",
        },
        {
            "question": "How do I connect {integration} for {customer}?",
            "answer": "Open Integrations, select {integration}, authenticate with the {customer} admin account, and map the default objects. Initial sync runs immediately.",
            "tags": "integrations,api",
            "priority": "high",
        },
        {
            "question": "What happens if {customer} exceeds the {threshold} storage threshold?",
            "answer": "We notify the {customer} admins when usage crosses {threshold}. They can archive data or purchase add-on storage from Billing > Storage.",
            "tags": "billing,storage",
            "priority": "medium",
        },
        {
            "question": "How do I enable {channel} notifications for {customer}?",
            "answer": "In Settings > Notifications turn on {channel} for each queue serving {customer}. Remind agents to verify their {channel} identity.",
            "tags": "notifications,channel",
            "priority": "low",
        },
        {
            "question": "How can I audit {workflow} activity for {customer}?",
            "answer": "Visit Compliance > Audit Logs, filter by {workflow}, and export the CSV to review every step completed for {customer}.",
            "tags": "compliance,audit",
            "priority": "high",
        },
        {
            "question": "How do I update the {product} branding for {customer}?",
            "answer": "Navigate to Settings > Branding, upload the {customer} logo, adjust colors, and publish to instantly refresh the {product}.",
            "tags": "branding,product",
            "priority": "medium",
        },
        {
            "question": "Can {customer} allow guest access during {workflow}?",
            "answer": "Yes, enable Guest Links under Settings > Access, specify the {workflow} workspace, and set an expiry for {customer}.",
            "tags": "access,security",
            "priority": "medium",
        },
        {
            "question": "How do I clone automation rules for {customer}?",
            "answer": "Go to Automations, open the rule, click Clone, select the {customer} workspace, and review the triggers before saving.",
            "tags": "automation,tasks",
            "priority": "low",
        },
        {
            "question": "What logs show {channel} delivery for {customer}?",
            "answer": "Use Messaging > Delivery Logs to filter by {channel} and the {customer} account to validate each event.",
            "tags": "messaging,logs",
            "priority": "medium",
        },
        {
            "question": "How do I reset API credentials for {customer}?",
            "answer": "Administrators can regenerate keys under Developers > API Keys. Inform {customer} to rotate credentials in downstream systems.",
            "tags": "api,security",
            "priority": "high",
        },
        {
            "question": "How do I route {channel} conversations to Tier 2 for {customer}?",
            "answer": "Create a routing rule in Support > Queues targeting {channel} and assign Tier 2 as the destination for {customer}.",
            "tags": "support,routing",
            "priority": "medium",
        },
        {
            "question": "Can I bulk import assets for {customer}?",
            "answer": "Yes, go to Assets > Import, download the template, fill it with the {customer} data, and upload. Validation errors will list row numbers.",
            "tags": "assets,import",
            "priority": "medium",
        },
        {
            "question": "How do I enforce IP restrictions for {customer}?",
            "answer": "Turn on IP Allow Lists under Security, enter the {customer} corporate ranges, and require re-login so the policy applies.",
            "tags": "security,network",
            "priority": "high",
        },
        {
            "question": "How do I create custom fields for {workflow} at {customer}?",
            "answer": "Open Settings > Custom Fields, add the field type, and assign it to the {workflow} forms within the {customer} workspace.",
            "tags": "customization,workflow",
            "priority": "low",
        },
        {
            "question": "How do I download historical {report} for {customer}?",
            "answer": "In Analytics > Reports choose {report}, set the date window, and export. The file includes every metric for {customer}.",
            "tags": "analytics,export",
            "priority": "low",
        },
        {
            "question": "How do I pause billing for {customer} during {workflow}?",
            "answer": "From Billing > Plans click Pause, pick {workflow} as the reason, and the system will freeze charges for {customer} until reactivated.",
            "tags": "billing,plan",
            "priority": "medium",
        },
        {
            "question": "How do I sync {integration} contacts for {customer}?",
            "answer": "Enable contact sync inside the {integration} settings, map the owner field to {customer}, and run a manual sync to verify records.",
            "tags": "integrations,contacts",
            "priority": "medium",
        },
        {
            "question": "How do I monitor uptime alerts for {customer}?",
            "answer": "Subscribe {customer} admins to status.example.com and enable webhook alerts under Monitoring > Alerts.",
            "tags": "status,alerts",
            "priority": "high",
        },
        {
            "question": "What API endpoint retrieves {feature} metrics for {customer}?",
            "answer": "Use GET /v1/metrics/{feature} with the {customer} workspace ID and include the API key in headers.",
            "tags": "api,metrics",
            "priority": "medium",
        },
    ]

    customers = [
        "Acme Retail",
        "Beta Logistics",
        "Cobalt Finance",
        "Delta Foods",
        "Echo Airlines",
        "Falcon Manufacturing",
        "Gemini Health",
        "Helix Labs",
        "Indigo Travel",
        "Jupiter Media",
        "Krypton Telecom",
        "Lumen Energy",
        "Metro Realty",
        "Nova Sports",
        "Orion Mobility",
        "Pulse Insurance",
        "Quantum Bank",
        "Radial Retail",
        "Solar Freight",
        "Titan Auto",
        "Unity Schools",
        "Vertex Pharma",
        "Willow Apparel",
        "Xenon Mining",
        "Yield Ventures",
        "Zephyr Hotels",
        "Apex Education",
        "Bright Telecom",
        "Canyon Estates",
        "Drift Mobility",
        "Everest Outdoors",
        "Forge Hardware",
        "Glide Mobility",
        "Harbor Logistics",
        "Ion Analytics",
        "Jigsaw Studios",
        "Keystone Legal",
        "Lyric Entertainment",
        "Mosaic Systems",
        "Northstar Staffing",
        "Opal Interior",
        "Prairie Foods",
        "Quasar Labs",
        "River Tech",
        "Summit Wellness",
        "Torrent Water",
        "Urban Transit",
        "Violet Beauty",
        "Westwind Travel",
        "Yonder Travel",
        "Zenith Metals",
        "Atlas Ventures",
        "Bluebird Finance",
        "Crimson Solar",
        "Discovery Health",
        "Ember Gaming",
        "Frontier Mobility",
        "Granite Housing",
        "Horizon Retail",
        "Inspire Fitness",
        "Jetstream Cargo",
        "Kindred Care",
        "Lattice Analytics",
        "Meridian Travel",
        "Nimbus Cloud",
        "Oakridge Health",
        "Paragon Apparel",
        "Quest Aviation",
        "Rockwell Media",
        "Sierra Construction",
        "Trident Logistics",
        "Uplink Studios",
        "Vanguard Realty",
        "Wildflower Spa",
        "Yukon Timber",
        "Zodiac Lighting",
    ]

    regions = ["APAC", "EMEA", "North America", "LATAM", "India", "Middle East"]
    plans = ["Starter", "Growth", "Professional", "Enterprise"]
    features = [
        "inventory sync",
        "smart routing",
        "AI summaries",
        "advanced analytics",
        "custom SLA",
        "auto tagging",
        "webhooks",
        "task automation",
        "voice intelligence",
        "knowledge insights",
    ]
    channels = ["email", "SMS", "WhatsApp", "Slack", "Teams", "Push"]
    reports = [
        "NPS trend",
        "SLA breach",
        "agent productivity",
        "billing variance",
        "churn risk",
        "usage adoption",
    ]
    integrations = [
        "Salesforce",
        "HubSpot",
        "Zendesk",
        "Freshdesk",
        "Shopify",
        "Stripe",
        "Notion",
        "Google Workspace",
        "Jira",
        "QuickBooks",
    ]
    workflows = [
        "onboarding",
        "renewal",
        "incident response",
        "escalation",
        "billing dispute",
        "migration",
        "launch",
        "audit",
    ]
    thresholds = ["80%", "85%", "90%", "95%", "98%", "99%"]
    products = [
        "customer portal",
        "mobile app",
        "admin console",
        "public API",
        "agent workspace",
        "partner hub",
    ]

    count_needed = 1000
    rows: list[dict[str, str]] = []
    for idx in range(count_needed):
        template = templates[idx % len(templates)]
        data = {
            "customer": customers[idx % len(customers)],
            "region": regions[idx % len(regions)],
            "plan": plans[idx % len(plans)],
            "feature": features[idx % len(features)],
            "channel": channels[idx % len(channels)],
            "report": reports[idx % len(reports)],
            "integration": integrations[idx % len(integrations)],
            "workflow": workflows[idx % len(workflows)],
            "threshold": thresholds[idx % len(thresholds)],
            "product": products[idx % len(products)],
        }
        question = template["question"].format(**data)
        answer = template["answer"].format(**data)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "tags": template["tags"],
                "priority": template["priority"],
            }
        )

    new_df = pd.DataFrame(rows)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"Added {len(new_df)} rows. New total: {len(df)}")


if __name__ == "__main__":
    main()



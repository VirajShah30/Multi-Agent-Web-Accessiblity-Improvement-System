import json

class AxeViolationsAgent:
    """
    Extracts and summarizes the axe violations from a raw JSON UI dump,
    providing id, impact, categories (cat.* tags), description, help text,
    and aggregated failureSummary per violation.
    """

    def __init__(self):
        pass

    def preprocess(self, raw_json: str) -> str:
        """
        1) Parse the JSON.
        2) Navigate to viewports[0].axe.violations.
        3) For each violation, extract:
           - id, impact
           - tags starting with "cat."
           - description, help
           - combined failureSummary of all nodes
        4) Return a compact multi‑line summary.
        """
        try:
            doc = json.loads(raw_json)
            vps = doc.get("viewports", [])
            if not vps:
                return "No viewports in JSON."
            violations = vps[0].get("axe", {}).get("violations", [])
            if not violations:
                return "No axe violations found."
        except Exception as e:
            return f"Error parsing JSON: {e}"

        lines = []
        for viol in violations:
            vid        = viol.get("id", "<no-id>")
            impact     = viol.get("impact", "unknown")
            tags       = viol.get("tags", [])
            cats       = [t for t in tags if t.startswith("cat.")]
            desc       = viol.get("description", "").strip()
            help_text  = viol.get("help", "").strip()

            # aggregate all node failureSummary fields
            failures = []
            for node in viol.get("nodes", []):
                fs = node.get("failureSummary", "").replace("\n  ", "; ").strip()
                if fs:
                    failures.append(fs)
            fs_combined = " | ".join(failures) if failures else "<no failureSummary>"

            lines.append(
                f"- {vid} (impact: {impact})\n"
                f"    Categories: {', '.join(cats) or 'none'}\n"
                f"    Description: {desc}\n"
                f"    Help: {help_text}\n"
                f"    FailureSummary: {fs_combined}"
            )

        return "\n".join(lines)

    def handle(self, raw_json: str) -> str:
        """
        Main entrypoint: takes the raw UI JSON string and returns the
        preprocessed axe‑violations summary.
        """
        return self.preprocess(raw_json)
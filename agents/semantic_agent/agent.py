# agents/semantic_agent/semantic_agent.py
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class SemanticAgent:
    def __init__(self, model_dir: str, device: str = None):
        """
        model_dir should point at the folder containing:
          - config.json, pytorch_model.bin
          - tokenizer files (vocab etc)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained("trusha88/t5-semantic-agent")
        self.model     = T5ForConditionalGeneration.from_pretrained("trusha88/t5-semantic-agent")
        self.model.to(self.device)

    def preprocess(self, raw_json: str) -> str:
        """
        1) Parse the full page JSON (one page may have multiple viewports).
        2) For each viewport: extract semantic context + filter only cat.semantics violations.
        3) Build the SAME `source_text` you used during training:
 
           Page: {page_id} | Viewport: {viewport}
           Semantic Context: { ...json of headings/images/links... }
           Violations:        { ...json of filtered violations... }

        If there are multiple viewports with violations, this example simply takes the first.
        You can easily extend it to loop over all and concatenate.
        """
        doc = json.loads(raw_json)
        page_id = doc.get("page_id")

        records = []
        for vp in doc.get("viewports", []):
            viewport = vp.get("viewport")
            sem      = vp.get("semantic", {})
            headings = sem.get("headings", [])
            images   = sem.get("images", [])
            links    = sem.get("links", [])

            # filter only the semantic-category violations
            sem_viol = [
                viol for viol in vp.get("axe", {}).get("violations", [])
                if any(tag.startswith("cat.semantics") for tag in viol.get("tags", []))
            ]
            if not sem_viol:
                continue

            records.append({
                "page_id": page_id,
                "viewport": viewport,
                "semantic": {
                    "headings": headings,
                    "images":    images,
                    "links":     links
                },
                "violations": sem_viol
            })

        if not records:
            raise ValueError(f"No semantic violations found in page {page_id}")

        # take the first viewport with violations
        rec = records[0]

        # build the prompt identical to your training's make_source()
        prompt = (
            f"Page: {rec['page_id']} | Viewport: {rec['viewport']}\n"
            f"Semantic Context: {json.dumps(rec['semantic'], ensure_ascii=False)}\n"
            f"Violations: {json.dumps(rec['violations'], ensure_ascii=False)}"
        )
        return prompt

    def generate_summary(
        self,
        prompt: str,
        max_input_len: int  = 512,
        max_output_len: int = 256,
        num_beams: int     = 4
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            max_length=max_input_len,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        out = self.model.generate(
            input_ids      = inputs.input_ids,
            attention_mask = inputs.attention_mask,
            max_length     = max_output_len,
            num_beams      = num_beams,
            early_stopping = True
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def handle(self, raw_json: str) -> str:
        """
        Entry point for your manager/dispatcher.
        Takes the raw JSON string of one page, and returns the T5‐generated summary.
        """
        prompt  = self.preprocess(raw_json)
        summary = self.generate_summary(prompt)
        return summary

import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ContrastAgent:
    def __init__(self, model_dir: str = "virajns2/contrast-violation-t5", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, raw_json: str) -> str:
        """
        1. Parse the JSON of one page.
        2. Extract all contrast violations.
        3. For each, build an input string used during training.
        4. This example returns only the first found; extend if needed.
        """
        doc = json.loads(raw_json)
        records = []

        for vp in doc.get("viewports", []):
            for c in vp.get("contrast", []):
                role = c.get("role", "unknown")
                fg = ",".join(map(str, c.get("fg", [0,0,0])))
                bg = ",".join(map(str, c.get("bg", [255,255,255])))
                contrast_val = c.get("contrast", 1.0)
                input_str = f"role: {role}, fg: {fg}, bg: {bg}, contrast: {contrast_val:.2f}"
                records.append(input_str)

        if not records:
            raise ValueError("No contrast violations found in JSON.")

        return records[0]  # Or modify to return all

    def generate_description(
        self,
        prompt: str,
        max_input_len: int = 64,
        max_output_len: int = 64,
        num_beams: int = 1
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            max_length=max_input_len,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_output_len,
                num_beams=num_beams,
                early_stopping=True
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def handle(self, raw_json: str) -> str:
        """
        Main entrypoint. Pass in a JSON string of a page.
        Returns the generated contrast violation description.
        """
        prompt = self.preprocess(raw_json)
        return self.generate_description(prompt)

from gliner import GLiNER

def run_ner():
    print("Loading peer-reviewed GLiNER-BioMed model from Hugging Face...")
    # This downloads and caches the model inside your workspace
    model = GLiNER.from_pretrained("Ihor/gliner-biomed-base-v1.0")

    text = """
    Patient John Doe, a 45-year-old male, was diagnosed with type 2 diabetes 
    and hypertension. He was prescribed Metformin 500mg twice daily.
    """

    # Because GLiNER is open-palette, you can add or change these labels anytime!
    labels = ["Patient name", "Age", "Gender", "Disease", "Drug", "Drug dosage"]

    print("Extracting entities...")
    entities = model.predict_entities(text, labels, threshold=0.4)

    print("\n--- Extraction Results ---")
    for entity in entities:
        print(f"[{entity['label']}] -> {entity['text']}")

if __name__ == "__main__":
    run_ner()
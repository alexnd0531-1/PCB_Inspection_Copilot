# PCB Inspection Copilot

An ECEN 250 project that uses machine learning to triage PCB images for manual review and suggest likely solder or assembly defect types.

## What it does

- Loads a YOLO-format PCB solder-defect dataset.
- Builds an image-level table with split, defect count, and defect types.
- Uses a Random Forest classifier to decide whether a board should be sent for manual inspection.
- Uses one-vs-rest Logistic Regression to suggest likely defect types.
- Provides typed tools for individual images, multi-image batches, and inspection-report drafts.
- Uses TAMU AI to select and call the trained-model tools through a capped agent loop.
- Includes a single-page Gradio portal with multi-image selection, visible tool calls, live inspection statistics, and a model-derived defect chart.
- Applies input, scope, batch-size, and human-review guardrails.

## Models

### Model A: inspection triage

The triage model uses brightness, color, edge, and HOG image features with a Random Forest classifier. It flags an image when its predicted defect probability is at least 0.50.

Latest test result:

- F1 score: 0.881
- Accuracy: 0.900
- Recall for images needing inspection: 1.000
- Training-majority baseline F1: approximately 0.548

### Model B: defect-type classification

The defect classifier uses one-vs-rest Logistic Regression to predict up to nine possible defect types. Its selected global prediction threshold is 0.80. Its test macro F1 is 0.504 across the eight evaluable classes, compared with a per-label training-majority baseline of 0.000. Oxidation cannot be evaluated because the validation and test splits contain no positive oxidation examples.

## Dataset

The project uses the [PCB Solder Defect Detection dataset](https://huggingface.co/datasets/ParsaVictor/pcb-solder-defect-detection). The dataset is not included in this repository because it is approximately 401 MB. The notebook downloads it from Hugging Face when run.

The nine defect classes are:

- insufficient solder
- excess solder
- solder bridge
- solder void
- missing component
- shifted component
- burn
- physical damage
- oxidation

## Run the project

The easiest option is to open `PCB_Inspection.ipynb` in Google Colab and run the cells in order.

The notebook installs its required packages:

```python
%pip install huggingface_hub pandas matplotlib scikit-learn scikit-image joblib gradio openai pillow
```

Add `TAMU_AI_KEY` to Colab Secrets before launching the portal. The TAMU AI connection uses `https://chat-api.tamu.ai/openai` and the default model `protected.gpt-4.1-mini`.

The first run downloads and extracts the dataset. Then run the notebook from top to bottom to train both models, calculate their baselines, register the tools, and launch the Gradio portal. Without an API key, the portal uses a clearly labeled local demonstration mode while keeping all model outputs grounded in the trained Python functions.

## Repository contents

- `PCB_Inspection.ipynb` - dataset preparation, model training, baseline evaluation, agent tools, guardrails, and Gradio portal.
- `README.md` - project overview and setup instructions.

## Limitations

- This is an educational prototype, not a production quality-control system.
- Model predictions should always be verified by a human inspector.
- The defect-type model is less reliable than the triage model, especially for classes with limited evaluation data.
- The portal currently selects images already present in the downloaded dataset; direct user image uploads are not yet implemented.
- The report tool drafts text only. Email delivery remains intentionally unavailable to the LLM and requires a separate human-approved implementation.
- The data and trained model artifacts are intentionally excluded from version control.


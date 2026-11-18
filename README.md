# PCB Inspection Copilot

An ECEN 250 project that uses machine learning to triage PCB images for manual review and suggest likely solder or assembly defect types.

## What it does

- Loads a YOLO-format PCB solder-defect dataset.
- Builds an image-level table with split, defect count, and defect types.
- Uses a Random Forest classifier to decide whether a board should be sent for manual inspection.
- Uses one-vs-rest Logistic Regression to suggest likely defect types.
- Provides helper tools for individual images, batches, and text inspection reports.
- Includes a Gradio interface for inspecting a selected dataset image.

## Models

### Model A: inspection triage

The triage model uses brightness, color, edge, and HOG image features with a Random Forest classifier. It flags an image when its predicted defect probability is at least 0.50.

Latest test result:

- F1 score: 0.881
- Accuracy: 0.900
- Recall for images needing inspection: 1.000

### Model B: defect-type classification

The defect classifier uses one-vs-rest Logistic Regression to predict up to nine possible defect types. Its selected global prediction threshold is 0.80. Oxidation cannot be evaluated because the validation and test splits contain no positive oxidation examples.

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
%pip install huggingface_hub pandas matplotlib scikit-learn scikit-image joblib gradio pillow
```

The first run downloads and extracts the dataset. Then run the notebook from top to bottom to train the models and launch the Gradio inspection interface.

## Repository contents

- `PCB_Inspection.ipynb` - dataset preparation, model training, evaluation, helper tools, and Gradio interface.
- `COMPLETION_CHECKLIST.md` - current project status and handoff notes.
- `README.md` - project overview and setup instructions.

## Limitations

- This is an educational prototype, not a production quality-control system.
- Model predictions should always be verified by a human inspector.
- The defect-type model is less reliable than the triage model, especially for classes with limited evaluation data.
- The data and trained model artifacts are intentionally excluded from version control.

## Next steps

- Re-evaluate Model B on the test set using its selected 0.80 threshold.
- Add the batch-review panel to the Gradio interface.
- Capture final screenshots and prepare the course presentation.

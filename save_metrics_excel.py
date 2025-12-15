import json
import os
from openpyxl import Workbook

def json_to_long_excel(json_path, sheet_name="baseline"):
    """
    Convert metrics.json into a long-format Excel sheet where each row includes:
    experiment info, assessment info, transcript, file_name, metric_name, model, value.
    Output Excel file is saved in the SAME DIRECTORY as json_path.
    """

    # ---------------------------------------------------------
    # Load JSON
    # ---------------------------------------------------------
    with open(json_path, "r") as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Extract experiment_info - all prompts (name and version)
    # ---------------------------------------------------------
    experiment_info = data.get("experiment_info", {})
    
    # Note prompt
    note_prompt_data = experiment_info.get("note_prompt", {})
    note_prompt_name = note_prompt_data.get("name", "")
    note_prompt_version = note_prompt_data.get("version", "")
    
    # Chunking prompt
    chunking_prompt_data = experiment_info.get("chunking_prompt", {})
    chunking_prompt_name = chunking_prompt_data.get("name", "")
    chunking_prompt_version = chunking_prompt_data.get("version", "")
    
    # Reasoning prompt
    reasoning_prompt_data = experiment_info.get("reasoning_prompt", {})
    reasoning_prompt_name = reasoning_prompt_data.get("name", "")
    reasoning_prompt_version = reasoning_prompt_data.get("version", "")
    
    experiment_timestamp = experiment_info.get("timestamp", "")

    assessment = experiment_info.get("assessment_questions", {})
    assessment_name = assessment.get("name", "")
    assessment_version = assessment.get("version", "")
    

    # Get all expected models from experiment_info
    all_models = experiment_info.get("models", [])
    
    transcripts = data["transcripts"]

    # ---------------------------------------------------------
    # Prepare output path (same folder as JSON)
    # ---------------------------------------------------------
    folder = os.path.dirname(os.path.abspath(json_path))
    output_filename = f"{sheet_name}_metrics.xlsx"
    output_path = os.path.join(folder, output_filename)

    # ---------------------------------------------------------
    # Initialize Workbook
    # ---------------------------------------------------------
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name[:31]  # Excel max length = 31 chars

    # ---------------------------------------------------------
    # Column order EXACTLY as required
    # ---------------------------------------------------------
    header = [
        "experiment_name",
        "note_prompt_name",
        "note_prompt_version",
        "chunking_prompt_name",
        "chunking_prompt_version",
        "reasoning_prompt_name",
        "reasoning_prompt_version",
        "experiment_timestamp",
        "assessment_name",
        "assessment_version",
        "transcript",
        "file_name",
        "metric",
        "model",
        "value"
    ]
    ws.append(header)


    # ---------------------------------------------------------
    # Helper to append a metric row
    # ---------------------------------------------------------
    def append_metric_row(transcript_key, file_name, metric_name, model, value):
        ws.append([
            sheet_name,
            note_prompt_name,
            note_prompt_version,
            chunking_prompt_name,
            chunking_prompt_version,
            reasoning_prompt_name,
            reasoning_prompt_version,
            experiment_timestamp,
            assessment_name,
            assessment_version,
            transcript_key,
            file_name,
            metric_name,
            model,
            value
        ])

    # ---------------------------------------------------------
    # Extract metrics from transcripts
    # Ensure all model-metric combinations exist, using "na" for missing values
    # ---------------------------------------------------------
    for transcript_key, tdata in transcripts.items():

        file_name = tdata.get("filename", "")
        if not file_name:
            file_name = tdata.get("file_name", "")

        m = tdata.get("metrics", {})

        # For each model, check all metrics
        for model in all_models:
            
            # -------------------------
            # BERTScore (precision, recall, f1)
            # -------------------------
            bertscore_data = m.get("bertscore", {}).get(model, {})
            append_metric_row(
                transcript_key, file_name, "bertscore_precision", model,
                bertscore_data.get("precision") if bertscore_data.get("precision") is not None else "na"
            )
            append_metric_row(
                transcript_key, file_name, "bertscore_recall", model,
                bertscore_data.get("recall") if bertscore_data.get("recall") is not None else "na"
            )
            append_metric_row(
                transcript_key, file_name, "bertscore_f1", model,
                bertscore_data.get("f1") if bertscore_data.get("f1") is not None else "na"
            )

            # -------------------------
            # ROUGE (precision, recall, fmeasure)
            # -------------------------
            rouge_data = m.get("rouge", {}).get(model, {})
            for rname in ["rouge1", "rouge2", "rougeL"]:
                rv = rouge_data.get(rname, {})
                append_metric_row(
                    transcript_key, file_name, f"{rname}_precision", model,
                    rv.get("precision") if rv.get("precision") is not None else "na"
                )
                append_metric_row(
                    transcript_key, file_name, f"{rname}_recall", model,
                    rv.get("recall") if rv.get("recall") is not None else "na"
                )
                append_metric_row(
                    transcript_key, file_name, f"{rname}_fmeasure", model,
                    rv.get("fmeasure") if rv.get("fmeasure") is not None else "na"
                )

            # -------------------------
            # Summarization (score + reason)
            # -------------------------
            summarization_data = m.get("summarization", {}).get(model, {})
            append_metric_row(
                transcript_key, file_name, "summarization_score", model,
                summarization_data.get("score") if summarization_data.get("score") is not None else "na"
            )
            append_metric_row(
                transcript_key, file_name, "summarization_reason", model,
                summarization_data.get("reason") if summarization_data.get("reason") is not None else "na"
            )

    # ---------------------------------------------------------
    # Save Excel file
    # ---------------------------------------------------------
    wb.save(output_path)
    print(f"Excel saved to: {output_path}")



if __name__ == "__main__":

    # output_folder = "outputs/baseline/20251203_195311" # Baseline
    # output_folder = "outputs/chunking/20251207_130615" # Chunking
    output_folder = "outputs/reasoning/20251209_112109" # Reasoning
    # output_folder = "" # Chunking + Reasoning
    
    json_to_long_excel(
        json_path=f"{output_folder}/metrics.json",
        sheet_name=output_folder.split("/")[-2]
    )

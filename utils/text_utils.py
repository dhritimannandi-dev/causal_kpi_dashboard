def extract_relevant_section(report_text: str, keyword: str) -> str:
    filtered_lines = [
        line for line in report_text.split("\n")
        if keyword.lower() in line.lower()
    ]
    return "\n".join(filtered_lines)


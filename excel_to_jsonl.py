import pandas as pd
import json
import os
from logger import setup_logger

logger = setup_logger(__name__)

def convert_excel_to_jsonl(excel_file_path, output_file_path=None):
    """
    Convert Excel file to JSONL format for training.

    Excel columns: System Prompt | Input | Output

    JSON format:
    {
        "messages": [
            {"role": "system", "content": "System Prompt"},
            {"role": "user", "content": "Input"},
            {"role": "assistant", "content": "Output"}
        ]
    }

    Args:
        excel_file_path (str): Path to the Excel file
        output_file_path (str): Path for the output JSONL file (optional)
    """

    # Read the Excel file
    try:
        df = pd.read_excel(excel_file_path)
        logger.info(f"成功加载Excel文件: {excel_file_path}")
        logger.info(f"数据行数: {len(df)}")
        logger.info(f"列名: {list(df.columns)}")
    except Exception as e:
        logger.error(f"读取Excel文件时出错: {e}")
        return

    # Expected columns
    expected_columns = {
        "system_prompt": ["system prompt", "system_prompt", "system"],
        "input": ["input", "user", "question"],
        "output": ["output", "assistant", "response", "answer"],
    }

    # Find matching columns
    column_mapping = {}
    df_columns_lower = [col.lower() for col in df.columns]

    for role, possible_names in expected_columns.items():
        found = False
        for possible_name in possible_names:
            if possible_name.lower() in df_columns_lower:
                actual_column = df.columns[
                    df_columns_lower.index(possible_name.lower())
                ]
                column_mapping[role] = actual_column
                found = True
                break

        if not found:
            logger.warning(
                f"警告: 无法找到'{role}'对应的列。可能的列名: {possible_names}"
            )
            logger.warning(f"可用的列: {list(df.columns)}")
            return

    logger.info(f"列映射关系: {column_mapping}")

    # Generate output file path if not provided
    if output_file_path is None:
        base_name = os.path.splitext(excel_file_path)[0]
        output_file_path = f"{base_name}_training_data.jsonl"

    logger.info(f"输出文件路径: {output_file_path}")

    # Convert to JSONL format
    jsonl_data = []
    for row_num, (index, row) in enumerate(df.iterrows(), start=1):
        # Skip rows with missing data
        if (
            bool(pd.isna(row[column_mapping["system_prompt"]]))
            or bool(pd.isna(row[column_mapping["input"]]))
            or bool(pd.isna(row[column_mapping["output"]]))
        ):
            logger.warning(f"跳过第{row_num}行: 数据缺失")
            continue

        # Create the conversation format for OpenAI
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": str(row[column_mapping["system_prompt"]]).strip(),
                },
                {"role": "user", "content": str(row[column_mapping["input"]]).strip()},
                {
                    "role": "assistant",
                    "content": str(row[column_mapping["output"]]).strip(),
                },
            ]
        }

        jsonl_data.append(conversation)

    # 写入JSONL文件
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item in jsonl_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"成功处理{len(jsonl_data)}条数据")

    except Exception as e:
        logger.error(f"写入JSONL文件时出错: {e}") 
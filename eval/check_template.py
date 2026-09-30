#!/usr/bin/env python3
"""学生の .cpp / .h が [DO NOT MODIFY] 区間を変更していないか検査する。

使い方:
  python check_template.py student.cpp                        # 416 版テンプレートと比較
  python check_template.py student.cpp --template tiny        # tiny 版テンプレートと比較
  python check_template.py student_utils.h --template utils   # utils.h を検査
  python check_template.py student.cpp student_utils.h        # 複数ファイルを一度に検査
"""

import argparse
import difflib
import os
import re
import sys


MARKER_BEGIN = re.compile(r"//\s*=====\s*\[DO NOT MODIFY\]\s*BEGIN:\s*(\S+)\s*=====")
MARKER_END = re.compile(r"//\s*=====\s*\[DO NOT MODIFY\]\s*END:\s*(\S+)\s*=====")

TEMPLATES = {
    "416": "yolov3_video_study.cpp",
    "tiny": "yolov3_video_study_tiny.cpp",
    "tcp_tiny": "yolov3_video_study_tcp_tiny_opt.cpp",
    "utils": "utils.h",
}


def extract_sections(filepath):
    """ファイルから [DO NOT MODIFY] 区間を抽出する。

    Returns: dict of {section_name: [lines...]} (コメント・空白を正規化済み)
    """
    sections = {}
    current = None
    buf = []

    with open(filepath) as f:
        for line in f:
            m_begin = MARKER_BEGIN.search(line)
            m_end = MARKER_END.search(line)

            if m_begin:
                current = m_begin.group(1)
                buf = []
            elif m_end:
                if current and m_end.group(1) == current:
                    sections[current] = buf
                current = None
                buf = []
            elif current is not None:
                buf.append(line)

    return sections


def normalize_line(line):
    """比較用に行を正規化する（末尾空白を除去、タブをスペースに）。"""
    return line.rstrip().replace("\t", "    ")


def normalize_section(lines):
    """区間の行リストを正規化する。"""
    return [normalize_line(l) for l in lines]


def check_file(student_path, template_path):
    """学生ファイルとテンプレートの [DO NOT MODIFY] 区間を比較する。

    Returns: (n_ok, n_fail, details)
    """
    if not os.path.exists(student_path):
        return 0, 0, [f"ERROR: ファイルが見つかりません: {student_path}"]
    if not os.path.exists(template_path):
        return 0, 0, [f"ERROR: テンプレートが見つかりません: {template_path}"]

    tmpl_sections = extract_sections(template_path)
    student_sections = extract_sections(student_path)

    if not tmpl_sections:
        return 0, 0, [f"WARNING: テンプレートに [DO NOT MODIFY] 区間がありません: {template_path}"]

    n_ok = 0
    n_fail = 0
    details = []

    for name, tmpl_lines in sorted(tmpl_sections.items()):
        if name not in student_sections:
            n_fail += 1
            details.append(f"  {name}: 変更あり（区間が見つかりません — マーカーが削除された可能性）")
            continue

        student_lines = student_sections[name]
        tmpl_norm = normalize_section(tmpl_lines)
        student_norm = normalize_section(student_lines)

        if tmpl_norm == student_norm:
            n_ok += 1
            details.append(f"  {name}: OK")
        else:
            n_fail += 1
            diff = list(difflib.unified_diff(
                tmpl_norm, student_norm,
                fromfile=f"template/{name}", tofile=f"student/{name}",
                lineterm="",
            ))
            details.append(f"  {name}: 変更あり")
            for d in diff:
                details.append(f"    {d}")

    return n_ok, n_fail, details


def guess_template(student_path):
    """ファイル名からテンプレートの種類を推定する。"""
    base = os.path.basename(student_path).lower()
    if "utils" in base and base.endswith(".h"):
        return "utils"
    if "tcp" in base and "tiny" in base:
        return "tcp_tiny"
    if "tiny" in base:
        return "tiny"
    return "416"


def main():
    parser = argparse.ArgumentParser(
        description="学生の .cpp/.h が [DO NOT MODIFY] 区間を変更していないか検査する"
    )
    parser.add_argument("files", nargs="+", help="検査するファイル")
    parser.add_argument(
        "--template", default=None,
        choices=["416", "tiny", "tcp_tiny", "utils"],
        help="テンプレートの種類（省略時はファイル名から推定）",
    )
    parser.add_argument(
        "--template-dir", default=None,
        help="テンプレートのディレクトリ (default: リポジトリの kv260/)",
    )
    args = parser.parse_args()

    template_dir = args.template_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "kv260",
    )

    all_ok = True

    for filepath in args.files:
        tmpl_type = args.template or guess_template(filepath)
        tmpl_file = os.path.join(template_dir, TEMPLATES[tmpl_type])

        print(f"\n検査: {filepath} (テンプレート: {tmpl_type})")
        n_ok, n_fail, details = check_file(filepath, tmpl_file)

        for d in details:
            print(d)

        if n_fail > 0:
            all_ok = False
            print(f"  結果: {n_ok} OK, {n_fail} 変更あり")
        elif n_ok > 0:
            print(f"  結果: 全 {n_ok} 区間 OK")
        else:
            print("  結果: 検査対象の区間なし")

    print()
    if all_ok:
        print("全ファイル OK")
    else:
        print("変更が検出されました。変更禁止区間を元に戻してください。")
        sys.exit(1)


if __name__ == "__main__":
    main()

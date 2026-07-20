#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TARGET_BLOCK = re.compile(
    r'^(?P<indent>[ \t]*)if\s*\(\s*present\(\s*timerept\s*\)\s*\.and\.\s*timerept\s*\)\s*then\s*(?:!.*)?$',
    re.IGNORECASE,
)
TARGET_ONE = re.compile(
    r'^(?P<indent>[ \t]*)if\s*\(\s*present\(\s*timerept\s*\)\s*\.and\.\s*timerept\s*\)\s*(?P<stmt>.+?)\s*$',
    re.IGNORECASE,
)
IF_START = re.compile(r'^[ \t]*if\s*\(.*\)\s*then\s*(?:!.*)?$', re.IGNORECASE)
END_IF = re.compile(r'^[ \t]*(?:end\s*if|endif)\b', re.IGNORECASE)
UNSAFE = re.compile(r'present\(\s*timerept\s*\)\s*\.and\.\s*timerept', re.IGNORECASE)


def find_matching_end(lines: list[str], start: int) -> int:
    depth = 1
    for j in range(start + 1, len(lines)):
        code = lines[j].split('!')[0].rstrip()
        if IF_START.match(code):
            depth += 1
        if END_IF.match(code):
            depth -= 1
            if depth == 0:
                return j
    raise RuntimeError(f'unmatched IF block starting at line {start + 1}')


def patch_lines(lines: list[str]) -> tuple[list[str], int, int]:
    out: list[str] = []
    n_block = 0
    n_one = 0
    i = 0
    while i < len(lines):
        raw = lines[i]
        code = raw.rstrip('\n')
        match = TARGET_BLOCK.match(code)
        if match:
            end = find_matching_end(lines, i)
            body, body_blocks, body_one = patch_lines(lines[i + 1:end])
            indent = match.group('indent')
            out.append(f'{indent}if (present(timerept)) then\n')
            out.append(f'{indent}  if (timerept) then\n')
            out.extend(body)
            out.append(f'{indent}  end if\n')
            out.append(f'{indent}end if\n')
            n_block += 1 + body_blocks
            n_one += body_one
            i = end + 1
            continue

        match = TARGET_ONE.match(code)
        if match and not re.search(r'\bthen\b', code, re.IGNORECASE):
            indent = match.group('indent')
            statement = match.group('stmt').strip()
            out.append(f'{indent}if (present(timerept)) then\n')
            out.append(f'{indent}  if (timerept) {statement}\n')
            out.append(f'{indent}end if\n')
            n_one += 1
            i += 1
            continue

        out.append(raw)
        i += 1

    return out, n_block, n_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source_root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    report = {
        'purpose': 'replace unsafe present(optional) .and. optional logical expressions with nested IF guards',
        'files': {},
        'block_guards': 0,
        'one_line_guards': 0,
    }

    for path in sorted((source_root / 'src').glob('*.F90')):
        lines = path.read_text().splitlines(keepends=True)
        patched, blocks, one_line = patch_lines(lines)
        if blocks or one_line:
            path.write_text(''.join(patched))
            report['files'][str(path.relative_to(source_root))] = {
                'block_guards': blocks,
                'one_line_guards': one_line,
            }
            report['block_guards'] += blocks
            report['one_line_guards'] += one_line

    remaining: list[str] = []
    for path in sorted((source_root / 'src').glob('*.F90')):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if UNSAFE.search(line):
                remaining.append(f'{path.relative_to(source_root)}:{line_number}:{line.strip()}')

    report['remaining_unsafe_guards'] = remaining
    report['passed'] = (
        report['block_guards'] > 0
        and report['one_line_guards'] > 0
        and not remaining
    )
    report['scientific_scope'] = 'Fortran optional-argument safety only; no R26 equation, closure, boundary formula, grid, or timestep is changed.'

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

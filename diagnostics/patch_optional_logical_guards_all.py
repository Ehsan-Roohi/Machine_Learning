#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path

UNSAFE = re.compile(
    r"present\s*\(\s*timerept\s*\)\s*\.and\.\s*timerept",
    re.IGNORECASE,
)
INLINE = re.compile(
    r"(?P<indent>^[ \t]*)if\s*\(\s*present\s*\(\s*timerept\s*\)\s*\.and\.\s*timerept\s*\)\s*"
    r"time_beg\s*=\s*ptime\s*\(\s*\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
BLOCK = re.compile(
    r"(?P<indent>^[ \t]*)if\s*\(\s*present\s*\(\s*timerept\s*\)\s*\.and\.\s*timerept\s*\)\s*then\s*\n"
    r"(?P<body>.*?)(?P<close>^(?P=indent)end\s*if\s*$)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

def patch_one(path: Path) -> dict:
    text=path.read_text()
    n_inline=len(list(INLINE.finditer(text)))
    text=INLINE.sub(
        lambda m:(
            f"{m.group('indent')}if (present(timerept)) then\n"
            f"{m.group('indent')}  if (timerept) time_beg=ptime()\n"
            f"{m.group('indent')}end if"
        ), text)
    n_block=len(list(BLOCK.finditer(text)))
    text=BLOCK.sub(
        lambda m:(
            f"{m.group('indent')}if (present(timerept)) then\n"
            f"{m.group('indent')}  if (timerept) then\n"
            f"{m.group('body')}"
            f"{m.group('indent')}  end if\n"
            f"{m.group('indent')}end if"
        ), text)
    remaining=len(UNSAFE.findall(text))
    if remaining:
        raise RuntimeError(f'{path}: {remaining} unsafe optional-logical expressions remain')
    if n_inline or n_block:
        path.write_text(text)
    return {'inline':n_inline,'block':n_block,'remaining':remaining}

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('source_root',type=Path)
    ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args()
    src=a.source_root/'src'
    by_file={}
    total_inline=total_block=0
    for p in sorted(src.rglob('*.F90')):
        c=patch_one(p)
        if c['inline'] or c['block']:
            by_file[p.name]=c
            total_inline+=c['inline']; total_block+=c['block']
    leftovers=[]
    for p in sorted(src.rglob('*.F90')):
        if UNSAFE.search(p.read_text()): leftovers.append(str(p))
    report={
        'purpose':'infrastructure-only Fortran optional-logical safety patch',
        'physics_changes':False,
        'files':by_file,
        'total_inline':total_inline,
        'total_block':total_block,
        'remaining':len(leftovers),
        'leftovers':leftovers,
        'expected_source_fingerprint':{'total_inline':40,'total_block':44},
    }
    report['all_passed']=(total_inline==40 and total_block==44 and not leftovers)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['all_passed']:
        raise SystemExit(2)

if __name__=='__main__':
    main()

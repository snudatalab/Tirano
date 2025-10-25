#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔹 entity2id.txt, relation2id.txt, train.txt → train_readable.txt

▪ entity2id.txt      “<entity_name><TAB><id>”
▪ relation2id.txt    “<relation_name><TAB><id>”
▪ train.txt          “<head_id><SPACE><relation_id><SPACE><tail_id><SPACE><timestamp>”

결과:   <head_name> <TAB> <relation_name> <TAB> <tail_name> <TAB> <timestamp>
"""

from pathlib import Path

def load_mapping(path: Path) -> dict[int, str]:
    """파일을 읽어 {id:int → name:str} 딕셔너리로 반환"""
    mapping: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue                      # 빈 줄 무시
            name, id_str = line.rstrip("\n").split("\t")
            mapping[int(id_str)] = name
    return mapping


# (1) 매핑 테이블 읽기 ──────────────────────────────────────────────────────────────
entity_map   = load_mapping(Path("entity2id.txt"))
relation_map = load_mapping(Path("relation2id.txt"))


# (2) train.txt 변환하여 저장 ───────────────────────────────────────────────────────
with Path("test.txt").open(encoding="utf-8") as fin, \
     Path("test_readable.txt").open("w", encoding="utf-8") as fout:

    for line in fin:
        if not line.strip():
            continue      # 빈 줄 건너뛰기

        head_id, rel_id, tail_id, ts = map(int, line.split())

        head = entity_map.get(head_id,  f"<UNK‑E:{head_id}>")
        rel  = relation_map.get(rel_id, f"<UNK‑R:{rel_id}>")
        tail = entity_map.get(tail_id,  f"<UNK‑E:{tail_id}>")

        fout.write(f"{head}\t{rel}\t{tail}\t{ts}\n")

print("✅  test_readable.txt  생성 완료")

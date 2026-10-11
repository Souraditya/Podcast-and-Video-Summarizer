from podsum.evaluate.segmentation import boundary_prf
from podsum.schema import Chapter


def ch(start):
    return Chapter(start=start, end=start + 10)


def test_perfect_match():
    gt = [ch(0), ch(60), ch(180)]
    pred = [ch(0), ch(61), ch(179)]
    result = boundary_prf(pred, gt, tolerance_s=5)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_missed_and_extra_boundaries():
    gt = [ch(0), ch(60), ch(180)]
    pred = [ch(0), ch(300)]  # misses 60 and 180, adds a spurious one at 300
    result = boundary_prf(pred, gt, tolerance_s=5)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 2
    assert 0 < result["precision"] < 1
    assert 0 < result["recall"] < 1


def test_each_ground_truth_matched_at_most_once():
    gt = [ch(100)]
    pred = [ch(98), ch(102)]  # both close to the same single ground-truth boundary
    result = boundary_prf(pred, gt, tolerance_s=5)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_load_chapters_accepts_result_json_wrapper(tmp_path):
    import json

    from podsum.schema import load_chapters

    chapters = [{"start": 0.0, "end": 10.0, "title": "a", "summary": "", "first_sentence": 0, "last_sentence": 1}]
    bare, wrapped = tmp_path / "bare.json", tmp_path / "result.json"
    bare.write_text(json.dumps(chapters))
    wrapped.write_text(json.dumps({"source": "x", "tldr": "t", "chapters": chapters}))
    assert load_chapters(bare) == load_chapters(wrapped)


def test_compare_table(tmp_path):
    import json

    from podsum.evaluate.compare import compare, format_table

    gt = [{"start": 0.0, "end": 60.0, "title": "intro"}, {"start": 60.0, "end": 120.0, "title": "main"}]
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "a.json").write_text(json.dumps(gt))
    rows = compare(str(tmp_path / "gt.json"), {"perfect": str(tmp_path / "a.json")})
    assert rows[0]["F1@15"] == 1.0
    assert "perfect" in format_table(rows)

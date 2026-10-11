import numpy as np

from podsum.segment.semantic import find_local_minima, similarity_curve


def _block_embeddings(block_sizes, dim=8, seed=0):
    """Build synthetic embeddings: each block is tightly clustered around its
    own random unit vector, so similarity should visibly dip between blocks."""
    rng = np.random.default_rng(seed)
    vecs = []
    for size in block_sizes:
        center = rng.normal(size=dim)
        center /= np.linalg.norm(center)
        for _ in range(size):
            noisy = center + rng.normal(scale=0.03, size=dim)
            vecs.append(noisy / np.linalg.norm(noisy))
    return np.array(vecs)


def test_similarity_curve_dips_between_topic_blocks():
    embeddings = _block_embeddings([10, 10])
    sims = similarity_curve(embeddings, window=3)
    boundary_region = sims[8:12]
    interior = np.concatenate([sims[2:6], sims[14:18]])
    assert boundary_region.min() < interior.min()


def test_find_local_minima_detects_three_blocks():
    embeddings = _block_embeddings([12, 12, 12])
    sims = similarity_curve(embeddings, window=3)
    boundaries = find_local_minima(sims, depth_threshold=0.05, min_gap=5)
    assert len(boundaries) == 2
    # boundaries should land near the true block edges (index 12 and 24)
    assert any(abs(b - 12) <= 3 for b in boundaries)
    assert any(abs(b - 24) <= 3 for b in boundaries)


def test_find_local_minima_respects_min_gap():
    embeddings = _block_embeddings([12, 12, 12])
    sims = similarity_curve(embeddings, window=3)
    boundaries = find_local_minima(sims, depth_threshold=0.05, min_gap=100)
    assert len(boundaries) <= 1


def test_boundaries_to_chapters_covers_every_sentence():
    from podsum.schema import Sentence
    from podsum.segment.semantic import boundaries_to_chapters

    sents = [Sentence(idx=i, text=f"s{i}", start=i * 10.0, end=i * 10.0 + 9) for i in range(12)]
    chapters = boundaries_to_chapters(sents, [4, 8])
    assert [(c.first_sentence, c.last_sentence) for c in chapters] == [(0, 3), (4, 7), (8, 11)]
    assert chapters[0].start == 0.0 and chapters[-1].end == 119.0
    assert boundaries_to_chapters(sents, [])[0].last_sentence == 11
    assert boundaries_to_chapters([], []) == []

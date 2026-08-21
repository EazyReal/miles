from miles.backends.training_utils.serialized_buckets import align_serialized_bucket_columns


def test_all_empty_ranks_yield_no_columns():
    assert align_serialized_bucket_columns([[], []]) == []


def test_short_rank_pads_with_none():
    remote = {"flattened_tensor": ("remote",), "metadata": ("w",)}
    assert align_serialized_bucket_columns([[], [remote]]) == [[None, remote]]


def test_ragged_dtype_counts_pad_later_columns():
    a0, a1, b0 = "a0", "a1", "b0"
    assert align_serialized_bucket_columns([[a0, a1], [b0]]) == [[a0, b0], [a1, None]]

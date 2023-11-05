import mrpt
import numpy as np
import time

from .utils import helpers

logger = helpers.make_logger(__name__)

def run(target, queries, k):

	n, d = target.shape
	test_n, test_d = queries.shape
	assert (test_d == d)
	target_recall = 1.0
	
	index = mrpt.MRPTIndex(target)
	# exact search
	# for q in queries:
	# 	index.exact_search(q, k, return_distances=False)
	
	index.build_autotune_sample(target_recall, k)
	
	search_start = time.perf_counter()
	for q in queries:
		index.ann(q, return_distances=False)
		
	search_elapsed = time.perf_counter() - search_start
	logger.info(f"Search time: {search_elapsed:.2e} s")
	logger.info(f"Search time per query: {search_elapsed/test_n:.2e} s")
	logger.info(f"Throughput: { test_n / search_elapsed:.3f}")
	


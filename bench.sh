#!/bin/sh

for k in 10 100 ; do
	for ds in deep-image fashion-mnist gist glove-25 glove-50 glove-100 glove-200 mnist sift lastfm nytimes ; do
		for rival in faiss hnsw mrpt ; do
			python -m rivals /scratch/data/ann-benchmarks/datasets $ds $rival $k > $ds_$rival_$k 2>&1;
		done
	done
done
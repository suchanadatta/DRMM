#!/bin/bash

# create interaction matrices both for train (from qrel) and test (from top retrieved doc set) split

cd ./InteractionMatrix/

echo "#######################################################################"
echo "################# Generate Interaction Matrices #######################"
echo "#######################################################################"

sh interaction.sh ../data/query_sample-1/train-query.txt ../data/query_sample-1/test-query.txt /store/index/trec678/ ../InteractionMatrix/resources/smart-stopwords 3 10 ../data/ /store/causalIR/drmm/data/trec678.vec.model.txt content ../data/interaction_matrix/ /store/qrels/trec-robust.qrel 

# Train DRMM model and generate reranked test file

../
cd ./drmm/

echo "#######################################################################"
echo "################# Test file reranked using DRMM #######################"
echo "#######################################################################"

# create train-test folds

python3 train_test_folds.py ../data/qrel ../data/LMDirichlet1000.0-D10-content.res ../data/query_sample-1/train-query.txt ../data/query_sample-1/test-query.txt ../data/

# DRMM model

python3 run_model.py drmm_trec ../data/drmm_fold.train ../data/interaction_matrix/foo.hist ../data/drmm_fold.test ../data/interaction_matrix/foo.hist ../data/




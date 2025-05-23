# BiKE-SPHOTA
Knowledge Graph Structure Prediction with a Hybrid Orientation of Textual Alignment. </br>
This repository contains the edited files from other repositories natuke and kims-bert along with some additional new codes. After cloning these three repositories, those codes are need to be copied to their corresponding directories. This will replace the original codes with the edited codes of modification <i> (These steps are included with cp command in the instructions below). </i> The errors that may occur during the requirements installation step is fixed using the methods written in  installation_errors_fixing.pdf . Some of these errors are system/environment/package specific and may not occur in appropriate setups. In such case, you can follow the flow of execution given below without interruption.

<b> Step 1 </b>
Create a folder BiKE
1. `cd BiKE/`  Change Directory to BiKE
2. `git clone https://github.com/AKSW/natuke.git`  Clone natuke repository for BiKE challenge data.
3. `git clone https://github.com/vincenzodeleo/kims-bert.git`  Clone kims-bert repository for K-BERT model.
4. `git clone https://github.com/bharathchand10/BiKE-SPHOTA.git`  Clone BiKE-SPHOTA repository for the edited codes.

<b> Step 2 </b>
Create Anaconda Environments
1. `conda create --name bike_1 python=3.8 ipykernel notebook ipywidgets -c conda-forge`  for running some natuke codes.
2. `conda create --name bike_2 python=3.8 ipykernel notebook ipywidgets -c conda-forge`  for running other natuke codes
3. `conda create --name k_bert python=3.8` for running k-bert codes.

<b> Step 3 </b>
Install the requirements 
1. `cd natuke/`
2. `conda activate bike_1`
3. `pip install -r requirements.txt`
4. `cd GraphEmbeddings/`
5. `python setup.py install`
6. `cd ../`
  
7. `conda deactivate`
8. `conda activate bike_2`
9. `pip install -r requirements_topic_cuda.txt` <b>  OR  </b> `pip install -r requirements_topic.txt`
10. `conda deactivate`
  
11. `cd ../kims-bert/CODE/K-BERT/`
12. `conda activate k_bert`
13. `pip install -r requirements.txt`
14. `conda deactivate`

<b> Step 4 </b> Generate Phrases, Entities and Triples
1. `cd ../../../natuke/`
2. `mkdir Data`
3. `cp ../BiKE-SPHOTA/flat-data.csv ./Data/`
4. `cp ../BiKE-SPHOTA/smiles_name.csv ./Data/`
5. `mkdir ./Data/pdfs` -- <i> (This is the folder from which the pdf inputs are taken) </i>
6. `conda activate bike_1`
8. clean-pdfs.ipynb -- <i> (change - path = 'path-to-data-repository' to path = 'Data/' AND 'query03-05.csv' to 'flat-data.csv' AND 'smiles_names03-05.csv' to 'smiles_name.csv'  ) </i>
10. phrase_flow.py
11. `conda deactivate`
12. `conda activate bike_2`
13. topic_generation.ipynb
14. topic_distribution.ipynb
15. `rm -rf ../kims-bert/CODE/K-BERT/datasets/scholarly_dataset/*`  -- <i> (Cleaning this directory by removing all the existing files here) </i>
16. `rm -rf ../kims-bert/CODE/K-BERT/brain/kgs/*`  -- <i> (Cleaning this directory by removing all the existing files here) </i>
17. `cp ../BiKE-SPHOTA/phrases_and_triples.ipynb ./`
18. phrases_and_triples.ipynb      -- <i> (This code will generate outputs that are to be used for K-BERT code in the next step (in directories /kims-bert/CODE/K-BERT/datasets/scholarly_dataset/ AND /kims-bert/CODE/K-BERT/brain/kgs/)) </i>
20. `conda deactivate`

<b> Step 5 </b> K-BERT Embedding
1. `cd ../kims-bert/CODE/K-BERT/`
2. `cp ../../../BiKE-SPHOTA/run_kbert_cls.py ./`
3. `cp ../../../BiKE-SPHOTA/run.sh ./`
4. `cp ../../../BiKE-SPHOTA/knowgraph.py ./brain/`
5. Download models.tar.gz from https://drive.google.com/file/d/157KliIkO3iYf7a7TCNzVABvsK5jgSj_g/view?usp=sharing and uncompress it in this directory.
6. `conda activate k_bert`
7. `bash run.sh`    -- <i> (This code will generate outputs that are of K-BERT embedding in the directory /natuke/ with the name "embeddings.parquet". This will be used for the node embedding of Knowledge Graph in the next step. Ignore the errors after "Embedding extraction completed." and "Saving embeddings to embeddings.parquet") </i>
8. `conda deactivate`

<b> Step 6 </b> Creating Knowledge Graph
1. `cd ../../../natuke/`
2. `cp ../BiKE-SPHOTA/including_k-bert_embeddings.ipynb ./`
3. `cp ../BiKE-SPHOTA/hin_generation_new.ipynb ./`
4. `conda activate bike_2`
5. including_k-bert_embeddings.ipynb
6. hin_generation_new.ipynb  -- <i> (This is a new modified code. Not to be confused with the original one in the same directory for execution) </i>
7. `conda deactivate`

<b> Step 7 </b> Final Results and Evaluation
1. `conda activate bike_1`
2. `mkdir Data/results`
3. `cp ../BiKE-SPHOTA/knn_dynamic_benchmark.py ./`
4. `cp ../BiKE-SPHOTA/natuke_utils.py ./`
5. knn_dynamic_benchmark.py -- <i> (This code is edited to call the regularization_2 function from the edited code </i> natuke_utils.py <i> where K-BERT embedding associated regularization is written in the function regularization_2) </i>
6. `mkdir Data/metric_results`
7. `cp ../BiKE-SPHOTA/dynamic_benchmark_evaluation.py ./`
8. dynamic_benchmark_evaluation.py
9. `cp ../BiKE-SPHOTA/Final_Outputs.ipynb ./Data/metric_results/`
10. `cd Data/metric_results/`
11. Final_Outputs.ipynb
12. `conda deactivate`





# BiKE-SPHOTA
Knowledge Graph Structure Prediction with a Hybrid Orientation of Textual Alignment

<b> Step 1 </b>
Create a folder BiKE
1. `cd BiKE`  Change Directory to BiKE
2. `git clone https://github.com/AKSW/natuke.git`  Clone natuke repository for BiKE challenge data.
3. `git clone https://github.com/vincenzodeleo/kims-bert.git`  Clone kims-bert repository for K-BERT model.
4. `git clone https://github.com/bharathchand10/BiKE-SPHOTA.git`  Clone BiKE-SPHOTA repository for the edited codes.

<b> Step 2 </b>
Create Anaconda Environments
1. `conda create -n bike_1 python=3.8 ipykernel notebook ipywidgets -c conda-forge`  for running some natuke codes.
2. `conda create -n bike_2 python=3.8 ipykernel notebook ipywidgets -c conda-forge`  for running other natuke codes
3. `conda create -n k_bert python=3.8 for running k-bert codes.

<b> Step 3 </b>
Install the requirements 
1. `cd natuke`
2. `conda activate bike_1`
3. `pip install -r requirements.txt`
  
4. `conda deactivate`
5. `conda activate bike_2`
6. `pip install -r requirements_topic_cuda.txt` <b>  OR  </b> `pip install -r requirements_topic.txt`
7. `conda deactivate`
  
8. `cd ../kims-bert/CODE/K-BERT`
9. `conda activate k_bert`
10. `pip install -r requirements.txt`
11. `conda deactivate`

<b> Step 4 </b> Generate Phrases, Entities and Triples
1. `cd ../../../natuke`
2. `mkdir Data`
3. `cp ../BiKE-SPHOTA/flat-data.csv ./Data/`
4. `cp ../BiKE-SPHOTA/smiles_name.csv ./Data/`
5. `mkdir ./Data/pdfs`
6. `conda activate bike_1`
7. change path = 'path-to-data-repository' to path = 'Data/' in all files
8. clean-pdfs.ipynb
9. phrase_flow.py
11. `conda deactivate`
12. `conda activate bike_2`
13. topic_generation.ipynb
14. topic_distribution.ipynb
15. `rm -rf ../kims-bert/CODE/K-BERT/datasets/scholarly_dataset/*`
16. `rm -rf ../kims-bert/CODE/K-BERT/brain/kgs/*`
17. `cp ../BiKE-SPHOTA/phrases_and_triples.ipynb ./`
18. phrases_and_triples.ipynb
20. `conda deactivate`

<b> Step 5 </b> 



<b> Step 6 </b>

<b> Step 7 </b>

<b> Step 8 </b>

<b> Step 9 </b>

<b> Step 10 </b>



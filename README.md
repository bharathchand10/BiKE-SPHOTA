# BiKE-SPHOTA
Knowledge Graph Structure Prediction with a Hybrid Orientation of Textual Alignment

<b> Step 1 </b>
Create a folder BiKE
- `cd BiKE`  Change Directory to BiKE
- `git clone https://github.com/AKSW/natuke.git`  Clone natuke repository for BiKE challenge data.
- `git clone https://github.com/vincenzodeleo/kims-bert.git`  Clone kims-bert repository for K-BERT model.
- `git clone https://github.com/bharathchand10/BiKE-SPHOTA.git`  Clone BiKE-SPHOTA repository for the edited codes.

<b> Step 2 </b>
Create Anaconda Environments
- `conda create -n bike_1 python=3.8 ipykernel notebook ipywidgets -c conda-forge`  for running some natuke codes.
- `conda create -n bike_2 python=3.8 ipykernel notebook ipywidgets -c conda-forge`  for running other natuke codes
- `conda create -n k_bert python=3.8 ipykernel notebook ipywidgets -c conda-forge`  for running k-bert codes.

<b> Step 3 </b>
Install the requirements <br>
- cd natuke
- conda activate bike_1
- pip install -r requirements.txt
<br>
- conda deactivate
- conda activate bike_2
- pip install -r requirements_topic_cuda.txt <b> OR </b> pip install -r requirements_topic.txt
- conda deactivate
<br>
cd ../kims-bert/CODE/K-BERT
- conda activate k_bert
- pip install -r requirements.txt
- conda deactivate




<b> Step 4 </b>

<b> Step 5 </b>

<b> Step 6 </b>

<b> Step 7 </b>

<b> Step 8 </b>

<b> Step 9 </b>

<b> Step 10 </b>



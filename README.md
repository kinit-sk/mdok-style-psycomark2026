# mdok-style @ PsyCoMark

The mdok-style systems submitted to the [SemEval-2026 Task 10](https://hide-ous.github.io/semeval2026-psycomark) shared task.

## Cite
If you use the data, code, or the information in this repository, cite the following paper.

TBA

## Source Code Structure
| File | Description |
| :- | :- |
|mdok-style.py|the script for training and inference of the pure mdok-style system|
|mdok-style_robust.py|the script for training and inference of the self-trained mdok-style system submitted to subtask 2 of PsyCoMark|

## Installation
Clone and install the [IMGTB framework](https://github.com/kinit-sk/IMGTB), activate the conda environment.
   ```
   git clone https://github.com/kinit-sk/IMGTB.git
   cd IMGTB
   conda env create -f environment.yaml
   conda activate IMGTB
   ```

## Code Usage
1. To retrain the Qwen3-32B model, run the enclosed code mdok-style.py.

2. Run predictions using the fine-tuned Qwen3-32B model on dev and test splits.

3. Retrain self-trained Qwen3-32B model by running the enclosed mdok-style_robust.py script.

4. To run just inference, append ```--test_only``` option to the enclosed scripts.

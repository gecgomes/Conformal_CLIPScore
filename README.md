# 🤖 A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates

*Official source code repository for the article accepted at* **Findings of ACL 2025**  
🔗 [Read the paper (PDF)](https://aclanthology.org/2025.findings-acl.638.pdf)

```bibtex
@inproceedings{gomes2025conformal,
  title={A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates},
  author={Gomes, Gon{\c{c}}alo and Martins, Bruno and Zerva, Chrysoula},
  booktitle={Findings of the Association for Computational Linguistics},
  year={2025},
}
```

---

## Introduction

This repository contains the implementation of the methods described in the paper. The codebase is structured to mirror the paper and is divided into two main components:

- **`foil_main.py`**  
  Focuses on **reliability and identifying incorrect words** (foil words) within captions.

- **`ci_main.py`**  
  Aims at **improving CLIPScore reliability** by calibrating confidence intervals produced by the evaluation system.

> ⚠️ Each main file accepts its own set of arguments — please read them carefully before running.

---

## Note on Model Initialization

When initializing, the **image and text models load separately**. You may see a warning like:

> _Some weights of `CLIPModelWithMask` were not initialized from the model checkpoint._

This warning is expected because these weights correspond to the text part, which is initialized independently. **You can safely ignore this.**

---

## Datasets

### 📥 Download

Download the dataset folder here: **[Dataset Link](https://drive.google.com/file/d/14cg33vIBIfODBmAGcmXEXkdaTK6T6mUi/view?usp=drive_link)**

- The first time you run the main code, it checks if processed datasets exist locally.
- If not found, the datasets will be **generated automatically** from the downloaded starting points.

---

### Folder Structure

Your workspace should look like this:

```
ConformalFoil
├── cache
├── data # Folder downloaded using the link above
├── Helpers
├── ci_main.py
└── foil_main.py
```

---

### Datasets Used

Please download the following datasets from their official sources:

| Dataset       | Download Link                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------ |
| **FOIL-it**   | [https://foilunitn.github.io](https://foilunitn.github.io)                                       |
| **FOIL-nocaps** | [GitHub](https://github.com/DavidMChan/aloha/tree/main/data)                                   |
| **Rich-HF**   | [GitHub](https://github.com/google-research-datasets/richhf-18k)                                |
| **Ex-8k & CF-8k** | [clipscore flickr8k example](https://github.com/jmhessel/clipscore/blob/main/flickr8k_example/download.py) |
| **Composite** | [Dropbox Composite Dataset](https://www.dropbox.com/scl/fi/pktyhl9th4ycluq/AMT_eval.zip?rlkey=r63cr4ff2tx7su6u5ckqk81rn&e=1&dl=0)  
| **Flickr-30K** | [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)                      |
| **VICR**      | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c0b91f9a3587bf35287f41dba5d20233-Abstract-Datasets_and_Benchmarks.html) |
| **Polaris**   | [https://yuiga.dev/polos/](https://yuiga.dev/polos/)                                            |

---

## 📩 Contact

For questions or collaboration inquiries, please reach out to:  
**goncaloecgomes@tecnico.ulisboa.pt**

---

*Thank you for your interest in our work!*

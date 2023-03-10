# Salient Mask-Guided Vision Transformer for Fine-Grained Classification
Official repository for the paper "[Salient Mask-Guided Vision Transformer for Fine-Grained Classification](https://www.researchgate.net/publication/366389604_Salient_Mask-Guided_Vision_Transformer_for_Fine-Grained_Classification)", <br>
accepted as a Full Paper to [VISAPP '23](https://visapp.scitevents.org/) (part of [VISIGRAPP '23](https://visigrapp.scitevents.org/)).

> [**Salient Mask-Guided Vision Transformer for Fine-Grained Classification**](https://www.researchgate.net/publication/366389604_Salient_Mask-Guided_Vision_Transformer_for_Fine-Grained_Classification)
> [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.researchgate.net/publication/366389604_Salient_Mask-Guided_Vision_Transformer_for_Fine-Grained_Classification)<br>
> [Dmitry Demidov](https://scholar.google.es/citations?hl=en&pli=1&user=k3euI0sAAAAJ), [Muhammad Hamza Sharif](https://www.researchgate.net/profile/Muhammad-Sharif-44), [Aliakbar Abdurahimov](https://www.researchgate.net/scientific-contributions/Aliakbar-Abdurahimov-2227848477), [Hisham Cholakkal](https://scholar.google.com/citations?user=bZ3YBRcAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)


## Approach

<p align="center"> 

Main Architecture          | Attention guiding (see Eq. 3)
:-------------------------:|:-------------------------:
<img src="docs/Images/2_main.jpg" width="750">  | <img src="docs/Images/attention_guiding.gif" width="220">
\- | blue, green, red bars -  <br> attention to salient patches

</p>

<p align="justify">  In this work, we introduce a simple yet effective approach to improve the performance of the standard Vision Transformer architecture at FGVC. Our method, named SalientMask-Guided Vision Transformer (SM-ViT), utilises a salient object detection module comprising an off-the-shelf saliency detector to produce a salient mask likely focusing on the potentially discriminative foreground object regions in an image. The saliency mask is then utilised within our ViT-like Salient Mask-Guided Encoder (SMGE) to boost the discriminabil-ity of the standard self-attention mechanism, thereby focusing on more distinguishable tokens. </p>


> **<p align="justify"> Abstract:** *Fine-grained visual classification (FGVC) is a challenging computer vision problem, where the task is to automatically recognise objects from subordinate categories. One of its main difficulties is capturing the most discriminative inter-class variances among visually similar classes. Recently, methods with Vision Transformer (ViT) have demonstrated noticeable achievements in FGVC, generally by employing the self-attention mechanism with additional resource-consuming techniques to distinguish potentially discriminative regions while disregarding the rest. However, such approaches may struggle to effectively focus on truly discriminative regions due to only relying on the inherent self-attention mechanism, resulting in the classification token likely aggregating global information from less-important background patches. Moreover, due to the immense lack of the datapoints, classifiers may fail to find the most helpful inter-class distinguishing features, since other unrelated but distinctive background regions may be falsely recognised as being valuable. To this end, we introduce a simple yet effective Salient Mask-Guided Vision Transformer (SM-ViT), where the discriminability of the standard ViT's attention maps is boosted through salient masking of potentially discriminative foreground regions. Extensive experiments demonstrate that with the standard training procedure our SM-ViT achieves state-of-the-art performance on popular FGVC benchmarks among existing ViT-based approaches while requiring fewer resources and lower input image resolution.* </p>




## Main Contributions

1) We introduce a simple yet effective approach to improve the performance of the standard Vision Transformer architecture at FGVC.
2) To the best of our knowledge, we are the first to explore the effective utilisation of saliency masks in order to extract more distinguishable information within the ViT encoder layers by boosting the discriminability of self-attention features for the FGVC task.  
3) Our extensive experiments on three popular FGVC datasets (Stanford Dogs, CUB, and NABirds) demonstrate that with the standard training procedure the proposed SM-ViT achieves state-of-the-art performance.
4) Important advantage of our solution is its integrability, since it can be fine-tuned on top of a ViT-based backbone or can be integrated into a Transformer-like architecture that leverages the standard self-attention mechanism.


<hr />


# ???? Model Zoo

All models in our experiments are first initialised with publicly available pre-trained [ViT/B-16 model's weights](https://console.cloud.google.com/storage/browser/vit_models;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) and then fine-tuned on the corresponding datasets.

### Main Models

| Model         |  Baseline  |  Input Size | St. Dogs | *Weights*   | CUB-200 | *Weights*    | NABirds | *Weights*  | 
|---------------|:----------:|:-----------:|:-------:|:-----------|:-------:|:-----------|:-------:|:---------|
| Vanilla ViT   | ViT-B/16   |  448x448    |  91.4   | -          |  90.6   | -          |  89.6   | -        |
| SM-ViT <br> (ours) | ViT-B/16   |  400x400    |  92.3   | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/ETgkV4GFNVtKjvADenZsZZsBCo07hWu5EazVuANq5_i3bQ?e=TCvN5Y)   |  91.6   | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/ESEHQadrOaJAo3NiW8Sok_IBn6j9m5V7-BfpCOO0yqbK7w?e=dDQtpb)   |  90.2   | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/EVyqOsO5o69CkkEtngxmSVkBfgyq5fqedZyHOrY-F_PUPw?e=MFVgRT) |
| SM-ViT <br> (ours) | ViT-B/16   |  448x448    |      |           |      |           |  90.5   | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/Ec7ZMacJlo9Kgmvp4lk_ppgBkDe1CAuaAPdNzukRlSpSxw?e=3yjM80) |
| SM-ViT <br> (ours) | ViT-B/16   |  560x560    |      |           |      |           |  90.7   | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/EeLk-WK5vfNNrIqu7FUN4c0Bn-QtFKRUUwxZLwAKfRyPUw?e=aCWWEH) |


#### Experimental Models (outside the paper)

| Model                         |  Input Size | St. Dogs | *Weights*   | CUB-200 | *Weights*    | NABirds | *Weights*  | 
|-------------------------------|:-----------:|:-------:|:-----------|:-------:|:-----------|:-------:|:---------|
| SM-ViT <br> + Advanced guiding | 400x400     |  -      | -   |  91.7   | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/EXfZdBmHZFtLpcwkli-cOfABtowHXUR2oX03TMdEtJcc6w?e=EWi56c)   |  90.7   | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/EUiGpYiy4mZPlHlFQImWHfwBdIrOvPLMZgW69p5ApoH3xA?e=CuCDmw) |


<hr />


# ???? How to start

## Installation 
For environment installation and pre-trained models preparation, please follow the instructions in [INSTALL.md](docs/INSTALL.md). 

## Data preparation
For datasets preparation, please follow the instructions in [DATASET.md](docs/DATASET.md).

## Training and Evaluation
For training and evaluation, please follow the instructions in [RUN.md](docs/RUN.md).


<hr />


# ???? News
* **(Dec 20, 2022)** 
  * Repo description added (README.md).

* **(Dec 30, 2022)** 
  * Pretrained models are released.
  * Code instructions added (INSTALL.md, DATASET.md, RUN.md).
  
* **(Jan 09, 2023)** 
  * Training and evaluation code is released.

* **(Soon)** 
  * Optimisation
 

<hr />


# ??????? Credits

## Citation
In case you would like to utilise or refer to our approach (source code, trained models, or results) in your research, please consider citing:

```
@conference{demidov2022smvit,
    author={Dmitry Demidov. and Muhammad Sharif. and Aliakbar Abdurahimov. and Hisham Cholakkal. and Fahad Khan.},
    title={Salient Mask-Guided Vision Transformer for Fine-Grained Classification},
    booktitle={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP,},
    year={2023},
    pages={27-38},
    publisher={SciTePress},
    organization={INSTICC},
    doi={10.5220/0011611100003417},
    isbn={978-989-758-634-7},
    issn={2184-4321},
}
```


## Contacts
In case you have a question or suggestion, please create an issue or contact us at _dmitry.demidov@mbzuai.ac.ae_ .


## Acknowledgements
Our code is partially based on [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch), [U2N](https://github.com/xuebinqin/U-2-Net), and [FFVT](https://github.com/Markin-Wang/FFVT) repositories and we thank the corresponding authors for releasing their code. If you use our derived code, please consider giving credits to these works as well.

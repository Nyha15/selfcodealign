# SelfCodeAlign: Self-Alignment for Code Generation

<p align="left">
    <a href="https://openreview.net/forum?id=xXRnUU7xTL"><img src="https://img.shields.io/badge/Paper-NeurIPS'24-a55fed.svg?style=for-the-badge" alt="Paper" class="img-fluid"/></a>
    <a href="https://arxiv.org/abs/2410.24198"><img src="https://img.shields.io/badge/arXiv-2410.24198-b31b1b.svg?style=for-the-badge">
    <!-- <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge"> -->
    <!-- <a href="https://huggingface.co/ise-uiuc/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ise--uiuc-%23ff8811.svg?style=for-the-badge"> -->
    <!-- <a href="https://twitter.com/magicoder_ai"><img src="https://img.shields.io/badge/@magicoder__ai-000000?style=for-the-badge&logo=x&logoColor=white"> -->
</p>

<p align="left">
    🧐&nbsp;<a href="#about">About</a>
    | ⭐️&nbsp;<a href="README-SC2INST.md">StarCoder2-Instruct</a>
    | 📝&nbsp;<a href="#quick-start">Citation</a>
    <!-- | 🚀&nbsp;<a href="#quick-start">Quick start</a> -->
    <!-- | 📚&nbsp;<a href="#data-generation-pipeline">Data generation</a> -->
    <!-- | 🧑‍💻&nbsp;<a href="#training-details">Training</a> -->
    <!-- | 📊&nbsp;<a href="#evaluation-on-evalplus-livecodebench-and-ds-1000">Evaluation</a> -->
    <!-- | ⚠️&nbsp;<a href="#bias-risks-and-limitations">Limitations</a> -->
</p>

## About

**SelfCodeAlign** is the first fully open and transparent pipeline that enhances a code language model without relying on human annotations or distilled data from large, proprietary models. This approach led to the creation of [StarCoder2-Instruct](README-SC2INST.md), a fully transparent, permissively licensed, self-aligned code model that achieves state-of-the-art performance in coding tasks.

**Authors:**
[Yuxiang Wei](https://yuxiang.cs.illinois.edu),
[Federico Cassano](https://federico.codes/),
[Jiawei Liu](https://jw-liu.xyz),
[Yifeng Ding](https://yifeng-ding.com),
[Naman Jain](https://naman-ntc.github.io),
[Zachary Mueller](https://muellerzr.github.io),
[Harm de Vries](https://www.harmdevries.com),
[Leandro von Werra](https://twitter.com/lvwerra),
[Arjun Guha](https://www.khoury.northeastern.edu/home/arjunguha/main/homehttps://www.khoury.northeastern.edu/home/arjunguha/main/home//),
[Lingming Zhang](https://lingming.cs.illinois.edu).

![self-alignment pipeline](https://huggingface.co/datasets/bigcode/starcoder2-instruct-assets/resolve/main/SelfCodeAlign.png)

## StarCoder2-Instruct

![Banner](https://huggingface.co/datasets/bigcode/starcoder2-instruct-assets/resolve/main/banner.png)

StarCoder2-Instruct is created with an [earlier version](https://github.com/bigcode-project/selfcodealign/tree/starcoder2-instruct) of SelfCodeAlign. It is the very first entirely self-aligned code Large Language Model (LLM) trained with a fully permissive and transparent pipeline. Our open-source pipeline uses StarCoder2-15B to generate thousands of instruction-response pairs, which are then used to fine-tune StarCoder-15B itself without any human annotations or distilled data from huge and proprietary LLMs.

- **Model:** [bigcode/starcoder2-15b-instruct-v0.1](https://huggingface.co/bigcode/starcoder2-instruct-15b-v0.1)
- **Code:** [bigcode-project/starcoder2-self-align](https://github.com/bigcode-project/starcoder2-self-align)
- **Dataset:** [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k/)

For more details, check [README-SC2INST.md](README-SC2INST.md).

## Citation

```bibtex
@article{wei2024selfcodealign,
  title={SelfCodeAlign: Self-Alignment for Code Generation}, 
  author={Yuxiang Wei and Federico Cassano and Jiawei Liu and Yifeng Ding and Naman Jain and Zachary Mueller and Harm de Vries and Leandro von Werra and Arjun Guha and Lingming Zhang},
  year={2024},
  journal={arXiv preprint arXiv:2410.24198}
}
```

# RealQA
Next Token Is Enough: Realistic Image Quality and Aesthetic Scoring with Multimodal Large Language Model.  
[[Paper]](https://arxiv.org/pdf/2503.06141) [[🏆 SOTA IQA]](https://paperswithcode.com/sota/image-quality-assessment-on-koniq-10k)  

![RealQA](figure/dataset.pdf)


# Abstract
The rapid expansion of mobile internet has resulted in a substantial increase in user-generated content (UGC) images, thereby making the thorough assessment of UGC images both urgent and essential. Recently, multimodal large language models (MLLMs) have shown great potential in image quality assessment (IQA) and image aesthetic assessment (IAA). Despite this progress, effectively scoring the quality and aesthetics of UGC images still faces two main challenges:   
1) A single score is inadequate to capture the hierarchical human perception.  
2) How to use MLLMs to output numerical scores, such as mean opinion scores (MOS), remains an open question.  

To address these challenges, we introduce a novel dataset, named Realistic image Quality and Aesthetic (RealQA), including **14,715** UGC images, each of which is annotated with **10 fine-grained attributes**. These attributes span three levels: low level (e.g., image clarity), middle level (e.g., subject integrity) and high level (e.g., composition). Besides, **we conduct a series of in-depth and comprehensive investigations into how to effectively predict numerical scores using MLLMs**. Surprisingly, by predicting just two extra significant digits, the next token paradigm can achieve SOTA performance. Furthermore, with the help of chain of thought (CoT) combined with the learnt fine-grained attributes, the proposed method can outperform SOTA methods on **five** public datasets for IQA and IAA with superior interpretability and show strong zero-shot generalization for video quality assessment (VQA). 



# TODO
- [ ] Release the dataset
- [ ] Release the checkpoint
- [ ] Release the inference code
- [ ] Release the evaluation code
- [ ] Release the training code



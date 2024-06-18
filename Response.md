# **Responses to the Editor:**

We appreciate the time and effort invested in reviewing our manuscript submitted to the Transactions on Industrial Informatics. We are also grateful for the constructive feedback provided by the reviewers. Taking into consideration the invaluable comments and suggestions, we have diligently revised the manuscript to address the identified concerns. The changes made aim to enhance the quality and relevance of our work to align more closely with the standards of the journal. We made the following changes:

- Improve the analysis and writing of the key contributions part. 

- Incorporate sub-section relationships and contribution descriptions to ensure a clear and coherent flow.

- Check the manuscript for grammatical errors and improve the writing of some paragraphs.

We believe that the revisions made have significantly improved the manuscript's overall quality and addressed the concerns raised during the initial review. With this, we would like to express our sincere interest in submitting the revised manuscript for your consideration.

------



# **Reviewer 1:**

***This paper is interesting and is an impressive piece of research work.***

**Response:** Thanks for your time to review our manuscript. We respond point by point to your constructive comments and suggestions, detailed below:



***Q1. The key contributions of models require improvement.***

**Response:** Thanks to the reviewer for valuable suggestions. We enhance the analysis and writing of the key contributions part. **A short description of the key contributions of this paper is given below:**

- **The key contribution of this work is combining the strengths of ConvNets and Transformers to develop an efficient and high-performance network.**

- **Based on this motivation, we thoroughly analyze the merits of Transformer framework,** including large-range relationships modeling, two-order features interaction, input self-adaptation and advanced components. **(Detailed analyses are provided in Section III-A.)** 

- **Based on the comprehensive analysis of the advantages of the Transformer architecture, we utilize advanced Transformer architecture characteristics to guide the design of a new ConvNet, which leads the Transformer-style ConvNet (TSCN), with efficiency and performance.** 

- **To realize this motivation, we propose two key modules in TSCN, including LMCM (Section III-C) and SADFF (Section III-D).** LMCM integrates the re-weighting process into the LCM and generates convolution features as weight matrices to self-adaptively recalibrate input features, achieving multi-order features interaction and large-range dependencies modeling. SADFF introduces spatial awareness and locality, improves feature diversity, and dynamically regulates the flow of information between layers to handle the absence of locality and channel redundancy compared to vanilla FFN. **(See Section III-C-2 for the analysis of why TSCN has the advantages of both ConvNet and Transformer.)**

- Finally, quantitative and qualitative results on synthetic and realistic datasets demonstrate that TSCN performs better than other recently advanced Transformer-based methods with lower complexity, latency, and higher FPS.

 

***Q2. The design idea of the model is similar to the attention or convolution-based works. To continue advancing the capabilities of related domains, the authors need to add related discussions in some parts.***

***(https://ieeexplore.ieee.org/document/10312808;https://www.sciencedirect.com/science/article/pii/S0031320322007130;https://ieeexplore.ieee.org/document/9982294)***

**Response:** Thanks to the reviewer for highly valuable comments. These works are very relevant to our paper, and we have cited them in the revised version. We add the following to the related work (Section II-B):

Like the multi-scale strategy and attention in our work, STFuse [1] designs a cross-feature enhancement module that utilizes self-attention and mutual-attention features to guide each branch to refine features. MCANet [2] can adaptively uncover the spatial-temporal contextual dependence information by novel multi-scale feature fusion strategy. GCR-Net [3] devises the novel exponential gated convolutional residual networks to extract multiple complex patterns of the spatial dimension from genomic sequences.

```
STFuse [1] ~ https://ieeexplore.ieee.org/document/10312808
MCANet [2] ~ https://www.sciencedirect.com/science/article/pii/S0031320322007130
GCR-Net [3] ~ https://ieeexplore.ieee.org/document/9982294
[1] Wang X, Guan Z, Qian W, et al. STFuse: Infrared and Visible Image Fusion via Semisupervised Transfer Learning[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.
[2] Guo Y, Zhou D, Li P, et al. Context-aware poly (a) signal prediction model via deep spatial–temporal neural networks[J]. IEEE Transactions on Neural Networks and Learning Systems, 2022.
[3] Li W, Guo Y, Wang B, et al. Learning spatiotemporal embedding with gated convolutional recurrent networks for translation initiation site prediction[J]. Pattern Recognition, 2023, 136: 109234.
```



# **Reviewer 2:**

***Q1. Although there are clear improvements, the manuscript is still very difficult to read. Aside from the many spelling and grammar issues, it is difficult to understand the ideas in the algorithm, results, and discussion sections. There is too much low level detail and no clear, coherent flow. It is also not clear what is the technical contribution and what is previous work.***

**Response:** Thanks for your comment. We carefully check the manuscript for grammatical errors and improve the writing of some paragraphs to improve readability. We incorporate sub-section relationships and contribution descriptions to ensure a clear and coherent flow. **For improved clarity, we provide a simplified version of the content in both the algorithm (Section III) and results (Section IV).**



- ### <font color=DarkRed>Section III Proposed Method</font>

> **First, we briefly describe the motivation:** The complementary relationship exists between Transformers' representation ability and ConvNets' efficiency, making both crucial for practical SISR applications. **Then, motivated by this, we first analyze the advantages of the Transformer architecture in Section III-A.** **These insights are then leveraged to guide the design of ConvNets, combining the strengths of both approaches to develop the proposed TSCN, as detailed in Section III-B.**


- #### <font color=DodgerBlue>Section III-A Revisit</font>

> - **Contribution:** We revisit Transformer architecture from four key aspects. Correspondingly, four network design motivations are proposed to guide TSCN design.
> - **We revisit Transformer architecture from four key aspects:** long-range spatial relationships modeling, two-order contextual information interaction, input content self-adaptation, and advanced macro structure and components.
> - **Correspondingly, four network design motivations are proposed**: Inspiration-LRM (Large-range modeling), Inspiration-MOI (Multi-order interaction), Inspiration-ISA (Input self-adaptation), and Inspiration-AMS (Advanced macro structure). 
> - Next is the detailed description of the four key benefits of the Transformer and the corresponding four points of motivation for the TSCN design.


- #### <font color=DodgerBlue>Section III-B Overall architecture</font>

> - **Contribution:** We propose the TSCN, which modifies the macro structure of the Transformer by replacing the SA and FFN with the proposed bottleneck-like LMCM and SADFF.
> - The overall structure of the TSCN contains three parts: shallow feature extraction, (2) deep feature extraction, and (3) image reconstruction. The core module TSCB in deep feature extraction is expressed as Equation 2, where LMCM(·) and SADFF(·) denote LMCM and SADFF, and norm(·) denotes norm layer. 
> - **Next, LMCM and SADFF are described in detail in Sections III-C and III-D.**


- #### <font color=DodgerBlue>Section III-C LMCM</font>

> - **Contribution:** The proposed LMCM integrates the re-weighting process into the LCM technology and utilizes the extracted convolutional features as weight matrices to self-adaptively re-calibrate the input representations, allowing for efficient large-range spatial relationships modeling and multi-order features interaction. 
> - LMCM contains three sub-processes: (1) Local perception, (2) Large-range spatial relationships modeling and multi-order contextual information interaction, and (3) Feature integration. The implementation of the three sub-processes is then described in detail.
> - Finally, in discussion part, we interpret why LMCM exhibits the characteristics of the Transformer architecture, *e.g.,* large-range spatial relationships modeling, multi-order context information interaction and input content adaptation. 


- #### <font color=DodgerBlue>Section III-D SADFF</font>

> - **Contribution**: The proposed SADFF introduces spatial awareness, locality, and dynamic information flow modulation between layers.
> - Firstly, two sub-optimal aspects of FFN are first analyzed, including the absence of locality modeling, and channel redundancy.
> - Secondly, we propose the SADFF to alleviate these problems by incorporating SAL, gating mechanism, and channel attention.



- ### <font color=DarkRed>Section IV Experiments</font>
- #### <font color=DodgerBlue>Section IV-B Ablation Study on Micro Design</font>

> - **Architecture configuration.** The relationship between model complexity and performance is explored by changing the number of TSCBs. (Figure 5 and Table 1 in the 'Ablation Study on Micro Design' in Anonymous Repository)
> - **Subordinate components in SADFF.** The impact of each component in SADFF is explored, including SAL, gating mechanism, and channel attention. (Figure 5 and Table II)
> - **Subordinate components in LMCM.** The impact of each component in LMCM is investigated, including the choice of layer sequences and the number of sub-branches. (Figure 5, Tables III and IV)


- #### <font color=DodgerBlue>Section IV-C SISR for Natural Images</font>

> - **Quantitative results:** Synthetic image experiment (Table V) and Real image experiment (Table VI).
> - **Qualitative results:** Synthetic image experiment (Figure 6) and Real image experiment (Figure 7).
> - **LAM analysis:** The LAM analysis aims to investigate the pixel utilization range in the input image during the reconstruction of a selected area, also known as receptive fields. (Figure 8)
> - **Operation efficiency analysis:** We analyze the operation efficiency of TSCN in terms of number of parameters, Multi-Adds, latency, and FPS. These metrics reflect disk storage, computation burden, inference speed, and efficiency, respectively. (Table VII)


- #### <font color=DodgerBlue>Section IV-D SISR for Industrial Images</font>

> - Our method can be generalized to the resolution enhancement of some industrial scene images to serve subsequent high-level vision tasks, such as autonomous driving scene, remote sensing industrial detection, etc. The results show that our method can help to improve the segmentation and identification accuracy of existing methods, bringing negligible computational burden due to the extremely lightweight structure.
> - **SISR for Printed Circuit Board** (Figure 9a)
> - **SISR for Automatic Driving** (Figure 9b and Figure 10)
> - **SISR for Remote Sensing Detection** (Figure 11)

 

***Q2. I thank the authors for addressing some of my comments including motivating the performance aspect for the remote sensing detection application. Also, the additional results are appreciated. Overall, the results are qualitatively sometimes very good and many times just "ok", but I still think it warrants publication. The proposed approach seems to be as good or better than state of the art in most cases. Quantitatively, the results are also impressive compared to SOTA.***

**Response:** We thank the reviewer for the positive comments.

 

***Q3. The following was not considered in my final recommendation, but another aspect that I felt was missing is that the proposed approach seeks to be efficient and high quality, but it is not clear which of the previous approaches consider inference runtime in their architecture design. I think the proposed approach could perform even better if there were less performance constraints. At least this context could be discussed even if there are no algorithmic changes (since the proposed approach already performs well even with the constraints).***

**Response:** Model complexity and inference time are critical considerations for every lightweight image super-resolution method. Accordingly, we also compare the model complexity (in terms of both the number of parameters and computational complexity) and the inference time in Tables V and VII of this work. The results demonstrate that our method achieves the best reconstruction performance while maintaining the lowest model complexity and inference time. This also indicates that our approach achieves the optimal balance between performance and complexity.

If the constraints on reconstruction performance are reduced, such as by relaxing the restrictions on model complexity and inference time, the model depth can be increased. This experiment has already been explored, as shown in Figure 5 under 'Block Number'. Due to space constraints, more detailed experiments are available in the anonymous repository (see Table 1 in the 'Ablation Study on Micro Design'). 

**The summary version is shown in Table 1 below,** and results show that increasing the model depth improves reconstruction performance. However, the rate of improvement slows after a certain depth. Such experimental phenomena are similar to other lightweight super-resolution methods, like SwinIR [1] and PILN [2].

Table 1: Comparison of model performance, complexity and inference time.

| Depth  | Params  (K) | Multi-Adds  (G) | Latency  (ms) | FPS  (image/s) | PSNR/SSIM        |
| ------ | ----------- | --------------- | ------------- | -------------- | ---------------- |
| 12     | 671         | 36.1            | 52.37         | 19.09          | 32.18/0.8950     |
| 14     | 777         | 41.8            | 58.77         | 17.02          | 32.33/0.8974     |
| **16** | **884**     | **47.5**        | **71.61**     | **13.97**      | **32.53/0.8998** |
| 18     | 991         | 53.3            | 81.08         | 12.33          | 32.56/0.9002     |
| 20     | 1096        | 59.1            | 88.11         | 11.35          | 32.57/0.9003     |

 

 ```
 [1] Liang J, Cao J, Sun G, et al. Swinir: Image restoration using swin transformer[C] //Proceedings of the IEEE/CVF international conference on computer vision. 2021: 1833-1844.
 [2] Qin J, Chen L, Jeon S, et al. Progressive interaction-learning network for lightweight single-image super-resolution in industrial applications[J]. IEEE Transactions on Industrial Informatics, 2022, 19(2): 2183-2191.
 ```



# **Reviewer 3:**

***Thanks for the revision of the authors, I have no further comments.***

**Response:** Thanks for the reviewer’s recommendation.

 

# **Reviewer 4:**

***All issues have been addressed and it can be acceptable in this version.*** 

**Response:** Thanks for the reviewer’s recommendation.

 
## Anomaly Generation
Anomaly Generation refers to the synthesis of artificial anomalous data to augment scarce real-world defect samples in industrial settings. This technique addresses the fundamental challenge of data imbalance in anomaly detection, where normal samples vastly outnumber anomalies, hindering model training.

### Why Anomaly Generation Matters
- **Academic Value**: It advances unsupervised and few-shot learning paradigms by providing diverse, controllable synthetic data for benchmarking and evaluation. Recent surveys highlight its role in exploring generative models (e.g., diffusion-based) and vision-language integration, fostering innovations in evaluation frameworks and cross-domain generalization [arXiv 2025].
- **Practical Production Value**: In manufacturing pipelines, it reduces reliance on costly manual labeling, accelerates model training, and improves detection accuracy by simulating rare defects, minimizing downtime and quality control costs.
- **Applicable Domains**: Widely used in industrial inspection (e.g., semiconductors, automotive parts), medical imaging (e.g., tumor simulation), autonomous driving (e.g., road hazard synthesis), and surveillance (e.g., behavioral anomalies).

### 📋 Hierarchical Subcategories
To clarify the structure, here's a numbered outline of the categories.

- **1. Improving Generation Speed**

- **2. Controllable Image Generation**   - 2.1 CutPaste Method Generation   - 2.2 GAN Generation   - 2.3 Diffusion Generation     - 2.3.1 Text-based Generation     - 2.3.2 Image-based Generation     - 2.3.3 Multi-Modal Generation       - 2.3.3.1 Text-Image Multi-Modal       - 2.3.3.2 Image-Depth Multi-Modal   - 2.4 Feature-level Anomaly Generation

- **3. Precise Mask**

- **4. Generation Quality Judgment and Evaluation System**

<a id="improving-generation-speed"></a>
## 1. Improving Generation Speed
*Enhancing generation speed is crucial for scaling anomaly synthesis to large datasets, enabling real-time augmentation during training and deployment in high-throughput industrial environments. This reduces computational overhead, making diffusion-based methods viable for resource-constrained settings without sacrificing diversity.*

- SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection [[ICPR 2024]][JIMS 2025][code]

<a id="controllable-image-generation"></a>
## 2. Controllable Image Generation
*Controllability allows precise specification of anomaly types, locations, and attributes via prompts or priors, ensuring synthetic data aligns with domain-specific needs. This boosts model generalization by simulating targeted scenarios, bridging the gap between generic augmentation and real defect variability.*

<a id="cutpaste-method-generation"></a>
### 2.1 CutPaste Method Generation
*CutPaste-inspired methods simulate anomalies through simple patch cutting and pasting from normal images, offering lightweight, label-free augmentation. This is vital for self-supervised anomaly detection, as it mimics realistic defects efficiently without requiring generative models, promoting accessibility in early-stage research and low-resource setups.*

- CutPaste: Self-supervised Learning for Anomaly Detection and Localization [(OCC)ICCV 2021][unofficial code]
- Natural Synthetic Anomalies for Self-supervised Anomaly Detection and Localization [ECCV 2022][code]

<a id="gan-generation"></a>
### 2.2 GAN Generation
*GAN-based approaches excel in producing high-fidelity, diverse anomalies by adversarially learning defect distributions from limited samples. Their importance lies in handling extreme class imbalance, enabling robust data augmentation for supervised fine-tuning and improving detection in domains like textiles where real defects are rare and varied.*

- Multistage GAN for Fabric Defect Detection [2019]
- GAN-based Defect Synthesis for Anomaly Detection in Fabrics [2020]
- Defect Image Sample Generation with GAN for Improving Defect Recognition [2020]
- Defective Samples Simulation through Neural Style Transfer for Automatic Surface Defect Segment [2020]
- Defect Transfer GAN: Diverse Defect Synthesis for Data Augmentation [BMVC 2022]
- Defect-GAN: High-fidelity Defect Synthesis for Automated Defect Inspection [2021]
- EID-GAN: Generative Adversarial Nets for Extremely Imbalanced Data Augmentation [TII 2022]

<a id="diffusion-generation"></a>
### 2.3 Diffusion Generation
*Diffusion models provide iterative denoising for superior sample quality and flexibility in anomaly synthesis. They are essential for modern controllable generation, allowing fine-grained control over defect attributes and enabling zero/few-shot adaptation, which drives advancements in scalable, high-resolution industrial simulations.*

<a id="text-based-generation"></a>
#### 2.3.1 Text-based Generation
*Text-based generation harnesses natural language prompts to specify anomaly types, locations, and attributes, offering intuitive and flexible control for zero-shot synthesis. This approach excels in scenarios requiring semantic guidance without visual exemplars, fostering diverse and semantically coherent anomaly creation through prompt engineering and language model integration in diffusion processes.*

- Component-aware Unsupervised Logical Anomaly Generation for Industrial Anomaly Detection [2025]
- Photovoltaic Defect Image Generator with Boundary Alignment Smoothing Constraint for Domain Shift Mitigation [2025]
- Anomaly Anything: Promptable Unseen Visual Anomaly Generation [CVPR 2025][code]

<a id="image-based-generation"></a>
#### 2.3.2 Image-based Generation
*Image-based generation conditions synthesis on visual cues like masks, bounding boxes, or reference images, enabling precise spatial localization and structural fidelity in anomaly placement. It is particularly effective for few-shot adaptation and boundary-aligned defects, enhancing realism in industrial simulations by leveraging existing visual priors to guide diffusion denoising.*

- AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [AAAI 2024][code]
- CAGEN: Controllable Anomaly Generator using Diffusion Model [ICASSP 2024]
- Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation [ECCV 2024][code]
- Bounding Box-Guided Diffusion for Synthesizing Industrial Images and Segmentation Map [CVPRW 2025][code]
- Enhancing Glass Defect Detection with Diffusion Models: Addressing Imbalanced Datasets in Manufacturing Quality Control [2025]
- Anodapter: A Unified Framework for Generating Aligned Anomaly Images and Masks Using Diffusion Models [2025]

<a id="multi-modal-generation"></a>
#### 2.3.3 Multi-Modal Generation
*Multi-modal synthesis integrates data from diverse sources (e.g., RGB + depth + text), capturing richer contextual cues for robust detection in complex scenes. It's vital for handling incomplete or noisy inputs in real-world applications, improving cross-modal fusion and overall system resilience—especially within diffusion frameworks, where modalities can be jointly denoised for coherent anomaly injection.*

<a id="text-image-multi-modal"></a>
##### 2.3.3.1 Text-Image Multi-Modal
*Text-image multi-modal generation combines natural language descriptions with RGB visuals to guide anomaly synthesis, enabling semantically rich and contextually aware defect creation. This fusion enhances controllability and realism by leveraging textual semantics to refine visual outputs, ideal for scenarios blending descriptive prompts with image priors.*

- AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion [2024][data]
- A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation [2024][code]
- AnomalyControl: Learning Cross-modal Semantic Features for Controllable Anomaly Synthesis [2024]
- AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer [AAAI 2026][code]
- Anomagic: Crossmodal Prompt-driven Zero-shot Anomaly Generation [AAAI 2026][code]
- AnomalyControl: Highly-Aligned Anomalous Image Generation with Controlled Diffusion Model [ACM MM 2025]

<a id="image-depth-multi-modal"></a>
##### 2.3.3.2 Image-Depth Multi-Modal
*Image-depth multi-modal generation fuses RGB images with depth maps to produce geometrically accurate anomalies, simulating 3D structural defects like deformations or occlusions. This approach is crucial for depth-sensitive industrial applications, ensuring spatial coherence and enhanced detection in 3D-aware environments through joint modality conditioning.*

- AnomalyHybrid: A Domain-agnostic Generative Framework for General Anomaly Detection [CVPR 2025 SyntaGen Workshop]

<a id="feature-level-generation"></a>
### 2.4 Feature-level Anomaly Generation
*Feature-level anomaly generation operates in latent or feature spaces to inject anomalies at higher abstractions, allowing for subtle and semantically meaningful defects without direct pixel-level manipulations. This method enhances efficiency, preserves global image consistency, and enables boundary-guided synthesis for more realistic industrial defect simulation.*

- Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection [TCSVT 2024][code]

<a id="precise-mask"></a>
## 3. Precise Mask
*Generating pixel-accurate masks ensures anomalies are spatially aligned with defects, facilitating supervised fine-tuning and precise localization. This is essential for pixel-level tasks like segmentation, reducing false positives and enhancing interpretability in downstream detection pipelines.*

- DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection [CVPR 2023][code]
- Anodapter: A Unified Framework for Generating Aligned Anomaly Images and Masks Using Diffusion Models [2025]
- Bounding Box-Guided Diffusion for Synthesizing Industrial Images and Segmentation Map [CVPRW 2025][code]
- DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup [ICCV 2025][code]

<a id="generation-quality-judgment-and-evaluation-system"></a>
## 4. Generation Quality Judgment and Evaluation System
*Robust evaluation metrics quantify synthetic data's fidelity, diversity, and utility, preventing domain shifts that degrade detection performance. This subcategory enables standardized benchmarking, guiding method selection and iterative improvements for trustworthy anomaly synthesis.*

- ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection [2025]

## 💌 Acknowledgement
We acknowledge the Awesome Industrial Anomaly Detection repository for its comprehensive paper list and datasets on industrial image anomaly/defect detection. Big thanks to this amazing open-source work!

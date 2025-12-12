# Awesome-FineGrained-Anomaly-Detection
This repository provides a hierarchical taxonomy of key paperson industrial anomaly detection, surpassing flat lists with fine-grained subcategories that delineate emerging hotspots

## Anomaly Generation

Anomaly Generation refers to the synthesis of artificial anomalous data to augment scarce real-world defect samples in industrial settings. This technique addresses the fundamental challenge of data imbalance in anomaly detection, where normal samples vastly outnumber anomalies, hindering model training.

### Why Anomaly Generation Matters
- **Academic Value**: It advances unsupervised and few-shot learning paradigms by providing diverse, controllable synthetic data for benchmarking and evaluation. Recent surveys highlight its role in exploring generative models (e.g., diffusion-based) and vision-language integration, fostering innovations in evaluation frameworks and cross-domain generalization [[arXiv 2025]](https://arxiv.org/abs/2502.16412).
- **Practical Production Value**: In manufacturing pipelines, it reduces reliance on costly manual labeling, accelerates model training, and improves detection accuracy by simulating rare defects, minimizing downtime and quality control costs.
- **Applicable Domains**: Widely used in industrial inspection (e.g., semiconductors, automotive parts), medical imaging (e.g., tumor simulation), autonomous driving (e.g., road hazard synthesis), and surveillance (e.g., behavioral anomalies).

### Subcategories
- [Improving Generation Speed](#improving-generation-speed)
- [Controllable Image Generation](#controllable-image-generation)
- [Multi-Modal Generation](#multi-modal-generation)
- [Precise Mask](#precise-mask)
- [Generation Quality Judgment and Evaluation System](#generation-quality-judgment-and-evaluation-system)

+ Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation [[arXiv 2025]](https://arxiv.org/abs/2505.09263)
+ Background-Aware Defect Generation for Robust Industrial Anomaly Detection [[arXiv 2024]](https://arxiv.org/abs/2411.16767)
+ SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning [[arXiv 2025]](https://arxiv.org/abs/2410.14987)
+ A Survey on Industrial Anomalies Synthesis [[arXiv 2025]](https://arxiv.org/abs/2502.16412)
+ AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer [[arXiv 2025]](https://arxiv.org/abs/2511.06687)

### Improving Generation Speed {#improving-generation-speed}
Enhancing generation speed is crucial for scaling anomaly synthesis to large datasets, enabling real-time augmentation during training and deployment in high-throughput industrial environments. This reduces computational overhead, making diffusion-based methods viable for resource-constrained settings without sacrificing diversity.

+ FAST: Foreground-Aware Diffusion with Accelerated Sampling [[arXiv 2025]](https://arxiv.org/abs/2509.20295)
+ SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning [[arXiv 2025]](https://arxiv.org/abs/2410.14987)
+ Double Helix Diffusion for Cross-Domain Anomaly Image Generation [[arXiv 2025]](https://arxiv.org/abs/2509.12787)

### Controllable Image Generation {#controllable-image-generation}
Controllability allows precise specification of anomaly types, locations, and attributes via prompts or priors, ensuring synthetic data aligns with domain-specific needs. This boosts model generalization by simulating targeted scenarios, bridging the gap between generic augmentation and real defect variability.

+ Learning Cross-modal Semantic Features for Controllable Anomaly Synthesis [[arXiv 2025]](https://arxiv.org/abs/2412.06510)
+ UniADC: A Unified Framework for Anomaly Detection and Classification [[arXiv 2025]](https://arxiv.org/abs/2511.06644)
+ Double Helix Diffusion for Cross-Domain Anomaly Image Generation [[arXiv 2025]](https://arxiv.org/abs/2509.12787)
+ AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer [[arXiv 2025]](https://arxiv.org/abs/2511.06687)

### Multi-Modal Generation {#multi-modal-generation}
Multi-modal synthesis integrates data from diverse sources (e.g., RGB + depth), capturing richer contextual cues for robust detection in complex scenes. It's vital for handling incomplete or noisy inputs in real-world applications, improving cross-modal fusion and overall system resilience.

+ AD-FM: Multimodal LLMs for Anomaly Detection via Multi-Stage Reasoning and Fine-Grained Reward Optimization [[arXiv 2025]](https://arxiv.org/abs/2508.04175)
+ Robust Modality-Incomplete Anomaly Detection [[arXiv 2025]](https://arxiv.org/abs/2410.01737)
+ OmniAD: Detect and Understand Industrial Anomaly via Multimodal Reasoning [[arXiv 2025]](https://arxiv.org/abs/2505.22039)

### Precise Mask {#precise-mask}
Generating pixel-accurate masks ensures anomalies are spatially aligned with defects, facilitating supervised fine-tuning and precise localization. This is essential for pixel-level tasks like segmentation, reducing false positives and enhancing interpretability in downstream detection pipelines.

+ Double Helix Diffusion for Cross-Domain Anomaly Image Generation [[arXiv 2025]](https://arxiv.org/abs/2509.12787)
+ SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning [[arXiv 2025]](https://arxiv.org/abs/2410.14987)
+ Region-Guided Few-Shot Anomaly Image-Mask Pair Synthesis [[arXiv 2025]](https://arxiv.org/abs/2507.09619)
+ UniADC: A Unified Framework for Anomaly Detection and Classification [[arXiv 2025]](https://arxiv.org/abs/2511.06644)

### Generation Quality Judgment and Evaluation System {#generation-quality-judgment-and-evaluation-system}
Robust evaluation metrics quantify synthetic data's fidelity, diversity, and utility, preventing domain shifts that degrade detection performance. This subcategory enables standardized benchmarking, guiding method selection and iterative improvements for trustworthy anomaly synthesis.

+ ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection [[arXiv 2025]](https://arxiv.org/abs/2510.07927)
+ A Survey on Industrial Anomalies Synthesis [[arXiv 2025]](https://arxiv.org/abs/2502.16412)
+ Formally Exploring Time-Series Anomaly Detection Evaluation Metrics [[arXiv 2025]](https://arxiv.org/abs/2510.17562)

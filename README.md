# Awesome Human Motion

An aggregation of human motion understanding research, feel free to contribute.

[Reviews & Surveys](#review) 

[Motion Generation](#motion-generation)  [Motion Editing](#motion-editing)  [Motion Stylization](#motion-stylization) 

[Human-Object Interaction](#hoi) [Human-Scene Interaction](#hsi)  [Human-Human Interaction](#hhi) 

[Datasets](#datasets) [Humanoid](#humanoid) [Bio-stuff](#bio)


<span id="review"></span>
<details open>
<summary><h2>Reviews & Surveys</h2></summary>
<ul style="margin-left: 5px;">
<li><b>(JEB 2025)</b> <a href="https://journals.biologists.com/jeb/article/228/Suppl_1/JEB248125/367009/Behavioural-energetics-in-human-locomotion-how">McAllister et al</a>: Behavioural energetics in human locomotion: how energy use influences how we move, McAllister et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.12763">Sui et al</a>: A Survey on Human Interaction Motion Generation, Sui et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.13120">Fan et al</a>: 3D Human Interaction Generation: A Survey, Fan et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2501.02116">Gu et al</a>: Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning, Gu et al.</li>
<li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/pdf/2412.10458">Zhao et al</a>: Motion Generation Review: Exploring Deep Learning for Lifelike Animation with Manifold, Zhao et al.</li>
<li><b>(T-PAMI 2023)</b> <a href="https://arxiv.org/abs/2307.10894">Zhu et al</a>: Human Motion Generation: A Survey, Zhu et al.</li>
</ul>
</details>

<span id="motion-generation"></span>
<details open>
<summary><h2>Motion Generation, Text/Speech/Music-Driven</h2></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
        <ul style="margin-left: 5px;">
        <li><b>(CVPR 2025)</b> <a href="https://seokhyeonhong.github.io/projects/salad/">SALAD</a>: SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing, Hong et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://boeun-kim.github.io/page-PersonaBooth/">PersonalBooth</a>: PersonaBooth: Personalized Text-to-Motion Generation, Kim et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2411.16575">MARDM</a>: Rethinking Diffusion for Text-Driven Human Motion Generation, Meng et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2503.04829">StickMotion</a>: StickMotion: Generating 3D Human Motions by Drawing a Stickman, Wang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2411.16805">LLaMo</a>: Human Motion Instruction Tuning, Li et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://star-uu-wang.github.io/HOP/">HOP</a>: HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation, Cheng et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://atom-motion.github.io/">AtoM</a>: AToM: Aligning Text-to-Motion Model at Event-Level with GPT-4Vision Reward, Han et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://jiro-zhang.github.io/EnergyMoGen/">EnergyMoGen</a>: EnergyMoGen: Compositional Human Motion Generation with Energy-Based Diffusion Model in Latent Space, Zhang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://languageofmotion.github.io/">Languate of Motion</a>: The Language of Motion: Unifying Verbal and Non-verbal Language of 3D Human Motion, Chen et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://shunlinlu.github.io/ScaMo/">ScaMo</a>: ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model, Lu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://hhsinping.github.io/Move-in-2D/">Move in 2D</a>: Move-in-2D: 2D-Conditioned Human Motion Generation, Huang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://solami-ai.github.io/">SOLAMI</a>: SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters, Jiang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://lijiaman.github.io/projects/mvlift/">MVLift</a>: Lifting Motion to the 3D World via 2D Diffusion, Li et al.</li>
        <li><b>(ACM Sensys 2025)</b> <a href="https://arxiv.org/pdf/2503.01768">SHADE-AD</a>: SHADE-AD: An LLM-Based Framework for Synthesizing Activity Data of Alzheimer’s Patients, Fu et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://arxiv.org/abs/2410.16623">MotionGlot</a>: MotionGlot: A Multi-Embodied Motion Generation Model, Harithas et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://guytevet.github.io/CLoSD-page/">CLoSD</a>: CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character control, Tevet et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://genforce.github.io/PedGen/">PedGen</a>: Learning to Generate Diverse Pedestrian Movements from Web Videos with Noisy Labels, Liu et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=IEul1M5pyk">HGM³</a>: HGM³: Hierarchical Generative Masked Motion Modeling with Hard Token Mining, Jeong et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=LYawG8YkPa">LaMP</a>: LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning, Li et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=d23EVDRJ6g">MotionDreamer</a>: MotionDreamer: One-to-Many Motion Synthesis with Localized Generative Masked Transformer, Wang et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=Oh8MuCacJW">Lyu et al</a>: Towards Unified Human Motion-Language Understanding via Sparse Interpretable Characterization, Lyu et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://zkf1997.github.io/DART/">DART</a>: DART: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control, Zhao et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://knoxzhao.github.io/Motion-Agent/">Motion-Agent</a>: Motion-Agent: A Conversational Framework for Human Motion Generation with LLMs, Wu et al.</li>
        <li><b>(IJCV 2025)</b> <a href="https://arxiv.org/pdf/2502.05534">Fg-T2M++</a>:。 Fg-T2M++: LLMs-Augmented Fine-Grained Text Driven Human Motion Generation, Wang et al.</li>
        <li><b>(TVCG 2025)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10891181/authors#authors">SPORT</a>: SPORT: From Zero-Shot Prompts to Real-Time Motion Generation, Ji et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="http://inwoohwang.me/SFControl">SFControl</a>: Motion Synthesis with Sparse and Flexible Keyjoint Control, Hwang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.14919">GenM3</a>: GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation, Shi et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://zju3dv.github.io/MotionStreamer/">MotionStreamer</a>: MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space, Xiao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.13859">Less Is More</a>: Less is More: Improving Motion Diffusion Models with Sparse Keyframes, Bae et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.13300">Zeng et al</a>:  Progressive Human Motion Generation Based on Text and Few Motion Frames, Zeng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://mjwei3d.github.io/ACMo/">ACMo</a>: ACMo: Attribute Controllable Motion Generation, Wei et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://jackyu6.github.io/HERO/">HERO</a>: HERO: Human Reaction Generation from Videos, Yu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.06151">BioMoDiffuse</a>: BioMoDiffuse: Physics-Guided Biomechanical Diffusion for Controllable and Authentic Human Motion Synthesis, Kang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.06499">ExGes</a>: ExGes: Expressive Human Motion Retrieval and Modulation for Audio-Driven Gesture Synthesis, Zhou et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.17327">AnyTop</a>: AnyTop: Character Animation Diffusion with Any Topology, Gat et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="hhttps://steve-zeyu-zhang.github.io/MotionAnything/">MotionAnything</a>: Motion Anything: Any to Motion Generation, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.18309">GCDance</a>: GCDance: Genre-Controlled 3D Full Body Dance Generation Driven By Music, Liu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://diouo.github.io/motionlab.github.io/">MotionLab</a>: MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm, Guo et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://cjerry1243.github.io/casim_t2m/">CASIM</a>: CASIM: Composite Aware Semantic Injection for Text to Motion Generation, Chang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2501.19083">MotionPCM</a>: MotionPCM: Real-Time Motion Synthesis with Phased Consistency Model, Jiang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://andypinxinliu.github.io/GestureLSM/">GestureLSM</a>: GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling, Liu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2501.18232">Free-T2M</a>: Free-T2M: Frequency Enhanced Text-to-Motion Diffusion Model With Consistency Loss, Chen et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2501.01449">LS-GAN</a>: LS-GAN: Human Motion Synthesis with Latent-space GANs, Amballa et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/html/2501.16778v1">FlexMotion</a>: FlexMotion: Lightweight, Physics-Aware, and Controllable Human Motion Generation, Tashakori et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.06897">HiSTF Mamba</a>: HiSTF Mamba: Hierarchical Spatiotemporal Fusion with Multi-Granular Body-Spatial Modeling for High-Fidelity Text-to-Motion Generation, Zhan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2501.16551">PackDiT</a>: PackDiT: Joint Human Motion and Text Generation via Mutual Prompting, Jiang et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://coral79.github.io/uni-motion/">Unimotion</a>: Unimotion: Unifying 3D Human Motion Synthesis and Understanding, Li et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://cyk990422.github.io/HoloGest.github.io//">HoloGest</a>: HoleGest: Decoupled Diffusion and Motion Priors for Generating Holisticly Expressive Co-speech Gestures, Cheng et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://hanyangclarence.github.io/unimumo_demo/">UniMuMo</a>: UniMuMo: Unified Text, Music and Motion Generation, Yang et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://arxiv.org/abs/2408.00352">ALERT-Motion</a>: Autonomous LLM-Enhanced Adversarial Attack for Text-to-Motion, Miao et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://cure-lab.github.io/MotionCraft/">MotionCraft</a>: MotionCraft: Crafting Whole-Body Motion with Plug-and-Play Multimodal Controls, Bian et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://arxiv.org/pdf/2412.11193">Light-T2M</a>: Light-T2M: A Lightweight and Fast Model for Text-to-Motion Generation, Zeng et al.</li>
        <li><b>(WACV 2025)</b> <a href="https://reindiffuse.github.io/">ReinDiffuse</a>: ReinDiffuse: Crafting Physically Plausible Motions with Reinforced Diffusion Model, Han et al.</li>
        <li><b>(WACV 2025)</b> <a href="https://motion-rag.github.io/">MoRAG</a>: MoRAG -- Multi-Fusion Retrieval Augmented Generation for Human Motion, Shashank et al.</li>
        <li><b>(WACV 2025)</b> <a href="https://arxiv.org/abs/2409.11920">Mandelli et al</a>: Generation of Complex 3D Human Motion by Temporal and Spatial Composition of Diffusion Models, Mandelli et al.</li>
    </ul></details>
    <details>
    <summary><h3>2024</h3></summary>
        <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://xiangyue-zhang.github.io/SemTalk">SemTalk</a>: SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://inter-dance.github.io/">InterDance</a>: InterDance: Reactive 3D Dance Generation with Realistic Duet Interactions, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://zju3dv.github.io/Motion-2-to-3/">Motion-2-to-3</a>: Motion-2-to-3: Leveraging 2D Motion Data to Boost 3D Motion Generation, Pi et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2412.07797">Mogo</a>: Mogo: RQ Hierarchical Causal Transformer for High-Quality 3D Human Motion Generation, Fu et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://gabrie-l.github.io/coma-page/">CoMA</a>: CoMA: Compositional Human Motion Generation with Multi-modal Agents, Sun et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://sopo-motion.github.io/">SoPo</a>: SoPo: Text-to-Motion Generation Using Semi-Online Preference Optimization, Tan et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/pdf/2412.04343">RMD</a>: RMD: A Simple Baseline for More General Human Motion Generation via Training-free Retrieval-Augmented Motion Diffuse, Liao et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/pdf/2412.00112">BiPO</a>: BiPO: Bidirectional Partial Occlusion Network for Text-to-Motion Synthesis, Hong et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://whwjdqls.github.io/discord.github.io/">DisCoRD</a>: DisCoRD: Discrete Tokens to Continuous Motion via Rectified Flow Decoding, Cho et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.19786">MoTe</a>: MoTe: Learning Motion-Text Diffusion Model for Multiple Generation Tasks, Wue et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.18303">InfiniDreamer</a>: InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation, Zhuo et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.17532">FTMoMamba</a>: FTMoMamba: Motion Generation with Frequency and Text State Space Models, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://andypinxinliu.github.io/KinMo/">KinMo</a>: KinMo: Kinematic-aware Human Motion Understanding and Generation, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.14951">Morph</a>: Morph: A Motion-free Physics Optimization Framework for Human Motion Generation, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://steve-zeyu-zhang.github.io/KMM">KMM</a>: KMM: Key Frame Mask Mamba for Extended Motion Generation, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.21747">MotionGPT-2</a>: MotionGPT-2: A General-Purpose Motion-Language Model for Motion Generation and Understanding, Wang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://li-ronghui.github.io/lodgepp">Lodge++</a>: Lodge++: High-quality and Long Dance Generation with Vivid Choreography Patterns, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.18977">MotionCLR</a>: MotionCLR: Motion Generation and Training-Free Editing via Understanding Attention Mechanisms, Chen et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.14508">LEAD</a>: LEAD: Latent Realignment for Human Motion Diffusion, Andreou et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.08931">Leite et al.</a>: Enhancing Motion Variation in Text-to-Motion Models via Pose and Video Conditioned Editing, Leite et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.06513">MotionRL</a>: MotionRL: Align Text-to-Motion Generation to Human Preferences with Multi-Reward Reinforcement Learning, Liu et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://lhchen.top/MotionLLM/">MotionLLM</a>: MotionLLM: Understanding Human Behaviors from Human Motions and Videos, Chen et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.03311">Wang et al</a>: Quo Vadis, Motion Generation? From Large Language Models to Large Motion Models, Wang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2409.13251">T2M-X</a>: T2M-X: Learning Expressive Text-to-Motion Generation from Partially Annotated Data, Liu et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://github.com/RohollahHS/BAD">BAD</a>: BAD: Bidirectional Auto-regressive Diffusion for Text-to-Motion Generation, Hosseyni et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://von31.github.io/synNsync/">synNsync</a>: Synergy and Synchrony in Couple Dances, Manukele et al.</li>
        <li><b>(EMNLP 2024)</b> <a href="https://aclanthology.org/2024.findings-emnlp.584/">Dong et al</a>: Word-Conditioned 3D American Sign Language Motion Generation, Dong et al.</li>
        <li><b>(NeurIPS D&B 2024)</b> <a href="https://nips.cc/virtual/2024/poster/97700">Text to blind motion</a>: Text to Blind Motion, Kim et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://github.com/xiyuanzh/UniMTS">UniMTS</a>: UniMTS: Unified Pre-training for Motion Time Series, Zhang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://openreview.net/forum?id=FsdB3I9Y24">Christopher et al.</a>: Constrained Synthesis with Projected Diffusion Models, Christopher et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://momu-diffusion.github.io/">MoMu-Diffusion</a>: MoMu-Diffusion: On Learning Long-Term Motion-Music Synchronization and Correspondence, You et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://aigc3d.github.io/mogents/">MoGenTS</a>: MoGenTS: Motion Generation based on Spatial-Temporal Joint Modeling, Yuan et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/abs/2405.16273">M3GPT</a>: M3GPT: An Advanced Multimodal, Multitask Framework for Motion Comprehension and Generation, Luo et al.</li>
        <li><b>(NeurIPS Workshop 2024)</b> <a href="https://openreview.net/forum?id=BTSnh5YdeI">Bikov et al</a>: Fitness Aware Human Motion Generation with Fine-Tuning, Bikov et al.</li>
        <li><b>(NeurIPS Workshop 2024)</b> <a href="https://arxiv.org/pdf/2502.20176">DGFM</a>: DGFM: Full Body Dance Generation Driven by Music Foundation Models, Liu et al.</li>
        <li><b>(ICPR 2024)</b> <a href="https://link.springer.com/chapter/10.1007/978-3-031-78104-9_30">FG-MDM</a>: FG-MDM: Towards Zero-Shot Human Motion Generation via ChatGPT-Refined Descriptions, Shi et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://bohongchen.github.io/SynTalker-Page/">SynTalker</a>: Enabling Synergistic Full-Body Control in Prompt-Based Co-Speech Motion Generation, Chen et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://dl.acm.org/doi/abs/10.1145/3664647.3681487">L3EM</a>: Towards Emotion-enriched Text-to-Motion Generation via LLM-guided Limb-level Emotion Manipulating. Yu et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://dl.acm.org/doi/abs/10.1145/3664647.3681657">StableMoFusion</a>: StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework, Huang et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://dl.acm.org/doi/abs/10.1145/3664647.3681034">SATO</a>: SATO: Stable Text-to-Motion Framework, Chen et al.</li>
        <li><b>(ICANN 2024)</b> <a href="https://link.springer.com/chapter/10.1007/978-3-031-72356-8_2">PIDM</a>: PIDM: Personality-Aware Interaction Diffusion Model for Gesture Generation, Shibasaki et al.</li>
        <li><b>(HFES 2024)</b> <a href="https://journals.sagepub.com/doi/full/10.1177/10711813241262026">Macwan et al</a>: High-Fidelity Worker Motion Simulation With Generative AI, Macwan et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://jpthu17.github.io/GuidedMotion-project/">Jin et al.</a>: Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation, Jin et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/100_ECCV_2024_paper.php">Motion Mamba</a>: Motion Mamba: Efficient and Long Sequence Motion Generation, Zhong et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://frank-zy-dou.github.io/projects/EMDM/index.html">EMDM</a>: EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Human Motion Generation, Zhou et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://yh2371.github.io/como/">CoMo</a>: CoMo: Controllable Motion Generation through Language Guided Pose Code Editing, Huang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://github.com/jsun57/CoMusion">CoMusion</a>: CoMusion: Towards Consistent Stochastic Human Motion Prediction via Motion Diffusion, Sun et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2405.18483">Shan et al.</a>: Towards Open Domain Text-Driven Synthesis of Multi-Person Motions, Shan et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://github.com/qrzou/ParCo">ParCo</a>: ParCo: Part-Coordinating Text-to-Motion Synthesis, Zou et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2407.11532">Sampieri et al.</a>: Length-Aware Motion Synthesis via Latent Diffusion, Sampieri et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://github.com/line/ChronAccRet">ChroAccRet</a>: Chronologically Accurate Retrieval for Temporal Grounding of Motion-Language Models, Fujiwara et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://idigitopia.github.io/projects/mhc/">MHC</a>: Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs, Liu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://github.com/moonsliu/Pro-Motion">ProMotion</a>: Plan, Posture and Go: Towards Open-vocabulary Text-to-Motion Generation, Liu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2406.10740">FreeMotion</a>: FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models, Zhang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://eccv.ecva.net/virtual/2024/poster/266">Text Motion Translator</a>: Text Motion Translator: A Bi-Directional Model for Enhanced 3D Human Motion Generation from Open-Vocabulary Descriptions, Qian et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://vankouf.github.io/FreeMotion/">FreeMotion</a>: FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis, Fan et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://foruck.github.io/KP/">Kinematic Phrases</a>: Bridging the Gap between Human Motion and Action Semantics via Kinematic Phrases, Liu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2404.01700">MotionChain</a>: MotionChain: Conversational Motion Controllers via Multimodal Prompts, Jiang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://neu-vi.github.io/SMooDi/">SMooDi</a>: SMooDi: Stylized Motion Diffusion Model, Zhong et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://exitudio.github.io/BAMM-page/">BAMM</a>: BAMM: Bidirectional Autoregressive Motion Model, Pinyoanuntapong et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://dai-wenxun.github.io/MotionLCM-page/">MotionLCM</a>: MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model, Dai et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2312.10993">Ren et al.</a>: Realistic Human Motion Generation with Cross-Diffusion Models, Ren et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2407.14502">M2D2M</a>: M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models, Chi et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://mingyuan-zhang.github.io/projects/LMM.html">Large Motion Model</a>: Large Motion Model for Unified Multi-Modal Motion Generation, Zhang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://research.nvidia.com/labs/toronto-ai/tesmo/">TesMo</a>: Generating Human Interaction Motions in Scenes with Text Control, Yi et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://tlcontrol.weilinwl.com/">TLcontrol</a>: TLcontrol: Trajectory and Language Control for Human Motion Synthesis, Wan et al.</li>
        <li><b>(ICME 2024)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10687922">ExpGest</a>: ExpGest: Expressive Speaker Generation Using Diffusion Model and Hybrid Audio-Text Guidance, Cheng et al.</li>
        <li><b>(ICME Workshop 2024)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10645445">Chen et al</a>: Anatomically-Informed Vector Quantization Variational Auto-Encoder for Text-to-Motion Generation, Chen et al.</li>
        <li><b>(ICML 2024)</b> <a href="https://github.com/LinghaoChan/HumanTOMATO">HumanTOMATO</a>: HumanTOMATO: Text-aligned Whole-body Motion Generation, Lu et al.</li>
        <li><b>(ICML 2024)</b> <a href="https://sites.google.com/view/gphlvm/">GPHLVM</a>: Bringing Motion Taxonomies to Continuous Domains via GPLVM on Hyperbolic Manifolds, Jaquier et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://setarehc.github.io/CondMDI/">CondMDI</a>: Flexible Motion In-betweening with Diffusion Models, Cohan et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://aiganimation.github.io/CAMDM/">CAMDM</a>: Taming Diffusion Probabilistic Models for Character Control, Chen et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://vcc.tech/research/2024/LGTM">LGTM</a>: LGTM: Local-to-Global Text-Driven Human Motion Diffusion Models, Sun et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://threedle.github.io/TEDi/">TEDi</a>: TEDi: Temporally-Entangled Diffusion for Long-Term Motion Synthesis, Zhang et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://github.com/Yi-Shi94/AMDM">A-MDM</a>: Interactive Character Control with Auto-Regressive Motion Diffusion Models, Shi et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://dl.acm.org/doi/10.1145/3658209">Starke et al.</a>: Categorical Codebook Matching for Embodied Character Controllers, Starke et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://arxiv.org/abs/2407.10481">SuperPADL</a>: SuperPADL: Scaling Language-Directed Physics-Based Control with Progressive Supervised Distillation, Juravsky et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://hanchaoliu.github.io/Prog-MoGen/">ProgMoGen</a>: Programmable Motion Generation for Open-set Motion Control Tasks, Liu et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://github.com/IDC-Flash/PacerPlus">PACER+</a>: PACER+: On-Demand Pedestrian Animation Controller in Driving Scenarios, Wang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://amuse.is.tue.mpg.de/">AMUSE</a>: Emotional Speech-driven 3D Body Animation via Disentangled Latent Diffusion, Chhatre et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://feifeifeiliu.github.io/probtalk/">Liu et al.</a>: Towards Variable and Coordinated Holistic Co-Speech Motion Generation, Liu et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://guytevet.github.io/mas-page/">MAS</a>: MAS: Multi-view Ancestral Sampling for 3D motion generation using 2D diffusion, Kapon et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://wandr.is.tue.mpg.de/">WANDR</a>: WANDR: Intention-guided Human Motion Generation, Diomataris et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://ericguo5513.github.io/momask/">MoMask</a>: MoMask: Generative Masked Modeling of 3D Human Motions, Guo et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://yfeng95.github.io/ChatPose/">ChapPose</a>: ChatPose: Chatting about 3D Human Pose, Feng et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://zixiangzhou916.github.io/AvatarGPT/">AvatarGPT</a>: AvatarGPT: All-in-One Framework for Motion Understanding, Planning, Generation and Beyond, Zhou et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://exitudio.github.io/MMM-page/">MMM</a>: MMM: Generative Masked Motion Model, Pinyoanuntapong et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Li_AAMDM_Accelerated_Auto-regressive_Motion_Diffusion_Model_CVPR_2024_paper.pdf">AAMDM</a>: AAMDM: Accelerated Auto-regressive Motion Diffusion Model, Li et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://tr3e.github.io/omg-page/">OMG</a>: OMG: Towards Open-vocabulary Motion Generation via Mixture of Controllers, Liang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://barquerogerman.github.io/FlowMDM/">FlowMDM</a>: FlowMDM: Seamless Human Motion Composition with Blended Positional Encodings, Barquero et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://digital-life-project.com/">Digital Life Project</a>: Digital Life Project: Autonomous 3D Characters with Social Intelligence, Cai et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://pantomatrix.github.io/EMAGE/">EMAGE</a>: EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling, Liu et al.</li>
        <li><b>(CVPR Workshop 2024)</b> <a href="https://xbpeng.github.io/projects/STMC/index.html">STMC</a>: Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation, Petrovich et al.</li>
        <li><b>(CVPR Workshop 2024)</b> <a href="https://github.com/THU-LYJ-Lab/InstructMotion">InstructMotion</a>: Exploring Text-to-Motion Generation with Human Preference, Sheng et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://sinmdm.github.io/SinMDM-page/">Single Motion Diffusion</a>: Raab et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://openreview.net/forum?id=sOJriBlOFd&noteId=KaJUBoveeo">NeRM</a>: NeRM: Learning Neural Representations for High-Framerate Human Motion Synthesis, Wei et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://priormdm.github.io/priorMDM-page/">PriorMDM</a>: PriorMDM: Human Motion Diffusion as a Generative Prior, Shafir et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://neu-vi.github.io/omnicontrol/">OmniControl</a>: OmniControl: Control Any Joint at Any Time for Human Motion Generation, Xie et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://openreview.net/forum?id=yQDFsuG9HP">Adiya et al.</a>: Bidirectional Temporal Diffusion Model for Temporally Consistent Human Animation, Adiya et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://lisiyao21.github.io/projects/Duolando/">Duolando</a>: Duolando: Follower GPT with Off-Policy Reinforcement Learning for Dance Accompaniment, Li et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://arxiv.org/abs/2312.12227">HuTuDiffusion</a>: HuTuMotion: Human-Tuned Navigation of Latent Motion Diffusion Models with Minimal Feedback, Han et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://arxiv.org/abs/2312.12763">AMD</a>: AMD: Anatomical Motion Diffusion with Interpretable Motion Decomposition and Fusion, Jing et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://nhathoang2002.github.io/MotionMix-page/">MotionMix</a>: MotionMix: Weakly-Supervised Diffusion for Controllable Motion Generation, Hoang et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://github.com/xiezhy6/B2A-HDM">B2A-HDM</a>: Towards Detailed Text-to-Motion Synthesis via Basic-to-Advanced Hierarchical Diffusion Model, Xie et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/27936">Everything2Motion</a>: Everything2Motion: Synchronizing Diverse Inputs via a Unified Framework for Human Motion Synthesis, Fan et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://qiqiapink.github.io/MotionGPT/">MotionGPT</a>: MotionGPT: Finetuned LLMs are General-Purpose Motion Generators, Zhang et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://arxiv.org/abs/2305.13773">Dong et al</a>: Enhanced Fine-grained Motion Diffusion for Text-driven Human Motion Synthesis, Dong et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://evm7.github.io/UNIMASKM-page/">UNIMASKM</a>: A Unified Masked Autoencoder with Patchified Skeletons for Motion Synthesis, Mascaro et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://arxiv.org/abs/2312.10960">B2A-HDM</a>: Towards Detailed Text-to-Motion Synthesis via Basic-to-Advanced Hierarchical Diffusion Model, Xie et al.</li>
        <li><b>(TPAMI 2024)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10399852">GUESS</a>: GUESS: GradUally Enriching SyntheSis for Text-Driven Human Motion Generation, Gao et al.</li>
        <li><b>(WACV 2024)</b> <a href="https://arxiv.org/pdf/2312.12917">Xie et al.</a>: Sign Language Production with Latent Motion Transformer, Xie et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(NeurIPS 2023)</b> <a href="https://github.com/jpthu17/GraphMotion">GraphMotion</a>: Act As You Wish: Fine-grained Control of Motion Diffusion Model with Hierarchical Semantic Graphs, Jin et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://motion-gpt.github.io/">MotionGPT</a>: MotionGPT: Human Motion as Foreign Language, Jiang et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://mingyuan-zhang.github.io/projects/FineMoGen.html">FineMoGen</a>: FineMoGen: Fine-Grained Spatio-Temporal Motion Generation and Editing, Zhang et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://jiawei-ren.github.io/projects/insactor/">InsActor</a>: InsActor: Instruction-driven Physics-based Characters, Ren et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://github.com/ZcyMonkey/AttT2M">AttT2M</a>: AttT2M: Text-Driven Human Motion Generation with Multi-Perspective Attention Mechanism, Zhong et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://mathis.petrovich.fr/tmr">TMR</a>: TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis, Petrovich et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://azadis.github.io/make-an-animation">MAA</a>: Make-An-Animation: Large-Scale Text-conditional 3D Human Motion Generation, Azadi et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://nvlabs.github.io/PhysDiff">PhysDiff</a>: PhysDiff: Physics-Guided Human Motion Diffusion Model, Yuan et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html">ReMoDiffusion</a>: ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model, Zhang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://barquerogerman.github.io/BeLFusion/">BelFusion</a>: BeLFusion: Latent Diffusion for Behavior-Driven Human Motion Prediction, Barquero et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://korrawe.github.io/gmd-project/">GMD</a>: GMD: Guided Motion Diffusion for Controllable Human Motion Synthesis, Karunratanakul et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Aliakbarian_HMD-NeMo_Online_3D_Avatar_Motion_Generation_From_Sparse_Observations_ICCV_2023_paper.html">HMD-NeMo</a>: HMD-NeMo: Online 3D Avatar Motion Generation From Sparse Observations, Aliakbarian et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://sinc.is.tue.mpg.de/">SINC</a>: SINC: Spatial Composition of 3D Human Motions for Simultaneous Action Generation, Athanasiou et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Kong_Priority-Centric_Human_Motion_Generation_in_Discrete_Latent_Space_ICCV_2023_paper.html">Kong et al.</a>: Priority-Centric Human Motion Generation in Discrete Latent Space, Kong et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Fg-T2M_Fine-Grained_Text-Driven_Human_Motion_Generation_via_Diffusion_Model_ICCV_2023_paper.html">FgT2M</a>: Fg-T2M: Fine-Grained Text-Driven Human Motion Generation via Diffusion Model, Wang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Qian_Breaking_The_Limits_of_Text-conditioned_3D_Motion_Synthesis_with_Elaborative_ICCV_2023_paper.html">EMS</a>: Breaking The Limits of Text-conditioned 3D Motion Synthesis with Elaborative Descriptions, Qian et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://weiyuli.xyz/GenMM/">GenMM</a>: Example-based Motion Synthesis via Generative Motion Matching, Li et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://pku-mocca.github.io/GestureDiffuCLIP-Page/">GestureDiffuCLIP</a>: GestureDiffuCLIP: Gesture Diffusion Model with CLIP Latents, Ao et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://i.cs.hku.hk/~taku/kunkun2023.pdf">BodyFormer</a>: BodyFormer: Semantics-guided 3D Body Gesture Synthesis with Transformer, Pang et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://www.speech.kth.se/research/listen-denoise-action/">Alexanderson et al.</a>: Listen, denoise, action! Audio-driven motion synthesis with diffusion models, Alexanderson et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://dulucas.github.io/agrol/">AGroL</a>: Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model, Du et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://talkshow.is.tue.mpg.de/">TALKSHOW</a>: Generating Holistic 3D Human Motion from Speech, Yi et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://mael-zys.github.io/T2M-GPT/">T2M-GPT</a>: T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations, Zhang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://zixiangzhou916.github.io/UDE/">UDE</a>: UDE: A Unified Driving Engine for Human Motion Generation, Zhou et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://github.com/junfanlin/oohmg">OOHMG</a>: Being Comes from Not-being: Open-vocabulary Text-to-Motion Generation with Wordless Training, Lin et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://edge-dance.github.io/">EDGE</a>: EDGE: Editable Dance Generation From Music, Tseng et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://chenxin.tech/mld">MLD</a>: Executing your Commands via Motion Diffusion in Latent Space, Chen et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://sigal-raab.github.io/MoDi">MoDi</a>: MoDi: Unconditional Motion Synthesis from Diverse Data, Raab et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://vcai.mpi-inf.mpg.de/projects/MoFusion/">MoFusion</a>: MoFusion: A Framework for Denoising-Diffusion-based Motion Synthesis, Dabral et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://arxiv.org/abs/2303.14926">Mo et al.</a>: Continuous Intermediate Token Learning with Implicit Motion Manifold for Keyframe Based Motion Interpolation, Mo et al.</li>
        <li><b>(ICLR 2023)</b> <a href="https://guytevet.github.io/mdm-page/">HMDM</a>: MDM: Human Motion Diffusion Model, Tevet et al.</li>
        <li><b>(TPAMI 2023)</b> <a href="https://mingyuan-zhang.github.io/projects/MotionDiffuse.html">MotionDiffuse</a>: MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model, Zhang et al.</li>
        <li><b>(TPAMI 2023)</b> <a href="https://www.mmlab-ntu.com/project/bailando/">Bailando++</a>: Bailando++: 3D Dance GPT with Choreographic Memory, Li et al.</li>
        <li><b>(ArXiv 2023)</b> <a href="https://zixiangzhou916.github.io/UDE-2/">UDE-2</a>: A Unified Framework for Multimodal, Multi-Part Human Motion Synthesis, Zhou et al.</li>
        <li><b>(ArXiv 2023)</b> <a href="https://pjyazdian.github.io/MotionScript/">Motion Script</a>: MotionScript: Natural Language Descriptions for Expressive 3D Human Motions, Yazdian et al.</li>
    </ul></details>
    <details>
    <summary><h3>2022 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/c-he/NeMF">NeMF</a>: NeMF: Neural Motion Fields for Kinematic Animation, He et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://github.com/nv-tlabs/PADL">PADL</a>: PADL: Language-Directed Physics-Based Character, Juravsky et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://pku-mocca.github.io/Rhythmic-Gesticulator-Page/">Rhythmic Gesticulator</a>: Rhythmic Gesticulator: Rhythm-Aware Co-Speech Gesture Synthesis with Hierarchical Neural Embeddings, Ao et al.</li>
        <li><b>(3DV 2022)</b> <a href="https://teach.is.tue.mpg.de/">TEACH</a>: TEACH: Temporal Action Composition for 3D Human, Athanasiou et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://github.com/PACerv/ImplicitMotion">Implicit Motion</a>: Implicit Neural Representations for Variable Length Human Motion Generation, Cervantes et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810707.pdf">Zhong et al.</a>: Learning Uncoupled-Modulation CVAE for 3D Action-Conditioned Human Motion Synthesis, Zhong et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://guytevet.github.io/motionclip-page/">MotionCLIP</a>: MotionCLIP: Exposing Human Motion Generation to CLIP Space, Tevet et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://europe.naverlabs.com/research/computer-vision/posegpt">PoseGPT</a>: PoseGPT: Quantizing human motion for large scale generative modeling, Lucas et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://mathis.petrovich.fr/temos/">TEMOS</a>: TEMOS: Generating diverse human motions from textual descriptions, Petrovich et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://ericguo5513.github.io/TM2T/">TM2T</a>: TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts, Guo et al.</li>
        <li><b>(SIGGRAPH 2022)</b> <a href="https://hongfz16.github.io/projects/AvatarCLIP.html">AvatarCLIP</a>: AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars, Hong et al.</li>
        <li><b>(SIGGRAPH 2022)</b> <a href="https://dl.acm.org/doi/10.1145/3528223.3530178">DeepPhase</a>: Deepphase: Periodic autoencoders for learning motion phase manifolds, Starke et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://ericguo5513.github.io/text-to-motion">Guo et al.</a>: Generating Diverse and Natural 3D Human Motions from Text, Guo et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://www.mmlab-ntu.com/project/bailando/">Bailando</a>: Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory, Li et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://mathis.petrovich.fr/actor/index.html">ACTOR</a>: Action-Conditioned 3D Human Motion Synthesis with Transformer VAE, Petrovich et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://google.github.io/aichoreographer/">AIST++</a>: AI Choreographer: Music Conditioned 3D Dance Generation with AIST++, Li et al.</li>
        <li><b>(SIGGRAPH 2021)</b> <a href="https://dl.acm.org/doi/10.1145/3450626.3459881">Starke et al.</a>: Neural animation layering for synthesizing martial arts movements, Starke et al.</li>
        <li><b>(CVPR 2021)</b> <a href="https://yz-cnsdqz.github.io/eigenmotion/MOJO/index.html">MOJO</a>: We are More than Our Joints: Predicting how 3D Bodies Move, Zhang et al.</li>
        <li><b>(ECCV 2020)</b> <a href="https://www.ye-yuan.com/dlow">DLow</a>: DLow: Diversifying Latent Flows for Diverse Human Motion Prediction, Yuan et al.</li>
        <li><b>(SIGGRAPH 2020)</b> <a href="https://www.ipab.inf.ed.ac.uk/cgvu/basketball.pdf">Starke et al.</a>: Local motion phases for learning multi-contact character movements, Starke et al.</li>
    </ul></details>
</ul></details>

<span id="motion-editing"></span>
<details open>
<summary><h2>Motion Editing</h2></summary>
<ul style="margin-left: 5px;">
    <li><b>(CVPR 2025)</b> <a href="https://kwanyun.github.io/AnyMoLe_page/">AnyMoLe</a>: AnyMoLe: Any Character Motion In-Betweening Leveraging Video Diffusion Models, Yun et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.08180">Dai et al</a>: Towards Synthesized and Editable Motion In-Betweening Through Part-Wise Phase Representation, Dai et al.</li>
    <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://motionfix.is.tue.mpg.de/">MotionFix</a>: MotionFix: Text-Driven 3D Human Motion Editing, Athanasiou et al.</li>
    <li><b>(NeurIPS 2024)</b> <a href="https://btekin.github.io/">CigTime</a>: CigTime: Corrective Instruction Generation Through Inverse Motion Editing, Fang et al.</li>
    <li><b>(SIGGRAPH 2024)</b> <a href="https://purvigoel.github.io/iterative-motion-editing/">Iterative Motion Editing</a>: Iterative Motion Editing with Natural Language, Goel et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://korrawe.github.io/dno-project/">DNO</a>: DNO: Optimizing Diffusion Noise Can Serve As Universal Motion Priors, Karunratanakul et al.</li>
</ul></details>

<span id="motion-stylization"></span>
<details open>
<summary><h2>Motion Stylization</h2></summary>
<ul style="margin-left: 5px;">
    <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/pdf/2412.09901">MulSMo</a>: MulSMo: Multimodal Stylized Motion Generation by Bidirectional Control Flow, Li et al.</li>
    <li><b>(TSMC 2024)</b> <a href="https://arxiv.org/pdf/2412.04097">D-LORD</a>: D-LORD for Motion Stylization, Gupta et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://otaheri.github.io/publication/2024_humos/">HUMOS</a>: HUMOS: Human Motion Model Conditioned on Body Shape, Tripathi et al.</li>
    <li><b>(SIGGRAPH 2024)</b> <a href="https://dl.acm.org/doi/10.1145/3641519.3657457">SMEAR</a>: SMEAR: Stylized Motion Exaggeration with ARt-direction, Basset et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://xingliangjin.github.io/MCM-LDM-Web/">MCM-LDM</a>: Arbitrary Motion Style Transfer with Multi-condition Motion Latent Diffusion Model, Song et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://boeun-kim.github.io/page-MoST/">MoST</a>: MoST: Motion Style Transformer between Diverse Action Contents, Kim et al.</li>
    <li><b>(ICLR 2024)</b> <a href="https://yxmu.foo/GenMoStyle/">GenMoStyle</a>: Generative Human Motion Stylization in Latent Space, Guo et al.</li>
</ul></details>

<span id="hoi"></span>
<details open>
<summary><h2>Human-Object Interaction</h2></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(Bioengineering 2025)</b> <a href="https://www.mdpi.com/2306-5354/12/3/317">MeLLO</a>: The Utah Manipulation and Locomotion of Large Objects (MeLLO) Data Library, Luttmer et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2503.13130">ChainHOI</a>: ChainHOI: Joint-based Kinematic Chain Modeling for Human-Object Interaction Generation, Zeng et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://4dvlab.github.io/project_page/semgeomo/">SemGeoMo</a>: SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance, Cong et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://liyitang22.github.io/phys-reach-grasp/">Phys-Reach-Grasp</a>: Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References, Li et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://jlogkim.github.io/parahome/">ParaHome</a>: ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions, Kim et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://sirui-xu.github.io/InterMimic/">InterMimic</a>: InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions, Xu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://core4d.github.io/">CORE4D</a>: CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement, Zhang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://ingrid789.github.io/SkillMimic/">SkillMimic</a>: SkillMimic: Learning Reusable Basketball Skills from Demonstrations, Wang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2501.04595">MobileH2R</a>: MobileH2R: Learning Generalizable Human to Mobile Robot Handover Exclusively from Scalable and Diverse Synthetic Data, Wang et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://iscas3dv.github.io/DiffGrasp/">DiffGrasp</a>: Diffgrasp: Whole-Body Grasping Synthesis Guided by Object Motion Using a Diffusion Model, Zhang et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://arxiv.org/pdf/2408.16770?">Paschalidis et al</a>: 3D Whole-body Grasp Synthesis with Directional Controllability, Paschalidis et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/InterTrack/">InterTrack</a>: InterTrack: Tracking Human Object Interaction without Object Templates, Xie et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://arxiv.org/abs/2403.11237">FORCE</a>: FORCE: Dataset and Method for Intuitive Physics Guided Human-object Interaction, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.05231">Kaiwu</a>: Kaiwu: A Multimodal Manipulation Dataset and Framework for Robot Learning and Human-Robot Interaction, Jiang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.00382">EigenActor</a>:  EigenActor: Variant Body-Object Interaction Generation Evolved from Invariant Action Basis Reasoning, Guo et al.</li>
    </ul></details>
    <details>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://syncdiff.github.io/">SyncDiff</a>: SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis, He et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/pdf/2412.06702">CHOICE</a>: CHOICE: Coordinated Human-Object Interaction in Cluttered Environments for Pick-and-Place Actions, Lu et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/tridi/">TriDi</a>: TriDi: Trilateral Diffusion of 3D Humans, Objects and Interactions, Petrov et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://nickk0212.github.io/ood-hoi/">OOD-HOI</a>: OOD-HOI: Text-Driven 3D Whole-Body Human-Object Interactions Generation Beyond Training Domains, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2409.20502">COLLAGE</a>: COLLAGE: Collaborative Human-Agent Interaction Generation using Hierarchical Latent Diffusion and Language Models, Daiya et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.16216">SMGDiff</a>: SMGDiff: Soccer Motion Generation using diffusion probabilistic models, Yang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://hoifhli.github.io/">Wu et al</a>: Human-Object Interaction from Human-Level Instructions, Wu et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/abs/2406.19972">HumanVLA</a>: HumanVLA: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid, Xu et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://www.zhengyiluo.com/Omnigrasp-Site/">OmniGrasp</a>: Grasping Diverse Objects with Simulated Humanoids, Luo et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://yyvhang.github.io/EgoChoir/">EgoChoir</a>: EgoChoir: Capturing 3D Human-Object Interaction Regions from Egocentric Views, Yang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://gao-jiawei.com/Research/CooHOI/">CooHOI</a>: CooHOI: Learning Cooperative Human-Object Interaction with Manipulated Object Dynamics, Gao et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/abs/2403.19652">InterDreamer</a>: InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction, Xu et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://pimforce.hcitech.org/">PiMForce</a>: Posture-Informed Muscular Force Learning for Robust Hand Pressure Estimation, Seo et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://sisidai.github.io/InterFusion/">InterFusion</a>: InterFusion: Text-Driven Generation of 3D Human-Object Interaction, Dai et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://lijiaman.github.io/projects/chois/">CHOIS</a>: Controllable Human-Object Interaction Synthesis, Li et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://f-hoi.github.io/">F-HOI</a>: F-HOI: Toward Fine-grained Semantic-Aligned 3D Human-Object Interactions, Yang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://lvxintao.github.io/himo/">HIMO</a>: HIMO: A New Benchmark for Full-Body Human Interacting with Multiple Objects, Lv et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://jiashunwang.github.io/PhysicsPingPong/">PhysicsPingPong</a>: Strategy and Skill Learning for Physics-based Table Tennis Animation, Wang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://nileshkulkarni.github.io/nifty/">NIFTY</a>: NIFTY: Neural Object Interaction Fields for Guided Human Motion Synthesis, Kulkarni et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://zxylinkstart.github.io/HOIAnimator-Web/">HOI Animator</a>: HOIAnimator: Generating Text-Prompt Human-Object Animations using Novel Perceptive Diffusion Models, Son et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://cg-hoi.christian-diller.de/#main">CG-HOI</a>: CG-HOI: Contact-Guided 3D Human-Object Interaction Generation, Diller et al.</li>
        <li><b>(IJCV 2024)</b> <a href="https://intercap.is.tue.mpg.de/">InterCap</a>: InterCap: Joint Markerless 3D Tracking of Humans and Objects in Interaction, Huang et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://eth-ait.github.io/phys-fullbody-grasp/">Phys-Fullbody-Grasp</a>: Physically Plausible Full-Body Hand-Object Interaction Synthesis, Braun et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://grip.is.tue.mpg.de">GRIP</a>: GRIP: Generating Interaction Poses Using Spatial Cues and Latent Consistency, Taheri et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://kailinli.github.io/FAVOR/">FAVOR</a>: Favor: Full-Body AR-driven Virtual Object Rearrangement Guided by Instruction Text, Li et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://github.com/lijiaman/omomo_release">OMOMO</a>: Object Motion Guided Human Motion Synthesis, Li et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://jnnan.github.io/project/chairs/">CHAIRS</a>: Full-Body Articulated Human-Object Interaction, Jiang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://zju3dv.github.io/hghoi">HGHOI</a>: Hierarchical Generation of Human-Object Interactions with Diffusion Probabilistic Models, Pi et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://sirui-xu.github.io/InterDiff/">InterDiff</a>: InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion, Xu et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/object_popup/">Object Pop Up</a>: Object pop-up: Can we infer 3D objects and their poses from human interactions alone? Petrov et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://arctic.is.tue.mpg.de/">ARCTIC</a>: A Dataset for Dexterous Bimanual Hand-Object Manipulation, Fan et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630001.pdf">TOCH</a>: TOCH: Spatio-Temporal Object-to-Hand Correspondence for Motion Refinement, Zhou et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/couch/">COUCH</a>: COUCH: Towards Controllable Human-Chair Interactions, Zhang et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://jiahaoplus.github.io/SAGA/saga.html">SAGA</a>: SAGA: Stochastic Whole-Body Grasping with Contact, Wu et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://goal.is.tue.mpg.de/">GOAL</a>: GOAL: Generating 4D Whole-Body Motion for Hand-Object Grasping, Taheri et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/behave/">BEHAVE</a>: BEHAVE: Dataset and Method for Tracking Human Object Interactions, Bhatnagar et al.</li>
        <li><b>(ECCV 2020)</b> <a href="https://grab.is.tue.mpg.de/">GRAB</a>: GRAB: A Dataset of Whole-Body Human Grasping of Objects, Taheri et al.</li>
    </ul></details>
</ul></details>

<span id="hsi"></span>
<details open>
<summary><h2>Human-Scene Interaction</h2></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2412.10235">EnvPoser</a>: EnvPoser: Environment-aware Realistic Human Motion Estimation from Sparse Observations with Uncertainty Modeling. Xia et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://github.com/WindVChen/Sitcom-Crafter">Sitcom-Crafter</a>: Sitcom-Crafter: A Plot-Driven Human Motion Generation System in 3D Scenes, Chen et al. </li>
        <li><b>(3DV 2025)</b> <a href="https://arxiv.org/pdf/2408.16770?">Paschalidis et al</a>: 3D Whole-body Grasp Synthesis with Directional Controllability, Paschalidis et al.</li>
        <li><b>(WACV 2025)</b> <a href="https://arxiv.org/abs/2405.18438">GHOST</a>: GHOST: Grounded Human Motion Generation with Open Vocabulary Scene-and-Text Contexts, Milacski et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="http://inwoohwang.me/SceneMI">SceneMI</a>: SceneMI: Motion In-Betweening for Modeling Human-Scene Interactions, Hwang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.12955">HIS-GPT</a>: HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding, Zhao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.00371">Gao et al</a>: Jointly Understand Your Command and Intention: Reciprocal Co-Evolution between Scene-Aware 3D Human Motion Synthesis and Analysis, Gao et al.</li>
    </ul></details>
    <details open>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://awfuact.github.io/zerohsi/">ZeroHSI</a>: ZeroHSI: Zero-Shot 4D Human-Scene Interaction by Video Generation, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://mimicking-bench.github.io/">Mimicking-Bench</a>: Mimicking-Bench: A Benchmark for Generalizable Humanoid-Scene Interaction Learning via Human Mimicking, Liu et al. </li>
        <li><b>(ArXiv 2024)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/scenic/">SCENIC</a>: SCENIC: Scene-aware Semantic Navigation with Instruction-guided Control, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://jingyugong.github.io/DiffusionImplicitPolicy/">Diffusion Implicit Policy</a>: Diffusion Implicit Policy for Unpaired Scene-aware Motion synthesis, Gong et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.19921">SIMS</a>: SIMS: Simulating Human-Scene Interactions with Real World Script Planning, Wang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://github.com/4DVLab/LaserHuman">LaserHuman</a>: LaserHuman: Language-guided Scene-aware Human Motion Generation in Free Environment, Cong et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://lingomotions.com/">LINGO</a>: Autonomous Character-Scene Interaction Synthesis from Text Instruction, Jiang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://sites.google.com/view/dimop3d">DiMoP3D</a>: Harmonizing Stochasticity and Determinism: Scene-responsive Diverse Human Motion Prediction, Lou et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/html/2312.02700v2">Liu et al.</a>: Revisit Human-Scene Interaction via Space Occupancy, Liu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://research.nvidia.com/labs/toronto-ai/tesmo/">TesMo</a>: Generating Human Interaction Motions in Scenes with Text Control, Yi et al.</li>
        <li><b>(ECCV 2024 Workshop)</b> <a href="https://github.com/felixbmuller/SAST">SAST</a>: Massively Multi-Person 3D Human Motion Forecasting with Scene Context, Mueller et al.</li>
        <li><b>(Eurographics 2024)</b> <a href="https://diglib.eg.org/server/api/core/bitstreams/f1072102-82a6-4228-a140-9ccf09f21077/content">Kang et al</a>: Learning Climbing Controllers for Physics-Based Characters, Kang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://afford-motion.github.io/">Afford-Motion</a>: Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance, Wang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://craigleili.github.io/projects/genzi/">GenZI</a>: GenZI: Zero-Shot 3D Human-Scene Interaction Generation, Li et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://zju3dv.github.io/text_scene_motion/">Cen et al.</a>: Generating Human Motion in 3D Scenes from Text Descriptions, Cen et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://jnnan.github.io/trumans/">TRUMANS</a>: Scaling Up Dynamic Human-Scene Interaction Modeling, Jiang et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://xizaoqu.github.io/unihsi/">UniHSI</a>: UniHSI: Unified Human-Scene Interaction via Prompted Chain-of-Contacts, Xiao et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://arxiv.org/pdf/2404.12942">Purposer</a>: Purposer: Putting Human Motion Generation in Context, Ugrinovic et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10550906">InterScene</a>: Synthesizing Physically Plausible Human Motions in 3D Scenes, Pan et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://arxiv.org/abs/2304.02061">Mir et al</a>: Generating Continual Human Motion in Diverse 3D Scenes, Mir et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ICCV 2023)</b> <a href="https://github.com/zkf1997/DIMOS">DIMOS</a>: DIMOS: Synthesizing Diverse Human Motions in 3D Indoor Scenes, Zhao et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://jiyewise.github.io/projects/LAMA/">LAMA</a>: Locomotion-Action-Manipulation: Synthesizing Human-Scene Interactions in Complex 3D Environments, Lee et al.</li>
        <li><b>(ICCV 2023)</b> <a href="http://cic.tju.edu.cn/faculty/likun/projects/Narrator">Narrator</a>: Narrator: Towards Natural Control of Human-Scene Interaction Generation via Relationship Reasoning, Xuan et al.</li>
        <li><b>(CVPR 2023)</b> <a href="http://www.lidarhumanmotion.net/cimi4d">CIMI4D</a>: CIMI4D: A Large Multimodal Climbing Motion Dataset under Human-Scene Interactions, Yan et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://people.mpi-inf.mpg.de/~jianwang/projects/sceneego/">Scene-Ego</a>: Scene-aware Egocentric 3D Human Pose Estimation, Wang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="http://www.lidarhumanmotion.net/sloper4d">SLOPER4D</a>: SLOPER4D: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments, Dai et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://stanford-tml.github.io/circle_dataset/">CIRCLE</a>: CIRCLE: Capture in Rich Contextual Environments, Araujo et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://scenediffuser.github.io/">SceneDiffuser</a>: Diffusion-based Generation, Optimization, and Planning in 3D Scenes, Huang et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://github.com/jinseokbae/pmp">PMP</a>: PMP: Learning to Physically Interact with Environments using Part-wise Motion Priors, Bae et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://dl.acm.org/doi/10.1145/3588432.3591504">QuestEnvSim</a>: QuestEnvSim: Environment-Aware Simulated Motion Tracking from Sparse Sensors, Lee et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://research.nvidia.com/publication/2023-08_synthesizing-physical-character-scene-interactions">Hassan et al.</a>: Synthesizing Physical Character-Scene Interactions, Hassan et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/wei-mao-2019/ContAwareMotionPred">Mao et al.</a>: Contact-Aware Human Motion Forecasting, Mao et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/Silverster98/HUMANISE">HUMANISE</a>: HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes, Wang et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/ZhengyiLuo/EmbodiedPose">EmbodiedPose</a>: Embodied Scene-aware Human Pose Estimation, Luo et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://github.com/y-zheng18/GIMO">GIMO</a>: GIMO: Gaze-Informed Human Motion Prediction in Context, Zheng et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://zkf1997.github.io/COINS/index.html">COINS</a>: COINS: Compositional Human-Scene Interaction Synthesis with Semantic Control, Zhao et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Towards_Diverse_and_Natural_Scene-Aware_3D_Human_Motion_Synthesis_CVPR_2022_paper.pdf">Wang et al.</a>: Towards Diverse and Natural Scene-aware 3D Human Motion Synthesis, Wang et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://yz-cnsdqz.github.io/eigenmotion/GAMMA/">GAMMA</a>: The Wanderings of Odysseus in 3D Scenes, Zhang et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://samp.is.tue.mpg.de/">SAMP</a>: Stochastic Scene-Aware Motion Prediction, Hassan et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://sanweiliti.github.io/LEMO/LEMO.html">LEMO</a>: Learning Motion Priors for 4D Human Body Capture in 3D Scenes, Zhang et al. </li>
        <li><b>(3DV 2020)</b> <a href="https://sanweiliti.github.io/PLACE/PLACE.html">PLACE</a>: PLACE: Proximity Learning of Articulation and Contact in 3D Environments, Zhang et al.</li>
        <li><b>(SIGGRAPH 2020)</b> <a href="https://www.ipab.inf.ed.ac.uk/cgvu/basketball.pdf">Starke et al.</a>: Local motion phases for learning multi-contact character movements, Starke et al.</li>
        <li><b>(CVPR 2020)</b> <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Generating_3D_People_in_Scenes_Without_People_CVPR_2020_paper.pdf">PSI</a>: Generating 3D People in Scenes without People, Zhang et al.</li>
        <li><b>(SIGGRAPH Asia 2019)</b> <a href="https://www.ipab.inf.ed.ac.uk/cgvu/nsm.pdf">NSM</a>: Neural State Machine for Character-Scene Interactions, Starke et al.</li>
        <li><b>(ICCV 2019)</b> <a href="https://prox.is.tue.mpg.de/">PROX</a>: Resolving 3D Human Pose Ambiguities with 3D Scene Constraints, Hassan et al.</li>
    </ul></details>
</ul></details>

<span id="hhi"></span>
<details open>
<summary><h2>Human-Human Interaction</h2></summary>
<ul style="margin-left: 5px;">
    <li><b>(CVPR 2025)</b> <a href="https://aigc-explorer.github.io/TIMotion-page/">TIMotion</a>: TIMotion: Temporal and Interactive Framework for Efficient Human-Human Motion Generation, Wang et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=UxzKcIZedp">Think Then React</a>: Think Then React: Towards Unconstrained Action-to-Reaction Motion Generation, Tan et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://zju3dv.github.io/ready_to_react/">Ready-to-React</a>: Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation, Cen et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://gohar-malik.github.io/intermask">InterMask</a>: InterMask: 3D Human Interaction Generation via Collaborative Masked Modelling, Javed et al.</li>
    <li><b>(3DV 2025)</b> <a href="https://arxiv.org/abs/2312.08983">Interactive Humanoid</a>: Interactive Humanoid: Online Full-Body Motion Reaction Synthesis with Social Affordance Canonicalization and Forecasting, Liu et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.13120">Fan et al</a>: 3D Human Interaction Generation: A Survey, Fan et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.04816">Invisible Strings</a>: Invisible Strings: Revealing Latent Dancer-to-Dancer Interactions with Graph Neural Networks, Zerkowski et al. </li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.11563">Leader and Follower</a>: Leader and Follower: Interactive Motion Generation under Trajectory Constraints, Wang et al. </li>
    <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/pdf/2412.16670">Two in One</a>: Two-in-One: Unified Multi-Person Interactive Motion Generation by Latent Diffusion Transformer, Li et al.</li>
    <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/pdf/2412.02419">It Takes Two</a>: It Takes Two: Real-time Co-Speech Two-person’s Interaction Generation via Reactive Auto-regressive Diffusion Model, Shi et al.</li>
    <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2409.20502">COLLAGE</a>: COLLAGE: Collaborative Human-Agent Interaction Generation using Hierarchical Latent Diffusion and Language Models, Daiya et al.</li>
    <li><b>(NeurIPS 2024)</b> <a href="https://jyuntins.github.io/harmony4d/">Harmony4D</a>: Harmony4D: A Video Dataset for In-The-Wild Close Human Interactions, Khirodkar et al.</li>
    <li><b>(NeurIPS 2024)</b> <a href="https://github.com/zhenzhiwang/intercontrol">InterControl</a>: InterControl: Generate Human Motion Interactions by Controlling Every Joint, Wang et al.</li>
    <li><b>(ACM MM 2024)</b> <a href="https://yunzeliu.github.io/PhysReaction/">PhysReaction</a>: PhysReaction: Physically Plausible Real-Time Humanoid Reaction Synthesis via Forward Dynamics Guided 4D Imitation, Liu et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2405.18483">Shan et al</a>: Towards Open Domain Text-Driven Synthesis of Multi-Person Motions, Shan et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://vcai.mpi-inf.mpg.de/projects/remos/">ReMoS</a>: ReMoS: 3D Motion-Conditioned Reaction Synthesis for Two-Person Interactions, Ghosh et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://liangxuy.github.io/inter-x/">Inter-X</a>: Inter-X: Towards Versatile Human-Human Interaction Analysis, Xu et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://github.com/liangxuy/ReGenNet">ReGenNet</a>: ReGenNet: Towards Human Action-Reaction Synthesis, Xu et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Fang_Capturing_Closely_Interacted_Two-Person_Motions_with_Reaction_Priors_CVPR_2024_paper.pdf">Fang et al.</a>: Capturing Closely Interacted Two-Person Motions with Reaction Priors, Fan et al.</li>
    <li><b>(CVPR Workshop 2024)</b> <a href="https://openaccess.thecvf.com/content/CVPR2024W/HuMoGen/html/Ruiz-Ponce_in2IN_Leveraging_Individual_Information_to_Generate_Human_INteractions_CVPRW_2024_paper.html">in2IN</a>: in2IN: Leveraging Individual Information to Generate Human INteractions, Ruiz-Ponce et al.</li>
    <li><b>(IJCV 2024)</b> <a href="https://tr3e.github.io/intergen-page/">InterGen</a>: InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions, Liang et al.</li>
    <li><b>(ICCV 2023)</b> <a href="https://liangxuy.github.io/actformer/">ActFormer</a>: ActFormer: A GAN-based Transformer towards General Action-Conditioned 3D Human Motion Generation, Xu et al.</li>
    <li><b>(ICCV 2023)</b> <a href="https://github.com/line/Human-Interaction-Generation">Tanaka et al.</a>: Role-aware Interaction Generation from Textual Description, Tanaka et al.</li>
    <li><b>(CVPR 2023)</b> <a href="https://yifeiyin04.github.io/Hi4D/">Hi4D</a>: Hi4D: 4D Instance Segmentation of Close Human Interaction, Yin et al.</li>
    <li><b>(CVPR 2022)</b> <a href="https://github.com/GUO-W/MultiMotion">ExPI</a>: Multi-Person Extreme Motion Prediction, Guo et al.</li>
    <li><b>(CVPR 2020)</b> <a href="https://ci3d.imar.ro/home">CHI3D</a>: Three-Dimensional Reconstruction of Human Interactions, Fieraru et al.</li>
</ul></details>

<span id="datasets"></span>
<details open>
<summary><h2>Datasets & Benchmarks</h2></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(Scientific Data 2024)</b> <a href="https://www.nature.com/articles/s41597-024-03144-z">MultiSenseBadminton</a>: MultiSenseBadminton: Wearable Sensor–Based Biomechanical Dataset for Evaluation of Badminton Performance, Seong et al.</li>
        <li><b>(Bioengineering 2025)</b> <a href="https://www.mdpi.com/2306-5354/12/3/317">MeLLO</a>: The Utah Manipulation and Locomotion of Large Objects (MeLLO) Data Library, Luttmer et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://atom-motion.github.io/">AtoM</a>: AToM: Aligning Text-to-Motion Model at Event-Level with GPT-4Vision Reward, Han et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://core4d.github.io/">CORE4D</a>: CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement, Zhang et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://motioncritic.github.io/">MotionCritic</a>: Aligning Human Motion Generation with Human Perceptions, Wang et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=9mBodivRIo">LocoVR</a>: LocoVR: Multiuser Indoor Locomotion Dataset in Virtual Reality, Takeyama et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://github.com/coding-rachal/PMRDataset">PMR</a>: Pedestrian Motion Reconstruction: A Large-scale Benchmark via Mixed Reality Rendering with Multiple Perspectives and Modalities, Wang et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://arxiv.org/abs/2408.17168">EMHI</a>: EMHI: A Multimodal Egocentric Human Motion Dataset with HMD and Body-Worn IMUs, Fan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.06522">SGA-INTERACT</a>: SGA-INTERACT: A3DSkeleton-based Benchmark for Group Activity Understanding in Modern Basketball Tactic, Yang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.05231">Kaiwu</a>: Kaiwu: A Multimodal Manipulation Dataset and Framework for Robot Learning and Human-Robot Interaction, Jiang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2501.05098">Motion-X++</a>: Motion-X++: A Large-Scale Multimodal 3D Whole-body Human Motion Dataset, Zhang et al.</li>
    </ul></details>
    <details open>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://mimicking-bench.github.io/">Mimicking-Bench</a>: Mimicking-Bench: A Benchmark for Generalizable Humanoid-Scene Interaction Learning via Human Mimicking, Liu et al. </li>
        <li><b>(ArXiv 2024)</b> <a href="https://github.com/4DVLab/LaserHuman">LaserHuman</a>: LaserHuman: Language-guided Scene-aware Human Motion Generation in Free Environment, Cong et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/scenic/">SCENIC</a>: SCENIC: Scene-aware Semantic Navigation with Instruction-guided Control, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://von31.github.io/synNsync/">synNsync</a>: Synergy and Synchrony in Couple Dances, Manukele et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://github.com/liangxuy/MotionBank">MotionBank</a>: MotionBank: A Large-scale Video Motion Benchmark with Disentangled Rule-based Annotations, Xu et al.</li>
        <li><b>(Github 2024)</b> <a href="https://github.com/fyyakaxyy/AnimationGPT">CMP & CMR</a>: AnimationGPT: An AIGC tool for generating game combat motion assets, Liao et al.</li>
        <li><b>(Scientific Data 2024)</b> <a href="https://www.nature.com/articles/s41597-024-04077-3?fromPaywallRec=false">Evans et al</a>: Synchronized Video, Motion Capture and Force Plate Dataset for Validating Markerless Human Movement Analysis, Evans et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://lingomotions.com/">LINGO</a>: Autonomous Character-Scene Interaction Synthesis from Text Instruction, Jiang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://jyuntins.github.io/harmony4d/">Harmony4D</a>: Harmony4D: A Video Dataset for In-The-Wild Close Human Interactions, Khirodkar et al.</li>
        <li><b>(NeurIPS D&B 2024)</b> <a href="https://siplab.org/projects/EgoSim">EgoSim</a>: EgoSim: An Egocentric Multi-view Simulator for Body-worn Cameras during Human Motion, Hollidt et al.</li>
        <li><b>(NeurIPS D&B 2024)</b> <a href="https://simplexsigil.github.io/mint">Muscles in Time</a>: Muscles in Time: Learning to Understand Human Motion by Simulating Muscle Activations, Schneider et al.</li>
        <li><b>(NeurIPS D&B 2024)</b> <a href="https://blindways.github.io/">Text to blind motion</a>: Text to blind motion, Kim et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://dl.acm.org/doi/abs/10.1145/3664647.3685523">CLaM</a>: CLaM: An Open-Source Library for Performance Evaluation of Text-driven Human Motion Generation, Chen et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://addbiomechanics.org/">AddBiomechanics</a>: AddBiomechanics Dataset: Capturing the Physics of Human Motion at Scale, Werling et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://4dvlab.github.io/project_page/LiveHPS2.html">LiveHPS++</a>: LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment, Ren et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://signavatars.github.io/">SignAvatars</a>: SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark, Yu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://www.projectaria.com/datasets/nymeria">Nymeria</a>: Nymeria: A massive collection of multimodal egocentric daily motion in the wild, Ma et al.</li>
        <li><b>(Multibody System Dynamics 2024)</b> <a href="https://github.com/ainlamyae/Human3.6Mplus">Human3.6M+</a>: Using musculoskeletal models to generate physically-consistent data for 3D human pose, kinematic, dynamic, and muscle estimation, Nasr et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://liangxuy.github.io/inter-x/">Inter-X</a>: Inter-X: Towards Versatile Human-Human Interaction Analysis, Xu et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_HardMo_A_Large-Scale_Hardcase_Dataset_for_Motion_Capture_CVPR_2024_paper.pdf">HardMo</a>: HardMo: ALarge-Scale Hardcase Dataset for Motion Capture, Liao et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://metaverse-ai-lab-thu.github.io/MMVP-Dataset/">MMVP</a>: MMVP: A Multimodal MoCap Dataset with Vision and Pressure Sensors, Zhang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="http://www.lidarhumanmotion.net/reli11d/">RELI11D</a>: RELI11D: A Comprehensive Multimodal Human Motion Dataset and Method, Yan et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://cs-people.bu.edu/xjhan/groundlink.html">GroundLink</a>: GroundLink: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics, Han et al.</li>
        <li><b>(NeurIPS D&B 2023)</b> <a href="https://hohdataset.github.io/">HOH</a>: HOH: Markerless Multimodal Human-Object-Human Handover Dataset with Large Object Count, Wiederhold et al.</li>
        <li><b>(NeurIPS D&B 2023)</b> <a href="https://motion-x-dataset.github.io/">Motion-X</a>: Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset, Lin et al.</li>
        <li><b>(NeurIPS D&B 2023)</b> <a href="https://github.com/jutanke/hik">Humans in Kitchens</a>: Humans in Kitchens: A Dataset for Multi-Person Human Motion Forecasting with Scene Context, Tanke et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://jnnan.github.io/project/chairs/">CHAIRS</a>: Full-Body Articulated Human-Object Interaction, Jiang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="http://www.lidarhumanmotion.net/cimi4d">CIMI4D</a>: CIMI4D: A Large Multimodal Climbing Motion Dataset under Human-Scene Interactions, Yan et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://andytang15.github.io/FLAG3D/">FLAG3D</a>: FLAG3D: A 3D Fitness Activity Dataset with Language Instruction, Tang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://yifeiyin04.github.io/Hi4D/">Hi4D</a>: Hi4D: 4D Instance Segmentation of Close Human Interaction, Yin et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://stanford-tml.github.io/circle_dataset/">CIRCLE</a>: CIRCLE: Capture in Rich Contextual Environments, Araujo et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/microsoft/MoCapAct">MoCapAct</a>: MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control, Wagener et al.</li>
        <li><b>(ACM MM 2022)</b> <a href="https://github.com/MichiganCOG/ForcePose?tab=readme-ov-file">ForcePose</a>: Learning to Estimate External Forces of Human Motion in Video, Louis et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://pantomatrix.github.io/BEAT/">BEAT</a>: BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis, Liu et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://github.com/dmoltisanti/brace">BRACE</a>: BRACE: The Breakdancing Competition Dataset for Dance Motion Synthesis, Moltisanti et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://sanweiliti.github.io/egobody/egobody.html">EgoBody</a>: Egobody: Human body shape and motion of interacting people from head-mounted devices, Zhang et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://github.com/y-zheng18/GIMO">GIMO</a>: GIMO: Gaze-Informed Human Motion Prediction in Context, Zheng et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://caizhongang.github.io/projects/HuMMan/">HuMMan</a>: HuMMan: Multi-Modal 4D Human Dataset for Versatile Sensing and Modeling, Cai et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://github.com/GUO-W/MultiMotion">ExPI</a>: Multi-Person Extreme Motion Prediction, Guo et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://ericguo5513.github.io/text-to-motion">HumanML3D</a>: Generating Diverse and Natural 3D Human Motions from Text, Guo et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/behave/">BEHAVE</a>: BEHAVE: Dataset and Method for Tracking Human Object Interactions, Bhatnagar et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://google.github.io/aichoreographer/">AIST++</a>: AI Choreographer: Music Conditioned 3D Dance Generation with AIST++, Li et al.</li>
        <li><b>(CVPR 2021)</b> <a href="https://fit3d.imar.ro/home">Fit3D</a>: AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training, Fieraru et al.</li>
        <li><b>(CVPR 2021)</b> <a href="https://babel.is.tue.mpg.de/">BABEL</a>: BABEL: Bodies, Action, and Behavior with English Labels, Punnakkal et al.</li>
        <li><b>(AAAI 2021)</b> <a href="https://sc3d.imar.ro/home">HumanSC3D</a>: Learning complex 3d human self-contact, Fieraru et al.</li>
        <li><b>(CVPR 2020)</b> <a href="https://ci3d.imar.ro/home">CHI3D</a>: Three-Dimensional Reconstruction of Human Interactions, Fieraru et al.</li>
        <li><b>(ICCV 2019)</b> <a href="https://prox.is.tue.mpg.de/">PROX</a>: Resolving 3D Human Pose Ambiguities with 3D Scene Constraints, Hassan et al.</li>
        <li><b>(ICCV 2019)</b> <a href="https://amass.is.tue.mpg.de/">AMASS</a>: AMASS: Archive of Motion Capture As Surface Shapes, Mahmood et al.</li>
    </ul></details>
</ul></details>

<span id="humanoid"></span>
<details open>
<summary><h2>Humanoid, Simulated or Real</h2></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(CVPR 2025)</b> <a href="https://sirui-xu.github.io/InterMimic/">InterMimic</a>: InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions, Xu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://ingrid789.github.io/SkillMimic/">SkillMimic</a>: SkillMimic: Learning Reusable Basketball Skills from Demonstrations, Wang et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://arxiv.org/abs/2410.21229">HOVER</a>: HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots, He et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://junfeng-long.github.io/PIM/">PIM</a>: Learning Humanoid Locomotion with Perceptive Internal Model, Long et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://arxiv.org/pdf/2502.18901">Think on your feet</a>: Think on your feet: Seamless Transition between Human-like Locomotion in Response to Changing Commands, Huang et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://rlpuppeteer.github.io/">Puppeteer</a>: Hierarchical World Models as Visual Whole-Body Humanoid Controllers, Hansen et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=9sOR0nYLtz">FB-CPR</a>: Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models, Tirinzoni et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=MWHIIWrWWu">MPC2</a>: Motion Control of High-Dimensional Musculoskeletal System with Hierarchical Model-Based Planning, Wei et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://guytevet.github.io/CLoSD-page/">CLoSD</a>: CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character control, Tevet et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://arxiv.org/pdf/2502.03122">HiLo</a>: HiLo: Learning Whole-Body Human-like Locomotion with Motion Tracking Controller, Zhang et al.</li>
        <li><b>(Github 2025)</b> <a href="https://github.com/NVlabs/MobilityGen">MobilityGen</a>: MobilityGen.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.14734">GR00T N1</a>: GR00T N1: An Open Foundation Model for Generalist Humanoid Robots, NVIDIA.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://styleloco.github.io/">StyleLoco</a>: StyleLoco: Generative Adversarial Distillation for Natural Humanoid Robot Locomotion, Ma et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.14637">KINESIS</a>: Reinforcement Learning-Based Motion Imitation for Physiologically Plausible Musculoskeletal Motor Control, Simos et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.11801">Diffuse-CLoC</a>: Diffuse-CLoC: Guided Diffusion for Physics-based Character Look-ahead Control, Huang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.12814">Bae et al</a>: Versatile Physics-based Character Control with Hybrid Latent Representation, Bae et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://sites.google.com/view/humanoid-gmp">GMP</a>: Natural Humanoid Robot Locomotion with Generative Motion Prior, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.00692">Sun et al</a>: Learning Perceptive Humanoid Locomotion over Challenging Terrain, Sun et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.00923">HWC-Loco</a>: HWC-Loco: AHierarchical Whole-Body Control Approach to Robust Humanoid Locomotion, Lin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://toruowo.github.io/recipe/">Lin et al</a>: Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids, Lin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://nvlabs.github.io/COMPASS/">COMPASS</a>: COMPASS: Cross-embOdiment Mobility Policy via ResiduAl RL and Skill Synthesis, Liu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://renjunli99.github.io/vbcom.github.io/">VB-COM</a>: VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception, Ren et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.14140">ModSkill</a>: ModSkill: Physical Character Skill Modularization, Huang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.14795">Humanoid-VLA</a>: Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration, Ding et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.13707">Li et al</a>: Human-Like Robot Impedance Regulation Skill Learning from Human-Human Demonstrations, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://humanoid-getup.github.io/">HumanUP</a>: Learning Getting-Up Policies for Real-World Humanoid Robots, He et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://humanoid-interaction.github.io/">RHINO</a>: RHINO: Learning Real-Time Humanoid-Human-Object Interaction from Human Demonstrations, Chen et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.13013">HOMIE</a>: HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit, Ben et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://why618188.github.io/beamdojo/">BeamDojo</a>: BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://taohuang13.github.io/humanoid-standingup.github.io/">HoST</a>: Learning Humanoid Standing-up Control across Diverse Postures, Huang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.01465">Embrace Collisions</a>: Embrace Collisions: Humanoid Shadowing for Deployable Contact-Agnostics Motion, Zhuang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://toddlerbot.github.io/">ToddlerBot</a>: ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation, Shi et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://agile.human2humanoid.com/">ASAP</a>: ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills, He et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2501.02116">Gu et al</a>: Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning, Gu et al.</li>
    </ul></details>
    <details>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://usc-gvl.github.io/UH-1/">UH-1</a>: Learning from Massive Human Videos for Universal Humanoid Pose Control, Mao et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://mimicking-bench.github.io/">Mimicking-Bench</a>: Mimicking-Bench: A Benchmark for Generalizable Humanoid-Scene Interaction Learning via Human Mimicking, Liu et al. </li>
        <li><b>(ArXiv 2024)</b> <a href="https://exbody2.github.io/">Exbody2</a>: Exbody2: Advanced Expressive Humanoid Whole-Body Control, Ji et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.19921">SIMS</a>: SIMS: Simulating Human-Scene Interactions with Real World Script Planning, Wang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://smplolympics.github.io/SMPLOlympics">Humanoidlympics</a>: Humanoidlympics: Sports Environments for Physically Simulated Humanoids, Luo et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://wyhuai.github.io/physhoi-page/">PhySHOI</a>: PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction, Wang et al.</li>
        <li><b>(RA-L 2024)</b> <a href="https://arxiv.org/pdf/2412.15166">Liu et al</a>: Human-Humanoid Robots Cross-Embodiment Behavior-Skill Transfer Using Decomposed Adversarial Learning from Demonstration, Liu et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://stanford-tml.github.io/PDP.github.io/">PDP</a>: PDP: Physics-Based Character Animation via Diffusion Policy, Truong et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://xbpeng.github.io/projects/MaskedMimic/index.html">MaskedMimic</a>: MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting, Tessler et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/abs/2406.19972">HumanVLA</a>: HumanVLA: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid, Xu et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://www.zhengyiluo.com/Omnigrasp-Site/">OmniGrasp</a>: Grasping Diverse Objects with Simulated Humanoids, Luo et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://github.com/zhenzhiwang/intercontrol">InterControl</a>: InterControl: Generate Human Motion Interactions by Controlling Every Joint, Wang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://gao-jiawei.com/Research/CooHOI/">CooHOI</a>: CooHOI: Learning Cooperative Human-Object Interaction with Manipulated Object Dynamics, Gao et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://humanoid-next-token-prediction.github.io/">Radosavovic et al.</a>: Humanoid Locomotion as Next Token Prediction, Radosavovic et al.</li>
        <li><b>(CoRL 2024)</b> <a href="https://ut-austin-rpl.github.io/Harmon/">HARMON</a>: Harmon: Whole-Body Motion Generation of Humanoid Robots from Language Descriptions, Jiang et al.</li>
        <li><b>(CoRL 2024)</b> <a href="https://ut-austin-rpl.github.io/OKAMI/">OKAMI</a>: OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation, Li et al.</li>
        <li><b>(CoRL 2024)</b> <a href="https://humanoid-ai.github.io/">HumanPlus</a>: HumanPlus: Humanoid Shadowing and Imitation from Humans, Fu et al.</li>
        <li><b>(CoRL 2024)</b> <a href="https://omni.human2humanoid.com/">OmniH2O</a>: OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning, He et al.</li>
        <li><b>(Humanoids 2024)</b> <a href="https://evm7.github.io/Self-AWare/">Self-Aware</a>: Know your limits! Optimize the behavior of bipedal robots through self-awareness, Mascaro et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://yunzeliu.github.io/PhysReaction/">PhysReaction</a>: PhysReaction: Physically Plausible Real-Time Humanoid Reaction Synthesis via Forward Dynamics Guided 4D Imitation, Liu et al.</li>
        <li><b>(IROS 2024)</b> <a href="https://human2humanoid.com/">H2O</a>: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation, He et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://idigitopia.github.io/projects/mhc/">MHC</a>: Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs, Shrestha et al.</li>
        <li><b>(ICML 2024)</b> <a href="https://arxiv.org/pdf/2405.14790">DIDI</a>: DIDI: Diffusion-Guided Diversity for Offline Behavioral Generation, Liu et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://moconvq.github.io/">MoConVQ</a>: MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations, Yao et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://jiashunwang.github.io/PhysicsPingPong/">PhysicsPingPong</a>: Strategy and Skill Learning for Physics-based Table Tennis Animation, Wang et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://arxiv.org/abs/2407.10481">SuperPADL</a>: SuperPADL: Scaling Language-Directed Physics-Based Control with Progressive Supervised Distillation, Juravsky et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://www.zhengyiluo.com/SimXR">SimXR</a>: Real-Time Simulated Avatar from Head-Mounted Sensors, Luo et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://anyskill.github.io/">AnySkill</a>: AnySkill: Learning Open-Vocabulary Physical Skill for Interactive Agents, Cui et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://github.com/ZhengyiLuo/PULSE">PULSE</a>: Universal Humanoid Motion Representations for Physics-Based Control, Luo et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://github.com/facebookresearch/hgap">H-GAP</a>: H-GAP: Humanoid Control with a Generalist Planner, Jiang et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://xizaoqu.github.io/unihsi/">UniHSI</a>: UniHSI: Unified Human-Scene Interaction via Prompted Chain-of-Contacts, Xiao et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://eth-ait.github.io/phys-fullbody-grasp/">Phys-Fullbody-Grasp</a>: Physically Plausible Full-Body Hand-Object Interaction Synthesis, Braun et al.</li>
        <li><b>(RSS 2024)</b> <a href="https://expressive-humanoid.github.io/">ExBody</a>: Expressive Whole-Body Control for Humanoid Robots, Cheng et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://vcai.mpi-inf.mpg.de/projects/FatiguedMovements/">Fatigued Movements</a>: Discovering Fatigued Movements for Virtual Character Animation, Cheema et al.</li>
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://frank-zy-dou.github.io/projects/CASE/index.html">CASE</a>: C·ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters, Dou et al.</li>
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://github.com/xupei0610/AdaptNet">AdaptNet</a>: AdaptNet: Policy Adaptation for Physics-Based Character Control, Xu et al.</li>
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://tencent-roboticsx.github.io/NCP/">NCP</a>: Neural Categorical Priors for Physics-Based Character Control, Zhu et al.</li>
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://stanford-tml.github.io/drop/">DROP</a>: DROP: Dynamics Responses from Human Motion Prior and Projective Dynamics, Jiang et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://jiawei-ren.github.io/projects/insactor/">InsActor</a>: InsActor: Instruction-driven Physics-based Characters, Ren et al.</li>
        <li><b>(CoRL 2023)</b> <a href="https://humanoid4parkour.github.io/">Humanoid4Parkour</a>: Humanoid Parkour Learning, Zhuang et al.</li>
        <li><b>(CoRL Workshop 2023)</b> <a href="https://www.kniranjankumar.com/words_into_action/">Words into Action</a>: Words into Action: Learning Diverse Humanoid Robot Behaviors using Language Guided Iterative Motion Refinement, Kumar et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://zhengyiluo.github.io/PHC/">PHC</a>: Perpetual Humanoid Control for Real-time Simulated Avatars, Luo et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://xbpeng.github.io/projects/Trace_Pace/index.html">Trace and Pace</a>: Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion, Rempe et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://research.nvidia.com/labs/toronto-ai/vid2player3d/">Vid2Player3D</a>: DiffMimic: Efficient Motion Mimicking with Differentiable Physics, Zhang et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://dl.acm.org/doi/10.1145/3588432.3591504">QuestEnvSim</a>: QuestEnvSim: Environment-Aware Simulated Motion Tracking from Sparse Sensors, Lee et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://research.nvidia.com/publication/2023-08_synthesizing-physical-character-scene-interactions">Hassan et al.</a>: Synthesizing Physical Character-Scene Interactions, Hassan et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://xbpeng.github.io/projects/CALM/index.html">CALM</a>: CALM: Conditional Adversarial Latent Models for Directable Virtual Characters, Tessler et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://github.com/xupei0610/CompositeMotion">Composite Motion</a>: Composite Motion Learning with Task Control, Xu et al.</li>
        <li><b>(ICLR 2023)</b> <a href="https://diffmimic.github.io/">DiffMimic</a>: DiffMimic: Efficient Motion Mimicking with Differentiable Physics, Ren et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/ZhengyiLuo/EmbodiedPose">EmbodiedPose</a>: Embodied Scene-aware Human Pose Estimation, Luo et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/microsoft/MoCapAct">MoCapAct</a>: MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control, Wagener et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://research.facebook.com/publications/motion-in-betweening-for-physically-simulated-characters/">Gopinath et al.</a>: Motion In-betweening for Physically Simulated Characters, Gopinath et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://dl.acm.org/doi/10.1145/3550082.3564207">AIP</a>: AIP: Adversarial Interaction Priors for Multi-Agent Physics-based Character Control, Younes et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://github.com/heyuanYao-pku/Control-VAE">ControlVAE</a>: ControlVAE: Model-Based Learning of Generative Controllers for Physics-Based Characters, Yao et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://dl.acm.org/doi/fullHtml/10.1145/3550469.3555411">QuestSim</a>: QuestSim: Human Motion Tracking from Sparse Sensors with Simulated Avatars, Winkler et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://github.com/nv-tlabs/PADL">PADL</a>: PADL: Language-Directed Physics-Based Character, Juravsky et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://dl.acm.org/doi/10.1145/3550454.3555490">Wang et al.</a>: Differentiable Simulation of Inertial Musculotendons, Wang et al.</li>
        <li><b>(SIGGRAPH 2022)</b> <a href="https://xbpeng.github.io/projects/ASE/index.html">ASE</a>: ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters, Peng et al. </li>
        <li><b>(Journal of Neuro-Engineering and Rehabilitation 2021)</b> <a href="https://xbpeng.github.io/projects/Learn_to_Move/index.html">Learn to Move</a>: Deep Reinforcement Learning for Modeling Human Locomotion Control in Neuromechanical Simulation, Peng et al. </li>
        <li><b>(NeurIPS 2021)</b> <a href="https://zhengyiluo.github.io/projects/kin_poly/">KinPoly</a>: Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation, Luo et al.</li>
        <li><b>(SIGGRAPH 2021)</b> <a href="https://xbpeng.github.io/projects/AMP/index.html">AMP</a>: AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control, Peng et al. </li>
        <li><b>(CVPR 2021)</b> <a href="https://www.ye-yuan.com/simpoe">SimPoE</a>: SimPoE: Simulated Character Control for 3D Human Pose Estimation, Yuan et al.</li>
        <li><b>(NeurIPS 2020)</b> <a href="https://www.ye-yuan.com/rfc">RFC</a>: Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis, Yuan et al.</li>
        <li><b>(ICLR 2020)</b> <a href="https://arxiv.org/abs/1907.04967">Yuan et al.</a>: Diverse Trajectory Forecasting with Determinantal Point Processes, Yuan et al.</li>
        <li><b>(ICCV 2019)</b> <a href="https://ye-yuan.com/ego-pose/">Ego-Pose</a>: Ego-Pose Estimation and Forecasting as Real-Time PD Control, Yuan et al.</li>
        <li><b>(SIGGRAPH 2018)</b> <a href="https://xbpeng.github.io/projects/DeepMimic/index.html">DeepMimic</a>: DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills, Peng et al.</li>
    </ul></details>
</ul></details>

<span id="bio"></span>
<details open>
<summary><h2>Bio-stuff: Human Anatomy, Biomechanics, Physiology</h2></summary>
<ul style="margin-left: 5px;">
    <li><b>(CVPR 2025)</b> <a href="https://foruck.github.io/HDyS">HDyS</a>: Homogeneous Dynamics Space for Heterogeneous Humans, Liu et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://foruck.github.io/ImDy">ImDy</a>: ImDy: Human Inverse Dynamics from Imitated Observations, Liu et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=MWHIIWrWWu">MPC2</a>: Motion Control of High-Dimensional Musculoskeletal System with Hierarchical Model-Based Planning, Wei et al.</li>
    <li><b>(ACM Sensys 2025)</b> <a href="https://arxiv.org/pdf/2503.01768">SHADE-AD</a>: SHADE-AD: An LLM-Based Framework for Synthesizing Activity Data of Alzheimer’s Patients, Fu et al.</li>
    <li><b>(JEB 2025)</b> <a href="https://journals.biologists.com/jeb/article/228/Suppl_1/JEB248125/367009/Behavioural-energetics-in-human-locomotion-how">McAllister et al</a>: Behavioural energetics in human locomotion: how energy use influences how we move, McAllister et al.</li>
    <li><b>(WACV 2025)</b> <a href="https://arxiv.org/abs/2406.09788">OpenCapBench</a>: A Benchmark to Bridge Pose Estimation and Biomechanics, Gozlan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2503.14637">KINESIS</a>: Reinforcement Learning-Based Motion Imitation for Physiologically Plausible Musculoskeletal Motor Control, Simos et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.13760">Wu et al</a>: Muscle Activation Estimation by Optimizing the Musculoskeletal Model for Personalized Strength and Conditioning Training, Wu et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/pdf/2502.06486">Cotton et al</a>: Biomechanical Reconstruction with Confidence Intervals from Multiview Markerless Motion Capture, Cotton et al.</li>
    <li><b>(BiorXiv 2024)</b> <a href="https://www.biorxiv.org/content/10.1101/2024.12.30.630841v1.full.pdf">Lai et al</a>: Mapping Grip Force to Muscular Activity Towards Understanding Upper Limb Musculoskeletal Intent using a Novel Grip Strength Model, Lai et al.</li>
    <li><b>(IROS 2024)</b> <a href="https://arxiv.org/pdf/2412.18869">Shahriari et al</a>:  Enhancing Robustness in Manipulability Assessment: The Pseudo-Ellipsoid Approach, Shahriari et al.</li>
    <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://cs-people.bu.edu/xjhan/bioDesign.html">BioDesign</a>: Motion-Driven Neural Optimizer for Prophylactic Braces Made by Distributed Microstructures, Han et al.</li>
    <li><b>(Scientific Data 2024)</b> <a href="https://www.nature.com/articles/s41597-024-04077-3?fromPaywallRec=false">Evans et al</a>: Synchronized Video, Motion Capture and Force Plate Dataset for Validating Markerless Human Movement Analysis, Evans et al.</li>
    <li><b>(NeurIPS D&B 2024)</b> <a href="https://simplexsigil.github.io/mint">Muscles in Time</a>: Muscles in Time: Learning to Understand Human Motion by Simulating Muscle Activations, Schneider et al.</li>
    <li><b>(CoRL 2024)</b> <a href="https://lnsgroup.cc/research/hdsafebo">Wei et al</a>: Safe Bayesian Optimization for the Control of High-Dimensional Embodied Systems, Wei et al.</li>
    <li><b>(HFES 2024)</b> <a href="https://journals.sagepub.com/doi/full/10.1177/10711813241262026">Macwan et al</a>: High-Fidelity Worker Motion Simulation With Generative AI, Macwan et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://addbiomechanics.org/">AddBiomechanics</a>: AddBiomechanics Dataset: Capturing the Physics of Human Motion at Scale, Werling et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00194.pdf">MANIKIN</a>: MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation, Jiang et al.</li>
    <li><b>(TOG 2024)</b> <a href="https://dl.acm.org/doi/pdf/10.1145/3658230">NICER</a>: NICER: A New and Improved Consumed Endurance and Recovery Metric to Quantify Muscle Fatigue of Mid-Air Interactions, Li et al.</li>
    <li><b>(ICML 2024)</b> <a href="https://www.beanpow.top/assets/pdf/dynsyn_poster.pdf">DynSyn</a>: DynSyn: Dynamical Synergistic Representation for Efficient Learning and Control in Overactuated Embodied Systems, He et al.</li>
    <li><b>(Multibody System Dynamics 2024)</b> <a href="https://github.com/ainlamyae/Human3.6Mplus">Human3.6M+</a>: Using musculoskeletal models to generate physically-consistent data for 3D human pose, kinematic, dynamic, and muscle estimation, Nasr et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://hit.is.tue.mpg.de/">HIT</a>: HIT: Estimating Internal Human Implicit Tissues from the Body Surface, Keller et al.</li>
    <li><b>(Frontiers in Neuroscience 2024)</b> <a href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1388742/full">Dai et al</a>: Full-body pose reconstruction and correction in virtual reality for rehabilitation training, Dai et al.</li>
    <li><b>(ICRA 2024)</b> <a href="https://arxiv.org/pdf/2312.05473.pdf">He et al.</a>: Self Model for Embodied Intelligence: Modeling Full-Body Human Musculoskeletal System and Locomotion Control with Hierarchical Low-Dimensional Representation, He et al.</li>
    <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://vcai.mpi-inf.mpg.de/projects/FatiguedMovements/">Fatigued Movements</a>: Discovering Fatigued Movements for Virtual Character Animation, Cheema et al.</li>
    <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://skel.is.tue.mpg.de/">SKEL</a>: From skin to skeleton: Towards biomechanically accurate 3d digital humans, Keller et al.</li>
    <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://pku-mocca.github.io/MuscleVAE-page/">MuscleVAE</a>: MuscleVAE: Model-Based Controllers of Muscle-Actuated Characters, Feng et al.</li>
    <li><b>(SIGGRAPH 2023)</b> <a href="https://github.com/namjohn10/BidirectionalGaitNet">Bidirectional GaitNet</a>: Bidirectional GaitNet, Park et al.</li>
    <li><b>(SIGGRAPH 2023)</b> <a href="https://arxiv.org/abs/2305.04995">Lee et al.</a>: Anatomically Detailed Simulation of Human Torso, Lee et al.</li>
    <li><b>(ICCV 2023)</b> <a href="https://musclesinaction.cs.columbia.edu/">MiA</a>: Muscles in Action, Chiquer et al.</li>
    <li><b>(CVPR 2022)</b> <a href="https://osso.is.tue.mpg.de/">OSSO</a>: OSSO: Obtaining Skeletal Shape from Outside, Keller et al.</li>
    <li><b>(Scientific Data 2022)</b> <a href="https://www.nature.com/articles/s41597-022-01188-7">Xing et al</a>: Functional movement screen dataset collected with two Azure Kinect depth sensors, Xing et al.</li>
    <li><b>(NCA 2020)</b> <a href="https://link.springer.com/article/10.1007/s00521-019-04658-z">Zell et al</a>: Learning inverse dynamics for human locomotion analysis, Zell et al.</li>
    <li><b>(ECCV 2020)</b> <a href="https://arxiv.org/pdf/2007.08969">Zell et al</a>: Weakly-supervised learning of human dynamics, Zell et al.</li>
    <li><b>(SIGGRAPH 2019)</b> <a href="https://github.com/jyf588/lrle">LRLE</a>: Synthesis of biologically realistic human motion using joint torque actuation, Jiang et al.</li>
    <li><b>(TII 2018)</b> <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8078194">Pham et al</a>: Multicontact Interaction Force Sensing From Whole-Body Motion Capture, Pham et al.</li>
    <li><b>(ICCV Workshop 2017)</b> <a href="http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w16/Zell_Learning-Based_Inverse_Dynamics_ICCV_2017_paper.pdf">Zell et al</a>: Learning-based inverse dynamics of human motion, Zell et al.</li>
    <li><b>(CVPR Workshop 2017)</b> <a href="http://openaccess.thecvf.com/content_cvpr_2017_workshops/w1/papers/Zell_Joint_3D_Human_CVPR_2017_paper.pdf">Zell et al</a>: Joint 3d human motion capture and physical analysis from monocular videos, Zell et al.</li>
    <li><b>(AIST 2017)</b> <a href="https://link.springer.com/chapter/10.1007/978-3-319-73013-4_12">HuGaDb</a>: HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks, Chereshnev et al.</li>
    <li><b>(SIGGRAPH 2016)</b> <a href="https://dl.acm.org/doi/10.1145/2980179.2982440">Lv et al</a>: Data-driven inverse dynamics for human motion, Lv et al.</li>
</ul></details>

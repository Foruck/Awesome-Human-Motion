# Awesome Human Motion

An aggregation of human motion understanding research; feel free to contribute.

- [Reviews & Surveys](#review) 
- [Motion Generation](#motion-generation)  
- [Motion Editing](#motion-editing)  
- [Motion Stylization](#motion-stylization) 
- [Human-Object Interaction](#hoi) 
- [Human-Scene Interaction](#hsi)  
- [Human-Human Interaction](#hhi) 
- [Datasets](#datasets) 
- [Humanoid](#humanoid) 
- [Bio-stuff](#bio)
- [Human Reconstruction](#motion-reconstruction)    
- [Human-Object/Scene/Human Interaction Reconstruction](#hoi/hsi-reconstruction)
- [Motion Controlled Image/Video Generation](#motion-video/image-generation)
- [Human Pose Estimation/Recognition](#pose-estimation)
- [Human Motion Understanding](#motion-understanding)

---

<span id="review"></span>
## Reviews & Surveys
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
<li><b>(JEB 2025)</b> <a href="https://journals.biologists.com/jeb/article/228/Suppl_1/JEB248125/367009/Behavioural-energetics-in-human-locomotion-how">McAllister et al</a>: Behavioural energetics in human locomotion: how energy use influences how we move, McAllister et al.</li>
<li><b>(ICER 2025)</b> <a href="https://arxiv.org/abs/2412.10458">Zhao et al</a>: Motion Generation Review: Exploring Deep Learning for Lifelike Animation with Manifold, Zhao et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.05419">Motion Generation</a>: A Survey of Generative Approaches and Benchmarks, Khani et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.03191">Segado et al</a>: Grounding Intelligence in Movement, Segado et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.02771">Multimodal Generative AI with Autoregressive LLMs for Human Motion Understanding and Generation</a>: A Way Forward, Islam et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2504.19056">Generative AI for Character Animation</a>: A Comprehensive Survey of Techniques, Applications, and Future Directions, Abootorabi et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.12763">Sui et al</a>: A Survey on Human Interaction Motion Generation, Sui et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.13120">3D Human Interaction Generation</a>: A Survey, Fan et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2502.08556">Human-Centric Foundation Models</a>: Perception, Generation and Agentic Modeling, Tang et al.</li>
<li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2501.02116">Humanoid Locomotion and Manipulation</a>: Current Progress and Challenges in Control, Planning, and Learning, Gu et al.</li>
<li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2407.08428">A Comprehensive Survey on Human Video Generation</a>: Challenges, Methods, and Insights, Lei et al.</li>
<li><b>(ArXiv 2024)</b> <a href="https://github.com/Winn1y/Awesome-Human-Motion-Video-Generation?tab=readme-ov-file">Human Motion Video Generation</a>: A survey, Xue et al.</li>
<li><b>(Neurocomputing)</b> <a href="https://github.com/Winn1y/Awesome-Human-Motion-Video-Generation?tab=readme-ov-file">Deep Learning for 3D Human Pose Estimation and Mesh Recovery</a>: A survey, Liu et al.</li>
<li><b>(TVCG 2024)</b> <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10230894">Loi et al</a>: Machine Learning Approaches for 3D Motion Synthesis and Musculoskeletal Dynamics Estimation: A Survey, Loi et al.</li>
<li><b>(T-PAMI 2023)</b> <a href="https://arxiv.org/abs/2307.10894">Zhu et al</a>: Human Motion Generation: A Survey, Zhu et al.</li>
</ul>
</details>

<span id="motion-generation"></span>
## Motion Generation, Text/Speech/Music-Driven
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
        <ul style="margin-left: 5px;">
        <li><b>(TMLR 2025)</b> <a href="https://xiyan-xu.github.io/MoReactWebPage/">MoReact</a>: Generating Reactive Motion from Textual Descriptions, Xu et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://chaitanya100100.github.io/UniEgoMotion/">UniEgoMotion</a>: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation, Patel et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2507.19850">FineMotion</a>: A Dataset and Benchmark with both Spatial and Temporal Annotation for Fine-grained Motion Generation and Editing, Wu et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2507.20170">PUMPS</a>: Skeleton-Agnostic Point-based Universal Motion Pre-Training for Synthesis in Human Motion Tasks, Mo et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://research.nvidia.com/labs/dair/genmo/">GENMO</a>: A GENeralist Model for Human MOtion, Li et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2411.18303">InfiniDreamer</a>: Arbitrarily Long Human Motion Generation via Segment Score Distillation, Zhuo et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://github.com/VankouF/MotionMillion-Codes">Go to Zero</a>: Towards Zero-shot Motion Generation with Million-scale Data, Fan et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2411.14951">Morph</a>: A Motion-free Physics Optimization Framework for Human Motion Generation, Li et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://whwjdqls.github.io/discord.github.io/">DisCoRD</a>: Discrete Tokens to Continuous Motion via Rectified Flow Decoding, Cho et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://xiangyuezhang.com/SemTalk/">SemTalk</a>: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis, Zhang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://andypinxinliu.github.io/KinMo/">KinMo</a>: Kinematic-aware Human Motion Understanding and Generation, Zhang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://andypinxinliu.github.io/GestureLSM/">GestureLSM</a>: Latent Shortcut-based Co-Speech Gesture Generation with Spatial-Temporal Modeling, Liu et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://zju3dv.github.io/Motion-2-to-3/">Motion-2-to-3</a>: Leveraging 2D Motion Data to Boost 3D Motion Generation, Pi et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://diouo.github.io/motionlab.github.io/">MotionLab</a>: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm, Guo et al.</li>
        <li><b>(ICCV 2025)</b> <a href="http://inwoohwang.me/SFControl">SFControl</a>: Motion Synthesis with Sparse and Flexible Keyjoint Control, Hwang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2503.13859">Less Is More</a>: Improving Motion Diffusion Models with Sparse Keyframes, Bae et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://www.ekkasit.com/ControlMM-page/">ControlMM</a>: Controllable Masked Motion Generation, Pinyoanuntapong et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://yz-cnsdqz.github.io/eigenmotion/PRIMAL/">PRIMAL</a>: Physically Reactive and Interactive Motor Model for Avatar Learning, Zhang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://jackyu6.github.io/HERO/">HERO</a>: Human Reaction Generation from Videos, Yu et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://zju3dv.github.io/MotionStreamer/">MotionStreamer</a>: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space, Xiao et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2503.14919">GenM3</a>: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation, Shi et al.</li>
        <li><b>(ACM MM 2025)</b> <a href="https://arxiv.org/abs/2507.19836">ChoreoMuse</a>: Robust Music-to-Dance Video Generation with Style Transfer and Beat-Adherent Motion, Wang et al.</li>
        <li><b>(ICML 2025)</b> <a href="https://arxiv.org/abs/2410.03311">Being-M0</a>: Scaling Motion Generation Models with Million-Level Human Motions, Wang et al.</li>
        <li><b>(TOG 2025)</b> <a href="https://zhongleilz.github.io/Sketch2Anim/">Sketch2Anim</a>: Towards Transferring Sketch Storyboards into 3D Animation, Zhong et al.</li>
        <li><b>(SIGGRAPH 2025)</b> <a href="https://robinwitch.github.io/MECo-Page/">MECo</a>: Motion-example-controlled Co-speech Gesture Generation Leveraging Large Language Models, Chen et al.</li>
        <li><b>(SIGGRAPH 2025)</b> <a href="https://arxiv.org/abs/2505.14087">Chang et al.</a>: Large-Scale Multi-Character Interaction Synthesis, Chang et al.</li>
        <li><b>(SIGGRAPH 2025)</b> <a href="https://arxiv.org/abs/2502.17327">AnyTop</a>: Character Animation Diffusion with Any Topology, Gat et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2505.00998">DSDFM</a>: Deterministic-to-Stochastic Diverse Latent Feature Mapping for Human Motion Synthesis, Hua et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://antgroup.github.io/ai/echomimic_v2/">EchoMimicV2</a>: Towards Striking, Simplified, and Semi-Body Human Animation, Hua et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://liyiheng23.github.io/UniPose-Page/">UniPose</a>: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing, Li et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2504.05265">From Sparse Signal to Smooth Motion</a>: Real-Time Motion Generation with Rolling Prediction Models, Barquero et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://shape-move.github.io/">Shape My Moves</a>: Text-Driven Shape-Aware Synthesis of Human Motions, Liao et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://github.com/CVI-SZU/MG-MotionLLM">MG-MotionLLM</a>: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities, Wu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://seokhyeonhong.github.io/projects/salad/">SALAD</a>: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing, Hong et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://boeun-kim.github.io/page-PersonaBooth/">PersonalBooth</a>: Personalized Text-to-Motion Generation, Kim et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2411.16575">MARDM</a>: Rethinking Diffusion for Text-Driven Human Motion Generation, Meng et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2503.04829">StickMotion</a>: Generating 3D Human Motions by Drawing a Stickman, Wang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2411.16805">LLaMo</a>: Human Motion Instruction Tuning, Li et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://star-uu-wang.github.io/HOP/">HOP</a>: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation, Cheng et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://atom-motion.github.io/">AtoM</a>: Aligning Text-to-Motion Model at Event-Level with GPT-4Vision Reward, Han et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://jiro-zhang.github.io/EnergyMoGen/">EnergyMoGen</a>: Compositional Human Motion Generation with Energy-Based Diffusion Model in Latent Space, Zhang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://languageofmotion.github.io/">The Languate of Motion</a>: Unifying Verbal and Non-verbal Language of 3D Human Motion, Chen et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://shunlinlu.github.io/ScaMo/">ScaMo</a>: Exploring the Scaling Law in Autoregressive Motion Generation Model, Lu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://hhsinping.github.io/Move-in-2D/">Move in 2D</a>: 2D-Conditioned Human Motion Generation, Huang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://solami-ai.github.io/">SOLAMI</a>: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters, Jiang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://lijiaman.github.io/projects/mvlift/">MVLift</a>: Lifting Motion to the 3D World via 2D Diffusion, Li et al.</li>
        <li><b>(CVPR 2025 Workshop)</b> <a href="https://arxiv.org/abs/2505.10810">MoCLIP</a>: Motion-Aware Fine-Tuning and Distillation of CLIP for Human Motion Generation, Maldonado et al.</li>
        <li><b>(CVPR 2025 Workshop)</b> <a href="https://arxiv.org/abs/2505.09827">Dyadic Mamba</a>: Long-term Dyadic Human Motion Synthesis, Tanke et al.</li>
        <li><b>(ACM Sensys 2025)</b> <a href="https://arxiv.org/abs/2503.01768">SHADE-AD</a>: An LLM-Based Framework for Synthesizing Activity Data of Alzheimer’s Patients, Fu et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://arxiv.org/abs/2410.16623">MotionGlot</a>: A Multi-Embodied Motion Generation Model, Harithas et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://guytevet.github.io/CLoSD-page/">CLoSD</a>: Closing the Loop between Simulation and Diffusion for Multi-Task Character Control, Tevet et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://genforce.github.io/PedGen/">PedGen</a>: Learning to Generate Diverse Pedestrian Movements from Web Videos with Noisy Labels, Liu et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=IEul1M5pyk">HGM³</a>: Hierarchical Generative Masked Motion Modeling with Hard Token Mining, Jeong et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=LYawG8YkPa">LaMP</a>: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning, Li et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=d23EVDRJ6g">MotionDreamer</a>: One-to-Many Motion Synthesis with Localized Generative Masked Transformer, Wang et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=Oh8MuCacJW">Lyu et al</a>: Towards Unified Human Motion-Language Understanding via Sparse Interpretable Characterization, Lyu et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://zkf1997.github.io/DART/">DART</a>: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control, Zhao et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://knoxzhao.github.io/Motion-Agent/">Motion-Agent</a>: A Conversational Framework for Human Motion Generation with LLMs, Wu et al.</li>
        <li><b>(TMM 2025)</b> <a href="https://arxiv.org/abs/2508.01590">MCG-IMM</a>: A Plug-and-Play Multi-Criteria Guidance for Diverse In-Betweening Human Motion Generation, Yu et al.</li>
        <li><b>(IJCV 2025)</b> <a href="https://arxiv.org/abs/2502.05534">Fg-T2M++</a>: LLMs-Augmented Fine-Grained Text Driven Human Motion Generation, Wang et al.</li>
        <li><b>(TCSVT 2025)</b> <a href="https://arxiv.org/abs/2503.13300">Zeng et al</a>: Progressive Human Motion Generation Based on Text and Few Motion Frames, Zeng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.24099">Pulp Motion</a>: Framing-aware multimodal camera and human motion generation, Courant et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2510.03200">MonSTeR</a>: a Unified Model for Motion, Scene, Text Retrieval, Collorone et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2510.02722">MoGIC</a>: Boosting Motion Generation via Intention Understanding and Visual Context, Shi et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.24099">Gupta et al</a>: Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow, Gupta et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.24469">LaMoGen</a>: Laban Movement-Guided Diffusion for Text-to-Motion Generation, Kim et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.25304">LUMA</a>: Low-Dimension Unified Motion Alignment with Dual-Path Anchoring for Text-to-Motion Diffusion Model, Jia et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.20927">SimDiff</a>: Simulator-constrained Diffusion Model for Physically Plausible Motion Generation, Watanabe et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.04058">SmooGPT</a>: Stylized Motion Generation using Large Language Models, Zhong et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.20604">Embracing Aleatoric Uncertainty</a>: Generating Diverse 3D Human Motion, Qin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.19527">MotionFLUX</a>: Efficient Text-Guided Motion Generation through Rectified Flow Matching and Preference Alignment, Gao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.12081">VimoRAG</a>: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models, Xu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.08991">MSQ</a>: Spatial-Temporal Multi-Scale Quantizationfor Flexible Motion Generation, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.05162">X-MoGen</a>: Unified Motion Generation across Humans and Animals, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://aigeeksgroup.github.io/ReMoMask/">ReMoMask</a>: Retrieval-Augmented Masked Motion Generation, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://omni-avatar.github.io/">OmniAvatar</a>: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation, Gan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://dorniwang.github.io/SpeakerVid-5M/">SpeakerVid-5M</a>: A Large-Scale High-Quality Dataset for audio-visual Dyadic Interactive Human Generation, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://www.arxiv.org/pdf/2507.03905">EchoMimicV3</a>: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation, Meng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href=https://arxiv.org/abs/2507.11949">MOSPA</a>: Human Motion Generation Driven by Spatial Audio, Xu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://snap-research.github.io/SnapMoGen/">SnapMoGen</a>: Human Motion Generation from Expressive Texts, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.06590">MOST</a>: Motion Diffusion Model for Rare Text via Temporal Clip Banzhaf Interaction, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://groundedgestures.github.io/">Grounded Gestures</a>: Language, Motion and Space, Deichler et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://github.com/OpenMotionLab/MotionGPT3">MotionGPT3</a>: Human Motion as a Second Modality, Zhu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.21912">HumanAttr</a>: Generating Attribute-Aware Human Motions from Textual Prompt, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.17912">PlanMoGPT</a>: Flow-Enhanced Progressive Planning for Text to Motion Synthesis, Jin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://motion-r1.github.io/">Motion-R1</a>: Chain-of-Thought Reasoning and Reinforcement Learning for Human Motion Generation, Ouyang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.02452">ANT</a>: Adaptive Neural Temporal-Aware Text-to-Motion Model, Chen et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.02661">MotionRAG-Diff</a>: A Retrieval-Augmented Diffusion Framework for Long-Term Music-to-Dance Generation, Huang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.21146">IKMo</a>: Image-Keyframed Motion Generation with Trajectory-Pose Conditioned Motion Diffusion Model, Zhao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.21531">Li et al</a>: How Much Do Large Language Models Know about Human Motion? A Case Study in 3D Avatar Control, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.21837">UniMoGen</a>: Universal Motion Generation, Khani et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.23465">Wang et al</a>: Semantics-Aware Human Motion Generation from Audio Instructions, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.19377">ACMDM</a>: Absolute Coordinates Make Motion Generation Easy, Meng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://mucunzhuzhu.github.io/PAMD-page/">PAMD</a>: Plausibility-Aware Motion Diffusion Model for Long Dance Generation, Zhu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.15197">Intentional Gesture</a>: Deliver Your Intentions with Gestures for Speech, Liu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.14222">MatchDance</a>: Collaborative Mamba-Transformer Architecture Matching for High-Quality 3D Dance Synthesis, Yang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.08293">M3G</a>: Multi-Granular Gesture Generator for Audio-Driven Full-Body Human Motion Synthesis, Yin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.05589">ReactDance</a>: Progressive-Granular Representation for Long-Term Coherent Reactive Dance Generation, Lin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://wengwanjiang.github.io/ReAlign-page/">ReAlign</a>: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment, Weng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2504.16722">PMG</a>: Progressive Motion Generation via Sparse Anchor Postures Curriculum Learning, Xi et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://foram-s1.github.io/DanceMosaic/">DanceMosaic</a>: High-Fidelity Dance Generation with Multimodal Editability, Shah et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://yong-xie-xy.github.io/ReCoM/">ReCoM</a>: Realistic Co-Speech Motion Generation with Recurrent Embedded Transformer, Xie et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://www.pinlab.org/hmu">HMU</a>: Human Motion Unlearning, Matteis et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://mjwei3d.github.io/ACMo/">ACMo</a>: Attribute Controllable Motion Generation, Wei et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.06151">BioMoDiffuse</a>: Physics-Guided Biomechanical Diffusion for Controllable and Authentic Human Motion Synthesis, Kang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.06499">ExGes</a>: Expressive Human Motion Retrieval and Modulation for Audio-Driven Gesture Synthesis, Zhou et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://steve-zeyu-zhang.github.io/MotionAnything/">Motion Anything</a>: Any to Motion Generation, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2502.18309">GCDance</a>: Genre-Controlled 3D Full Body Dance Generation Driven By Music, Liu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://cjerry1243.github.io/casim_t2m/">CASIM</a>: Composite Aware Semantic Injection for Text to Motion Generation, Chang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2501.19083">MotionPCM</a>: Real-Time Motion Synthesis with Phased Consistency Model, Jiang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2501.18232">Free-T2M</a>: Frequency Enhanced Text-to-Motion Diffusion Model With Consistency Loss, Chen et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/html/2501.16778v1">FlexMotion</a>: Lightweight, Physics-Aware, and Controllable Human Motion Generation, Tashakori et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.06897">HiSTF Mamba</a>: Hierarchical Spatiotemporal Fusion with Multi-Granular Body-Spatial Modeling for High-Fidelity Text-to-Motion Generation, Zhan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2501.16551">PackDiT</a>: Joint Human Motion and Text Generation via Mutual Prompting, Jiang et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://coral79.github.io/uni-motion/">Unimotion</a>: Unifying 3D Human Motion Synthesis and Understanding, Li et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://cyk990422.github.io/HoloGest.github.io//">HoloGest</a>: Decoupled Diffusion and Motion Priors for Generating Holisticly Expressive Co-speech Gestures, Cheng et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://yu1ut.com/ReMoGPT-HP/">RemoGPT</a>: Part-Level Retrieval-Augmented Motion-Language Models, Yu et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://hanyangclarence.github.io/unimumo_demo/">UniMuMo</a>: Unified Text, Music and Motion Generation, Yang et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://antgroup.github.io/ai/echomimic/">EchoMimic</a>: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning, Chen et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://arxiv.org/abs/2408.00352">ALERT-Motion</a>: Autonomous LLM-Enhanced Adversarial Attack for Text-to-Motion, Miao et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://cure-lab.github.io/MotionCraft/">MotionCraft</a>: Crafting Whole-Body Motion with Plug-and-Play Multimodal Controls, Bian et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://arxiv.org/abs/2412.11193">Light-T2M</a>: A Lightweight and Fast Model for Text-to-Motion Generation, Zeng et al.</li>
        <li><b>(WACV 2025 Worhshop)</b> <a href="https://arxiv.org/abs/2501.01449">LS-GAN</a>: Human Motion Synthesis with Latent-space GANs, Amballa et al.</li>
        <li><b>(WACV 2025)</b> <a href="https://reindiffuse.github.io/">ReinDiffuse</a>: Crafting Physically Plausible Motions with Reinforced Diffusion Model, Han et al.</li>
        <li><b>(WACV 2025)</b> <a href="https://motion-rag.github.io/">MoRAG</a>: Multi-Fusion Retrieval Augmented Generation for Human Motion, Shashank et al.</li>
        <li><b>(WACV 2025)</b> <a href="https://arxiv.org/abs/2409.11920">Mandelli et al</a>: Generation of Complex 3D Human Motion by Temporal and Spatial Composition of Diffusion Models, Mandelli et al.</li>
    </ul></details>
    <details>
    <summary><h3>2024</h3></summary>
        <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://mmofusion.github.io/">MMoFusion</a>: Multi-modal Co-Speech Motion Generation with Diffusion Model, Wang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://inter-dance.github.io/">InterDance</a>: Reactive 3D Dance Generation with Realistic Duet Interactions, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2412.07797">Mogo</a>: RQ Hierarchical Causal Transformer for High-Quality 3D Human Motion Generation, Fu et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://gabrie-l.github.io/coma-page/">CoMA</a>: Compositional Human Motion Generation with Multi-modal Agents, Sun et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://sopo-motion.github.io/">SoPo</a>: Text-to-Motion Generation Using Semi-Online Preference Optimization, Tan et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2412.04343">RMD</a>: A Simple Baseline for More General Human Motion Generation via Training-free Retrieval-Augmented Motion Diffuse, Liao et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2412.00112">BiPO</a>: Bidirectional Partial Occlusion Network for Text-to-Motion Synthesis, Hong et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.19786">MoTe</a>: Learning Motion-Text Diffusion Model for Multiple Generation Tasks, Wue et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2411.17532">FTMoMamba</a>: Motion Generation with Frequency and Text State Space Models, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://steve-zeyu-zhang.github.io/KMM">KMM</a>: Key Frame Mask Mamba for Extended Motion Generation, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.21747">MotionGPT-2</a>: A General-Purpose Motion-Language Model for Motion Generation and Understanding, Wang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://li-ronghui.github.io/lodgepp">Lodge++</a>: High-quality and Long Dance Generation with Vivid Choreography Patterns, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.18977">MotionCLR</a>: Motion Generation and Training-Free Editing via Understanding Attention Mechanisms, Chen et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.14508">LEAD</a>: Latent Realignment for Human Motion Diffusion, Andreou et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.08931">Leite et al.</a> Enhancing Motion Variation in Text-to-Motion Models via Pose and Video Conditioned Editing, Leite et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2410.06513">MotionRL</a>: Align Text-to-Motion Generation to Human Preferences with Multi-Reward Reinforcement Learning, Liu et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://lhchen.top/MotionLLM/">MotionLLM</a>: Understanding Human Behaviors from Human Motions and Videos, Chen et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2409.13251">T2M-X</a>: Learning Expressive Text-to-Motion Generation from Partially Annotated Data, Liu et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://github.com/RohollahHS/BAD">BAD</a>: Bidirectional Auto-regressive Diffusion for Text-to-Motion Generation, Hosseyni et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://von31.github.io/synNsync/">synNsync</a>: Synergy and Synchrony in Couple Dances, Manukele et al.</li>
        <li><b>(EMNLP 2024)</b> <a href="https://aclanthology.org/2024.findings-emnlp.584/">Dong et al</a>: Word-Conditioned 3D American Sign Language Motion Generation, Dong et al.</li>
        <li><b>(NeurIPS D&B 2024)</b> <a href="https://nips.cc/virtual/2024/poster/97700">Kim et al</a>: Text to Blind Motion, Kim et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://github.com/xiyuanzh/UniMTS">UniMTS</a>: Unified Pre-training for Motion Time Series, Zhang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://openreview.net/forum?id=FsdB3I9Y24">Christopher et al.</a>: Constrained Synthesis with Projected Diffusion Models, Christopher et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://momu-diffusion.github.io/">MoMu-Diffusion</a>: On Learning Long-Term Motion-Music Synchronization and Correspondence, You et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://aigc3d.github.io/mogents/">MoGenTS</a>: Motion Generation based on Spatial-Temporal Joint Modeling, Yuan et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/abs/2405.16273">M3GPT</a>: An Advanced Multimodal, Multitask Framework for Motion Comprehension and Generation, Luo et al.</li>
        <li><b>(NeurIPS Workshop 2024)</b> <a href="https://openreview.net/forum?id=BTSnh5YdeI">Bikov et al</a>: Fitness Aware Human Motion Generation with Fine-Tuning, Bikov et al.</li>
        <li><b>(NeurIPS Workshop 2024)</b> <a href="https://arxiv.org/abs/2502.20176">DGFM</a>: Full Body Dance Generation Driven by Music Foundation Models, Liu et al.</li>
        <li><b>(ICPR 2024)</b> <a href="https://link.springer.com/chapter/10.1007/978-3-031-78104-9_30">FG-MDM</a>: Towards Zero-Shot Human Motion Generation via ChatGPT-Refined Descriptions, Shi et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://bohongchen.github.io/SynTalker-Page/">SynTalker</a>: Enabling Synergistic Full-Body Control in Prompt-Based Co-Speech Motion Generation, Chen et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://dl.acm.org/doi/abs/10.1145/3664647.3681487">L3EM</a>: Towards Emotion-enriched Text-to-Motion Generation via LLM-guided Limb-level Emotion Manipulating. Yu et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://dl.acm.org/doi/abs/10.1145/3664647.3681657">StableMoFusion</a>: Towards Robust and Efficient Diffusion-based Motion Generation Framework, Huang et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://dl.acm.org/doi/abs/10.1145/3664647.3681034">SATO</a>: Stable Text-to-Motion Framework, Chen et al.</li>
        <li><b>(ICANN 2024)</b> <a href="https://link.springer.com/chapter/10.1007/978-3-031-72356-8_2">PIDM</a>: Personality-Aware Interaction Diffusion Model for Gesture Generation, Shibasaki et al.</li>
        <li><b>(HFES 2024)</b> <a href="https://journals.sagepub.com/doi/full/10.1177/10711813241262026">Macwan et al</a>: High-Fidelity Worker Motion Simulation With Generative AI, Macwan et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://jpthu17.github.io/GuidedMotion-project/">Jin et al</a>: Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation, Jin et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/100_ECCV_2024_paper.php">Motion Mamba</a>: Efficient and Long Sequence Motion Generation, Zhong et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://frank-zy-dou.github.io/projects/EMDM/index.html">EMDM</a>: Efficient Motion Diffusion Model for Fast, High-Quality Human Motion Generation, Zhou et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://yh2371.github.io/como/">CoMo</a>: Controllable Motion Generation through Language Guided Pose Code Editing, Huang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://github.com/jsun57/CoMusion">CoMusion</a>: Towards Consistent Stochastic Human Motion Prediction via Motion Diffusion, Sun et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2405.18483">Shan et al</a>: Towards Open Domain Text-Driven Synthesis of Multi-Person Motions, Shan et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://github.com/qrzou/ParCo">ParCo</a>: Part-Coordinating Text-to-Motion Synthesis, Zou et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2407.11532">Sampieri et al</a>: Length-Aware Motion Synthesis via Latent Diffusion, Sampieri et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://github.com/line/ChronAccRet">ChroAccRet</a>: Chronologically Accurate Retrieval for Temporal Grounding of Motion-Language Models, Fujiwara et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://idigitopia.github.io/projects/mhc/">MHC</a>: Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs, Liu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://github.com/moonsliu/Pro-Motion">ProMotion</a>: Plan, Posture and Go: Towards Open-vocabulary Text-to-Motion Generation, Liu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2406.10740">FreeMotion</a>: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models, Zhang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://eccv.ecva.net/virtual/2024/poster/266">Text Motion Translator</a>: A Bi-Directional Model for Enhanced 3D Human Motion Generation from Open-Vocabulary Descriptions, Qian et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://vankouf.github.io/FreeMotion/">FreeMotion</a>: A Unified Framework for Number-free Text-to-Motion Synthesis, Fan et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://foruck.github.io/KP/">Kinematic Phrases</a>: Bridging the Gap between Human Motion and Action Semantics via Kinematic Phrases, Liu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2404.01700">MotionChain</a>: Conversational Motion Controllers via Multimodal Prompts, Jiang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://neu-vi.github.io/SMooDi/">SMooDi</a>: Stylized Motion Diffusion Model, Zhong et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://exitudio.github.io/BAMM-page/">BAMM</a>: Bidirectional Autoregressive Motion Model, Pinyoanuntapong et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://dai-wenxun.github.io/MotionLCM-page/">MotionLCM</a>: Real-time Controllable Motion Generation via Latent Consistency Model, Dai et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2312.10993">Ren et al</a>: Realistic Human Motion Generation with Cross-Diffusion Models, Ren et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2407.14502">M2D2M</a>: Multi-Motion Generation from Text with Discrete Diffusion Models, Chi et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://mingyuan-zhang.github.io/projects/LMM.html">LMM</a>: Large Motion Model for Unified Multi-Modal Motion Generation, Zhang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://research.nvidia.com/labs/toronto-ai/tesmo/">TesMo</a>: Generating Human Interaction Motions in Scenes with Text Control, Yi et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://tlcontrol.weilinwl.com/">TLcontrol</a>: Trajectory and Language Control for Human Motion Synthesis, Wan et al.</li>
        <li><b>(ICME 2024)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10687922">ExpGest</a>: Expressive Speaker Generation Using Diffusion Model and Hybrid Audio-Text Guidance, Cheng et al.</li>
        <li><b>(ICME Workshop 2024)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10645445">Chen et al</a>: Anatomically-Informed Vector Quantization Variational Auto-Encoder for Text-to-Motion Generation, Chen et al.</li>
        <li><b>(ICML 2024)</b> <a href="https://github.com/LinghaoChan/HumanTOMATO">HumanTOMATO</a>: Text-aligned Whole-body Motion Generation, Lu et al.</li>
        <li><b>(ICML 2024)</b> <a href="https://sites.google.com/view/gphlvm/">GPHLVM</a>: Bringing Motion Taxonomies to Continuous Domains via GPLVM on Hyperbolic Manifolds, Jaquier et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://diffposetalk.github.io/">DiffPoseTalk</a>: Speech-Driven Stylistic 3D Facial Animation and Head Pose Generation via Diffusion Models, Sun et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://setarehc.github.io/CondMDI/">CondMDI</a>: Flexible Motion In-betweening with Diffusion Models, Cohan et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://aiganimation.github.io/CAMDM/">CAMDM</a>: Taming Diffusion Probabilistic Models for Character Control, Chen et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://vcc.tech/research/2024/LGTM">LGTM</a>: Local-to-Global Text-Driven Human Motion Diffusion Models, Sun et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://threedle.github.io/TEDi/">TEDi</a>: Temporally-Entangled Diffusion for Long-Term Motion Synthesis, Zhang et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://github.com/Yi-Shi94/AMDM">A-MDM</a>: Interactive Character Control with Auto-Regressive Motion Diffusion Models, Shi et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://dl.acm.org/doi/10.1145/3658209">Starke et al</a>: Categorical Codebook Matching for Embodied Character Controllers, Starke et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://arxiv.org/abs/2407.10481">SuperPADL</a>: Scaling Language-Directed Physics-Based Control with Progressive Supervised Distillation, Juravsky et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://hanchaoliu.github.io/Prog-MoGen/">ProgMoGen</a>: Programmable Motion Generation for Open-set Motion Control Tasks, Liu et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://github.com/IDC-Flash/PacerPlus">PACER+</a>: On-Demand Pedestrian Animation Controller in Driving Scenarios, Wang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://amuse.is.tue.mpg.de/">AMUSE</a>: Emotional Speech-driven 3D Body Animation via Disentangled Latent Diffusion, Chhatre et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://feifeifeiliu.github.io/probtalk/">Liu et al</a>: Towards Variable and Coordinated Holistic Co-Speech Motion Generation, Liu et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://guytevet.github.io/mas-page/">MAS</a>: Multi-view Ancestral Sampling for 3D motion generation using 2D diffusion, Kapon et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://wandr.is.tue.mpg.de/">WANDR</a>: Intention-guided Human Motion Generation, Diomataris et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://ericguo5513.github.io/momask/">MoMask</a>: Generative Masked Modeling of 3D Human Motions, Guo et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://yfeng95.github.io/ChatPose/">ChatPose</a>: Chatting about 3D Human Pose, Feng et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://zixiangzhou916.github.io/AvatarGPT/">AvatarGPT</a>: All-in-One Framework for Motion Understanding, Planning, Generation and Beyond, Zhou et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://exitudio.github.io/MMM-page/">MMM</a>: Generative Masked Motion Model, Pinyoanuntapong et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Li_AAMDM_Accelerated_Auto-regressive_Motion_Diffusion_Model_CVPR_2024_paper.pdf">AAMDM</a>: Accelerated Auto-regressive Motion Diffusion Model, Li et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://tr3e.github.io/omg-page/">OMG</a>: Towards Open-vocabulary Motion Generation via Mixture of Controllers, Liang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://barquerogerman.github.io/FlowMDM/">FlowMDM</a>: Seamless Human Motion Composition with Blended Positional Encodings, Barquero et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://digital-life-project.com/">Digital Life Project</a>: Autonomous 3D Characters with Social Intelligence, Cai et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://pantomatrix.github.io/EMAGE/">EMAGE</a>: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling, Liu et al.</li>
        <li><b>(CVPR Workshop 2024)</b> <a href="https://xbpeng.github.io/projects/STMC/index.html">STMC</a>: Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation, Petrovich et al.</li>
        <li><b>(CVPR Workshop 2024)</b> <a href="https://github.com/THU-LYJ-Lab/InstructMotion">InstructMotion</a>: Exploring Text-to-Motion Generation with Human Preference, Sheng et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://sinmdm.github.io/SinMDM-page/">Single Motion Diffusion</a>: Raab et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://openreview.net/forum?id=sOJriBlOFd&noteId=KaJUBoveeo">NeRM</a>: Learning Neural Representations for High-Framerate Human Motion Synthesis, Wei et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://priormdm.github.io/priorMDM-page/">PriorMDM</a>: Human Motion Diffusion as a Generative Prior, Shafir et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://neu-vi.github.io/omnicontrol/">OmniControl</a>: Control Any Joint at Any Time for Human Motion Generation, Xie et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://openreview.net/forum?id=yQDFsuG9HP">Adiya et al.</a>: Bidirectional Temporal Diffusion Model for Temporally Consistent Human Animation, Adiya et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://lisiyao21.github.io/projects/Duolando/">Duolando</a>: Follower GPT with Off-Policy Reinforcement Learning for Dance Accompaniment, Li et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://arxiv.org/abs/2312.12227">HuTuDiffusion</a>: Human-Tuned Navigation of Latent Motion Diffusion Models with Minimal Feedback, Han et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://arxiv.org/abs/2312.12763">AMD</a>: Anatomical Motion Diffusion with Interpretable Motion Decomposition and Fusion, Jing et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://nhathoang2002.github.io/MotionMix-page/">MotionMix</a>: Weakly-Supervised Diffusion for Controllable Motion Generation, Hoang et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://github.com/xiezhy6/B2A-HDM">B2A-HDM</a>: Towards Detailed Text-to-Motion Synthesis via Basic-to-Advanced Hierarchical Diffusion Model, Xie et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/27936">Everything2Motion</a>: Everything2Motion: Synchronizing Diverse Inputs via a Unified Framework for Human Motion Synthesis, Fan et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://qiqiapink.github.io/MotionGPT/">MotionGPT</a>: Finetuned LLMs are General-Purpose Motion Generators, Zhang et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://arxiv.org/abs/2305.13773">Dong et al</a>: Enhanced Fine-grained Motion Diffusion for Text-driven Human Motion Synthesis, Dong et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://evm7.github.io/UNIMASKM-page/">UNIMASKM</a>: A Unified Masked Autoencoder with Patchified Skeletons for Motion Synthesis, Mascaro et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://arxiv.org/abs/2312.10960">B2A-HDM</a>: Towards Detailed Text-to-Motion Synthesis via Basic-to-Advanced Hierarchical Diffusion Model, Xie et al.</li>
        <li><b>(TPAMI 2024)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10399852">GUESS</a>: GradUally Enriching SyntheSis for Text-Driven Human Motion Generation, Gao et al.</li>
        <li><b>(WACV 2024)</b> <a href="https://arxiv.org/abs/2312.12917">Xie et al.</a>: Sign Language Production with Latent Motion Transformer, Xie et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(NeurIPS 2023)</b> <a href="https://github.com/jpthu17/GraphMotion">GraphMotion</a>: Act As You Wish: Fine-grained Control of Motion Diffusion Model with Hierarchical Semantic Graphs, Jin et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://motion-gpt.github.io/">MotionGPT</a>: Human Motion as Foreign Language, Jiang et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://mingyuan-zhang.github.io/projects/FineMoGen.html">FineMoGen</a>: Fine-Grained Spatio-Temporal Motion Generation and Editing, Zhang et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://jiawei-ren.github.io/projects/insactor/">InsActor</a>: Instruction-driven Physics-based Characters, Ren et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://github.com/ZcyMonkey/AttT2M">AttT2M</a>: Text-Driven Human Motion Generation with Multi-Perspective Attention Mechanism, Zhong et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://mathis.petrovich.fr/tmr">TMR</a>: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis, Petrovich et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://azadis.github.io/make-an-animation">MAA</a>: Make-An-Animation: Large-Scale Text-conditional 3D Human Motion Generation, Azadi et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://nvlabs.github.io/PhysDiff">PhysDiff</a>: Physics-Guided Human Motion Diffusion Model, Yuan et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html">ReMoDiffuse</a>: Retrieval-Augmented Motion Diffusion Model, Zhang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://barquerogerman.github.io/BeLFusion/">BelFusion</a>: Latent Diffusion for Behavior-Driven Human Motion Prediction, Barquero et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://korrawe.github.io/gmd-project/">GMD</a>: Guided Motion Diffusion for Controllable Human Motion Synthesis, Karunratanakul et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Aliakbarian_HMD-NeMo_Online_3D_Avatar_Motion_Generation_From_Sparse_Observations_ICCV_2023_paper.html">HMD-NeMo</a>: Online 3D Avatar Motion Generation From Sparse Observations, Aliakbarian et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://sinc.is.tue.mpg.de/">SINC</a>: Spatial Composition of 3D Human Motions for Simultaneous Action Generation, Athanasiou et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Kong_Priority-Centric_Human_Motion_Generation_in_Discrete_Latent_Space_ICCV_2023_paper.html">Kong et al.</a>: Priority-Centric Human Motion Generation in Discrete Latent Space, Kong et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Fg-T2M_Fine-Grained_Text-Driven_Human_Motion_Generation_via_Diffusion_Model_ICCV_2023_paper.html">Fg-T2M</a>: Fine-Grained Text-Driven Human Motion Generation via Diffusion Model, Wang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Qian_Breaking_The_Limits_of_Text-conditioned_3D_Motion_Synthesis_with_Elaborative_ICCV_2023_paper.html">EMS</a>: Breaking The Limits of Text-conditioned 3D Motion Synthesis with Elaborative Descriptions, Qian et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://weiyuli.xyz/GenMM/">GenMM</a>: Example-based Motion Synthesis via Generative Motion Matching, Li et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://pku-mocca.github.io/GestureDiffuCLIP-Page/">GestureDiffuCLIP</a>: Gesture Diffusion Model with CLIP Latents, Ao et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://i.cs.hku.hk/~taku/kunkun2023.pdf">BodyFormer</a>: Semantics-guided 3D Body Gesture Synthesis with Transformer, Pang et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://www.speech.kth.se/research/listen-denoise-action/">Alexanderson et al.</a>: Listen, denoise, action! Audio-driven motion synthesis with diffusion models, Alexanderson et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://dulucas.github.io/agrol/">AGroL</a>: Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model, Du et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://talkshow.is.tue.mpg.de/">TALKSHOW</a>: Generating Holistic 3D Human Motion from Speech, Yi et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://mael-zys.github.io/T2M-GPT/">T2M-GPT</a>: Generating Human Motion from Textual Descriptions with Discrete Representations, Zhang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://zixiangzhou916.github.io/UDE/">UDE</a>: A Unified Driving Engine for Human Motion Generation, Zhou et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://github.com/junfanlin/oohmg">OOHMG</a>: Being Comes from Not-being: Open-vocabulary Text-to-Motion Generation with Wordless Training, Lin et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://edge-dance.github.io/">EDGE</a>: Editable Dance Generation From Music, Tseng et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://chenxin.tech/mld">MLD</a>: Executing your Commands via Motion Diffusion in Latent Space, Chen et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://sigal-raab.github.io/MoDi">MoDi</a>: Unconditional Motion Synthesis from Diverse Data, Raab et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://vcai.mpi-inf.mpg.de/projects/MoFusion/">MoFusion</a>: A Framework for Denoising-Diffusion-based Motion Synthesis, Dabral et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://arxiv.org/abs/2303.14926">Mo et al.</a>: Continuous Intermediate Token Learning with Implicit Motion Manifold for Keyframe Based Motion Interpolation, Mo et al.</li>
        <li><b>(ICLR 2023)</b> <a href="https://guytevet.github.io/mdm-page/">HMDM</a>: Human Motion Diffusion Model, Tevet et al.</li>
        <li><b>(TPAMI 2023)</b> <a href="https://mingyuan-zhang.github.io/projects/MotionDiffuse.html">MotionDiffuse</a>: Text-Driven Human Motion Generation with Diffusion Model, Zhang et al.</li>
        <li><b>(TPAMI 2023)</b> <a href="https://www.mmlab-ntu.com/project/bailando/">Bailando++</a>: 3D Dance GPT with Choreographic Memory, Li et al.</li>
        <li><b>(ArXiv 2023)</b> <a href="https://zixiangzhou916.github.io/UDE-2/">UDE-2</a>: A Unified Framework for Multimodal, Multi-Part Human Motion Synthesis, Zhou et al.</li>
        <li><b>(ArXiv 2023)</b> <a href="https://pjyazdian.github.io/MotionScript/">Motion Script</a>: Natural Language Descriptions for Expressive 3D Human Motions, Yazdian et al.</li>
    </ul></details>
    <details>
    <summary><h3>2022 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/c-he/NeMF">NeMF</a>: Neural Motion Fields for Kinematic Animation, He et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://github.com/nv-tlabs/PADL">PADL</a>: Language-Directed Physics-Based Character, Juravsky et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://pku-mocca.github.io/Rhythmic-Gesticulator-Page/">Rhythmic Gesticulator</a>: Rhythm-Aware Co-Speech Gesture Synthesis with Hierarchical Neural Embeddings, Ao et al.</li>
        <li><b>(3DV 2022)</b> <a href="https://teach.is.tue.mpg.de/">TEACH</a>: Temporal Action Composition for 3D Human, Athanasiou et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://github.com/PACerv/ImplicitMotion">Implicit Motion</a>: Implicit Neural Representations for Variable Length Human Motion Generation, Cervantes et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810707.pdf">Zhong et al.</a>: Learning Uncoupled-Modulation CVAE for 3D Action-Conditioned Human Motion Synthesis, Zhong et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://guytevet.github.io/motionclip-page/">MotionCLIP</a>: Exposing Human Motion Generation to CLIP Space, Tevet et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://europe.naverlabs.com/research/computer-vision/posegpt">PoseGPT</a>: Quantizing human motion for large scale generative modeling, Lucas et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://mathis.petrovich.fr/temos/">TEMOS</a>: Generating diverse human motions from textual descriptions, Petrovich et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://ericguo5513.github.io/TM2T/">TM2T</a>: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts, Guo et al.</li>
        <li><b>(SIGGRAPH 2022)</b> <a href="https://hongfz16.github.io/projects/AvatarCLIP.html">AvatarCLIP</a>: Zero-Shot Text-Driven Generation and Animation of 3D Avatars, Hong et al.</li>
        <li><b>(SIGGRAPH 2022)</b> <a href="https://dl.acm.org/doi/10.1145/3528223.3530178">DeepPhase</a>: Periodic autoencoders for learning motion phase manifolds, Starke et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://ericguo5513.github.io/text-to-motion">Guo et al.</a>: Generating Diverse and Natural 3D Human Motions from Text, Guo et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://www.mmlab-ntu.com/project/bailando/">Bailando</a>: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory, Li et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://mathis.petrovich.fr/actor/index.html">ACTOR</a>: Action-Conditioned 3D Human Motion Synthesis with Transformer VAE, Petrovich et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://google.github.io/aichoreographer/">AIST++</a>: AI Choreographer: Music Conditioned 3D Dance Generation with AIST++, Li et al.</li>
        <li><b>(SIGGRAPH 2021)</b> <a href="https://dl.acm.org/doi/10.1145/3450626.3459881">Starke et al.</a>: Neural animation layering for synthesizing martial arts movements, Starke et al.</li>
        <li><b>(CVPR 2021)</b> <a href="https://yz-cnsdqz.github.io/eigenmotion/MOJO/index.html">MOJO</a>: We are More than Our Joints: Predicting how 3D Bodies Move, Zhang et al.</li>
        <li><b>(ECCV 2020)</b> <a href="https://www.ye-yuan.com/dlow">DLow</a>: Diversifying Latent Flows for Diverse Human Motion Prediction, Yuan et al.</li>
        <li><b>(SIGGRAPH 2020)</b> <a href="https://www.ipab.inf.ed.ac.uk/cgvu/basketball.pdf">Starke et al.</a>: Local motion phases for learning multi-contact character movements, Starke et al.</li>
    </ul></details>
</ul></details>


<span id="motion-editing"></span>
## Motion Editing
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <li><b>(IVA 2025)</b> <a href="https://arxiv.org/abs/2507.00792">TF-JAX-IK</a>: Real-Time Inverse Kinematics for Generating Multi-Constrained Movements of Virtual Human Characters, Voss et al.</li>
    <li><b>(ICCV 2025)</b> <a href="https://yz-cnsdqz.github.io/eigenmotion/PRIMAL/">PRIMAL</a>: Physically Reactive and Interactive Motor Model for Avatar Learning, Zhang et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://seokhyeonhong.github.io/projects/salad/">SALAD</a>: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing, Hong et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://www.pabloruizponce.com/papers/MixerMDM">MixerMDM</a>: Learnable Composition of Human Motion Diffusion Models, Ruiz-Ponce et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://kwanyun.github.io/AnyMoLe_page/">AnyMoLe</a>: Any Character Motion In-Betweening Leveraging Video Diffusion Models, Yun et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://github.com/lzhyu/SimMotionEdit">SimMotionEdit</a>: Text-Based Human Motion Editing with Motion Similarity Prediction, Li et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://awfuact.github.io/motionrefit/">MotionReFit</a>: Dynamic Motion Blending for Versatile Motion Editing, Jiang et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.03154">StableMotion</a>: Training Motion Cleanup Models with Unpaired Corrupted Data, Mu et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.08180">Dai et al</a>: Towards Synthesized and Editable Motion In-Betweening Through Part-Wise Phase Representation, Dai et al.</li>
    <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://motionfix.is.tue.mpg.de/">MotionFix</a>: Text-Driven 3D Human Motion Editing, Athanasiou et al.</li>
    <li><b>(NeurIPS 2024)</b> <a href="https://btekin.github.io/">CigTime</a>: Corrective Instruction Generation Through Inverse Motion Editing, Fang et al.</li>
    <li><b>(SIGGRAPH 2024)</b> <a href="https://purvigoel.github.io/iterative-motion-editing/">Iterative Motion Editing</a>: Iterative Motion Editing with Natural Language, Goel et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://korrawe.github.io/dno-project/">DNO</a>: Optimizing Diffusion Noise Can Serve As Universal Motion Priors, Karunratanakul et al.</li>
</ul></details>

<span id="motion-stylization"></span>
## Motion Stylization
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <li><b>(ICCV 2025)</b> <a href="https://stylemotif.github.io/">StyleMotif</a>: Multi-Modal Motion Stylization using Style-Content Cross Fusion, Guo et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://cvlab-kaist.github.io/Visual-Persona/">Visual Persona</a>: Foundation Model for Full-Body Human Customization, Nam et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.00173">MotionPersona</a>: Characteristics-aware Locomotion Control, Shi et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://haimsaw.github.io/LoRA-MDM/">Dance Like a Chicken</a>: Low-Rank Stylization for Human Motion Diffusion, Sawdayee et al.</li>
    <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2412.09901">MulSMo</a>: Multimodal Stylized Motion Generation by Bidirectional Control Flow, Li et al.</li>
    <li><b>(TSMC 2024)</b> <a href="https://arxiv.org/abs/2412.04097">D-LORD</a>: D-LORD for Motion Stylization, Gupta et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://otaheri.github.io/publication/2024_humos/">HUMOS</a>: Human Motion Model Conditioned on Body Shape, Tripathi et al.</li>
    <li><b>(SIGGRAPH 2024)</b> <a href="https://dl.acm.org/doi/10.1145/3641519.3657457">SMEAR</a>: Stylized Motion Exaggeration with ARt-direction, Basset et al.</li>
    <li><b>(SIGGRAPH 2024)</b> <a href="https://onethousandwu.com/portrait3d.github.io/">Portrait3D</a>: Text-Guided High-Quality 3D Portrait Generation Using Pyramid Representation and GANs Prior, Wu et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://xingliangjin.github.io/MCM-LDM-Web/">MCM-LDM</a>: Arbitrary Motion Style Transfer with Multi-condition Motion Latent Diffusion Model, Song et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://boeun-kim.github.io/page-MoST/">MoST</a>: Motion Style Transformer between Diverse Action Contents, Kim et al.</li>
    <li><b>(ICLR 2024)</b> <a href="https://yxmu.foo/GenMoStyle/">GenMoStyle</a>: Generative Human Motion Stylization in Latent Space, Guo et al.</li>
</ul></details>

<span id="hoi"></span>
## Human-Object Interaction
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ACM MM 2025)</b> <a href="https://arxiv.org/abs/2508.06205">PA-HOI</a>: A Physics-Aware Human and Object Interaction Dataset, Wang et al.</li>
        <li><b>(ACM MM 2025)</b> <a href="https://arxiv.org/abs/2509.12250">OnlineHOI</a>: Towards Online Human-Object Interaction Generation and Perception, Ji et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://liangxuy.github.io/InterVLA/">Perceiving and Acting in First-Person</a>: A Dataset and Benchmark for Egocentric Human-Object-Human Interactions, Xu et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/tridi/">TriDi</a>: Trilateral Diffusion of 3D Humans, Objects and Interactions, Petrov et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2411.16216">SMGDiff</a>: Soccer Motion Generation using diffusion probabilistic models, Yang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://syncdiff.github.io/">SyncDiff</a>: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis, He et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://hoifhli.github.io/">Wu et al</a>: Human-Object Interaction from Human-Level Instructions, Wu et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://jiaxin-lu.github.io/humoto/">HUMOTO</a>: A 4D Dataset of Mocap Human Object Interactions, Lu et al.</li>
        <li><b>(SIGGRAPH 2025)</b> <a href="hhttps://arxiv.org/abs/2504.21216">PhysicsFC</a>: Learning User-Controlled Skills for a Physics-Based Football Player Controller, Kim et al.</li>
        <li><b>(SIGGRAPH 2025)</b> <a href="https://arxiv.org/abs/2505.02094">SkillMimic-v2</a>: Learning Robust and Generalizable Interaction Skills from Sparse and Noisy Demonstrations, Yu et al.</li>
        <li><b>(Bioengineering 2025)</b> <a href="https://www.mdpi.com/2306-5354/12/3/317">MeLLO</a>: The Utah Manipulation and Locomotion of Large Objects (MeLLO) Data Library, Luttmer et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2503.13130">ChainHOI</a>: Joint-based Kinematic Chain Modeling for Human-Object Interaction Generation, Zeng et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://www.mingzhenhuang.com/projects/hoigpt.html">HOIGPT</a>: Learning Long Sequence Hand-Object Interaction with Language Models, Huang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2503.18134">Hui et al</a>: An Image-like Diffusion Method for Human-Object Interaction Detection, Hui et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2501.05823">PersonaHOI</a>: Effortlessly Improving Personalized Face with Human-Object Interaction Generation, Hu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://interactvlm.is.tue.mpg.de">InteractVLM</a>: 3D Interaction Reasoning from 2D Foundational Models, Dwivedi et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://pico.is.tue.mpg.de">PICO</a>: Reconstructing 3D People In Contact with Objects, Cseke et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://lym29.github.io/EasyHOI-page/">EasyHOI</a>: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild, Liu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://vision.cs.utexas.edu/projects/FIction/">FIction</a>: 4D Future Interaction Prediction from Video, Ashutosh et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://lalalfhdh.github.io/rog_page/">ROG</a>: Guiding Human-Object Interactions with Rich Geometry and Relations, Xue et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://4dvlab.github.io/project_page/semgeomo/">SemGeoMo</a>: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance, Cong et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://liyitang22.github.io/phys-reach-grasp/">Phys-Reach-Grasp</a>: Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References, Li et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://jlogkim.github.io/parahome/">ParaHome</a>: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions, Kim et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://sirui-xu.github.io/InterMimic/">InterMimic</a>: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions, Xu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://core4d.github.io/">CORE4D</a>: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement, Zhang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://jinluzhang.site/projects/interactanything/">InteractAnything</a>: Zero-shot Human Object-Interaction Synthesis via LLM Feedback and Object Affordance Parsing, Zhang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://ingrid789.github.io/SkillMimic/">SkillMimic</a>: Learning Reusable Basketball Skills from Demonstrations, Wang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2501.04595">MobileH2R</a>: Learning Generalizable Human to Mobile Robot Handover Exclusively from Scalable and Diverse Synthetic Data, Wang et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://arxiv.org/abs/2503.16801">ARDHOI</a>: Auto-Regressive Diffusion for Generating 3D Human-Object Interactions, Geng et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://iscas3dv.github.io/DiffGrasp/">DiffGrasp</a>: Whole-Body Grasping Synthesis Guided by Object Motion Using a Diffusion Model, Zhang et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://arxiv.org/abs/2408.16770?">Paschalidis et al</a>: 3D Whole-body Grasp Synthesis with Directional Controllability, Paschalidis et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/InterTrack/">InterTrack</a>: Tracking Human Object Interaction without Object Templates, Xie et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://arxiv.org/abs/2403.11237">FORCE</a>: Dataset and Method for Intuitive Physics Guided Human-object Interaction, Zhang et al.</li>
        <li><b>(PAMI 2025)</b> <a href="https://arxiv.org/abs/2509.23635">MotionVerse</a>: A Unified Multimodal Framework for Motion Comprehension, Generation and Editing, Hou et al.</li>
        <li><b>(PAMI 2025)</b> <a href="https://arxiv.org/abs/2503.00382">EigenActor</a>: Variant Body-Object Interaction Generation Evolved from Invariant Action Basis Reasoning, Guo et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.23612">InteractMove</a>: Text-Controlled Human-Object Interaction Generation in 3D Scenes with Movable Objects, Cai et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://mael-zys.github.io/InterPose/">InterPose</a>: Learning to Generate Human-Object Interactions from Large-Scale Web Videos, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.21556">ECHO</a>: Ego-Centric modeling of Human-Object interactions, Petrov et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.07162">CoopDiff</a>: Anticipating 3D Human-object Interactions via Contact-consistent Decoupled Diffusion, Lin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.01737">HOI-Dyn</a>: Learning Interaction Dynamics for Human-Object Motion Diffusion, Wu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://hoidini.github.io/">HOIDiNi</a>: Human-Object Interaction through Diffusion Noise Optimization, Ron et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.15483">GenHOI</a>: Generalizing Text-driven 4D Human-Object Interaction Synthesis for Unseen Objects, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://hoipage.github.io/">HOI-PAGE</a>: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://yw0208.github.io/hosig/">HOSIG</a>: Full-Body Human-Object-Scene Interaction Generation, Yao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://phj128.github.io/page/CoDA/index.html">CoDA</a>: Coordinated Diffusion Noise Optimization for Whole-Body Manipulation of Articulated Objects, Pi et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.19086">MaskedManipulator</a>: Versatile Whole-Body Control for Loco-Manipulation, Tessler et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.12774">UniHM</a>: Universal Human Motion Generation with Object Interactions in Indoor Scenes, Geng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.23121">EJIM</a>: Efficient Explicit Joint-level Interaction Modeling with Mamba for Text-guided HOI Generation, Huang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://thorin666.github.io/projects/ZeroHOI/">ZeroHOI</a>: Zero-Shot Human-Object Interaction Synthesis with Multimodal Priors, Lou et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://rmd-hoi.github.io/">RMD-HOI</a>: Human-Object Interaction with Vision-Language Model Guided Relative Movement Dynamics, Deng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.05231">Kaiwu</a>: A Multimodal Manipulation Dataset and Framework for Robot Learning and Human-Robot Interaction, Jiang et al.</li>
    </ul></details>
    <details>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2412.06702">CHOICE</a>: Coordinated Human-Object Interaction in Cluttered Environments for Pick-and-Place Actions, Lu et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://nickk0212.github.io/ood-hoi/">OOD-HOI</a>: Text-Driven 3D Whole-Body Human-Object Interactions Generation Beyond Training Domains, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2409.20502">COLLAGE</a>: Collaborative Human-Agent Interaction Generation using Hierarchical Latent Diffusion and Language Models, Daiya et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/abs/2406.19972">HumanVLA</a>: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid, Xu et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://www.zhengyiluo.com/Omnigrasp-Site/">OmniGrasp</a>: Grasping Diverse Objects with Simulated Humanoids, Luo et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://yyvhang.github.io/EgoChoir/">EgoChoir</a>: Capturing 3D Human-Object Interaction Regions from Egocentric Views, Yang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://gao-jiawei.com/Research/CooHOI/">CooHOI</a>: Learning Cooperative Human-Object Interaction with Manipulated Object Dynamics, Gao et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/abs/2403.19652">InterDreamer</a>: Zero-Shot Text to 3D Dynamic Human-Object Interaction, Xu et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://pimforce.hcitech.org/">PiMForce</a>: Posture-Informed Muscular Force Learning for Robust Hand Pressure Estimation, Seo et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://sisidai.github.io/InterFusion/">InterFusion</a>: Text-Driven Generation of 3D Human-Object Interaction, Dai et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://lijiaman.github.io/projects/chois/">CHOIS</a>: Controllable Human-Object Interaction Synthesis, Li et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://f-hoi.github.io/">F-HOI</a>: Toward Fine-grained Semantic-Aligned 3D Human-Object Interactions, Yang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://lvxintao.github.io/himo/">HIMO</a>: A New Benchmark for Full-Body Human Interacting with Multiple Objects, Lv et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://jiashunwang.github.io/PhysicsPingPong/">PhysicsPingPong</a>: Strategy and Skill Learning for Physics-based Table Tennis Animation, Wang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://nileshkulkarni.github.io/nifty/">NIFTY</a>: Neural Object Interaction Fields for Guided Human Motion Synthesis, Kulkarni et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://zxylinkstart.github.io/HOIAnimator-Web/">HOI Animator</a>: Generating Text-Prompt Human-Object Animations using Novel Perceptive Diffusion Models, Son et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://cg-hoi.christian-diller.de/#main">CG-HOI</a>: Contact-Guided 3D Human-Object Interaction Generation, Diller et al.</li>
        <li><b>(IJCV 2024)</b> <a href="https://intercap.is.tue.mpg.de/">InterCap</a>: Joint Markerless 3D Tracking of Humans and Objects in Interaction, Huang et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://eth-ait.github.io/phys-fullbody-grasp/">Phys-Fullbody-Grasp</a>: Physically Plausible Full-Body Hand-Object Interaction Synthesis, Braun et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://grip.is.tue.mpg.de">GRIP</a>: Generating Interaction Poses Using Spatial Cues and Latent Consistency, Taheri et al.</li>
        <li><b>(AAAI 2024)</b> <a href="https://kailinli.github.io/FAVOR/">FAVOR</a>: Full-Body AR-driven Virtual Object Rearrangement Guided by Instruction Text, Li et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://github.com/lijiaman/omomo_release">OMOMO</a>: Object Motion Guided Human Motion Synthesis, Li et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://jnnan.github.io/project/chairs/">CHAIRS</a>: Full-Body Articulated Human-Object Interaction, Jiang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://zju3dv.github.io/hghoi">HGHOI</a>: Hierarchical Generation of Human-Object Interactions with Diffusion Probabilistic Models, Pi et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://sirui-xu.github.io/InterDiff/">InterDiff</a>: Generating 3D Human-Object Interactions with Physics-Informed Diffusion, Xu et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/object_popup/">Object Pop Up</a>: Can we infer 3D objects and their poses from human interactions alone? Petrov et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://arctic.is.tue.mpg.de/">ARCTIC</a>: A Dataset for Dexterous Bimanual Hand-Object Manipulation, Fan et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630001.pdf">TOCH</a>: Spatio-Temporal Object-to-Hand Correspondence for Motion Refinement, Zhou et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/couch/">COUCH</a>: Towards Controllable Human-Chair Interactions, Zhang et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://jiahaoplus.github.io/SAGA/saga.html">SAGA</a>: Stochastic Whole-Body Grasping with Contact, Wu et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://goal.is.tue.mpg.de/">GOAL</a>: Generating 4D Whole-Body Motion for Hand-Object Grasping, Taheri et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/behave/">BEHAVE</a>: Dataset and Method for Tracking Human Object Interactions, Bhatnagar et al.</li>
        <li><b>(ECCV 2020)</b> <a href="https://grab.is.tue.mpg.de/">GRAB</a>: A Dataset of Whole-Body Human Grasping of Objects, Taheri et al.</li>
    </ul></details>
</ul></details>

<span id="hsi"></span>
## Human-Scene Interaction
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ICCV 2025)</b> <a href="https://beingbeyond.github.io/Being-M0.5/">Being-M0.5</a>: A Real-Time Controllable Vision-Language-Motion Model, Cao et al.</li>
        <li><b>(ICCV 2025)</b> <a href="http://inwoohwang.me/SceneMI">SceneMI</a>: Motion In-Betweening for Modeling Human-Scene Interactions, Hwang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2411.19921">SIMS</a>: Simulating Human-Scene Interactions with Real World Script Planning, Wang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://rms0329.github.io/Event-Driven-Storytelling/">Lim et al</a>: Event-Driven Storytelling with Multiple Lifelike Humans in a 3D scene, Lim et al.</li>
        <li><b>(ICME 2025)</b> <a href="https://tstmotion.github.io/">TSTMotion</a>: Training-free Scene-aware Text-to-motion Generation, Guo et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://openaccess.thecvf.com//content/CVPR2025/papers/Wang_HSI-GPT_A_General-Purpose_Large_Scene-Motion-Language_Model_for_Human_Scene_Interaction_CVPR_2025_paper.pdf">HSI-GPT</a>: A General-Purpose Large Scene-Motion-Language Model for Human Scene Interaction. Wang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://gap3ds.github.io/">Vision-Guided Action</a>: Enhancing 3D Human Motion Prediction with Gaze-informed Affordance in 3D Scenes. Yu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://egoallo.github.io/">Yi et al</a>: Estimating Body and Hand Motion in an Ego‑sensed World, Yi et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2412.10235">EnvPoser</a>: Environment-aware Realistic Human Motion Estimation from Sparse Observations with Uncertainty Modeling. Xia et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://liangpan99.github.io/TokenHSI/">TokenHSI</a>: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization, Pan et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://github.com/WindVChen/Sitcom-Crafter">Sitcom-Crafter</a>: A Plot-Driven Human Motion Generation System in 3D Scenes, Chen et al. </li>
        <li><b>(3DV 2025)</b> <a href="https://arxiv.org/abs/2408.16770?">Paschalidis et al</a>: 3D Whole-body Grasp Synthesis with Directional Controllability, Paschalidis et al.</li>
        <li><b>(WACV 2025)</b> <a href="https://arxiv.org/abs/2405.18438">GHOST</a>: Grounded Human Motion Generation with Open Vocabulary Scene-and-Text Contexts, Milacski et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://fantasy-amap.github.io/fantasy-hsi/">FantasyHSI</a>: Video-Generation-Centric 4D Human Synthesis In Any Scene through A Graph-based Multi-Agent Framework, Mu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://lisiyao21.github.io/projects/Half-Physics/">Half-Physics</a>: Enabling Kinematic 3D Human Model with Physical Interactions, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.19840">GenHSI</a>: Controllable Generation of Human-Scene Interaction Videos, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://rmd-hoi.github.io/">RMD-HOI</a>: Human-Object Interaction with Vision-Language Model Guided Relative Movement Dynamics, Deng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.12955">HIS-GPT</a>: Towards 3D Human-In-Scene Multimodal Understanding, Zhao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.00371">Jointly Understand Your Command and Intention</a>: Reciprocal Co-Evolution between Scene-Aware 3D Human Motion Synthesis and Analysis, Gao et al.</li>
    </ul></details>
    <details open>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://awfuact.github.io/zerohsi/">ZeroHSI</a>: Zero-Shot 4D Human-Scene Interaction by Video Generation, Li et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://mimicking-bench.github.io/">Mimicking-Bench</a>: A Benchmark for Generalizable Humanoid-Scene Interaction Learning via Human Mimicking, Liu et al. </li>
        <li><b>(ArXiv 2024)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/scenic/">SCENIC</a>: Scene-aware Semantic Navigation with Instruction-guided Control, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://jingyugong.github.io/DiffusionImplicitPolicy/">Diffusion Implicit Policy</a>: Diffusion Implicit Policy for Unpaired Scene-aware Motion synthesis, Gong et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://github.com/4DVLab/LaserHuman">LaserHuman</a>: Language-guided Scene-aware Human Motion Generation in Free Environment, Cong et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://lingomotions.com/">LINGO</a>: Autonomous Character-Scene Interaction Synthesis from Text Instruction, Jiang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://sites.google.com/view/dimop3d">DiMoP3D</a>: Harmonizing Stochasticity and Determinism: Scene-responsive Diverse Human Motion Prediction, Lou et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/html/2312.02700v2">MOB</a>: Revisit Human-Scene Interaction via Space Occupancy, Liu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://research.nvidia.com/labs/toronto-ai/tesmo/">TesMo</a>: Generating Human Interaction Motions in Scenes with Text Control, Yi et al.</li>
        <li><b>(ECCV 2024 Workshop)</b> <a href="https://github.com/felixbmuller/SAST">SAST</a>: Massively Multi-Person 3D Human Motion Forecasting with Scene Context, Mueller et al.</li>
        <li><b>(Eurographics 2024)</b> <a href="https://diglib.eg.org/server/api/core/bitstreams/f1072102-82a6-4228-a140-9ccf09f21077/content">Kang et al</a>: Learning Climbing Controllers for Physics-Based Characters, Kang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://afford-motion.github.io/">Afford-Motion</a>: Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance, Wang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://craigleili.github.io/projects/genzi/">GenZI</a>: Zero-Shot 3D Human-Scene Interaction Generation, Li et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://zju3dv.github.io/text_scene_motion/">Cen et al.</a>: Generating Human Motion in 3D Scenes from Text Descriptions, Cen et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://jnnan.github.io/trumans/">TRUMANS</a>: Scaling Up Dynamic Human-Scene Interaction Modeling, Jiang et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://xizaoqu.github.io/unihsi/">UniHSI</a>: Unified Human-Scene Interaction via Prompted Chain-of-Contacts, Xiao et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://arxiv.org/abs/2404.12942">Purposer</a>: Putting Human Motion Generation in Context, Ugrinovic et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://ieeexplore.ieee.org/abstract/document/10550906">InterScene</a>: Synthesizing Physically Plausible Human Motions in 3D Scenes, Pan et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://arxiv.org/abs/2304.02061">Mir et al</a>: Generating Continual Human Motion in Diverse 3D Scenes, Mir et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ICCV 2023)</b> <a href="https://github.com/zkf1997/DIMOS">DIMOS</a>: Synthesizing Diverse Human Motions in 3D Indoor Scenes, Zhao et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://jiyewise.github.io/projects/LAMA/">LAMA</a>: Locomotion-Action-Manipulation: Synthesizing Human-Scene Interactions in Complex 3D Environments, Lee et al.</li>
        <li><b>(ICCV 2023)</b> <a href="http://cic.tju.edu.cn/faculty/likun/projects/Narrator">Narrator</a>: Towards Natural Control of Human-Scene Interaction Generation via Relationship Reasoning, Xuan et al.</li>
        <li><b>(CVPR 2023)</b> <a href="http://www.lidarhumanmotion.net/cimi4d">CIMI4D</a>: A Large Multimodal Climbing Motion Dataset under Human-Scene Interactions, Yan et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://people.mpi-inf.mpg.de/~jianwang/projects/sceneego/">Scene-Ego</a>: Scene-aware Egocentric 3D Human Pose Estimation, Wang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="http://www.lidarhumanmotion.net/sloper4d">SLOPER4D</a>: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments, Dai et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://stanford-tml.github.io/circle_dataset/">CIRCLE</a>: Capture in Rich Contextual Environments, Araujo et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://scenediffuser.github.io/">SceneDiffuser</a>: Diffusion-based Generation, Optimization, and Planning in 3D Scenes, Huang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://mime.is.tue.mpg.de/">MIME</a>: Human-Aware 3D Scene Generation, Yi et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://github.com/jinseokbae/pmp">PMP</a>: Learning to Physically Interact with Environments using Part-wise Motion Priors, Bae et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://dl.acm.org/doi/10.1145/3588432.3591504">QuestEnvSim</a>: Environment-Aware Simulated Motion Tracking from Sparse Sensors, Lee et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://research.nvidia.com/publication/2023-08_synthesizing-physical-character-scene-interactions">Hassan et al.</a>: Synthesizing Physical Character-Scene Interactions, Hassan et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/wei-mao-2019/ContAwareMotionPred">Mao et al.</a>: Contact-Aware Human Motion Forecasting, Mao et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/Silverster98/HUMANISE">HUMANISE</a>: Language-conditioned Human Motion Generation in 3D Scenes, Wang et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/ZhengyiLuo/EmbodiedPose">EmbodiedPose</a>: Embodied Scene-aware Human Pose Estimation, Luo et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://github.com/y-zheng18/GIMO">GIMO</a>: Gaze-Informed Human Motion Prediction in Context, Zheng et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://zkf1997.github.io/COINS/index.html">COINS</a>: Compositional Human-Scene Interaction Synthesis with Semantic Control, Zhao et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Towards_Diverse_and_Natural_Scene-Aware_3D_Human_Motion_Synthesis_CVPR_2022_paper.pdf">Wang et al.</a>: Towards Diverse and Natural Scene-aware 3D Human Motion Synthesis, Wang et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://yz-cnsdqz.github.io/eigenmotion/GAMMA/">GAMMA</a>: The Wanderings of Odysseus in 3D Scenes, Zhang et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://samp.is.tue.mpg.de/">SAMP</a>: Stochastic Scene-Aware Motion Prediction, Hassan et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://sanweiliti.github.io/LEMO/LEMO.html">LEMO</a>: Learning Motion Priors for 4D Human Body Capture in 3D Scenes, Zhang et al. </li>
        <li><b>(3DV 2020)</b> <a href="https://sanweiliti.github.io/PLACE/PLACE.html">PLACE</a>: Proximity Learning of Articulation and Contact in 3D Environments, Zhang et al.</li>
        <li><b>(SIGGRAPH 2020)</b> <a href="https://www.ipab.inf.ed.ac.uk/cgvu/basketball.pdf">Starke et al.</a>: Local motion phases for learning multi-contact character movements, Starke et al.</li>
        <li><b>(CVPR 2020)</b> <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Generating_3D_People_in_Scenes_Without_People_CVPR_2020_paper.pdf">PSI</a>: Generating 3D People in Scenes without People, Zhang et al.</li>
        <li><b>(SIGGRAPH Asia 2019)</b> <a href="https://www.ipab.inf.ed.ac.uk/cgvu/nsm.pdf">NSM</a>: Neural State Machine for Character-Scene Interactions, Starke et al.</li>
        <li><b>(ICCV 2019)</b> <a href="https://prox.is.tue.mpg.de/">PROX</a>: Resolving 3D Human Pose Ambiguities with 3D Scene Constraints, Hassan et al.</li>
    </ul></details>
</ul></details>

<span id="hhi"></span>
## Human-Human Interaction
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <li><b>(ICCV 2025)</b> <a href="https://liangxuy.github.io/InterVLA/">Perceiving and Acting in First-Person</a>: A Dataset and Benchmark for Egocentric Human-Object-Human Interactions, Xu et al.</li>
    <li><b>(ICCV 2025)</b> <a href="https://humanx-interaction.github.io/">Towards Immersive Human-X Interaction</a>: A Real-Time Framework for Physically Plausible Motion Synthesis, Ji et al.</li>
    <li><b>(ICCV 2025)</b> <a href="https://sinc865.github.io/pino/">PINO</a>: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups, Ota et al.</li>
    <li><b>(SIGGRAPH 2025)</b> <a href="https://arxiv.org/abs/2505.17860">Xu et al</a>: Multi-Person Interaction Generation from Two-Person Motion Priors, Xu et al.</li>
    <li><b>(SIGGRAPH 2025)</b> <a href="https://arxiv.org/pdf/2506.18680">DuetGen</a>: Music Driven Two-Person Dance Generation via Hierarchical Masked Modeling, Ghosh et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://aigc-explorer.github.io/TIMotion-page/">TIMotion</a>: Temporal and Interactive Framework for Efficient Human-Human Motion Generation, Wang et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=UxzKcIZedp">Think Then React</a>: Towards Unconstrained Action-to-Reaction Motion Generation, Tan et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://zju3dv.github.io/ready_to_react/">Ready-to-React</a>: Online Reaction Policy for Two-Character Interaction Generation, Cen et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://gohar-malik.github.io/intermask">InterMask</a>: 3D Human Interaction Generation via Collaborative Masked Modelling, Javed et al.</li>
    <li><b>(3DV 2025)</b> <a href="https://arxiv.org/abs/2312.08983">Interactive Humanoid</a>: Online Full-Body Motion Reaction Synthesis with Social Affordance Canonicalization and Forecasting, Liu et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.05747">InterAct</a>: A Large-Scale Dataset of Dynamic, Expressive and Interactive Activities between Two People in Daily Scenarios, Ho et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.22554">E-React</a>: Towards Emotionally Controlled Synthesis of Human Reactions, Zhu et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.22554">Seamless Interaction</a>: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset, Agrawal et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.13040">MAMMA</a>: Markerless & Automatic Multi-Person Motion Action Capture, Cuevas-Velasquez et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://yw0208.github.io/physiinter/">PhysInter</a>: Integrating Physical Mapping for High-Fidelity Human Interaction Generation, Yao et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.03084">InterMamba</a>: Efficient Human-Human Interaction Generation with Adaptive Spatio-Temporal Mamba, Wu et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.11334">MARRS</a>: MaskedAutoregressive Unit-based Reaction Synthesis, Wang et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://socialgenx.github.io/">SocialGen</a>: Modeling Multi-Human Social Interaction with Language Models, Yu et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arflow2025.github.io/">ARFlow</a>: Human Action-Reaction Flow Matching with Physical Guidance, Jiang et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.13120">Fan et al</a>: 3D Human Interaction Generation: A Survey, Fan et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.04816">Invisible Strings</a>: Revealing Latent Dancer-to-Dancer Interactions with Graph Neural Networks, Zerkowski et al. </li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2502.11563">Leader and Follower</a>: Interactive Motion Generation under Trajectory Constraints, Wang et al. </li>
    <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2412.16670">Two in One</a>: Unified Multi-Person Interactive Motion Generation by Latent Diffusion Transformer, Li et al.</li>
    <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2412.02419">It Takes Two</a>: Real-time Co-Speech Two-person’s Interaction Generation via Reactive Auto-regressive Diffusion Model, Shi et al.</li>
    <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/abs/2409.20502">COLLAGE</a>: Collaborative Human-Agent Interaction Generation using Hierarchical Latent Diffusion and Language Models, Daiya et al.</li>
    <li><b>(NeurIPS 2024)</b> <a href="https://github.com/zhenzhiwang/intercontrol">InterControl</a>: Generate Human Motion Interactions by Controlling Every Joint, Wang et al.</li>
    <li><b>(ACM MM 2024)</b> <a href="https://yunzeliu.github.io/PhysReaction/">PhysReaction</a>: Physically Plausible Real-Time Humanoid Reaction Synthesis via Forward Dynamics Guided 4D Imitation, Liu et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2405.18483">Shan et al</a>: Towards Open Domain Text-Driven Synthesis of Multi-Person Motions, Shan et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://vcai.mpi-inf.mpg.de/projects/remos/">ReMoS</a>: 3D Motion-Conditioned Reaction Synthesis for Two-Person Interactions, Ghosh et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://liangxuy.github.io/inter-x/">Inter-X</a>: Towards Versatile Human-Human Interaction Analysis, Xu et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://github.com/liangxuy/ReGenNet">ReGenNet</a>: Towards Human Action-Reaction Synthesis, Xu et al.</li>
    <li><b>(CVPR Workshop 2024)</b> <a href="https://openaccess.thecvf.com/content/CVPR2024W/HuMoGen/html/Ruiz-Ponce_in2IN_Leveraging_Individual_Information_to_Generate_Human_INteractions_CVPRW_2024_paper.html">in2IN</a>: in2IN: Leveraging Individual Information to Generate Human INteractions, Ruiz-Ponce et al.</li>
    <li><b>(IJCV 2024)</b> <a href="https://tr3e.github.io/intergen-page/">InterGen</a>: Diffusion-based Multi-human Motion Generation under Complex Interactions, Liang et al.</li>
    <li><b>(ICCV 2023)</b> <a href="https://liangxuy.github.io/actformer/">ActFormer</a>: A GAN-based Transformer towards General Action-Conditioned 3D Human Motion Generation, Xu et al.</li>
    <li><b>(ICCV 2023)</b> <a href="https://github.com/line/Human-Interaction-Generation">Tanaka et al.</a>: Role-aware Interaction Generation from Textual Description, Tanaka et al.</li>
    <li><b>(CVPR 2023)</b> <a href="https://yifeiyin04.github.io/Hi4D/">Hi4D</a>: 4D Instance Segmentation of Close Human Interaction, Yin et al.</li>
    <li><b>(CVPR 2022)</b> <a href="https://github.com/GUO-W/MultiMotion">ExPI</a>: Multi-Person Extreme Motion Prediction, Guo et al.</li>
    <li><b>(CVPR 2020)</b> <a href="https://ci3d.imar.ro/home">CHI3D</a>: Three-Dimensional Reconstruction of Human Interactions, Fieraru et al.</li>
</ul></details>

<span id="datasets"></span>
## Datasets & Benchmarks
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2508.08179">PP-Motion</a>: Physical-Perceptual Fidelity Evaluation for Human Motion Generation, Zhao et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://gprerit96.github.io/mdd-page/">MDD</a>: A Dataset for Text-and-Music Conditioned Duet Dance Generation, Gupta et al.</li>
        <li><b>(ACM MM 2025)</b> <a href="https://liangxuy.github.io/InterVLA/">Perceiving and Acting in First-Person</a>: A Dataset and Benchmark for Egocentric Human-Object-Human Interactions, Xu et al.</li>
        <li><b>(Bioengineering 2025)</b> <a href="https://www.mdpi.com/2306-5354/12/3/317">MeLLO</a>: The Utah Manipulation and Locomotion of Large Objects (MeLLO) Data Library, Luttmer et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://fudan-generative-vision.github.io/OpenHumanVid/#/">OpenHumanVid</a>: A Large-Scale High-Quality Dataset for Enhancing Human-Centric Video Generation, Xu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://github.com/wzyabcas/InterAct">InterAct</a>: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation, Xu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://nju-cite-mocaphumanoid.github.io/MotionPRO/">MotionPro</a>: Exploring the Role of Pressure in Human MoCap and Beyond, Ren et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://shape-move.github.io/">GORP</a>: Real-Time Motion Generation with Rolling Prediction Models, Barquero et al.</li>
        <li><b>(CVPR 2025)</b> <a href="http://www.lidarhumanmotion.net/climbingcap/">ClimbingCap</a>: ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate, Yan et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://atom-motion.github.io/">AtoM</a>: AToM: Aligning Text-to-Motion Model at Event-Level with GPT-4Vision Reward, Han et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://core4d.github.io/">CORE4D</a>: CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement, Zhang et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://motioncritic.github.io/">MotionCritic</a>: Aligning Human Motion Generation with Human Perceptions, Wang et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=9mBodivRIo">LocoVR</a>: LocoVR: Multiuser Indoor Locomotion Dataset in Virtual Reality, Takeyama et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://github.com/coding-rachal/PMRDataset">PMR</a>: Pedestrian Motion Reconstruction: A Large-scale Benchmark via Mixed Reality Rendering with Multiple Perspectives and Modalities, Wang et al.</li>
        <li><b>(AAAI 2025)</b> <a href="https://arxiv.org/abs/2408.17168">EMHI</a>: EMHI: A Multimodal Egocentric Human Motion Dataset with HMD and Body-Worn IMUs, Fan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.20491">CaddieSet</a>: A Golf Swing Dataset with Human Joint Features and Ball Information, Jung et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://github.com/GuangxunZhu/Waymo-3DSkelMo">Waymo-3DSkelMo</a>: A Multi-Agent 3D Skeletal Motion Dataset for Pedestrian Interaction Modeling in Autonomous Driving, Zhu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://dorniwang.github.io/SpeakerVid-5M/">SpeakerVid-5M</a>: A Large-Scale High-Quality Dataset for audio-visual Dyadic Interactive Human Generation, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.12905">AthleticsPose</a>: Authentic Sports Motion Dataset on Athletic Field and Evaluation of Monocular 3D Pose Estimation Ability, Suzuki et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://mmhu-benchmark.github.io/">MMHU</a>: A Massive-Scale Multimodal Benchmark for Human Behavior Understanding, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.00043">FLEX</a>: A Large-Scale Multi-Modal Multi-Action Dataset for Fitness Action Quality Assessment, Yin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.00043">From Motion to Behavior</a>: Hierarchical Modeling of Humanoid Generative Behavior Control, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.23301">Rekik et al</a>: Quality assessment of 3D human animation: Subjective and objective evaluation, Rekik et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://k2muse.github.io/">K2MUSE</a>: A Large-scale Human Lower limb Dataset of Kinematics, Kinetics, amplitude Mode Ultrasound and Surface Electromyography, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://rmd-hoi.github.io/">RMD-HOI</a>: Human-Object Interaction with Vision-Language Model Guided Relative Movement Dynamics, Deng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.06522">SGA-INTERACT</a>: SGA-INTERACT: A3DSkeleton-based Benchmark for Group Activity Understanding in Modern Basketball Tactic, Yang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.05231">Kaiwu</a>: Kaiwu: A Multimodal Manipulation Dataset and Framework for Robot Learning and Human-Robot Interaction, Jiang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2501.05098">Motion-X++</a>: Motion-X++: A Large-Scale Multimodal 3D Whole-body Human Motion Dataset, Zhang et al.</li>
    </ul></details>
    <details open>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://mimicking-bench.github.io/">Mimicking-Bench</a>: A Benchmark for Generalizable Humanoid-Scene Interaction Learning via Human Mimicking, Liu et al. </li>
        <li><b>(ArXiv 2024)</b> <a href="https://github.com/4DVLab/LaserHuman">LaserHuman</a>: Language-guided Scene-aware Human Motion Generation in Free Environment, Cong et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/scenic/">SCENIC</a>: Scene-aware Semantic Navigation with Instruction-guided Control, Zhang et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://von31.github.io/synNsync/">synNsync</a>: Synergy and Synchrony in Couple Dances, Manukele et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://github.com/liangxuy/MotionBank">MotionBank</a>: A Large-scale Video Motion Benchmark with Disentangled Rule-based Annotations, Xu et al.</li>
        <li><b>(Github 2024)</b> <a href="https://github.com/fyyakaxyy/AnimationGPT">CMP & CMR</a>: AnimationGPT: An AIGC tool for generating game combat motion assets, Liao et al.</li>
        <li><b>(Scientific Data 2024)</b> <a href="https://www.nature.com/articles/s41597-024-04077-3?fromPaywallRec=false">Evans et al</a>: Synchronized Video, Motion Capture and Force Plate Dataset for Validating Markerless Human Movement Analysis, Evans et al.</li>
        <li><b>(Scientific Data 2024)</b> <a href="https://www.nature.com/articles/s41597-024-03144-z">MultiSenseBadminton</a>: MultiSenseBadminton: Wearable Sensor–Based Biomechanical Dataset for Evaluation of Badminton Performance, Seong et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://lingomotions.com/">LINGO</a>: Autonomous Character-Scene Interaction Synthesis from Text Instruction, Jiang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://jyuntins.github.io/harmony4d/">Harmony4D</a>: Harmony4D: A Video Dataset for In-The-Wild Close Human Interactions, Khirodkar et al.</li>
        <li><b>(NeurIPS D&B 2024)</b> <a href="https://siplab.org/projects/EgoSim">EgoSim</a>: EgoSim: An Egocentric Multi-view Simulator for Body-worn Cameras during Human Motion, Hollidt et al.</li>
        <li><b>(NeurIPS D&B 2024)</b> <a href="https://simplexsigil.github.io/mint">Muscles in Time</a>: Muscles in Time: Learning to Understand Human Motion by Simulating Muscle Activations, Schneider et al.</li>
        <li><b>(NeurIPS D&B 2024)</b> <a href="https://blindways.github.io/">Text to blind motion</a>: Text to blind motion, Kim et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://dl.acm.org/doi/abs/10.1145/3664647.3685523">CLaM</a>: CLaM: An Open-Source Library for Performance Evaluation of Text-driven Human Motion Generation, Chen et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://addbiomechanics.org/">AddBiomechanics</a>: AddBiomechanics Dataset: Capturing the Physics of Human Motion at Scale, Werling et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://4dvlab.github.io/project_page/LiveHPS2.html">LiveHPS++</a>: Robust and Coherent Motion Capture in Dynamic Free Environment, Ren et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://signavatars.github.io/">SignAvatars</a>: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark, Yu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://www.projectaria.com/datasets/nymeria">Nymeria</a>: A massive collection of multimodal egocentric daily motion in the wild, Ma et al.</li>
        <li><b>(Multibody System Dynamics 2024)</b> <a href="https://github.com/ainlamyae/Human3.6Mplus">Human3.6M+</a>: Using musculoskeletal models to generate physically-consistent data for 3D human pose, kinematic, dynamic, and muscle estimation, Nasr et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://liangxuy.github.io/inter-x/">Inter-X</a>: Towards Versatile Human-Human Interaction Analysis, Xu et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_HardMo_A_Large-Scale_Hardcase_Dataset_for_Motion_Capture_CVPR_2024_paper.pdf">HardMo</a>: A Large-Scale Hardcase Dataset for Motion Capture, Liao et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/">Xie et al</a>: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation, Xie et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://metaverse-ai-lab-thu.github.io/MMVP-Dataset/">MMVP</a>: MMVP: A Multimodal MoCap Dataset with Vision and Pressure Sensors, Zhang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="http://www.lidarhumanmotion.net/reli11d/">RELI11D</a>: RELI11D: A Comprehensive Multimodal Human Motion Dataset and Method, Yan et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://cs-people.bu.edu/xjhan/groundlink.html">GroundLink</a>: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics, Han et al.</li>
        <li><b>(NeurIPS D&B 2023)</b> <a href="https://hohdataset.github.io/">HOH</a>: Markerless Multimodal Human-Object-Human Handover Dataset with Large Object Count, Wiederhold et al.</li>
        <li><b>(NeurIPS D&B 2023)</b> <a href="https://motion-x-dataset.github.io/">Motion-X</a>: A Large-scale 3D Expressive Whole-body Human Motion Dataset, Lin et al.</li>
        <li><b>(NeurIPS D&B 2023)</b> <a href="https://github.com/jutanke/hik">Humans in Kitchens</a>: A Dataset for Multi-Person Human Motion Forecasting with Scene Context, Tanke et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://jnnan.github.io/project/chairs/">CHAIRS</a>: Full-Body Articulated Human-Object Interaction, Jiang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://eth-ait.github.io/emdb/">EMDB</a>: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild, Kaufmann et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://ipman.is.tue.mpg.de/">MOYO</a>: 3D Human Pose Estimation via Intuitive Physics, Tripathi et al.</li>
        <li><b>(CVPR 2023)</b> <a href="http://www.lidarhumanmotion.net/cimi4d">CIMI4D</a>: A Large Multimodal Climbing Motion Dataset under Human-Scene Interactions, Yan et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://andytang15.github.io/FLAG3D/">FLAG3D</a>: A 3D Fitness Activity Dataset with Language Instruction, Tang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://yifeiyin04.github.io/Hi4D/">Hi4D</a>: 4D Instance Segmentation of Close Human Interaction, Yin et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://stanford-tml.github.io/circle_dataset/">CIRCLE</a>: Capture in Rich Contextual Environments, Araujo et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://bedlam.is.tue.mpg.de/">BEDLAM</a>: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion, Black et al.</li>
        <li><b>(CVPR 2023)</b> <a href="http://www.lidarhumanmotion.net/sloper4d">SLOPER4D</a>: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments, Dai et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://mime.is.tue.mpg.de/">MIME</a>: Human-Aware 3D Scene Generation, Yi et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/microsoft/MoCapAct">MoCapAct</a>: A Multi-Task Dataset for Simulated Humanoid Control, Wagener et al.</li>
        <li><b>(ACM MM 2022)</b> <a href="https://github.com/MichiganCOG/ForcePose?tab=readme-ov-file">ForcePose</a>: Learning to Estimate External Forces of Human Motion in Video, Louis et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://pantomatrix.github.io/BEAT/">BEAT</a>: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis, Liu et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://github.com/dmoltisanti/brace">BRACE</a>: The Breakdancing Competition Dataset for Dance Motion Synthesis, Moltisanti et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://sanweiliti.github.io/egobody/egobody.html">EgoBody</a>: Human body shape and motion of interacting people from head-mounted devices, Zhang et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://github.com/y-zheng18/GIMO">GIMO</a>: Gaze-Informed Human Motion Prediction in Context, Zheng et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://caizhongang.github.io/projects/HuMMan/">HuMMan</a>: Multi-Modal 4D Human Dataset for Versatile Sensing and Modeling, Cai et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://github.com/GUO-W/MultiMotion">ExPI</a>: Multi-Person Extreme Motion Prediction, Guo et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://ericguo5513.github.io/text-to-motion">HumanML3D</a>: Generating Diverse and Natural 3D Human Motions from Text, Guo et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://www.yusun.work/BEV/BEV.html">Putting People in their Place</a>: Monocular Regression of 3D People in Depth, Sun et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/behave/">BEHAVE</a>: Dataset and Method for Tracking Human Object Interactions, Bhatnagar et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://google.github.io/aichoreographer/">AIST++</a>: AI Choreographer: Music Conditioned 3D Dance Generation with AIST++, Li et al.</li>
        <li><b>(CVPR 2021)</b> <a href="https://fit3d.imar.ro/home">Fit3D</a>: AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training, Fieraru et al.</li>
        <li><b>(CVPR 2021)</b> <a href="https://babel.is.tue.mpg.de/">BABEL</a>: Bodies, Action, and Behavior with English Labels, Punnakkal et al.</li>
        <li><b>(AAAI 2021)</b> <a href="https://sc3d.imar.ro/home">HumanSC3D</a>: Learning complex 3d human self-contact, Fieraru et al.</li>
        <li><b>(CVPR 2020)</b> <a href="https://ci3d.imar.ro/home">CHI3D</a>: Three-Dimensional Reconstruction of Human Interactions, Fieraru et al.</li>
        <li><b>(ICCV 2019)</b> <a href="https://prox.is.tue.mpg.de/">PROX</a>: Resolving 3D Human Pose Ambiguities with 3D Scene Constraints, Hassan et al.</li>
        <li><b>(ICCV 2019)</b> <a href="https://amass.is.tue.mpg.de/">AMASS</a>: Archive of Motion Capture As Surface Shapes, Mahmood et al.</li>
    </ul></details>
</ul></details>

<span id="humanoid"></span>
## Humanoid, Simulated or Real
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(CoRL 2025)</b> <a href="https://robot-trains-robot.github.io/">Robot Trains Robot</a>: Automatic Real-World Policy Adaptation and Learning for Humanoids, Hu et al.</li>
        <li><b>(CoRL 2025)</b> <a href="https://hub-robot.github.io/">HuB</a>: Learning Extreme Humanoid Balance, Zhang et al.</li>
        <li><b>(CoRL 2025)</b> <a href="https://humanoid-clone.github.io/">CLONE</a>: Closed-Loop Whole-Body Humanoid Teleoperation for Long-Horizon Tasks, Li et al.</li>
        <li><b>(CoRL 2025)</b> <a href="https://arxiv.org/abs/2508.03068">Hand-Eye Autonomous Delivery</a>: Learning Humanoid Navigation, Locomotion and Reaching, Ye et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2411.19921">SIMS</a>: Simulating Human-Scene Interactions with Real World Script Planning, Wang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://arxiv.org/abs/2502.14140">ModSkill</a>: Physical Character Skill Modularization, Huang et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://wuyan01.github.io/uniphys-project/">UniPhys</a>: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control, Wu et al.</li>
        <li><b>(RSS 2025)</b> <a href="https://arxiv.org/abs/2502.13013">HOMIE</a>: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit, Ben et al.</li>
        <li><b>(RSS 2025)</b> <a href="https://why618188.github.io/beamdojo/">BeamDojo</a>: Learning Agile Humanoid Locomotion on Sparse Footholds, Wang et al.</li>
        <li><b>(RSS 2025)</b> <a href="https://agile.human2humanoid.com/">ASAP</a>: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills, He et al.</li>
        <li><b>(RSS 2025)</b> <a href="https://humanoid-getup.github.io/">HumanUP</a>: Learning Getting-Up Policies for Real-World Humanoid Robots, He et al.</li>
        <li><b>(RSS 2025)</b> <a href="https://arxiv.org/abs/2504.17249">Demonstrating Berkeley Humanoid Lite</a>: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot, Chi et al.</li>
        <li><b>(RSS 2025)</b> <a href="https://amo-humanoid.github.io/">AMO</a>: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control, Li et al.</li>
        <li><b>(RSS 2025)</b> <a href="https://taohuang13.github.io/humanoid-standingup.github.io/">HoST</a>: Learning Humanoid Standing-up Control across Diverse Postures, Huang et al.</li>
        <li><b>(RSS 2025 Workshop)</b> <a href="https://exbody2.github.io/">Exbody2</a>: Advanced Expressive Humanoid Whole-Body Control, Ji et al.</li>
        <li><b>(SIGGRPAH 2025)</b> <a href="https://arxiv.org/abs/2503.11801">Diffuse-CLoC</a>: Guided Diffusion for Physics-based Character Look-ahead Control, Huang et al.</li>
        <li><b>(SIGGRAPH 2025)</b> <a href="https://arxiv.org/abs/2505.23708">AMOR</a>: Adaptive Character Control through Multi-Objective Reinforcement Learning, Alegre et al.</li>
        <li><b>(SIGGRAPH 2025)</b> <a href="https://arxiv.org/abs/2505.04002">PARC</a>: Physics-based Augmentation with Reinforcement Learning for Character Controllers, Xu et al.</li>
        <li><b>(SIGGRAPH 2025)</b> <a href="https://arxiv.org/abs/2505.02094">SkillMimic-v2</a>: Learning Robust and Generalizable Interaction Skills from Sparse and Noisy Demonstrations, Yu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://openaccess.thecvf.com//content/CVPR2025/papers/Ji_POMP_Physics-consistent_Motion_Generative_Model_through_Phase_Manifolds_CVPR_2025_paper.pdf">POMP</a>: Physics-constrainable Motion Generative Model through Phase Manifolds, Ji et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://lego-h-humanoidrobothiking.github.io/">Let Humanoids Hike!</a> Integrative Skill Development on Complex Trails, Lin et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://jiemingcui.github.io/grove/">GROVE</a>: A Generalized Reward for Learning Open-Vocabulary Physical Skill, Cui et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://sirui-xu.github.io/InterMimic/">InterMimic</a>: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions, Xu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://ingrid789.github.io/SkillMimic/">SkillMimic</a>: Learning Reusable Basketball Skills from Demonstrations, Wang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2504.07095">Neural Motion Simulator</a>: Pushing the Limit of World Models in Reinforcement Learning, Hao et al.</li>
        <li><b>(Eurographics 2025)</b> <a href="https://arxiv.org/abs/2503.12814">Bae et al</a>: Versatile Physics-based Character Control with Hybrid Latent Representation, Bae et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://arxiv.org/abs/2505.05773">Boguslavskii et al</a>: Human-Robot Collaboration for the Remote Control of Mobile Humanoid Robots with Torso-Arm Coordination, Boguslavskii et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://arxiv.org/abs/2410.21229">HOVER</a>: Versatile Neural Whole-Body Controller for Humanoid Robots, He et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://junfeng-long.github.io/PIM/">PIM</a>: Learning Humanoid Locomotion with Perceptive Internal Model, Long et al.</li>
        <li><b>(ICRA 2025)</b> <a href="https://arxiv.org/abs/2502.18901">Think on your feet</a>: Seamless Transition between Human-like Locomotion in Response to Changing Commands, Huang et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://robo-mimiclabs.github.io/">MimicLabs</a>: What Matters in Learning from Large-Scale Datasets for Robot Manipulation, Saxena et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://rlpuppeteer.github.io/">Puppeteer</a>: Hierarchical World Models as Visual Whole-Body Humanoid Controllers, Hansen et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=9sOR0nYLtz">FB-CPR</a>: Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models, Tirinzoni et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=MWHIIWrWWu">MPC2</a>: Motion Control of High-Dimensional Musculoskeletal System with Hierarchical Model-Based Planning, Wei et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://guytevet.github.io/CLoSD-page/">CLoSD</a>: Closing the Loop between Simulation and Diffusion for multi-task character control, Tevet et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://arxiv.org/abs/2502.03122">HiLo</a>: Learning Whole-Body Human-like Locomotion with Motion Tracking Controller, Zhang et al.</li>
        <li><b>(Github 2025)</b> <a href="https://github.com/NVlabs/MobilityGen">MobilityGen</a>: MobilityGen.</li>
        <li><b>(ArXiv 2025)</b> <a href="hhttps://arxiv.org/abs/2510.05070">ResMimic</a>: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning, Zhao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="hhttps://arxiv.org/abs/2510.02252">Retargeting Matters</a>: General Motion Retargeting for Humanoid Motion Tracking, Ara´ujo et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="hhttps://arxiv.org/abs/2510.01708">PolySim</a>: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization, Lei et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="hhttps://arxiv.org/abs/2509.24697">D'Elia et al</a>: Stabilizing Humanoid Robot Trajectory Generation via Physics-Informed Learning and Control-Informed Steering, D'Elia et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://omniretarget.github.io/">OmniRetarget</a>: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction, Yang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.25600">MoReFlow</a>: Motion Retargeting Learning through Unsupervised Flow Matching, Kim et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.21690">Towards Versatile Humanoid Table Tennis</a>: Unified Reinforcement Learning with Prediction Augmentation, Hu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.21231">SEEC</a>: Stable End-Effector Control with Model-Enhanced Residual Learning for Humanoid Loco-Manipulation, Jang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.20696">RuN</a>: Residual Policy for Natural Humanoid Locomotion, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.20717">RobotDancing</a>: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking, Sun et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://visualmimic.github.io/">VisualMimic</a>: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation, Yin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.19573">Chasing Stability</a>: Humanoid Running via Control Lyapunov Function Guided Reinforcement Learning, Olkin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.19545">RoMoCo</a>: Robotic Motion Control Toolbox for Reduced-Order Model-Based Locomotion on Bipedal and Humanoid Robots, Dai et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://hdmi-humanoid.github.io/">HDMI</a>: Learning Interactive Humanoid Whole-Body Control from Human Videos, Weng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.18046">HuMam</a>: Humanoid Motion Control via End-to-End Deep Reinforcement Learning with Mamba, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.16638">KungfuBot2</a>: Learning Versatile Motion Skills for Humanoid Whole-Body Contro, Han et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.15443">IKMR</a>: Implicit Kinodynamic Motion Retargeting for Human-to-humanoid Imitation Learning, Chen et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.14353">DreamControl</a>: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion, Kalaria et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.13780">BFM</a>: Behavior Foundation Model for Humanoid Robots, Zeng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.13534">Zheng et al</a>: Embracing Bulky Objects with Humanoid Robots: Whole-Body Manipulation with Reinforcement Learning, Zheng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.13200">StageACT</a>: Stage-Conditioned Imitation for Robust Humanoid Door Opening, Lee et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.11839">TrajBooster</a>: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning, Liu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.05581">Learning to Walk in Costume</a>: Adversarial Motion Priors for Aesthetically Constrained Humanoids, Alvarez et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2509.04722">Ghansah et al</a>: Hierarchical Reduced-Order Model Predictive Control for Robust Locomotion on Humanoid Robots, Ghansah et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://humanoid-table-tennis.github.io/">HITTER</a>: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning, Su et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.20661">Traversing the Narrow Path</a>: A Two-Stage Reinforcement Learning Framework for Humanoid Beam Walking, Huang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.19926">FARM</a>: Frame-Accelerated Augmentation and Residual Mixture-of-Experts for Physics-Based High-Dynamic Humanoid Control, Jing et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.18066">Arnold</a>: A Generalist Muscle Transformer Policy, Chiappa et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://haozhuo-zhang.github.io/HumanoidVerse-project-page/">HumanoidVerse</a>: A Versatile Humanoid for Vision-Language Guided Multi-Object Rearrangement, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.14099">Ciebielski et al</a>: Task and Motion Planning for Humanoid Loco-manipulation, Ciebielski et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.14120">SimGenHOI</a>: Physically Realistic Whole-Body Humanoid-Object Interaction via Generative Modeling and Reinforcement Learning, Lin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://sites.google.com/stanford.edu/lookout">LookOut</a>: Real-World Humanoid Egocentric Navigation, Pan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://rhea-mal.github.io/humanoidsynergies.io/">Malhotra et al</a>: Humanoid Motion Scripting with Postural Synergies, Malhotra et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.09960">GBC</a>: Generalized Behavior-Cloning Framework for Whole-Body Humanoid Imitation, Yao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.07611">SafeHumanoidsPolicy</a>: End-to-End Humanoid Robot Safe and Comfortable Locomotion Policy, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.08241">BeyondMimic</a>: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion, Truong et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.01247">SE-Policy</a>: Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy, Nie et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.00362">Chen et al</a>: A Whole-Body Motion Imitation Framework from Human Data for Full-Size Humanoid Robot, Chen et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.20217">Humanoid Occupancy</a>: Enabling A Generalized Multimodal Occupancy Perception System on Humanoid Robots, Cui et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.17141">Astribot</a>: Towards Human-level Intelligence via Human-like Whole-Body Manipulation, Astribot Team.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.15649">EMP</a>: Executable Motion Prior for Humanoid Robot Standing Upper-body Motion Imitation, Xu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://rchalyang.github.io/EgoVLA/">EgoVLA</a>: Learning Vision-Language-Action Models from Egocentric Human Videos, Yang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://robot-drummer.github.io/">Robot Drummer</a>: Learning Rhythmic Skills for Humanoid Drumming, Shahid et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.08303">PL-CAP</a>: Learning Robust Motion Skills via Critical Adversarial Attacks for Humanoid Robots, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.07356">UniTracker</a>: Learning Universal Whole-Body Motion Tracker for Humanoid Robots, Yin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://ulc-humanoid.github.io/">ULC</a>: A Unified and Fine-Grained Controller for Humanoid Loco-Manipulation, Sun et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.22827">Schakkal et al</a>: Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation, Schakkal et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.20343">PIMBS</a>: Efficient Body Schema Learning for Musculoskeletal Humanoids with Physics-Informed Neural Networks, Kawaharazuka et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.20487">Behavior Foundation Model</a>: Towards Next-Generation Whole-Body Control System of Humanoid Robots, Yuan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://beingbeyond.github.io/RLPF/">RL from Physical Feedback</a>: Aligning Large Motion Models with Humanoid Control, Yue et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.12779">From Experts to a Generalist</a>: Toward General Whole-Body Control for Humanoid Robots, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.13751">LeVERB</a>: Humanoid Whole-Body Control with Latent Vision-Language Instruction, Xue et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://gmt-humanoid.github.io/">GMT</a>: General Motion Tracking for Humanoid Whole-Body Control, Chen et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://kungfu-bot.github.io/">KungfuBot</a>: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills, Xie et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://usc-gvl.github.io/SkillBlender-web/">SkillBlender</a>: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending, Kuang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://lecar-lab.github.io/SoFTA/">Hold My Beer🍻</a>: Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://younggyo.me/fast_td3/">FastTD3</a>: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control, Seo et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.20619">Peng et al</a>: Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion, Peng et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://dreampolicy.github.io/">One Policy but Many Worlds</a>: A Scalable Unified Policy for Versatile Humanoid Locomotion, Fan et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://smap-project.github.io/">SMAP</a>: Self-supervised Motion Adaptation for Physically Plausible Humanoid Whole-body Control, Zhao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.19086">MaskedManipulator</a>: Versatile Whole-Body Control for Loco-Manipulation, Tessler et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://h2compact.github.io/h2compact/">H2-COMPAXT</a>: Human–Humanoid Co-Manipulation via Adaptive Contact Trajectory Policies, Bethala et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.12619">HIL</a>: Hybrid Imitation Learning of Diverse Parkour Skills from Videos, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.12679">Dribble Master</a>: Learning Agile Humanoid Dribbling Through Legged Locomotion, Wang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.13436">KinTwin</a>: Imitation Learning with Torque and Muscle Driven Biomechanical Models Enables Precise Replication of Able-Bodied and Impaired Movement from Markerless Motion Capture, Cotton et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="http://www.zhengyiluo.com/PDC-Site/">PDC</a>: Emergent Active Perception and Dexterity of Simulated Humanoids from Visual Reinforcement Learning, Luo et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://zzk273.github.io/R2S2/">R2S2</a>: Unleashing Humanoid Reaching Potential via Real-world-Ready Skill Space, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.11495">Bracing for Impact</a>: Robust Humanoid Push Recovery and Locomotion with Reduced Order Model, Yang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.11494">SHIELD</a>: Safety on Humanoids via CBFs In Expectation on Learned Dynamics, Yang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.06584">JAEGER</a>: Dual-Level Humanoid Whole-Body Controller, Ding et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://lecar-lab.github.io/falcon-humanoid/">FALCON</a>: Learning Force-Adaptive Humanoid Loco-Manipulation, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.04961">ADD</a>: Physics-Based Motion Imitation with Adversarial Differential Discriminators, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://www.videomimic.net/">VideoMimic</a>: Visual imitation enables contextual humanoid control, Allshire et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2505.02833">TWIST</a>: Teleoperated Whole-Body Imitation System, Ze et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://bit-bots.github.io/SoccerDiffusion/">SoccerDiffusion</a>: Toward Learning End-to-End Humanoid Robot Soccer from Gameplay Recordings, Vahl et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://almi-humanoid.github.io/">ALMI</a>: Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning, Shi et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2504.09532">Hao et al</a>: Embodied Chain of Action Reasoning with Multi-Modal Foundation Model for Humanoid Loco-manipulation, Hao et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2504.09833">PreCi</a>: Pre-training and Continual Improvement of Humanoid Locomotion via Model-Assumption-based Regularization, Jung et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2504.10390">Teacher Motion Priors</a>: Enhancing Robot Locomotion over Challenging Terrain, Lin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2504.06585">Cha et al</a>: Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection, Cha et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://xianqi-zhang.github.io/FLAM/">FLAM</a>: Foundation Model-Based Body Stabilization for Humanoid Locomotion and Manipulation, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.22459">Lutz et al</a>: Control of Humanoid Robots with Parallel Mechanisms using Kinematic Actuation Models, Lutz et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.14734">GR00T N1</a>: An Open Foundation Model for Generalist Humanoid Robots, NVIDIA.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://styleloco.github.io/">StyleLoco</a>: Generative Adversarial Distillation for Natural Humanoid Robot Locomotion, Ma et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.14637">KINESIS</a>: Reinforcement Learning-Based Motion Imitation for Physiologically Plausible Musculoskeletal Motor Control, Simos et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://sites.google.com/view/humanoid-gmp">GMP</a>: Natural Humanoid Robot Locomotion with Generative Motion Prior, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.00692">Sun et al</a>: Learning Perceptive Humanoid Locomotion over Challenging Terrain, Sun et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.00923">HWC-Loco</a>: A Hierarchical Whole-Body Control Approach to Robust Humanoid Locomotion, Lin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://toruowo.github.io/recipe/">Lin et al</a>: Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids, Lin et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://nvlabs.github.io/COMPASS/">COMPASS</a>: Cross-embOdiment Mobility Policy via ResiduAl RL and Skill Synthesis, Liu et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://renjunli99.github.io/vbcom.github.io/">VB-COM</a>: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception, Ren et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2502.14795">Humanoid-VLA</a>: Towards Universal Humanoid Control with Visual Integration, Ding et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2502.13707">Li et al</a>: Human-Like Robot Impedance Regulation Skill Learning from Human-Human Demonstrations, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://humanoid-interaction.github.io/">RHINO</a>: Learning Real-Time Humanoid-Human-Object Interaction from Human Demonstrations, Chen et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2502.01465">Embrace Collisions</a>: Humanoid Shadowing for Deployable Contact-Agnostics Motion, Zhuang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://toddlerbot.github.io/">ToddlerBot</a>: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation, Shi et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2501.02116">Gu et al</a>: Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning, Gu et al.</li>
    </ul></details>
    <details>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ArXiv 2024)</b> <a href="https://usc-gvl.github.io/UH-1/">UH-1</a>: Learning from Massive Human Videos for Universal Humanoid Pose Control, Mao et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://mimicking-bench.github.io/">Mimicking-Bench</a>: A Benchmark for Generalizable Humanoid-Scene Interaction Learning via Human Mimicking, Liu et al. </li>
        <li><b>(ArXiv 2024)</b> <a href="https://smplolympics.github.io/SMPLOlympics">Humanoidlympics</a>: Sports Environments for Physically Simulated Humanoids, Luo et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://wyhuai.github.io/physhoi-page/">PhySHOI</a>: Physics-Based Imitation of Dynamic Human-Object Interaction, Wang et al.</li>
        <li><b>(RA-L 2024)</b> <a href="https://arxiv.org/abs/2505.19580">Murooka et al</a>:  Whole-body Multi-contact Motion Control for Humanoid Robots Based on Distributed Tactile Sensors, Murooka et al.</li>
        <li><b>(RA-L 2024)</b> <a href="https://arxiv.org/abs/2412.15166">Liu et al</a>: Human-Humanoid Robots Cross-Embodiment Behavior-Skill Transfer Using Decomposed Adversarial Learning from Demonstration, Liu et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://stanford-tml.github.io/PDP.github.io/">PDP</a>: Physics-Based Character Animation via Diffusion Policy, Truong et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://xbpeng.github.io/projects/MaskedMimic/index.html">MaskedMimic</a>: MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting, Tessler et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/abs/2406.19972">HumanVLA</a>: Towards Vision-Language Directed Object Rearrangement by Physical Humanoid, Xu et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://www.zhengyiluo.com/Omnigrasp-Site/">OmniGrasp</a>: Grasping Diverse Objects with Simulated Humanoids, Luo et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://github.com/zhenzhiwang/intercontrol">InterControl</a>: Generate Human Motion Interactions by Controlling Every Joint, Wang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://gao-jiawei.com/Research/CooHOI/">CooHOI</a>: Learning Cooperative Human-Object Interaction with Manipulated Object Dynamics, Gao et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://humanoid-next-token-prediction.github.io/">Radosavovic et al.</a>: Humanoid Locomotion as Next Token Prediction, Radosavovic et al.</li>
        <li><b>(CoRL 2024)</b> <a href="https://ut-austin-rpl.github.io/Harmon/">HARMON</a>: Whole-Body Motion Generation of Humanoid Robots from Language Descriptions, Jiang et al.</li>
        <li><b>(CoRL 2024)</b> <a href="https://ut-austin-rpl.github.io/OKAMI/">OKAMI</a>: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation, Li et al.</li>
        <li><b>(CoRL 2024)</b> <a href="https://humanoid-ai.github.io/">HumanPlus</a>: Humanoid Shadowing and Imitation from Humans, Fu et al.</li>
        <li><b>(CoRL 2024)</b> <a href="https://omni.human2humanoid.com/">OmniH2O</a>: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning, He et al.</li>
        <li><b>(Humanoids 2024)</b> <a href="https://evm7.github.io/Self-AWare/">Self-Aware</a>: Know your limits! Optimize the behavior of bipedal robots through self-awareness, Mascaro et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://yunzeliu.github.io/PhysReaction/">PhysReaction</a>: Physically Plausible Real-Time Humanoid Reaction Synthesis via Forward Dynamics Guided 4D Imitation, Liu et al.</li>
        <li><b>(IROS 2024)</b> <a href="https://human2humanoid.com/">H2O</a>: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation, He et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://idigitopia.github.io/projects/mhc/">MHC</a>: Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs, Shrestha et al.</li>
        <li><b>(ICML 2024)</b> <a href="https://arxiv.org/abs/2405.14790">DIDI</a>: Diffusion-Guided Diversity for Offline Behavioral Generation, Liu et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://moconvq.github.io/">MoConVQ</a>: Unified Physics-Based Motion Control via Scalable Discrete Representations, Yao et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://jiashunwang.github.io/PhysicsPingPong/">PhysicsPingPong</a>: Strategy and Skill Learning for Physics-based Table Tennis Animation, Wang et al.</li>
        <li><b>(SIGGRAPH 2024)</b> <a href="https://arxiv.org/abs/2407.10481">SuperPADL</a>: Scaling Language-Directed Physics-Based Control with Progressive Supervised Distillation, Juravsky et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://www.zhengyiluo.com/SimXR">SimXR</a>: Real-Time Simulated Avatar from Head-Mounted Sensors, Luo et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://anyskill.github.io/">AnySkill</a>: Learning Open-Vocabulary Physical Skill for Interactive Agents, Cui et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://github.com/ZhengyiLuo/PULSE">PULSE</a>: Universal Humanoid Motion Representations for Physics-Based Control, Luo et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://github.com/facebookresearch/hgap">H-GAP</a>: Humanoid Control with a Generalist Planner, Jiang et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://xizaoqu.github.io/unihsi/">UniHSI</a>: Unified Human-Scene Interaction via Prompted Chain-of-Contacts, Xiao et al.</li>
        <li><b>(3DV 2024)</b> <a href="https://eth-ait.github.io/phys-fullbody-grasp/">Phys-Fullbody-Grasp</a>: Physically Plausible Full-Body Hand-Object Interaction Synthesis, Braun et al.</li>
        <li><b>(RSS 2024)</b> <a href="https://expressive-humanoid.github.io/">ExBody</a>: Expressive Whole-Body Control for Humanoid Robots, Cheng et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://vcai.mpi-inf.mpg.de/projects/FatiguedMovements/">Fatigued Movements</a>: Discovering Fatigued Movements for Virtual Character Animation, Cheema et al.</li>
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://frank-zy-dou.github.io/projects/CASE/index.html">C·ASE</a>: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters, Dou et al.</li>
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://github.com/xupei0610/AdaptNet">AdaptNet</a>: Policy Adaptation for Physics-Based Character Control, Xu et al.</li>
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://tencent-roboticsx.github.io/NCP/">NCP</a>: Neural Categorical Priors for Physics-Based Character Control, Zhu et al.</li>
        <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://stanford-tml.github.io/drop/">DROP</a>: Dynamics Responses from Human Motion Prior and Projective Dynamics, Jiang et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://jiawei-ren.github.io/projects/insactor/">InsActor</a>: InsActor: Instruction-driven Physics-based Characters, Ren et al.</li>
        <li><b>(CoRL 2023)</b> <a href="https://humanoid4parkour.github.io/">Humanoid4Parkour</a>: Humanoid Parkour Learning, Zhuang et al.</li>
        <li><b>(CoRL Workshop 2023)</b> <a href="https://www.kniranjankumar.com/words_into_action/">Words into Action</a>: Words into Action: Learning Diverse Humanoid Robot Behaviors using Language Guided Iterative Motion Refinement, Kumar et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://zhengyiluo.github.io/PHC/">PHC</a>: Perpetual Humanoid Control for Real-time Simulated Avatars, Luo et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://xbpeng.github.io/projects/Trace_Pace/index.html">Trace and Pace</a>: Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion, Rempe et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://research.nvidia.com/labs/toronto-ai/vid2player3d/">Vid2Player3D</a>: Learning Physically Simulated Tennis Skills from Broadcast Videos, Zhang et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://dl.acm.org/doi/10.1145/3588432.3591504">QuestEnvSim</a>: Environment-Aware Simulated Motion Tracking from Sparse Sensors, Lee et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://research.nvidia.com/publication/2023-08_synthesizing-physical-character-scene-interactions">Hassan et al.</a>: Synthesizing Physical Character-Scene Interactions, Hassan et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://xbpeng.github.io/projects/CALM/index.html">CALM</a>: Conditional Adversarial Latent Models for Directable Virtual Characters, Tessler et al.</li>
        <li><b>(SIGGRAPH 2023)</b> <a href="https://github.com/xupei0610/CompositeMotion">Composite Motion</a>: Composite Motion Learning with Task Control, Xu et al.</li>
        <li><b>(ICLR 2023)</b> <a href="https://diffmimic.github.io/">DiffMimic</a>: Efficient Motion Mimicking with Differentiable Physics, Ren et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/ZhengyiLuo/EmbodiedPose">EmbodiedPose</a>: Embodied Scene-aware Human Pose Estimation, Luo et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://github.com/microsoft/MoCapAct">MoCapAct</a>: A Multi-Task Dataset for Simulated Humanoid Control, Wagener et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://research.facebook.com/publications/motion-in-betweening-for-physically-simulated-characters/">Gopinath et al.</a>: Motion In-betweening for Physically Simulated Characters, Gopinath et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://dl.acm.org/doi/10.1145/3550082.3564207">AIP</a>: Adversarial Interaction Priors for Multi-Agent Physics-based Character Control, Younes et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://github.com/heyuanYao-pku/Control-VAE">ControlVAE</a>: Model-Based Learning of Generative Controllers for Physics-Based Characters, Yao et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://dl.acm.org/doi/fullHtml/10.1145/3550469.3555411">QuestSim</a>: Human Motion Tracking from Sparse Sensors with Simulated Avatars, Winkler et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://github.com/nv-tlabs/PADL">PADL</a>: Language-Directed Physics-Based Character, Juravsky et al.</li>
        <li><b>(SIGGRAPH Asia 2022)</b> <a href="https://dl.acm.org/doi/10.1145/3550454.3555490">Wang et al.</a>: Differentiable Simulation of Inertial Musculotendons, Wang et al.</li>
        <li><b>(SIGGRAPH 2022)</b> <a href="https://xbpeng.github.io/projects/ASE/index.html">ASE</a>: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters, Peng et al. </li>
        <li><b>(Journal of Neuro-Engineering and Rehabilitation 2021)</b> <a href="https://xbpeng.github.io/projects/Learn_to_Move/index.html">Learn to Move</a>: Deep Reinforcement Learning for Modeling Human Locomotion Control in Neuromechanical Simulation, Peng et al. </li>
        <li><b>(NeurIPS 2021)</b> <a href="https://zhengyiluo.github.io/projects/kin_poly/">KinPoly</a>: Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation, Luo et al.</li>
        <li><b>(SIGGRAPH 2021)</b> <a href="https://xbpeng.github.io/projects/AMP/index.html">AMP</a>: Adversarial Motion Priors for Stylized Physics-Based Character Control, Peng et al. </li>
        <li><b>(CVPR 2021)</b> <a href="https://www.ye-yuan.com/simpoe">SimPoE</a>: Simulated Character Control for 3D Human Pose Estimation, Yuan et al.</li>
        <li><b>(NeurIPS 2020)</b> <a href="https://www.ye-yuan.com/rfc">RFC</a>: Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis, Yuan et al.</li>
        <li><b>(ICLR 2020)</b> <a href="https://arxiv.org/abs/1907.04967">Yuan et al.</a>: Diverse Trajectory Forecasting with Determinantal Point Processes, Yuan et al.</li>
        <li><b>(ICCV 2019)</b> <a href="https://ye-yuan.com/ego-pose/">Ego-Pose</a>: Ego-Pose Estimation and Forecasting as Real-Time PD Control, Yuan et al.</li>
        <li><b>(SIGGRAPH 2018)</b> <a href="https://xbpeng.github.io/projects/DeepMimic/index.html">DeepMimic</a>: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills, Peng et al.</li>
    </ul></details>
</ul></details>

<span id="bio"></span>
## Bio-stuff: Human Anatomy, Biomechanics, Physiology
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <li><b>(npj 2025)</b> <a href="https://www.nature.com/articles/s41746-025-01677-0">Xiang et al</a>: Integrating personalized shape prediction, biomechanical modeling, and wearables for bone stress prediction in runners, Xiang et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://isshikihugh.github.io/HSMR/">HSMR</a>: Reconstructing Humans with A Biomechanically Accurate Skeleton, Xia et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://foruck.github.io/HDyS">HDyS</a>: Homogeneous Dynamics Space for Heterogeneous Humans, Liu et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://foruck.github.io/ImDy">ImDy</a>: Human Inverse Dynamics from Imitated Observations, Liu et al.</li>
    <li><b>(ICLR 2025)</b> <a href="https://openreview.net/forum?id=MWHIIWrWWu">MPC2</a>: Motion Control of High-Dimensional Musculoskeletal System with Hierarchical Model-Based Planning, Wei et al.</li>
    <li><b>(ACM Sensys 2025)</b> <a href="https://arxiv.org/abs/2503.01768">SHADE-AD</a>: An LLM-Based Framework for Synthesizing Activity Data of Alzheimer’s Patients, Fu et al.</li>
    <li><b>(JEB 2025)</b> <a href="https://journals.biologists.com/jeb/article/228/Suppl_1/JEB248125/367009/Behavioural-energetics-in-human-locomotion-how">McAllister et al</a>: Behavioural energetics in human locomotion: how energy use influences how we move, McAllister et al.</li>
    <li><b>(WACV 2025)</b> <a href="https://arxiv.org/abs/2406.09788">OpenCapBench</a>: A Benchmark to Bridge Pose Estimation and Biomechanics, Gozlan et al.</li>
    <li><b>(RA-L 2025)</b> <a href="https://arxiv.org/abs/2504.10102">Mobedi et al</a>: A Framework for Adaptive Load Redistribution in Human-Exoskeleton-Cobot Systems, Mobedi et al.</li>
    <li><b>(Preprint 2025)</b> <a href="https://github.com/stanfordnmbl/GaitDynamics">GaitDynamics</a>: A Foundation Model for Analyzing Gait Dynamics, Tan et al.</li>
    <li><b>(bioRxiv 2025)</b> <a href="https://www.biorxiv.org/content/10.1101/2025.06.24.660703v1.full.pdf">Richards et al</a>: Visualising joint force-velocity proper es in musculoskeletal models, Richards et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://intelligentsensingandrehabilitation.github.io/MonocularBiomechanics/">CADS</a>: A Comprehensive Anatomical Dataset and Segmentation for Whole-Body Anatomy in Computed Tomography, Xu et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://intelligentsensingandrehabilitation.github.io/MonocularBiomechanics/">Portable Biomechanics Laboratory</a>: Clinically Accessible Movement Analysis from a Handheld Smartphone, Peiffer et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.01340">Le et al</a>: Physics-informed Ground Reaction Dynamics from Human Motion Capture, Le et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2506.00071">SMS-Human</a>: Human sensory-musculoskeletal modeling and control of whole-body movements, Zuo et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://k2muse.github.io/">K2MUSE</a>: A Large-scale Human Lower limb Dataset of Kinematics, Kinetics, amplitude Mode Ultrasound and Surface Electromyography, Li et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2504.10102">A Human-Sensitive Controller</a>: Adapting to Human Ergonomics and Physical Constraints via Reinforcement Learning, Almeida et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2504.10294">Ankle Exoskeletons in Walking and Load-Carrying Tasks</a>: Insights into Biomechanics and Human-Robot Interaction, Almeida et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://vadeli.github.io/GAITGen/">GAITGen</a>: Disentangled Motion-Pathology Impaired Gait Generative Model, Adeli et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2503.14637">KINESIS</a>: Reinforcement Learning-Based Motion Imitation for Physiologically Plausible Musculoskeletal Motor Control, Simos et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2502.06486">Cotton et al</a>: Biomechanical Reconstruction with Confidence Intervals from Multiview Markerless Motion Capture, Cotton et al.</li>
    <li><b>(BiorXiv 2024)</b> <a href="https://www.biorxiv.org/content/10.1101/2024.12.30.630841v1.full.pdf">Lai et al</a>: Mapping Grip Force to Muscular Activity Towards Understanding Upper Limb Musculoskeletal Intent using a Novel Grip Strength Model, Lai et al.</li>
    <li><b>(ROBIO 2024)</b> <a href="https://arxiv.org/abs/2502.13760">Wu et al</a>: Muscle Activation Estimation by Optimizing the Musculoskeletal Model for Personalized Strength and Conditioning Training, Wu et al.</li>
    <li><b>(IROS 2024)</b> <a href="https://arxiv.org/abs/2412.18869">Shahriari et al</a>:  Enhancing Robustness in Manipulability Assessment: The Pseudo-Ellipsoid Approach, Shahriari et al.</li>
    <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://cs-people.bu.edu/xjhan/bioDesign.html">BioDesign</a>: Motion-Driven Neural Optimizer for Prophylactic Braces Made by Distributed Microstructures, Han et al.</li>
    <li><b>(Scientific Data 2024)</b> <a href="https://www.nature.com/articles/s41597-024-04077-3?fromPaywallRec=false">Evans et al</a>: Synchronized Video, Motion Capture and Force Plate Dataset for Validating Markerless Human Movement Analysis, Evans et al.</li>
    <li><b>(NeurIPS D&B 2024)</b> <a href="https://simplexsigil.github.io/mint">Muscles in Time</a>: Learning to Understand Human Motion by Simulating Muscle Activations, Schneider et al.</li>
    <li><b>(CoRL 2024)</b> <a href="https://lnsgroup.cc/research/hdsafebo">Wei et al</a>: Safe Bayesian Optimization for the Control of High-Dimensional Embodied Systems, Wei et al.</li>
    <li><b>(HFES 2024)</b> <a href="https://journals.sagepub.com/doi/full/10.1177/10711813241262026">Macwan et al</a>: High-Fidelity Worker Motion Simulation With Generative AI, Macwan et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://addbiomechanics.org/">AddBiomechanics Dataset</a>: Capturing the Physics of Human Motion at Scale, Werling et al.</li>
    <li><b>(ECCV 2024)</b> <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00194.pdf">MANIKIN</a>: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation, Jiang et al.</li>
    <li><b>(TOG 2024)</b> <a href="https://dl.acm.org/doi/pdf/10.1145/3658230">NICER</a>: A New and Improved Consumed Endurance and Recovery Metric to Quantify Muscle Fatigue of Mid-Air Interactions, Li et al.</li>
    <li><b>(TVCG 2024)</b> <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10230894">Loi et al</a>: Machine Learning Approaches for 3D Motion Synthesis and Musculoskeletal Dynamics Estimation: A Survey, Loi et al.</li>
    <li><b>(ICML 2024)</b> <a href="https://www.beanpow.top/assets/pdf/dynsyn_poster.pdf">DynSyn</a>: Dynamical Synergistic Representation for Efficient Learning and Control in Overactuated Embodied Systems, He et al.</li>
    <li><b>(Multibody System Dynamics 2024)</b> <a href="https://github.com/ainlamyae/Human3.6Mplus">Human3.6M+</a>: Using musculoskeletal models to generate physically-consistent data for 3D human pose, kinematic, dynamic, and muscle estimation, Nasr et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://hit.is.tue.mpg.de/">HIT</a>: Estimating Internal Human Implicit Tissues from the Body Surface, Keller et al.</li>
    <li><b>(Frontiers in Neuroscience 2024)</b> <a href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1388742/full">Dai et al</a>: Full-body pose reconstruction and correction in virtual reality for rehabilitation training, Dai et al.</li>
    <li><b>(ICRA 2024)</b> <a href="https://arxiv.org/abs/2312.05473.pdf">Self Model for Embodied Intelligence</a>: Modeling Full-Body Human Musculoskeletal System and Locomotion Control with Hierarchical Low-Dimensional Representation, He et al.</li>
    <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://vcai.mpi-inf.mpg.de/projects/FatiguedMovements/">Fatigued Movements</a>: Discovering Fatigued Movements for Virtual Character Animation, Cheema et al.</li>
    <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://skel.is.tue.mpg.de/">SKEL</a>: From skin to skeleton: Towards biomechanically accurate 3d digital humans, Keller et al.</li>
    <li><b>(SIGGRAPH Asia 2023)</b> <a href="https://pku-mocca.github.io/MuscleVAE-page/">MuscleVAE</a>: Model-Based Controllers of Muscle-Actuated Characters, Feng et al.</li>
    <li><b>(SIGGRAPH 2023)</b> <a href="https://github.com/namjohn10/BidirectionalGaitNet">Bidirectional GaitNet</a>: Bidirectional GaitNet, Park et al.</li>
    <li><b>(SIGGRAPH 2023)</b> <a href="https://arxiv.org/abs/2305.04995">Lee et al.</a>: Anatomically Detailed Simulation of Human Torso, Lee et al.</li>
    <li><b>(ICCV 2023)</b> <a href="https://musclesinaction.cs.columbia.edu/">MiA</a>: Muscles in Action, Chiquer et al.</li>
    <li><b>(CVPR 2022)</b> <a href="https://osso.is.tue.mpg.de/">OSSO</a>: Obtaining Skeletal Shape from Outside, Keller et al.</li>
    <li><b>(Scientific Data 2022)</b> <a href="https://www.nature.com/articles/s41597-022-01188-7">Xing et al</a>: Functional movement screen dataset collected with two Azure Kinect depth sensors, Xing et al.</li>
    <li><b>(NCA 2020)</b> <a href="https://link.springer.com/article/10.1007/s00521-019-04658-z">Zell et al</a>: Learning inverse dynamics for human locomotion analysis, Zell et al.</li>
    <li><b>(ECCV 2020)</b> <a href="https://arxiv.org/abs/2007.08969">Zell et al</a>: Weakly-supervised learning of human dynamics, Zell et al.</li>
    <li><b>(SIGGRAPH 2019)</b> <a href="https://github.com/jyf588/lrle">LRLE</a>: Synthesis of biologically realistic human motion using joint torque actuation, Jiang et al.</li>
    <li><b>(TII 2018)</b> <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8078194">Pham et al</a>: Multicontact Interaction Force Sensing From Whole-Body Motion Capture, Pham et al.</li>
    <li><b>(ICCV Workshop 2017)</b> <a href="http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w16/Zell_Learning-Based_Inverse_Dynamics_ICCV_2017_paper.pdf">Zell et al</a>: Learning-based inverse dynamics of human motion, Zell et al.</li>
    <li><b>(CVPR Workshop 2017)</b> <a href="http://openaccess.thecvf.com/content_cvpr_2017_workshops/w1/papers/Zell_Joint_3D_Human_CVPR_2017_paper.pdf">Zell et al</a>: Joint 3d human motion capture and physical analysis from monocular videos, Zell et al.</li>
    <li><b>(AIST 2017)</b> <a href="https://link.springer.com/chapter/10.1007/978-3-319-73013-4_12">HuGaDb</a>: HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks, Chereshnev et al.</li>
    <li><b>(SIGGRAPH 2016)</b> <a href="https://dl.acm.org/doi/10.1145/2980179.2982440">Lv et al</a>: Data-driven inverse dynamics for human motion, Lv et al.</li>
</ul></details>

<span id="motion-reconstruction"></span>
## Human Reconstruction, Motion/Interaction/Avatar
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>2025</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(ICCV 2025)</b> <a href="https://diffuman4d.github.io/">Diffuman4D</a>: 4D Consistent Human View Synthesis from Sparse-View Videos with Spatio-Temporal Diffusion Models, Jin et al.</li>
        <li><b>(ICCV 2025)</b> <a href="https://boqian-li.github.io/ETCH/">ETCH</a>: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness, Li et al.</li>
        <li><b>(ICML 2025)</b> <a href="https://arxiv.org/pdf/2505.10250">ADHMR</a>: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization, Shen et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://muelea.github.io/hsfm/">HSFM</a>: Reconstructing People, Places, and Cameras, M\"uller et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://prosepose.github.io/">Subramanian et al</a>: Pose Priors from Language Models, Subramanian et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://yufu-wang.github.io/phmr-page/">PromptHMR</a>: Embodied Promptable Human Mesh Recovery, Wang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://zhangyuhong01.github.io/HumanMM/">HumanMM</a>: Global Human Motion Recovery from Multi-shot Videos, Zhang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://isshikihugh.github.io/HSMR/">HSMR</a>: Reconstructing Humans with A Biomechanically Accurate Skeleton, Xia et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://g-fiche.github.io/research-pages/mega/">MEGA</a>: Masked Generative Autoencoder for Human Mesh Recovery, Fiche et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://jing-g2.github.io/DiSRT-In-Bed/">DiSRT-In-Bed</a>: TDiffusion-Based Sim-to-Real Transfer Framework for In-Bed Human Mesh Recovery, Gao et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Motions_as_Queries_One-Stage_Multi-Person_Holistic_Human_Motion_Capture_CVPR_2025_paper.pdf">Motions as Queries</a>: One-Stage Multi-Person Holistic Human Motion Capture, Liu et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://actionlab-cv.github.io/HMoRe/">H-MoRe</a>: Learning Human-centric Motion Representation for Action Analysis, Huang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://yiyuzhuang.github.io/IDOL/">IDOL</a>: Instant Photorealistic 3D Human Creation from a Single Image, Zhuang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://multigohuman.github.io/">MultiGO</a>: Towards Multi-Level Geometry Learning for Monocular 3D Textured Human Reconstruction, Zhang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Tu_GBC-Splat_Generalizable_Gaussian-Based_Clothed_Human_Digitalization_under_Sparse_RGB_Cameras_CVPR_2025_paper.pdf">GBC-Splat</a>: Generalizable Gaussian-Based Clothed Human Digitalization under Sparse RGB Cameras, Tu et al.</li>
        <li><b>(CVPR 2025 workshop)</b> <a href="https://arxiv.org/pdf/2409.17671">Ludwig et al</a>: Leveraging Anthropometric Measurements to Improve Human Mesh Estimation and Ensure Consistent Body Shapes, Ludwig et al.</li>
        <li><b>(CVPR 2025 workshop)</b> <a href="https://arxiv.org/pdf/2504.09953">Ludwig et al</a>: Efficient 2D to Full 3D Human Pose Uplifting including Joint Rotations, Ludwig et al.</li>
        <li><b>(ICLR 2025)</b> <a href="https://www.liuisabella.com/DG-Mesh/">Dynamic Gaussians Mesh</a>: Consistent Mesh Reconstruction from Dynamic Scenes, Liu et al.</li>
        <li><b>(3DV 2025)</b> <a href="https://camerahmr.is.tue.mpg.de/">CameraHMR</a>: Aligning People with Perspective, Patel et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2508.03313">BaroPoser</a>: Real-time Human Motion Tracking from IMUs and Barometers in Everyday Devices, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://intelligentsensingandrehabilitation.github.io/MonocularBiomechanics/">Portable Biomechanics Laboratory</a>: Clinically Accessible Movement Analysis from a Handheld Smartphone, Peiffer et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://caizhongang.com/projects/SMPLer-X/">SMPLest-X</a>: CUltimate Scaling for Expressive Human Pose and Shape Estimation, Yin et al.</li>
    </ul></details>
    <details open>
    <summary><h3>2024</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://zju3dv.github.io/gvhmr/">GVHMR</a>: World-Grounded Human Motion Recovery via Gravity-View Coordinates, Shen et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://evahuman.github.io/">EVAHuman</a>: Expressive Gaussian Human Avatars from Monocular RGB Video, Hu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://yufu-wang.github.io/tram4d/">TRAM</a>: Global Trajectory and Motion of 3D Humans from in-the-wild Videos, Wang et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://moygcc.github.io/ReLoo/">ReLoo</a>: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild, Guo et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://wqyin.github.io/projects/WHAC/">WHAC</a>: World-grounded Humans and Cameras, Yin et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/pdf/2402.14654">Multi-HMR</a>: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot, Baradel et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://www.meta.com/emerging-tech/codec-avatars/sapiens/?utm_source=about.meta.com&utm_medium=redirect">Sapiens</a>: Foundation for Human Vision Models, Khirodkar et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://smplcap.github.io/AiOS/">AiOS</a>: All-in-One-Stage Expressive Human Pose and Shape Estimation, Sun et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://muelea.github.io/buddi/">Generative Proxemics</a>: A Prior for 3D Social Interaction from Images, M{\“u}ller et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://github.com/MartaYang/KITRO">KITRO</a>: Refining Human Mesh by 2D Clues and Kinematic-tree Rotation, Yang et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://tokenhmr.is.tue.mpg.de/">TokenHMR</a>: Advancing Human Mesh Recovery with a Tokenized Pose Representation, Dwivedi et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://wham.is.tue.mpg.de/">WHAM</a>: Reconstructing World-grounded Humans with Accurate 3D Motion, Shin et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://www.iri.upc.edu/people/nugrinovic/multiphys/">MultiPhys</a>: Multi-Person Physics-aware 3D Motion Estimation, Ugrinovic et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://sanweiliti.github.io/ROHM/ROHM.html">RoHM</a>: Robust Human Motion Reconstruction via Diffusion, Zhang et al.</li>
    </ul></details>
    <details>
    <summary><h3>2023 and earlier</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(NeurIPS 2023)</b> <a href="https://caizhongang.com/projects/SMPLer-X/#target1">SMPLer-X</a>: Scaling Up Expressive Human Pose and Shape Estimation, Cai et al.</li>
        <li><b>(NeurIPS 2023)</b> <a href="https://arxiv.org/pdf/2312.08730">RoboSMPLX</a>: Towards Robust and Expressive Whole-body Human Pose and Shape Estimation, Pang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://arxiv.org/pdf/2303.13796">Zolly</a>: Zoom Focal Length Correctly for Perspective-Distorted Human Mesh Reconstruction, Wang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://motionbert.github.io/">MotionBERT</a>: A Unified Perspective on Learning Human Motion Representations, Zhu et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://yufu-wang.github.io/refit_humans/">ReFit</a>: Recurrent Fitting Network for 3D Human Recovery, Wang et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://shubham-goel.github.io/4dhumans/">Humans in 4D</a>: Reconstructing and Tracking Humans with Transformers, Goel et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://eth-ait.github.io/emdb/">EMDB</a>: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild, Kaufmann et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://vye16.github.io/slahmr/">Ye et al</a>: Decoupling Human and Camera Motion from Videos in the Wild, Ye et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://bedlam.is.tue.mpg.de/">BEDLAM</a>: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion, Black et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://www.yusun.work/TRACE/TRACE.html">TRACE</a>: 5D Temporal Regression of Avatars with Dynamic Cameras in 3D Environments, Sun et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://ipman.is.tue.mpg.de/">IPMAN</a>: 3D Human Pose Estimation via Intuitive Physics, Tripathi et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://people.eecs.berkeley.edu/~jathushan/LART/">Rajasegaran et al</a>: On the Benefits of 3D Pose and Tracking for Human Action Recognition, Rajasegaran et al.</li>
        <li><b>(NeurIPS 2022)</b> <a href="https://arxiv.org/pdf/2209.10529">Pang et al</a>: Benchmarking and Analyzing 3D Human Pose and Shape Estimation Beyond Algorithms, Pang et al.</li>
        <li><b>(ECCV 2022)</b> <a href="https://ethanweber.me/sitcoms3D/">Pavlakos et al</a>: The One Where They Reconstructed 3D Humans and Environments in TV Shows, Pavlakos et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://www.yusun.work/BEV/BEV.html">Putting People in their Place</a>: Monocular Regression of 3D People in Depth, Sun et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://geopavlakos.github.io/multishot/">Pavlakos et al</a>: Human Mesh Recovery from Multiple Shots, Pavlakos et al.</li>
        <li><b>(CVPR 2022 Workshop)</b> <a href="https://arxiv.org/pdf/2011.11232">NeuralAnnot</a>: Neural Annotator for 3D Human Mesh Training Sets, Moon et al.</li>
        <li><b>(NeurIPS 2021)</b> <a href="https://people.eecs.berkeley.edu/~jathushan/T3DP/">Rajasegaran et al</a>: Tracking People with 3D Representations, Rajasegaran et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://www.nikoskolot.com/projects/prohmr/">Kolotouros et al</a>: Probabilistic Modeling for Human Mesh Recovery, Kolotouros et al.</li>
        <li><b>(ICCV 2021)</b> <a href="https://arxiv.org/pdf/2008.12272">ROMP</a>: Monocular, One-stage, Regression of Multiple 3D People, Sun et al.</li>
        <li><b>(ECCV 2020)</b> <a href="https://expose.is.tue.mpg.de/">ExPose</a>: Monocular Expressive Body Regression through Body-Driven Attention, Choutas et al.</li>
        <li><b>(CVPR 2020)</b> <a href="https://jiangwenpl.github.io/multiperson/">Jiang et al</a>: Coherent Reconstruction of Multiple Humans from a Single Image, Jiang et al.</li>
        <li><b>(CVPR 2020)</b> <a href="https://wuminye.github.io/NHR/">NHR</a>: Multi-view Neural Human Rendering, Wu et al.</li>
        <li><b>(CVPR 2019)</b> <a href="https://smpl-x.is.tue.mpg.de/">Expressive Body Capture</a>: 3D Hands, Face, and Body from a Single Image, Pavlakos et al.</li>
    </ul></details>
</ul></details>


<span id="hoi/hsi-reconstruction"></span>
## Human-Object/Scene/Human Interaction Reconstruction
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>HOI Reconstruction</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(CVPR 2025)</b> <a href="https://wenboran2002.github.io/3dhoi/">Wen et al</a>: Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions, Wen et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://interactvlm.is.tue.mpg.de">InteractVLM</a>: 3D Interaction Reasoning from 2D Foundational Models, Dwivedi et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://pico.is.tue.mpg.de">PICO</a>: Reconstructing 3D People In Contact with Objects, Cseke et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://lym29.github.io/EasyHOI-page/">EasyHOI</a>: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild, Liu et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://zc-alexfan.github.io/hold">HOLD</a>:Category-agnostic 3D Reconstruction of Interacting Hands and Objects from Video, Fan et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/">Xie et al</a>: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation, Xie et al.</li>
        <li><b>(ACM MM 2024)</b> <a href="https://huochf.github.io/WildHOI/">WildHOI</a>: Monocular Human-Object Reconstruction in the Wild, Huo et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://deco.is.tue.mpg.de/">DECO</a>: Dense Estimation of 3D Human-Scene COntact in the Wild, Tripathi1 et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://virtualhumans.mpi-inf.mpg.de/behave/">BEHAVE</a>: Dataset and Method for Tracking Human Object Interactions, Bhatnagar et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://mover.is.tue.mpg.de/">MOVER</a>: Human-Aware Object Placement for Visual Environment Reconstruction, Yi et al.</li>
    </ul></details>
    <details open>
    <summary><h3>HSI Reconstruction</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(CVPR 2025)</b> <a href="https://eth-ait.github.io/ODHSR/">ODHSR et al</a>: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos, Zhang et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://genforce.github.io/JOSH/">JOSH</a>: Joint Optimization for 4D Human-Scene Reconstruction in the Wild, Liu et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://dreamscene4d.github.io/">DreamScene4D</a>: Dynamic Multi-Object Scene Generation from Monocular Videos, Chu et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://lxxue.github.io/human-scene-recon/">HSR</a>: Holistic 3D Human-Scene Reconstruction from Monocular Videos, Xue et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://machinelearning.apple.com/research/hugs">HUGS</a>: Human Gaussian Splats, Kocabas et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://yongtaoge.github.io/projects/humanwild/">Ge et al</a>: 3D Human Reconstrution in the Wild with Synthetic Data using Generative Models, Ge et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://deco.is.tue.mpg.de/">DECO</a>: Dense Estimation of 3D Human-Scene COntact in the Wild, Tripathi1 et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://sanweiliti.github.io/egohmr/egohmr.html">EgoHMR</a>: Probabilistic Human Mesh Recovery in 3D Scenes from Egocentric Views, Zhang et al.</li>
        <li><b>(CVPR 2023)</b> <a href="http://www.lidarhumanmotion.net/sloper4d">SLOPER4D</a>: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments, Dai et al.</li>
        <li><b>(CVPR 2022)</b> <a href="https://mover.is.tue.mpg.de/">MOVER</a>: Human-Aware Object Placement for Visual Environment Reconstruction, Yi et al.</li>
    </ul></details>
    <details open>
    <summary><h3>HHI Reconstruction</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(CVPR 2025)</b> <a href="https://www.buzhenhuang.com/publications/papers/CVPR2025-CloseApp.pdf">Huang et al</a>: Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning, Huang et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://jyuntins.github.io/harmony4d/">Harmony4D</a>: A Video Dataset for In-The-Wild Close Human Interactions, Khirodkar et al.</li>
        <li><b>(ECCV 2024)</b> <a href="https://arxiv.org/abs/2408.02110">AvatarPose</a>: Avatar-guided 3D Pose Estimation of Close Human Interaction from Sparse Multi-view Videos, Lu et al.</li>
        <li><b>(CVPR 2024)</b> <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Fang_Capturing_Closely_Interacted_Two-Person_Motions_with_Reaction_Priors_CVPR_2024_paper.pdf">Fang et al.</a>: Capturing Closely Interacted Two-Person Motions with Reaction Priors, Fan et al.</li>
        <li><b>(SIGGRAPH Asia 2024)</b> <a href="https://arxiv.org/abs/2401.16173">Shuai et al.</a>: Reconstructing Close Human Interactions from Multiple Views, Shuai et al.</li>
    </ul></details>
</ul></details>

<span id="motion-video/image-generation"></span>
## Motion Controlled Image/Video Generation
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <details open>
    <summary><h3>video</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(CVPR 2025)</b> <a href="https://humandreamer.github.io/">HumanDreamer</a>: Generating Controllable Human-Motion Videos via Decoupled Generation, Wang et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://smartdianlab.github.io/projects-FinePhys/">FinePhys</a>: Fine-grained Human Action Generation by Explicitly Incorporating Physical Laws for Effective Skeletal Guidance, Shao et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2405.20216">Na et al</a>: Boost Your Human Image Generation Model via Direct Preference Optimization, Na et al.</li>
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2504.08181">TokenMotion</a>: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation, Li et al.</li>
        <li><b>(ArXiv 2025)</b> <a href="https://byteaigc.github.io/X-Unimotion/">X-UniMotion</a>: Animating Human Images with Expressive, Unified and Identity-Agnostic Motion Latents, Song et al.</li>
    </ul></details>
    <details open>
    <summary><h3>image</h3></summary>
    <ul style="margin-left: 5px;">
        <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2405.20216">Na et al</a>: Boost Your Human Image Generation Model via Direct Preference Optimization, Na et al.</li>
        <li><b>(NeurIPS 2024)</b> <a href="https://arxiv.org/pdf/2406.02485">Stable-Pose</a>: Leveraging Transformers for Pose-Guided Text-to-Image Generation, Wang et al.</li>
        <li><b>(ICLR 2024)</b> <a href="https://arxiv.org/pdf/2310.06313">Shen et al</a>: Advancing Pose-Guided Image Synthesis with Progressive Conditional Diffusion Models, Shen et al.</li>
        <li><b>(ArXiv 2024)</b> <a href="https://arxiv.org/pdf/2411.12872">From Text to Pose to Image</a>: Improving Diffusion Model Control and Quality, Bonnet et al.</li>
        <li><b>(TNNLS 2023)</b> <a href="https://ieeexplore.ieee.org/document/9732175">Verbal-Person Nets</a>: Pose-Guided Multi-Granularity Language-to-Person Generation, Liu et al.</li>
        <li><b>(ICCV 2023)</b> <a href="https://idea-research.github.io/HumanSD/">HumanSD</a>: A Native Skeleton-Guided Diffusion Model for Human Image Generation, Ju et al.</li>
        <li><b>(CVPR 2023)</b> <a href="https://ankanbhunia.github.io/PIDM/">Bhunia et al</a>: Person Image Synthesis via Denoising Diffusion Model, Bhunia et al.</li>
    </ul></details>
</ul></details>

<span id="pose-estimation"></span>
## Human Pose Estimation/Recognition
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/pdf/2504.04708">SapiensID</a>: Foundation for Human Recognition, Kim et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://aviralchharia.github.io/MV-SSM/">MV-SSM</a>: Multi-View State Space Modeling for 3D Human Pose Estimation, Chharia et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://arxiv.org/abs/2412.10235">EnvPoser</a>: Environment-aware Realistic Human Motion Estimation from Sparse Observations with Uncertainty Modeling, Xia et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://mirapurkrabek.github.io/ProbPose/">ProbPose</a>: A Probabilistic Approach to 2D Human Pose Estimation, Purkrabek et al.</li>
    <li><b>(ArXiv 2025)</b> <a href="https://arxiv.org/abs/2507.17406">Aytekin et al</a>: Physics-based Human Pose Estimation from a Single Moving RGB Camera, Aytekin et al.</li>  
    <li><b>(ECCV 2024)</b> <a href="https://www.meta.com/emerging-tech/codec-avatars/sapiens/?utm_source=about.meta.com&utm_medium=redirect">Sapiens</a>: Foundation for Human Vision Models, Khirodkar et al.</li>
    <li><b>(ICCV 2023)</b> <a href="https://arxiv.org/pdf/2308.07313">Group Pose</a>: A Simple Baseline for End-to-End Multi-person Pose Estimation, Liu et al.</li>
    <li><b>(CVPR 2023)</b> <a href="https://arxiv.org/pdf/2211.08834">GenVIS</a>: A Generalized Framework for Video Instance Segmentation, Heo et al.</li>
    <li><b>(ArXiv 2022)</b> <a href="https://arxiv.org/pdf/2207.04320">Snipper</a>: A Spatiotemporal Transformer for Simultaneous Multi-Person 3D Pose Estimation Tracking and Forecasting on a Video Snippet, Zou et al.</li>
</ul></details>

<span id="motion-understanding"></span>
## Human Motion Understanding
<details open>
<summary></summary>
<ul style="margin-left: 5px;">
    <li><b>(CVPR 2025)</b> <a href="https://chathuman.github.io/">ChatHuman</a>: Chatting about 3D Humans with Tools, Lin et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://qhfang.github.io/papers/humocon.html">HuMoCon</a>: Concept Discovery for Human Motion Understanding, Fang et al.</li>
    <li><b>(CVPR 2025)</b> <a href="https://vision.cs.utexas.edu/projects/ExpertAF/">ExpertAF</a>: Expert Actionable Feedback from Video, Ashutosh et al.</li>
    <li><b>(ArXiv 2024)</b> <a href="https://lhchen.top/MotionLLM/">MotionLLM</a>: Understanding Human Behaviors from Human Motions and Videos, Chen et al.</li>
    <li><b>(CVPR 2024)</b> <a href="https://yfeng95.github.io/ChatPose/">ChatPose</a>: Chatting about 3D Human Pose, Feng et al.</li>
</ul></details>

---
# Contributors
This paper list is mainly contributed by <a href="https://foruck.github.io/">Xinpeng Liu</a> and <a href="https://github.com/Daydreamer-f">Yusu Fang</a>, feel free to contact us if you have any questions or suggestions!

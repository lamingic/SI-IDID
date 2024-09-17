**Project Outlines: SI-IDID**

SI-IDID represents an innovative extension of the established Ev-IDID toolkit~(please refer to https://github.com/lamingic/Ev-IDID), specifically tailored to address complex I-DID (Interactive Dynamic Influence Diagram) problems with enhanced precision and capabilities. Building upon the solid foundation laid by Ev-IDID, SI-IDID refines and augments its predecessor's functionalities, offering a superior approach to I-DID modeling.

Furthermore, the repository functions as supplementary material for our paper titled "**Improved Response Strategies for Unknown Behaviours in Multiagent Interaction**" authored by an esteemed team comprising Yinghui Pan (Member, IEEE), Mengen Zhou, Yifeng Zeng (Member, IEEE), Biyang Ma (Member, IEEE), Yew-soon Ong (Fellow, IEEE), and Guoquan Liu.

Within this paper, we harness two distinct simulation intelligence (SI) techniques to devise innovative behaviors for interacting agents, thereby empowering the subject agent to optimize its response strategies. These novel behaviors are designed to accurately mirror the authentic behaviors of other agents, enabling the subject agent to react appropriately and effectively. By leveraging the Interactive Dynamic Influence Diagrams framework, we undertake a theoretical analysis to explore how these algorithms impact the decision quality of the subject agent. Additionally, we empirically validate the performance of these algorithms across two widely recognized problem domains, showcasing their robustness and applicability.

**Outline of this repository**:

**Supplemental Material for "Improved Response Strategies for Unknown Behaviours in Multiagent Interaction"**

**Frameworks and Components**:

- **PSB (PSO-Enabled behaviours ) and ACB (ACO-based behavior) Frameworks**: We comprehensively outline the theoretical foundations and structural compositions of the PSB and ACB frameworks. These frameworks serve as the backbone for devising novel response strategies to handle unknown behaviors in multiagent interactions.

- **PSB Operators**: We detail the four operators employed within the PSB framework. These operators are designed to manipulate and optimize the behavior models of agents, enabling the subject agent to better anticipate and react to unknown behaviors. Adapted from PSO for policy tree representation, these operators are detailed in Section (Generation of New Behaviors through PSO) and used in Algorithm PSB to update particle position and velocity. We present the pseudocode for the four operators in PSB:
We present the pseudocode for the four operators in PSB:

- $\ominus$: operator $\ominus$ calculates particle velocity from particle positions.
- $\oplus$: operator $\oplus$ calculates a new particle position.
- $\otimes$ : operator $\otimes$ scales velocity $v_1$ by scalar $\omega$.
- $\uplus$ : operator $\uplus$ combines two velocities $v_1$ and $v_2$.

These operators, adapted from the PSO mechanism and developed for policy tree representation, are described in two situations in Section~(Generation of New  Behaviours through PSO) and used to update particle position and velocity in Algorithm PSB.

- **Shared Functionalities**: Both the PSB and ACB algorithms share four core operators outlined in PSB and ACB. The **formalise** operator generates an action sequence from a given policy tree, while the **transform** operator constructs a policy tree from a given action sequence. These two operators act as translators between policy trees and particles (or ants) in PSB (or ACB). The **fitness** operator evaluates the fitness of a particle's action sequence in PSB using the decision tool GeNIe[^1], which is the most computationally intensive operator in the algorithms. The **evaluate** operator calculates the pheromone value when an ant in ACB holds a policy encoded as an ordered sequence of actions over positions.
[^1]: https://www.bayesfusion.com)

- **Formalise**: Generates an action sequence from a policy tree.  
- **Transform**: Converts action sequences to policy trees. 
- **Fitness**: Evaluates particle sequences using GeNIe decision tool[^1].  
- **Evaluate**: Calculates pheromone for ant policies.

**Scripts**:

- **Implementations**: We present the complete implementations of the algorithms and techniques used in our research. These include source code, function libraries, and scripts necessary for replicating our experiments and results.

- **Datasets**: We provide detailed descriptions and access points to the datasets used in our experiments. These datasets encompass a wide range of multiagent scenarios and interaction patterns, ensuring the robustness and generalizability of our findings.

- **Vital Resources**: We also share other essential resources, such as simulation environments, data visualization tools, and documentation, to support further research and development in this field.

**Experimental Results**:

- **Comparative Studies**: This section presents a comprehensive set of comparative studies, evaluating the performance of our proposed response strategies against existing methods. We conduct these studies across multiple benchmarks and problem domains to ensure a fair and rigorous comparison.
  
- **Parameter Sensitivity**: We delve deeper into the influence of parameter variations within the Algorithms section, specifically focusing on the Impact of Parameters in the PSB and ACB Algorithms. In our parameter sensitivity experiments, we adopt standard Particle Swarm Optimization (PSO) parameters: $\omega=0.7$, $c_1=1.4$, and $c_2=1.4$, to validate their suitability for the enhanced PSO algorithm presented in this work. For the ACB algorithm, $\rho$ regulates the retention of actions in new behaviors, while $\epsilon$ governs the greedy exploration of novel actions. Through these experiments, we aim to gain a comprehensive understanding of how these parameters affect the performance and behavior of our proposed algorithms.

- **Statistical Analyses**: We provide detailed statistical analyses of the experimental results, including significance tests, effect sizes, and confidence intervals. These analyses enable us to draw meaningful conclusions about the effectiveness and advantages of our proposed approaches.

- **Insights and Implications**: Finally, we discuss the key insights and implications of our findings. We highlight the limitations of existing methods and how our proposed strategies can overcome them. We also discuss the potential applications and future directions of our research.

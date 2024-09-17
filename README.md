**Project Outlines: SI-IDID**

SI-IDID represents an innovative extension of the established Ev-IDID toolkit~(please refer to https://github.com/lamingic/Ev-IDID), specifically tailored to address complex I-DID (Interactive Dynamic Influence Diagram) problems with enhanced precision and capabilities. Building upon the solid foundation laid by Ev-IDID, SI-IDID refines and augments its predecessor's functionalities, offering a superior approach to I-DID modeling.

Furthermore, the repository functions as supplementary material for our paper titled "Improved Response Strategies for Unknown Behaviours in Multiagent Interaction," authored by an esteemed team comprising Yinghui Pan (Member, IEEE), Mengen Zhou, Yifeng Zeng (Member, IEEE), Biyang Ma (Member, IEEE), Yew-soon Ong (Fellow, IEEE), and Guoquan Liu.

Within this paper, we harness two distinct simulation intelligence (SI) techniques to devise innovative behaviors for interacting agents, thereby empowering the subject agent to optimize its response strategies. These novel behaviors are designed to accurately mirror the authentic behaviors of other agents, enabling the subject agent to react appropriately and effectively. By leveraging the Interactive Dynamic Influence Diagrams framework, we undertake a theoretical analysis to explore how these algorithms impact the decision quality of the subject agent. Additionally, we empirically validate the performance of these algorithms across two widely recognized problem domains, showcasing their robustness and applicability.

**title:**
**Supplemental Material for ''Improved Response Strategies for Unknown Behaviours in Multiagent Interaction''**
In this Supplementary Materials, we detail the frameworks of PSB and ACB, PSB operators, and shared functionalities. %Sensitivity analysis $t$-tests strengthen experiments. All implementations are included in our I-DID toolkit~\footnote{https://github.com/lamingic/SI-IDID} .

A: PSB and ACB Implementation

We introduce PSB and ACB frameworks in Figs. \ref{fig:PSO} and \ref{fig:ACO}, respectively, and elaborate on PSB operators and their shared functionalities with ACB.

A.1: [The framework of  PSB and ACB]
 
Fig. \ref{fig:PSO} shows our PSB (Particle Swarm Optimization-enabled Behavior) framework. First, known policy trees become particles $\sigma=(p,v)$ forming $pop$ (\textcircled{\small{1}}). Fitness $F(\sigma)$ is calculated (\textcircled{\small{2}}) to find local best $\bar{\sigma}$ and global best $\sigma^*$ (\textcircled{\small{3}}). Over $N$ iterations, positions and velocities update (\textcircled{\small{4}}-\textcircled{\small{5}}), fitness recalculates (\textcircled{\small{6}}), and best particles update (\textcircled{\small{7}}). Particles move towards $\sigma^*$ or $\bar{\sigma}$ (\textcircled{\small{8}}). Finally, particles transform back to policy trees and decode into behaviors via $transform$ (\textcircled{\small{9}}), selecting top-$K$ based on fitness (\textcircled{\small{10}}). Behaviors, initialized with varied beliefs, optimize over iterations.

Fig. \ref{fig:ACO} presents our ACB (Ant Colony Optimization-enabled Behavior) framework. Ant populations and pheromone tables initialize (\textcircled{\small{1}}-\textcircled{\small{2}}). Pheromones update based on ant actions (\textcircled{\small{3}}-\textcircled{\small{4}}). Ants choose next actions epsilon-greedily, influenced by pheromones (\textcircled{\small{5}}). After $N$ iterations (\textcircled{\small{6}}), the population converts to policy trees, selecting top-$K$ behaviors (\textcircled{\small{7}}-\textcircled{\small{8}}).

# **Learning2Disjunctive**

This project focuses on accelerating the overall resolution time in branch-and-bound tree algorithms by leveraging machine learning for variable and node selection.

There are three primary algorithms explored:

1. **Vanilla Cutting-Plane Tree (CPT) Algorithm:**

   - For detailed information, please refer to the research paper: [Link](https://www.sciencedirect.com/science/article/pii/S0167637711001143)
   - You can find the implementation in `vanilla_CPT_alg.ipynb`.
     - In each iteration, we first **identify** the LP optimal solution within the CPT and then **choose** a variable for branching:
       - Two methods for **variable selection**:
         - *Maximum Fractionality Rule* (completed)
         - Machine Learning-based Approach

2. **Naive CPT Algorithm with Fixed Number of Branching Variables:**

   - You can find the implementation in `naive_CPT_alg.ipynb`.

   - In each iteration, we construct a **new** branch-and-bound tree based on the LP optimal solution and specific **criteria**:
     - Two approaches for **variable selection**:
       - *Maximum Fractionality Rule* (completed)
       - Machine Learning-based Method

3. **Branch-and-CPT Algorithm:**

   This part of the project is currently is developed:

   - Branch-and-Bound Framework

     - Branching Node Selection: `nodeSelectionModel`

       1. `BBR`: Best Bound Rule

       2. `DNFR`: *Deepest Node First Rule (default after an incumbent is founded)*

       3. `RAND`: *Randomly choose a node*
     - Branching Variable Selection: 
       1. *Maximum Fractionality Rule*: according to the solution info in the selected node (default)
       2. RL-based method (<font color="red">Partially solved in Naive-CPT algorithm</font>) 
   - Multi-term Disjunctive Cut Framework

     - CGLP Node Selection: `cglpNodeSelectionModel`

       1. `bound-based`: *use the nodes with the best bounds to generate cuts*
          - `nodeNumber`: the number of nodes used to generate cuts

       2. `parentnode-based`: *use the nodes from the same parent node to generate cuts. Reason: inherit the cuts from previous iterations (default)*
          - Mainly use the new nodes generated in Branching procedures

       3. `RL-based`: Machine Learning-based Method(<font color="red">Waiting</font>)


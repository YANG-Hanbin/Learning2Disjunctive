# **Learning2Disjunctive**

This project focuses on accelerating the overall resolution time in branch-and-bound tree algorithms by leveraging machine learning for variable and node selection.

There are three primary algorithms explored:

1. **Vanilla Cutting-Plane Tree (CPT) Algorithm:**

   - For detailed information, please refer to the research paper: [Link](https://www.sciencedirect.com/science/article/pii/S0167637711001143)
   - You can find the implementation in `vanilla_CPT_alg.ipynb`.
     - In each iteration, we first **identify** the LP optimal solution within the CPT and then **choose** a variable for branching:
       - Two methods for **variable selection**:
         - *Maximum Fractionality Rule* (completed)
         - Machine Learning-based Approach (<font color="red">Waiting</font>)
2. **Naive CPT Algorithm with Fixed Number of Branching Variables:**

   - You can find the implementation in `naive_CPT_alg.ipynb`.

   - In each iteration, we construct a **new** branch-and-bound tree based on the LP optimal solution and specific **criteria**:
     - Two approaches for **variable selection**:
       - *Maximum Fractionality Rule* (completed)
       - Machine Learning-based Method (<font color="red">Waiting</font>)
3. **Branch-and-Bound (B&B) with Multi-term Disjunctive Cuts (<font color="red">Waiting</font>):**

   - This part of the project is currently is developed:
     - Variable Selection:
       - *Maximum Fractionality Rule*
     - Node Selection:
       - *Best Bound Rule*
       - *Deepest Node First Rule(<font color="red">Waiting</font>)*
       - Machine Learning-based Method(<font color="red">Waiting</font>)

